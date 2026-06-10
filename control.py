#!/usr/bin/env python3
"""
Head Tracker — Algorithm 1 (quasi-IBVS)

Subscribes to /subject_tracker/landmarks published by publisher.py.
Every UPDATE_EVERY seconds, computes the average 2D body center across
the buffered frames, then nudges head_pan/head_tilt to chase it.

Body center = intersection (in XY image plane) of:
  - left_shoulder  (lm 11) → right_hip (lm 24)
  - right_shoulder (lm 12) → left_hip  (lm 23)

Safe servo ranges: pan [300, 700], tilt [400, 600].
"""
import json
import math
import time
import threading

import rospy
from std_msgs.msg import String
from ainex_kinematics.motion_manager import MotionManager

# ── Tunable parameters ────────────────────────────────────────────────────────
TRACKING_DURATION = 5.0    # seconds per skeleton capture cycle
STABILITY_N       = 3      # Gaussian neighbor half-window for stability scoring
STABILITY_SIGMA   = 1.0    # Gaussian σ for stability scoring
REQUIRED_LANDMARKS = {11, 12, 13, 14, 15, 16}  # shoulders, elbows, wrists

UPDATE_EVERY  = 0.8    # seconds between head-move commands (must be > motor move time)
MOVE_DURATION = 600    # ms passed to set_servos_position for head moves
STEP          = 100    # servo units per unit of normalised image error (gain)

PAN_CENTER  = 500
TILT_CENTER = 500

PAN_MIN,  PAN_MAX  = 300, 700
TILT_MIN, TILT_MAX = 400, 600

# Servo IDs for head joints
HEAD_PAN_ID  = 23
HEAD_TILT_ID = 24

# Minimum per-landmark presence/visibility to accept a frame
PRESENCE_THRESH    = 0.4
VISIBILITY_THRESH  = 0.4

# Landmark indices (MediaPipe BlazePose)
L_SHOULDER = 11
R_SHOULDER = 12
L_HIP      = 23
R_HIP      = 24

L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16

MAIN_LANDMARKS = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]

# Arm servo IDs and positions
LEFT_ARM_SERVO_ID  = 15
RIGHT_ARM_SERVO_ID = 16
LEFT_ARM_RAISED_POS  = 500
LEFT_ARM_LOWERED_POS = 830
RIGHT_ARM_RAISED_POS  = 500
RIGHT_ARM_LOWERED_POS = 170

# Non-main landmarks grouped by BFS hop-distance from MAIN_LANDMARKS.
# Precomputed once so each callback is O(n) list scan, no graph work.
# dist=None (face cluster, 0-10) is treated as lowest priority.
FALLBACK_BY_PRIORITY = [
    [13, 14, 25, 26],                            # 1 hop  (sho/hip neighbours)
    [15, 16, 27, 28],                            # 2 hops (elbow/knee)
    [17, 18, 19, 20, 21, 22, 29, 30, 31, 32],   # 3 hops (wrist/ankle/foot)
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],         # disconnected (face)
]
# ─────────────────────────────────────────────────────────────────────────────


# ── Skeleton stability ────────────────────────────────────────────────────────


def _frame_distance(a, b):
    total = 0.0
    for la, lb in zip(a, b):
        dx, dy, dz = la['x']-lb['x'], la['y']-lb['y'], la['z']-lb['z']
        total += math.sqrt(dx*dx + dy*dy + dz*dz)
    return total

    # Below is a version implemented with only required landmarks
    # total = 0.0
    # for i in REQUIRED_LANDMARKS:
    #     la, lb = a[i], b[i]
    #     dx, dy, dz = la['x']-lb['x'], la['y']-lb['y'], la['z']-lb['z']
    #     total += math.sqrt(dx*dx + dy*dy + dz*dz)
    # return total


def _stability_score(frames, idx, N, sigma):
    """
    Looks at the N frames before and after and computers 
    
    """
    score = 0.0
    for k in range(-N, N+1):
        if k == 0:
            continue
        j = idx + k
        if 0 <= j < len(frames):
            score += _frame_distance(frames[idx], frames[j]) * math.exp(-k*k / (2.0*sigma*sigma))
    return score

def most_stable_frame(frames, N=STABILITY_N, sigma=STABILITY_SIGMA):
    if len(frames) < 2*N+1:
        return len(frames) // 2
    scores = [_stability_score(frames, i, N, sigma) for i in range(N, len(frames)-N)]
    return N + scores.index(min(scores))

def _is_valid_skel_frame(norm_lm):
    return bool(norm_lm) and all(_lm_valid(norm_lm, i) for i in REQUIRED_LANDMARKS if i < len(norm_lm))


# ── Head tracking geometry ────────────────────────────────────────────────────

def _line_intersection_2d(ax, ay, bx, by, cx, cy, dx, dy):
    """
    Return the XY intersection of lines AB and CD, or None if parallel.
    Uses parametric form: P = A + t*(B-A), t = ((C-A) x (D-C)) / ((B-A) x (D-C)).
    """
    dab_x, dab_y = bx - ax, by - ay
    dcd_x, dcd_y = dx - cx, dy - cy

    denom = dab_x * dcd_y - dab_y * dcd_x
    if abs(denom) < 1e-9:
        return None

    t = ((cx - ax) * dcd_y - (cy - ay) * dcd_x) / denom
    return ax + t * dab_x, ay + t * dab_y


def _lm_valid(norm_lm, idx):
    if idx >= len(norm_lm):
        return False
    lm = norm_lm[idx]
    return (lm.get('presence', 0) >= PRESENCE_THRESH and
            lm.get('visibility', 0) >= VISIBILITY_THRESH)


def _body_center(norm_lm, last_center=(0.5, 0.5)):
    """
    Return the best 2D body-center estimate, with graceful fallback:
      1. Diagonal intersection of all four main torso landmarks (best).
      2. Centroid of whichever main landmarks are visible.
      3. Single non-main landmark at the lowest hop-distance from main,
         tiebroken by Euclidean distance to last_center.
    Returns (cx, cy) in normalised [0,1] coords, or None if nothing visible.
    """
    # ── 1. Full intersection ─────────────────────────────────────────────────
    if all(_lm_valid(norm_lm, i) for i in MAIN_LANDMARKS):
        ls = norm_lm[L_SHOULDER]
        rs = norm_lm[R_SHOULDER]
        lh = norm_lm[L_HIP]
        rh = norm_lm[R_HIP]
        pt = _line_intersection_2d(
            ls['x'], ls['y'], rh['x'], rh['y'],
            rs['x'], rs['y'], lh['x'], lh['y'],
        )
        if pt is not None:
            return pt

    # ── 2. Centroid of visible main landmarks ────────────────────────────────
    visible_main = [
        (norm_lm[i]['x'], norm_lm[i]['y'])
        for i in MAIN_LANDMARKS if _lm_valid(norm_lm, i)
    ]
    if visible_main:
        return (
            sum(p[0] for p in visible_main) / len(visible_main),
            sum(p[1] for p in visible_main) / len(visible_main),
        )

    # ── 3. Closest visible non-main landmark at lowest priority level ────────
    cx, cy = last_center
    for group in FALLBACK_BY_PRIORITY:
        candidates = [
            (norm_lm[i]['x'], norm_lm[i]['y'])
            for i in group if _lm_valid(norm_lm, i)
        ]
        if candidates:
            return min(candidates, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)

    return None


class HeadTracker:
    def __init__(self):
        rospy.init_node('head_tracker', anonymous=False)

        self.motion_manager = MotionManager()

        self._lock         = threading.Lock()
        self._centers      = []          # list of (cx, cy) collected this cycle
        self._last_center  = (0.5, 0.5) # last computed average, used for fallback tiebreak
        self._cycle_start  = time.time()

        self._pan  = PAN_CENTER
        self._tilt = TILT_CENTER

        # Move to centre on startup
        self.motion_manager.set_servos_position(
            MOVE_DURATION,
            [[HEAD_PAN_ID, self._pan], [HEAD_TILT_ID, self._tilt]],
        )

        self._skel_pub      = rospy.Publisher('/head_tracker/skeleton',  String, queue_size=1)
        self._cycle_end_pub = rospy.Publisher('/head_tracker/cycle_end', String, queue_size=1)
        self._skel_buffer      = []
        self._current_skel     = None
        self._skel_cycle       = 0
        self._skel_cycle_start = time.time()

        rospy.Subscriber('/stable_gest/landmarks', String, self._landmarks_cb)
        rospy.loginfo('[HeadTracker] ready — update every %.2fs, step=%d', UPDATE_EVERY, STEP)

    def _landmarks_cb(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        
        rospy.loginfo("recevied landmark")

        norm_lm = payload.get('norm_landmarks', [])
        center  = _body_center(norm_lm, self._last_center)

        with self._lock:
            now = time.time()

            # ── Skeleton cycle rollover ──────────────────────────────────────
            if now - self._skel_cycle_start >= TRACKING_DURATION:
                final_skel = self._current_skel
                self._cycle_end_pub.publish(json.dumps({
                    'cycle':     self._skel_cycle,
                    'skeleton':  final_skel,
                    'had_human': final_skel is not None,
                }))
                if final_skel:
                    self._apply_arm_poses(final_skel)
                self._skel_buffer      = []
                self._current_skel     = None
                self._skel_cycle      += 1
                self._skel_cycle_start = now

                

            # ── Skeleton frame ingestion ─────────────────────────────────────
            if _is_valid_skel_frame(norm_lm):
                self._skel_buffer.append(norm_lm)
                self._current_skel = self._skel_buffer[most_stable_frame(self._skel_buffer)]
                self._skel_pub.publish(json.dumps(self._current_skel))

            # ── Head tracking cycle ──────────────────────────────────────────
            if center is not None:
                self._centers.append(center)
            if now - self._cycle_start >= UPDATE_EVERY:
                self._update()
                self._centers     = []
                self._cycle_start = now

    def _apply_arm_poses(self, skel):
        """At cycle end, match arm servos to whether each wrist is raised above its shoulder."""
        def arm_raised(shoulder_idx, wrist_idx):
            if not (_lm_valid(skel, shoulder_idx) and _lm_valid(skel, wrist_idx)):
                return False
            return skel[wrist_idx]['y'] < skel[shoulder_idx]['y']

        left_raised  = arm_raised(L_SHOULDER, L_WRIST)
        right_raised = arm_raised(R_SHOULDER, R_WRIST)

        self.motion_manager.set_servos_position(MOVE_DURATION, [
            [LEFT_ARM_SERVO_ID,  LEFT_ARM_RAISED_POS  if left_raised  else LEFT_ARM_LOWERED_POS],
            [RIGHT_ARM_SERVO_ID, RIGHT_ARM_RAISED_POS if right_raised else RIGHT_ARM_LOWERED_POS],
        ])
        rospy.loginfo('[HeadTracker] arms: left=%s right=%s',
                      'raised' if left_raised else 'down',
                      'raised' if right_raised else 'down')

    def _update(self):
        """Compute average body center, derive error, nudge head. Called under lock."""
        if not self._centers:
            return

        avg_cx = sum(c[0] for c in self._centers) / len(self._centers)
        avg_cy = sum(c[1] for c in self._centers) / len(self._centers)
        self._last_center = (avg_cx, avg_cy)

        # Image center in normalised coords is (0.5, 0.5).
        # Positive horiz error → subject is to the right → pan should increase.
        # Positive vert error  → subject is below centre → tilt should decrease
        # (tilt value decreases as head looks down on Ainex).
        horiz_err = avg_cx - 0.5
        vert_err  = avg_cy - 0.5

        new_pan  = int(self._pan  + STEP * horiz_err)
        new_tilt = int(self._tilt - STEP * vert_err)

        new_pan  = max(PAN_MIN,  min(PAN_MAX,  new_pan))
        new_tilt = max(TILT_MIN, min(TILT_MAX, new_tilt))

        if new_pan != self._pan or new_tilt != self._tilt:
            self.motion_manager.set_servos_position(
                MOVE_DURATION,
                [[HEAD_PAN_ID, new_pan], [HEAD_TILT_ID, new_tilt]],
            )
            rospy.loginfo(
                '[HeadTracker] body=(%.3f,%.3f) err=(%.3f,%.3f) pan %d→%d tilt %d→%d',
                avg_cx, avg_cy, horiz_err, vert_err,
                self._pan, new_pan, self._tilt, new_tilt,
            )
            self._pan  = new_pan
            self._tilt = new_tilt

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    import sys
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    print('[HeadTracker] starting...', flush=True)
    try:
        node = HeadTracker()
        node.run()
    except Exception as e:
        print('[HeadTracker] ERROR: %s' % e, flush=True)
        import traceback
        traceback.print_exc()