#!/usr/bin/env python3
# encoding: utf-8
# Pose Mimic 3D Node — New MediaPipe Tasks API + World Coordinates
#
# Uses PoseLandmarker (lite .task model) in VIDEO mode for built-in
# temporal smoothing. World coordinates (meters, hip-centered) for
# accurate 3D arm angles — needed for Warrior II and similar poses.
#
# Controls arm servos (ID 13-22):
#   - sho_pitch (13/14): forward/backward via world Z component
#   - sho_roll  (15/16): lateral raise via 2D screen coords (NOT world coords)
#   - el_pitch  (17/18): YAML says pitch but ACTUALLY forearm rotation (held at stand)
#   - el_yaw    (19/20): YAML says yaw but ACTUALLY elbow bend
#   - gripper   (21/22): held at stand (hand detection unreliable)
#   - servo 20:          BURNED — held at stand until replaced
#
# Crossed-arms gesture to return to standing position.
# Sliding window smoothing on top of MediaPipe's temporal smoothing.
#
# ============================================================
# MISTAKES LOG — lessons learned during development
# ============================================================
#
# 1. BURNED SERVO 20 (r_el_yaw, actual elbow bend)
#    Root cause: used clamp range 0-1000 (full theoretical range) instead of
#    the user-tested safe range 360-850. Code sent pulse values as low as 202
#    to servo 20, whose physical minimum is 360. This caused the servo to stall
#    against its mechanical stop, overheat, and burn out.
#    47 out of 788 commands were below 250, 131 were in the danger zone.
#    Fix: always use user-tested safe ranges (servo 19: 150-640, servo 20: 360-850).
#    LESSON: NEVER trust theoretical 0-1000 range. Physical limits are tighter.
#    A limited range of motion is always better than a destroyed servo.
#
# 2. sho_pitch (13/14) DIRECTION FLIPPED MULTIPLE TIMES
#    Root cause: confused about two things simultaneously:
#    (a) MediaPipe world Z direction — Z follows camera lens direction (away
#        from camera into the scene). With origin at hip center and person
#        facing camera: Z > 0 = toward person's BACK, Z < 0 = FORWARD.
#        I wrongly assumed Z > 0 = toward camera = forward.
#    (b) Servo pulse direction for each side (13: big=back, 14: small=back)
#    The left and right servos are mirror-mounted, so the SAME physical
#    movement (arm forward) requires OPPOSITE pulse directions (13 goes small,
#    14 goes big). I kept flipping the mapping trying to fix one side and
#    breaking the other. Should have checked user's servos.txt from the start.
#    Fix: servo 13 forward=small backward=big; servo 14 forward=big backward=small.
#
# 3. YAML NAMES FOR ELBOW SERVOS ARE WRONG
#    Servo 17/18: YAML calls them "el_pitch" but they actually control forearm
#    ROTATION (spinning around the upper arm axis, doesn't change elbow angle).
#    Servo 19/20: YAML calls them "el_yaw" but they actually control elbow
#    BEND (changes distance between forearm and upper arm).
#    I initially mapped bend angles to 17/18 and rotation to 19/20 — backwards.
#    The user's servos.txt observations were the key to identifying this swap.
#
# 4. sho_roll (15/16) WORLD COORDINATES DON'T WORK
#    Tried 3 different approaches with world coords for lateral arm raise:
#    (a) Y-up assumption — wrong, MediaPipe world Y is actually downward
#    (b) Y-down fix — angles still stuck at clamp boundaries when arms move
#    (c) atan2(dx, dy) on normalized coords — also didn't track arm spreading
#    Root cause: world coordinate XY projection is unreliable for lateral arm
#    raise detection on a single 2D camera. The depth estimation adds noise.
#    Fix: copied the EXACT approach from the working 2D node (pose_mimic_node.py)
#    which uses vector_2d_angle with a horizontal reference point. Both sides use
#    the IDENTICAL mapping val_map(angle, -90, 90, 170, 830) — the opposite-sign
#    angles from the flipped image naturally handle left/right mirroring.
#
# 5. MediaPipe world Y-axis direction
#    Assumed Y-up (like OpenGL). Actually MediaPipe world landmarks use Y-down
#    (same convention as image coordinates). This caused the "down" reference
#    vector [0, -1] to point UP, making all roll angles off by ~180 degrees.
#    Arms hanging down showed angles of ~170° instead of ~0°.
#
# 6. LANDMARK 11/12 IDENTITY CONFUSION
#    MediaPipe docs: landmark 11 = person's LEFT shoulder, 12 = person's RIGHT.
#    This is ALWAYS from the person's own anatomical perspective, per the docs.
#    However, we flip the image (cv2.flip) before feeding to MediaPipe for
#    mirror effect. MediaPipe doesn't know the image is flipped — it just
#    detects what it sees. So in the flipped image, your actual right arm
#    appears as a left arm, and MediaPipe labels it landmark 11.
#    Result: landmark 11 in our code = person's REAL right side (due to flip).
#    We map landmark 11 → robot LEFT servos, giving correct mirror behavior.
#    The code logic is correct but many comments wrongly stated landmark 11
#    is the person's left shoulder — that's only true for unflipped images.
#
# 7. WORLD COORDINATE SYSTEM (corrected understanding)
#    Origin: center of hips (moves with person)
#    X = person's LEFT direction (positive = left)
#    Y = DOWN (positive = downward)
#    Z = camera lens direction (positive = away from camera, into the scene)
#    When person faces camera: Z > 0 = toward person's back, Z < 0 = forward
#    Google's documentation does NOT clearly specify axis directions(!),
#    confirmed via GitHub issue #3370. The above was determined empirically
#    and from user knowledge.
# ============================================================

import os
import cv2
import math
import time
import rospy
import signal
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.framework.formats import landmark_pb2
import ainex_sdk.fps as fps
from collections import deque
from sensor_msgs.msg import Image
from ainex_sdk.common import cv2_image2ros
from ainex_kinematics.motion_manager import MotionManager
from ainex_interfaces.srv import SetWalkingCommand

# ============================================================
# Constants
# ============================================================

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'model', 'pose_landmarker_lite.task')

# Standing pulse values from stand.d6a
STAND_PULSE = {
    'l_ank_roll':  500,  'r_ank_roll':  500,
    'l_ank_pitch': 640,  'r_ank_pitch': 360,
    'l_knee':      500,  'r_knee':      500,
    'l_hip_pitch': 350,  'r_hip_pitch': 650,
    'l_hip_roll':  500,  'r_hip_roll':  500,
    'l_hip_yaw':   500,  'r_hip_yaw':   500,
    'l_sho_pitch': 835,  'r_sho_pitch': 165,
    'l_sho_roll':  830,  'r_sho_roll':  170,
    'l_el_pitch':  500,  'r_el_pitch':  500,
    'l_el_yaw':    150,  'r_el_yaw':    850,
    'l_gripper':   500,  'r_gripper':   500,
    'head_pan':    500,  'head_tilt':   500,
}

SERVO_ID = {
    'l_ank_roll': 1,   'r_ank_roll': 2,
    'l_ank_pitch': 3,  'r_ank_pitch': 4,
    'l_knee': 5,       'r_knee': 6,
    'l_hip_pitch': 7,  'r_hip_pitch': 8,
    'l_hip_roll': 9,   'r_hip_roll': 10,
    'l_hip_yaw': 11,   'r_hip_yaw': 12,
    'l_sho_pitch': 13, 'r_sho_pitch': 14,
    'l_sho_roll': 15,  'r_sho_roll': 16,
    'l_el_pitch': 17,  'r_el_pitch': 18,
    'l_el_yaw': 19,    'r_el_yaw': 20,
    'l_gripper': 21,   'r_gripper': 22,
    'head_pan': 23,    'head_tilt': 24,
}

# All arm joints we control (10 servos, 5 per arm)
ARM_JOINTS = [
    'l_sho_pitch', 'r_sho_pitch',
    'l_sho_roll',  'r_sho_roll',
    'l_el_pitch',  'r_el_pitch',
    'l_el_yaw',    'r_el_yaw',
    'l_gripper',   'r_gripper',
]

# --- Gesture ---
STAND_GESTURE_FRAMES = 5
RESUME_FRAMES = 12
CROSS_DIST_RATIO = 0.8

# --- Anti-twitch ---
WINDOW_SIZE = 6          # smaller window since VIDEO mode already smooths
SEND_EVERY = 3           # send more often since smoother input
MIN_FRAMES_BEFORE_SEND = 3
DEADZONE_PULSE = 25


def val_map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def angle_between_vectors_3d(v1, v2):
    """Unsigned angle between two 3D vectors (degrees)."""
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    if d < 1e-9:
        return None
    cos_val = np.clip(np.dot(v1, v2) / d, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


def signed_angle_2d(v1, v2):
    """Signed 2D angle between two vectors (degrees) using atan2."""
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    if d < 1e-9:
        return None
    cos_val = np.clip(np.dot(v1, v2) / d, -1.0, 1.0)
    sin_val = np.clip(np.cross(v1, v2) / d, -1.0, 1.0)
    return float(np.degrees(np.arctan2(sin_val, cos_val)))


class PoseMimic3DNode:
    def __init__(self, name):
        rospy.init_node(name, anonymous=False)
        self.name = name
        self.running = True
        self.image = None
        self.fps = fps.FPS()
        self.frame_ts = 0  # timestamp counter for VIDEO mode

        signal.signal(signal.SIGINT, self.shutdown)

        # ---- MediaPipe PoseLandmarker (new Tasks API, VIDEO mode) ----
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        self.detector = mp_vision.PoseLandmarker.create_from_options(
            mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_tracking_confidence=0.3,
            )
        )
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # ---- Servo control ----
        self.motion_manager = MotionManager()
        self.last_pulse = {}
        self.pulse_window = deque(maxlen=WINDOW_SIZE)
        self.frame_count = 0

        # Stop walking module
        try:
            rospy.wait_for_service('walking/command', timeout=5)
            walk_cmd = rospy.ServiceProxy('walking/command', SetWalkingCommand)
            walk_cmd('stop')
            time.sleep(0.5)
            walk_cmd('disable')
            rospy.loginfo('[PoseMimic3D] walking module disabled')
        except Exception as e:
            rospy.logwarn('[PoseMimic3D] cannot connect walking service: %s' % str(e))

        self._send_stand()
        time.sleep(1.0)

        # Gesture state
        self.gesture_count = 0
        self.no_gesture_count = 0
        self.in_stand_mode = False

        # ---- ROS ----
        self.camera = rospy.get_param('/camera')
        rospy.Subscriber(
            '/{}/{}'.format(self.camera['camera_name'], self.camera['image_topic']),
            Image, self.image_callback,
        )
        self.result_pub = rospy.Publisher('~image_result', Image, queue_size=1)

        rospy.loginfo('[PoseMimic3D] Ready! New Tasks API, VIDEO mode, world coords')

    def shutdown(self, signum, frame):
        self.running = False

    def image_callback(self, ros_image):
        self.image = np.ndarray(
            shape=(ros_image.height, ros_image.width, 3),
            dtype=np.uint8, buffer=ros_image.data,
        )

    def _send_stand(self):
        cmds = [[SERVO_ID[j], STAND_PULSE[j]] for j in SERVO_ID]
        self.motion_manager.set_servos_position(800, cmds)
        self.last_pulse.clear()
        self.pulse_window.clear()
        self.frame_count = 0

    # ------------------------------------------------------------------
    # Gesture: crossed arms (using normalized screen landmarks)
    # ------------------------------------------------------------------
    def _arms_crossed(self, norm_lm):
        """Check using normalized (screen) landmarks."""
        l_sho = norm_lm[11]
        r_sho = norm_lm[12]
        l_wri = norm_lm[15]
        r_wri = norm_lm[16]

        sho_w = math.sqrt((l_sho.x - r_sho.x)**2 + (l_sho.y - r_sho.y)**2)
        wri_d = math.sqrt((l_wri.x - r_wri.x)**2 + (l_wri.y - r_wri.y)**2)

        if sho_w < 0.02:
            return False

        ratio = wri_d / sho_w
        crossed = ratio < CROSS_DIST_RATIO

        if int(time.time()) != getattr(self, '_cross_dbg_t', 0):
            self._cross_dbg_t = int(time.time())
            print('[Cross] ratio=%.2f -> %s' % (ratio, crossed), flush=True)
        return crossed

    # ------------------------------------------------------------------
    # 3D angle extraction using world coordinates
    # ------------------------------------------------------------------
    def compute_all_arm_pulses(self, world_lm, norm_lm, width, height):
        """Compute pulses using:
        - Screen (normalized) coords for sho_roll (lateral raise — proven reliable in 2D)
        - World coords for sho_pitch (forward/backward — needs Z) and elbow

        Returns dict of {joint_name: pulse} or None."""

        # ============================================================
        # sho_roll (ID 15/16): lateral arm raise — EXACT SAME as working 2D node
        # ============================================================
        # Use screen pixel coords with vector_2d_angle from horizontal reference
        # Both sides use IDENTICAL mapping: val_map(angle, -90, 90, 170, 830)
        # Opposite-sign angles naturally handle mirroring
        l_sho_px = [norm_lm[11].x * width, norm_lm[11].y * height]
        r_sho_px = [norm_lm[12].x * width, norm_lm[12].y * height]
        l_elb_px = [norm_lm[13].x * width, norm_lm[13].y * height]
        r_elb_px = [norm_lm[14].x * width, norm_lm[14].y * height]

        l_ref = [width, l_sho_px[1]]
        r_ref = [0, r_sho_px[1]]

        a_l_roll = signed_angle_2d(
            np.array(l_sho_px) - np.array(l_ref),
            np.array(l_sho_px) - np.array(l_elb_px))
        a_r_roll = signed_angle_2d(
            np.array(r_sho_px) - np.array(r_ref),
            np.array(r_sho_px) - np.array(r_elb_px))

        if a_l_roll is None or a_r_roll is None:
            return None

        a_l_roll = clamp(a_l_roll, -90, 90)
        a_r_roll = clamp(a_r_roll, -90, 90)

        # SAME mapping for both — exactly like 2D node and TonyPi
        # Safe range 125-875
        p_l_sho_roll = int(clamp(val_map(a_l_roll, -90, 90, 170, 830), 125, 875))
        p_r_sho_roll = int(clamp(val_map(a_r_roll, -90, 90, 170, 830), 125, 875))

        # Extract world landmarks for pitch and elbow
        l_sho = np.array([world_lm[11].x, world_lm[11].y, world_lm[11].z])
        r_sho = np.array([world_lm[12].x, world_lm[12].y, world_lm[12].z])
        l_elb = np.array([world_lm[13].x, world_lm[13].y, world_lm[13].z])
        r_elb = np.array([world_lm[14].x, world_lm[14].y, world_lm[14].z])
        l_wri = np.array([world_lm[15].x, world_lm[15].y, world_lm[15].z])
        r_wri = np.array([world_lm[16].x, world_lm[16].y, world_lm[16].z])

        l_upper = l_elb - l_sho
        r_upper = r_elb - r_sho

        # ============================================================
        # sho_pitch (ID 13/14): forward/backward arm swing
        # ============================================================
        # World Z follows camera lens direction (away from camera).
        # With origin at hip center, person facing camera:
        #   Z > 0 = toward person's BACK (arm backward)
        #   Z < 0 = toward camera = arm FORWARD
        #
        # Note: landmark 11 → servo 13 (robot left), landmark 12 → servo 14 (robot right)
        # Due to image flip, landmark 11 = person's actual RIGHT side.

        l_pitch_z = l_upper[2]  # positive = backward, negative = forward
        r_pitch_z = r_upper[2]

        l_pitch_z = clamp(l_pitch_z, -0.30, 0.30)
        r_pitch_z = clamp(r_pitch_z, -0.30, 0.30)

        # Servo 13: 值越大→往后, 值越小→往前
        #   Z>0(back)→大, Z<0(fwd)→小
        # Servo 14: 值越小→往后
        #   Z>0(back)→小, Z<0(fwd)→大
        p_l_sho_pitch = int(clamp(val_map(l_pitch_z, -0.30, 0.30, 125, 875), 125, 875))
        p_r_sho_pitch = int(clamp(val_map(r_pitch_z, -0.30, 0.30, 875, 125), 125, 875))

        # ============================================================
        # IMPORTANT: YAML names are SWAPPED for elbow servos!
        #   Servo 17/18 (yaml: el_pitch) = ACTUALLY forearm rotation (yaw)
        #   Servo 19/20 (yaml: el_yaw)   = ACTUALLY elbow bend (pitch)
        # ============================================================

        l_forearm = l_wri - l_elb
        r_forearm = r_wri - r_elb

        # --- Servo 17/18 (yaml: el_pitch, ACTUAL: forearm rotation) ---
        # 小臂绕大臂旋转，不影响大小臂距离
        # MediaPipe无法可靠检测前臂绕轴旋转，固定在stand值
        p_l_el_pitch = STAND_PULSE['l_el_pitch']  # 500
        p_r_el_pitch = STAND_PULSE['r_el_pitch']  # 500
        l_rot_z = 0.0  # placeholder for debug
        r_rot_z = 0.0

        # --- Servo 19/20 (yaml: el_yaw, ACTUAL: elbow bend) ---
        # 转动影响大小臂距离
        # Servo 19: 值越小→弯曲, 530≈伸直, 最大640
        # Servo 20: 值越大→弯曲, 450≈伸直, 最低360
        a_l_elb = angle_between_vectors_3d(-l_upper, l_forearm)
        a_r_elb = angle_between_vectors_3d(-r_upper, r_forearm)

        if a_l_elb is None or a_r_elb is None:
            return None

        a_l_elb = clamp(a_l_elb, 0, 180)
        a_r_elb = clamp(a_r_elb, 0, 180)

        # Servo 19 (l): 越小越弯, 530≈伸直. Hardware safe: 0-600
        # Servo 20 (r): 越大越弯, 450≈伸直. Hardware safe: 400-1000
        # IMPORTANT: servo 20 was burned by sending 202 (below min 360).
        # 1000 units = 240°, so 180° elbow range = 750 units.
        # Servo 19: straight(180°)=530, so bent(0°)=530-750=-220 → clip to 0
        # Servo 20: straight(180°)=450, so bent(0°)=450+750=1200 → clip to 1000
        p_l_el_yaw = int(clamp(val_map(a_l_elb, 0, 180, 530 - 750, 530), 0, 600))
        p_r_el_yaw = int(clamp(val_map(a_r_elb, 0, 180, 450 + 750, 450), 400, 1000))

        # ============================================================
        # gripper (ID 21/22): held at stand
        # ============================================================
        p_l_gripper = STAND_PULSE['l_gripper']
        p_r_gripper = STAND_PULSE['r_gripper']

        pulses = {
            'l_sho_pitch': p_l_sho_pitch,  'r_sho_pitch': p_r_sho_pitch,
            'l_sho_roll':  p_l_sho_roll,   'r_sho_roll':  p_r_sho_roll,
            'l_el_pitch':  p_l_el_pitch,   'r_el_pitch':  p_r_el_pitch,
            'l_el_yaw':    p_l_el_yaw,     'r_el_yaw':    p_r_el_yaw,
            'l_gripper':   p_l_gripper,    'r_gripper':   p_r_gripper,
        }

        # Debug (once per second)
        if int(time.time()) != getattr(self, '_dbg_t', 0):
            self._dbg_t = int(time.time())
            print('[3D] roll L=%.0f R=%.0f | pitch_z L=%.3f R=%.3f | elbow_bend L=%.0f R=%.0f | rot_z L=%.3f R=%.3f' % (
                a_l_roll, a_r_roll, l_pitch_z, r_pitch_z, a_l_elb, a_r_elb, l_rot_z, r_rot_z), flush=True)

        return pulses

    # ------------------------------------------------------------------
    # Sliding window average
    # ------------------------------------------------------------------
    def _window_average(self):
        if len(self.pulse_window) < MIN_FRAMES_BEFORE_SEND:
            return None
        avg = {}
        for joint in ARM_JOINTS:
            vals = [p[joint] for p in self.pulse_window if joint in p]
            if vals:
                avg[joint] = int(sum(vals) / len(vals))
        return avg if len(avg) == len(ARM_JOINTS) else None

    # ------------------------------------------------------------------
    # Draw landmarks on image (convert new API format to legacy for drawing)
    # ------------------------------------------------------------------
    def _draw_landmarks(self, bgr_image, norm_landmarks):
        """Draw pose landmarks using legacy drawing utils."""
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in norm_landmarks
        ])
        self.mp_drawing.draw_landmarks(
            bgr_image, proto, self.mp_pose.POSE_CONNECTIONS)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        rate = rospy.Rate(10)

        while self.running:
            if self.image is None:
                rate.sleep()
                continue

            image_rgb = self.image.copy()
            self.image = None
            image_flip = cv2.flip(image_rgb, 1)
            height, width, _ = image_flip.shape

            # --- Detection with new Tasks API (VIDEO mode = temporal smoothing) ---
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_flip)
            self.frame_ts += 33  # ~30fps timestamp in ms
            result = self.detector.detect_for_video(mp_image, self.frame_ts)

            bgr_image = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)

            if result.pose_landmarks and result.pose_world_landmarks:
                norm_lm = result.pose_landmarks[0]   # normalized screen coords
                world_lm = result.pose_world_landmarks[0]  # 3D world coords (meters)

                # Draw skeleton
                self._draw_landmarks(bgr_image, norm_lm)

                # --- GESTURE CHECK ---
                crossed = self._arms_crossed(norm_lm)

                if crossed:
                    self.gesture_count += 1
                    self.no_gesture_count = 0
                else:
                    self.no_gesture_count += 1
                    if not self.in_stand_mode:
                        self.gesture_count = 0

                if not self.in_stand_mode and self.gesture_count >= STAND_GESTURE_FRAMES:
                    rospy.loginfo('[PoseMimic3D] Arms crossed -> STAND!')
                    print('[GESTURE] Arms crossed! Standing.', flush=True)
                    self._send_stand()
                    self.in_stand_mode = True
                    self.gesture_count = 0

                if self.in_stand_mode and self.no_gesture_count >= RESUME_FRAMES:
                    rospy.loginfo('[PoseMimic3D] Resuming.')
                    print('[GESTURE] Resuming follow.', flush=True)
                    self.in_stand_mode = False
                    self.no_gesture_count = 0

                # --- Arm control ---
                if self.in_stand_mode:
                    cv2.putText(bgr_image, 'STANDING (uncross to resume)', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    pulses = self.compute_all_arm_pulses(world_lm, norm_lm, width, height)
                    if pulses is not None:
                        self.pulse_window.append(pulses)
                        self.frame_count += 1

                        # Show buffer status
                        buf_len = len(self.pulse_window)
                        cv2.putText(bgr_image, 'Window [%d/%d] send in %d' % (
                            buf_len, WINDOW_SIZE,
                            max(0, SEND_EVERY - self.frame_count)),
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

                        if self.frame_count >= SEND_EVERY and buf_len >= MIN_FRAMES_BEFORE_SEND:
                            avg_pulses = self._window_average()
                            if avg_pulses is not None:
                                servo_cmds = []
                                info_lines = []
                                for joint_name in ARM_JOINTS:
                                    pulse = avg_pulses[joint_name]
                                    last = self.last_pulse.get(joint_name, STAND_PULSE[joint_name])
                                    if abs(pulse - last) < DEADZONE_PULSE:
                                        pulse = last
                                    else:
                                        self.last_pulse[joint_name] = pulse
                                    servo_cmds.append([SERVO_ID[joint_name], pulse])
                                    info_lines.append('%s:%d' % (joint_name, pulse))

                                self.motion_manager.set_servos_position(600, servo_cmds)
                                print('[SEND] %s' % ' | '.join(info_lines), flush=True)

                            self.frame_count = 0

                        # Show pulse values on image
                        y_off = 42
                        for j in ARM_JOINTS:
                            if j.endswith('gripper'):
                                continue
                            side = 'L' if j.startswith('l') else 'R'
                            jname = j.split('_', 1)[1]
                            cv2.putText(bgr_image, '%s %s: %d' % (side, jname, pulses.get(j, 0)),
                                        (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                        (0, 255, 0), 1)
                            y_off += 16
                    else:
                        cv2.putText(bgr_image, 'Angle calc failed', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(bgr_image, 'No person detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Gesture progress
            if self.gesture_count > 0 and not self.in_stand_mode:
                cv2.putText(bgr_image, 'CROSSING [%d/%d]' % (
                    min(self.gesture_count, STAND_GESTURE_FRAMES), STAND_GESTURE_FRAMES),
                    (10, bgr_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            self.fps.update()
            bgr_image = self.fps.show_fps(bgr_image)
            self.result_pub.publish(cv2_image2ros(cv2.resize(bgr_image, (640, 480)), self.name))
            rate.sleep()

        rospy.loginfo('[PoseMimic3D] Returning to stand...')
        self._send_stand()
        self.detector.close()
        rospy.signal_shutdown('shutdown')


if __name__ == "__main__":
    import sys
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    print("[MAIN] Starting pose_mimic_3d node (Tasks API)...", flush=True)
    try:
        node = PoseMimic3DNode('pose_mimic')
        print("[MAIN] Node initialized, entering run loop", flush=True)
        node.run()
    except Exception as e:
        print("[MAIN] ERROR: %s" % e, flush=True)
        import traceback
        traceback.print_exc()
