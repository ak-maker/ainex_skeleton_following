#!/usr/bin/env python3
# encoding: utf-8
# Skeleton Pose Mimic v3
#
# Standing pulse values from stand.d6a
# No head servos (not present on hardware)
# Stand gesture: cross both wrists together in front of chest
#   - Detected via Pose landmarks (no separate hand model needed)
#   - Hold crossed wrists for ~1s -> robot returns to stand
#   - Separate wrists -> resume skeleton following
# pulse = STAND_PULSE[joint] + offset * coef

import cv2
import math
import time
import rospy
import signal
import numpy as np
import mediapipe as mp
import ainex_sdk.fps as fps
from sensor_msgs.msg import Image
from ainex_sdk.common import cv2_image2ros
from ainex_kinematics.motion_manager import MotionManager
from ainex_interfaces.srv import SetWalkingCommand

# ============================================================
# Constants
# ============================================================
ENCODER_TICKS_PER_RADIAN = 180 / math.pi / 240 * 1000  # ~238.73

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
}

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
}

JOINT_COEF = {
    'l_ank_roll':   ENCODER_TICKS_PER_RADIAN,
    'r_ank_roll':   ENCODER_TICKS_PER_RADIAN,
    'l_ank_pitch':  ENCODER_TICKS_PER_RADIAN,
    'r_ank_pitch':  ENCODER_TICKS_PER_RADIAN,
    'l_knee':       ENCODER_TICKS_PER_RADIAN,
    'r_knee':       ENCODER_TICKS_PER_RADIAN,
    'l_hip_pitch':  ENCODER_TICKS_PER_RADIAN,
    'r_hip_pitch':  ENCODER_TICKS_PER_RADIAN,
    'l_hip_roll':   ENCODER_TICKS_PER_RADIAN,
    'r_hip_roll':   ENCODER_TICKS_PER_RADIAN,
    'l_hip_yaw':    ENCODER_TICKS_PER_RADIAN,
    'r_hip_yaw':    ENCODER_TICKS_PER_RADIAN,
    'l_sho_pitch': -ENCODER_TICKS_PER_RADIAN,
    'r_sho_pitch': -ENCODER_TICKS_PER_RADIAN,
    'l_sho_roll':   ENCODER_TICKS_PER_RADIAN,
    'r_sho_roll':   ENCODER_TICKS_PER_RADIAN,
    'l_el_pitch':   ENCODER_TICKS_PER_RADIAN,
    'r_el_pitch':   ENCODER_TICKS_PER_RADIAN,
    'l_el_yaw':     ENCODER_TICKS_PER_RADIAN,
    'r_el_yaw':     ENCODER_TICKS_PER_RADIAN,
    'l_gripper':    ENCODER_TICKS_PER_RADIAN,
    'r_gripper':    ENCODER_TICKS_PER_RADIAN,
}

SERVO_PULSE_MIN = 50
SERVO_PULSE_MAX = 950

CONTROLLED_JOINTS = [
    'l_sho_pitch', 'r_sho_pitch', 'l_sho_roll', 'r_sho_roll',
    'l_el_pitch', 'r_el_pitch', 'l_el_yaw', 'r_el_yaw',
    'l_hip_pitch', 'r_hip_pitch', 'l_hip_roll', 'r_hip_roll',
    'l_hip_yaw', 'r_hip_yaw',
    'l_knee', 'r_knee',
    'l_ank_pitch', 'r_ank_pitch', 'l_ank_roll', 'r_ank_roll',
]

# How many consecutive frames to trigger stand gesture
STAND_GESTURE_FRAMES = 5
# Frames needed to resume following after stand
RESUME_FRAMES = 12
# Crossed arms: wrist-to-wrist distance threshold (normalized, relative to shoulder width)
CROSS_DIST_RATIO = 0.8  # wrist dist < 0.8 * shoulder_width = crossed

# --- Anti-twitch settings ---
# Accumulate N frames and average before sending (reduces MediaPipe noise)
ACCUM_FRAMES = 8
# Deadzone: ignore offset changes smaller than this from last sent value (radians)
DEADZONE_RAD = 0.08  # ~4.6 degrees


def offset_to_pulse(joint_name, offset_rad):
    pulse = STAND_PULSE[joint_name] + int(round(offset_rad * JOINT_COEF[joint_name]))
    return max(SERVO_PULSE_MIN, min(SERVO_PULSE_MAX, pulse))


def vec_angle(v1, v2):
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return math.acos(np.clip(cos_a, -1.0, 1.0))


def points_angle(a, b, c):
    return vec_angle(np.array(a) - np.array(b), np.array(c) - np.array(b))


class SkeletonMimicNode:
    def __init__(self, name):
        rospy.init_node(name, anonymous=False)
        self.name = name
        self.running = True
        self.image = None
        self.fps = fps.FPS()

        signal.signal(signal.SIGINT, self.shutdown)

        # ---- MediaPipe Pose ----
        # NOTE: static_image_mode=True required on Raspberry Pi
        # (streaming mode fails to detect on this ARM platform)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.PL = self.mp_pose.PoseLandmark
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.3,
        )

        # ---- Servo control ----
        self.motion_manager = MotionManager()
        self.last_offsets = {}

        # Stop walking module
        try:
            rospy.wait_for_service('walking/command', timeout=5)
            walk_cmd = rospy.ServiceProxy('walking/command', SetWalkingCommand)
            walk_cmd('stop')
            time.sleep(0.5)
            walk_cmd('disable')
            rospy.loginfo('[SkeletonMimic] walking module disabled')
        except Exception as e:
            rospy.logwarn('[SkeletonMimic] cannot connect walking service: %s' % str(e))

        # Go to standing position
        self._send_stand()
        time.sleep(1.0)

        self.smooth_factor = 0.15
        self.servo_duration = 200

        # Gesture state
        self.gesture_count = 0
        self.no_gesture_count = 0
        self.in_stand_mode = False

        # Frame accumulation: average N frames before sending
        self.accum_buf = []          # list of offset dicts
        self.committed_offsets = {}  # last offsets actually sent to servos

        # ---- ROS ----
        self.camera = rospy.get_param('/camera')
        rospy.Subscriber(
            '/{}/{}'.format(self.camera['camera_name'], self.camera['image_topic']),
            Image, self.image_callback,
        )
        self.result_pub = rospy.Publisher('~image_result', Image, queue_size=1)

        rospy.loginfo('[SkeletonMimic] Ready!')
        rospy.loginfo('[SkeletonMimic] Cross arms in front -> stand. Uncross -> resume follow.')

    def shutdown(self, signum, frame):
        self.running = False

    def image_callback(self, ros_image):
        self.image = np.ndarray(
            shape=(ros_image.height, ros_image.width, 3),
            dtype=np.uint8, buffer=ros_image.data,
        )

    def lm(self, landmarks, idx):
        p = landmarks[idx]
        return [p.x, p.y, p.z], p.visibility

    def _send_stand(self):
        """Send all servos to stand.d6a values."""
        cmds = [[SERVO_ID[j], STAND_PULSE[j]] for j in SERVO_ID]
        self.motion_manager.set_servos_position(800, cmds)
        self.last_offsets.clear()

    # ------------------------------------------------------------------
    # Gesture: crossed arms in front of chest
    # ------------------------------------------------------------------
    def _arms_crossed(self, landmarks):
        """Return True if both wrists are close together (arms crossed in front).
        Uses shoulder width as reference to be distance-independent."""
        MIN_VIS = 0.3
        PL = self.PL
        l_wri, l_wri_v = self.lm(landmarks, PL.LEFT_WRIST.value)
        r_wri, r_wri_v = self.lm(landmarks, PL.RIGHT_WRIST.value)
        l_sho, l_sho_v = self.lm(landmarks, PL.LEFT_SHOULDER.value)
        r_sho, r_sho_v = self.lm(landmarks, PL.RIGHT_SHOULDER.value)

        if min(l_wri_v, r_wri_v, l_sho_v, r_sho_v) < MIN_VIS:
            return False

        shoulder_w = math.sqrt((l_sho[0] - r_sho[0])**2 + (l_sho[1] - r_sho[1])**2)
        wrist_dist = math.sqrt((l_wri[0] - r_wri[0])**2 + (l_wri[1] - r_wri[1])**2)

        if shoulder_w < 0.01:
            return False

        ratio = wrist_dist / shoulder_w
        crossed = ratio < CROSS_DIST_RATIO

        if int(time.time()) != getattr(self, '_cross_dbg_t', 0):
            self._cross_dbg_t = int(time.time())
            print('[Cross] wrist_dist=%.3f sho_w=%.3f ratio=%.2f thresh=%.2f -> %s' % (
                wrist_dist, shoulder_w, ratio, CROSS_DIST_RATIO, crossed), flush=True)
        return crossed

    # ------------------------------------------------------------------
    # Angle extraction
    # ------------------------------------------------------------------
    def extract_offsets(self, landmarks):
        result = {}
        MIN_VIS = 0.4

        l_sho, l_sho_v = self.lm(landmarks, self.PL.LEFT_SHOULDER.value)
        r_sho, r_sho_v = self.lm(landmarks, self.PL.RIGHT_SHOULDER.value)
        l_elb, l_elb_v = self.lm(landmarks, self.PL.LEFT_ELBOW.value)
        r_elb, r_elb_v = self.lm(landmarks, self.PL.RIGHT_ELBOW.value)
        l_wri, l_wri_v = self.lm(landmarks, self.PL.LEFT_WRIST.value)
        r_wri, r_wri_v = self.lm(landmarks, self.PL.RIGHT_WRIST.value)
        l_hip, l_hip_v = self.lm(landmarks, self.PL.LEFT_HIP.value)
        r_hip, r_hip_v = self.lm(landmarks, self.PL.RIGHT_HIP.value)
        l_kne, l_kne_v = self.lm(landmarks, self.PL.LEFT_KNEE.value)
        r_kne, r_kne_v = self.lm(landmarks, self.PL.RIGHT_KNEE.value)
        l_ank, l_ank_v = self.lm(landmarks, self.PL.LEFT_ANKLE.value)
        r_ank, r_ank_v = self.lm(landmarks, self.PL.RIGHT_ANKLE.value)

        # ---- Left arm ----
        if l_sho_v > MIN_VIS and l_elb_v > MIN_VIS:
            torso_yz = np.array([0, l_hip[1] - l_sho[1], l_hip[2] - l_sho[2]])
            arm_yz = np.array([0, l_elb[1] - l_sho[1], l_elb[2] - l_sho[2]])
            pitch = vec_angle(torso_yz, arm_yz)
            if l_elb[2] < l_sho[2]:
                result['l_sho_pitch'] = pitch
            else:
                result['l_sho_pitch'] = -pitch

            torso_xy = np.array([0, l_hip[1] - l_sho[1]])
            arm_xy = np.array([l_elb[0] - l_sho[0], l_elb[1] - l_sho[1]])
            roll = vec_angle(torso_xy, arm_xy)
            result['l_sho_roll'] = roll

        if l_sho_v > MIN_VIS and l_elb_v > MIN_VIS and l_wri_v > MIN_VIS:
            elbow = points_angle(l_sho, l_elb, l_wri)
            result['l_el_pitch'] = -(math.pi - elbow)

        # ---- Right arm ----
        if r_sho_v > MIN_VIS and r_elb_v > MIN_VIS:
            torso_yz = np.array([0, r_hip[1] - r_sho[1], r_hip[2] - r_sho[2]])
            arm_yz = np.array([0, r_elb[1] - r_sho[1], r_elb[2] - r_sho[2]])
            pitch = vec_angle(torso_yz, arm_yz)
            if r_elb[2] < r_sho[2]:
                result['r_sho_pitch'] = pitch
            else:
                result['r_sho_pitch'] = -pitch

            torso_xy = np.array([0, r_hip[1] - r_sho[1]])
            arm_xy = np.array([r_elb[0] - r_sho[0], r_elb[1] - r_sho[1]])
            roll = vec_angle(torso_xy, arm_xy)
            result['r_sho_roll'] = -roll

        if r_sho_v > MIN_VIS and r_elb_v > MIN_VIS and r_wri_v > MIN_VIS:
            elbow = points_angle(r_sho, r_elb, r_wri)
            result['r_el_pitch'] = (math.pi - elbow)

        # ---- Left leg ----
        if l_hip_v > MIN_VIS and l_kne_v > MIN_VIS:
            torso_up = np.array([0, l_sho[1] - l_hip[1], l_sho[2] - l_hip[2]])
            thigh_yz = np.array([0, l_kne[1] - l_hip[1], l_kne[2] - l_hip[2]])
            hip_pitch = vec_angle(torso_up, thigh_yz)
            human_offset = hip_pitch - math.pi
            if l_kne[2] < l_hip[2]:
                human_offset = -abs(human_offset)
            result['l_hip_pitch'] = human_offset

            torso_down_xy = np.array([0, 1])
            thigh_xy = np.array([l_kne[0] - l_hip[0], l_kne[1] - l_hip[1]])
            hip_roll = vec_angle(torso_down_xy, thigh_xy)
            if l_kne[0] < l_hip[0]:
                hip_roll = abs(hip_roll)
            else:
                hip_roll = -abs(hip_roll)
            result['l_hip_roll'] = hip_roll

        if l_hip_v > MIN_VIS and l_kne_v > MIN_VIS and l_ank_v > MIN_VIS:
            knee_angle = points_angle(l_hip, l_kne, l_ank)
            knee_bend = math.pi - knee_angle
            result['l_knee'] = knee_bend
            result['l_ank_pitch'] = knee_bend * 0.5

        # ---- Right leg ----
        if r_hip_v > MIN_VIS and r_kne_v > MIN_VIS:
            torso_up = np.array([0, r_sho[1] - r_hip[1], r_sho[2] - r_hip[2]])
            thigh_yz = np.array([0, r_kne[1] - r_hip[1], r_kne[2] - r_hip[2]])
            hip_pitch = vec_angle(torso_up, thigh_yz)
            human_offset = hip_pitch - math.pi
            if r_kne[2] < r_hip[2]:
                human_offset = -abs(human_offset)
            result['r_hip_pitch'] = -human_offset

            torso_down_xy = np.array([0, 1])
            thigh_xy = np.array([r_kne[0] - r_hip[0], r_kne[1] - r_hip[1]])
            hip_roll = vec_angle(torso_down_xy, thigh_xy)
            if r_kne[0] > r_hip[0]:
                hip_roll = -abs(hip_roll)
            else:
                hip_roll = abs(hip_roll)
            result['r_hip_roll'] = hip_roll

        if r_hip_v > MIN_VIS and r_kne_v > MIN_VIS and r_ank_v > MIN_VIS:
            knee_angle = points_angle(r_hip, r_kne, r_ank)
            knee_bend = math.pi - knee_angle
            result['r_knee'] = -knee_bend
            result['r_ank_pitch'] = -knee_bend * 0.5

        return result

    def smooth_offset(self, joint_name, target_rad):
        if joint_name in self.last_offsets:
            smoothed = self.last_offsets[joint_name] + self.smooth_factor * (target_rad - self.last_offsets[joint_name])
        else:
            smoothed = target_rad
        self.last_offsets[joint_name] = smoothed
        return smoothed

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        rate = rospy.Rate(12)

        while self.running:
            if self.image is None:
                rate.sleep()
                continue

            image_rgb = self.image.copy()
            self.image = None

            image_flip = cv2.flip(image_rgb, 1)

            # --- Pose detection ---
            results = self.pose.process(image_flip)

            bgr_image = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)

            # --- GESTURE CHECK: highest priority, every frame ---
            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark
                crossed = self._arms_crossed(lm)

                if crossed:
                    self.gesture_count += 1
                    self.no_gesture_count = 0
                else:
                    self.no_gesture_count += 1
                    if not self.in_stand_mode:
                        self.gesture_count = 0

                # Thumbs up detected -> IMMEDIATELY stand
                if not self.in_stand_mode and self.gesture_count >= STAND_GESTURE_FRAMES:
                    rospy.loginfo('[SkeletonMimic] Arms crossed -> STAND!')
                    print('[GESTURE] Arms crossed! Standing.', flush=True)
                    self._send_stand()
                    self.in_stand_mode = True
                    self.gesture_count = 0
                    self.accum_buf.clear()

                # In stand mode, wait for no thumbs up for a while to resume
                if self.in_stand_mode and self.no_gesture_count >= RESUME_FRAMES:
                    rospy.loginfo('[SkeletonMimic] Resuming skeleton follow.')
                    print('[GESTURE] Resuming follow.', flush=True)
                    self.in_stand_mode = False
                    self.no_gesture_count = 0

            if self.in_stand_mode:
                cv2.putText(bgr_image, 'STANDING (uncross arms to resume)', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            elif results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    bgr_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )

                lm = results.pose_landmarks.landmark
                offsets = self.extract_offsets(lm)

                # --- Accumulate frames and average to reduce noise ---
                self.accum_buf.append(offsets)
                info_lines = []
                if len(self.accum_buf) < ACCUM_FRAMES:
                    # Not enough frames yet, skip sending
                    pass
                else:
                    # Average the accumulated offsets
                    avg_offsets = {}
                    for jn in CONTROLLED_JOINTS:
                        vals = [f.get(jn, 0.0) for f in self.accum_buf]
                        avg_offsets[jn] = sum(vals) / len(vals)
                    self.accum_buf.clear()

                    servo_cmds = []
                    info_lines = []
                    any_change = False

                    for joint_name in CONTROLLED_JOINTS:
                        offset_rad = self.smooth_offset(joint_name, avg_offsets[joint_name])

                        # Deadzone: skip tiny changes from last sent value
                        committed = self.committed_offsets.get(joint_name, 0.0)
                        if abs(offset_rad - committed) < DEADZONE_RAD:
                            offset_rad = committed
                        else:
                            any_change = True

                        pulse = offset_to_pulse(joint_name, offset_rad)
                        servo_cmds.append([SERVO_ID[joint_name], pulse])

                        if avg_offsets.get(joint_name, 0.0) != 0.0:
                            info_lines.append(
                                '{}: {:.0f}d p:{}'.format(
                                    joint_name, math.degrees(offset_rad), pulse
                                )
                            )

                    servo_cmds.append([SERVO_ID['l_gripper'], STAND_PULSE['l_gripper']])
                    servo_cmds.append([SERVO_ID['r_gripper'], STAND_PULSE['r_gripper']])

                    if any_change and servo_cmds:
                        self.motion_manager.set_servos_position(self.servo_duration, servo_cmds)
                        self.committed_offsets = {jn: self.last_offsets.get(jn, 0.0)
                                                  for jn in CONTROLLED_JOINTS}
                        print('[Send] %d joints' % len(servo_cmds), flush=True)

                for i, line in enumerate(info_lines):
                    cv2.putText(bgr_image, line, (10, 22 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            else:
                cv2.putText(bgr_image, 'No person detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show accumulation indicator
            if not self.in_stand_mode and results.pose_landmarks:
                n = len(self.accum_buf)
                if n > 0:
                    cv2.putText(bgr_image, 'BUF [{}/{}]'.format(n, ACCUM_FRAMES),
                                (bgr_image.shape[1] - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    cv2.putText(bgr_image, 'FOLLOWING', (bgr_image.shape[1] - 150, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Show gesture progress
            if self.gesture_count > 0 and not self.in_stand_mode:
                cv2.putText(bgr_image, 'CROSSING [{}/{}]'.format(
                    min(self.gesture_count, STAND_GESTURE_FRAMES), STAND_GESTURE_FRAMES),
                    (10, bgr_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            self.fps.update()
            bgr_image = self.fps.show_fps(bgr_image)
            self.result_pub.publish(cv2_image2ros(cv2.resize(bgr_image, (640, 480)), self.name))

            rate.sleep()

        rospy.loginfo('[SkeletonMimic] Returning to stand...')
        self._send_stand()
        self.pose.close()
        rospy.signal_shutdown('shutdown')


if __name__ == "__main__":
    import sys
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # line-buffered
    print("[MAIN] Starting skeleton_mimic node...", flush=True)
    try:
        node = SkeletonMimicNode('skeleton_mimic')
        print("[MAIN] Node initialized, entering run loop", flush=True)
        node.run()
    except Exception as e:
        print("[MAIN] ERROR: %s" % e, flush=True)
        import traceback
        traceback.print_exc()
