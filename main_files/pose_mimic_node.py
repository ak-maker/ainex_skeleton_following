#!/usr/bin/env python3
# encoding: utf-8
# Pose Mimic Node v3 — 2D approach with sliding window smoothing
#
# Uses MediaPipe Pose with 2D pixel angles (TonyPi style).
# Sliding window (8 frames) for smooth, responsive servo control.
# Crossed-arms gesture to return to standing position.

import cv2
import math
import copy
import time
import rospy
import signal
import numpy as np
import mediapipe as mp
import ainex_sdk.fps as fps
from collections import deque
from sensor_msgs.msg import Image
from ainex_sdk.common import cv2_image2ros
from ainex_kinematics.motion_manager import MotionManager
from ainex_interfaces.srv import SetWalkingCommand

# ============================================================
# Constants
# ============================================================

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

# Joints we actually control
# NOTE: 2D camera angle maps to sho_ROLL (arm up/down), NOT sho_pitch (forward/backward)
ARM_JOINTS = ['l_sho_roll', 'r_sho_roll', 'l_el_pitch', 'r_el_pitch']

# --- Gesture settings ---
STAND_GESTURE_FRAMES = 5
RESUME_FRAMES = 12
CROSS_DIST_RATIO = 0.8

# --- Anti-twitch: sliding window ---
WINDOW_SIZE = 8          # sliding window of last 8 frames
SEND_EVERY = 4           # send averaged command every 4 frames (~1s)
MIN_FRAMES_BEFORE_SEND = 4  # need at least 4 frames before first send
DEADZONE_PULSE = 30      # ignore changes smaller than this (like TonyPi's 30)

# Arm segment lengths (for safety check, like TonyPi)
L1 = 0.06
L2 = 0.11


def val_map(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def vector_2d_angle(v1, v2):
    """Signed 2D angle between two vectors (degrees)."""
    d = np.linalg.norm(v1) * np.linalg.norm(v2)
    if d == 0:
        return None
    cos_val = np.clip(np.dot(v1, v2) / d, -1.0, 1.0)
    sin_val = np.clip(np.cross(v1, v2) / d, -1.0, 1.0)
    return float(np.degrees(np.arctan2(sin_val, cos_val)))


class PoseMimicNode:
    def __init__(self, name):
        rospy.init_node(name, anonymous=False)
        self.name = name
        self.running = True
        self.image = None
        self.fps = fps.FPS()

        signal.signal(signal.SIGINT, self.shutdown)

        # ---- MediaPipe Pose (lighter than Holistic, reliable on RPi) ----
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

        # Last sent pulse values (for deadzone)
        self.last_pulse = {}

        # Sliding window: stores pulse dicts, does NOT clear after sending
        self.pulse_window = deque(maxlen=WINDOW_SIZE)
        self.frame_count = 0  # counts frames since last send

        # Stop walking module
        try:
            rospy.wait_for_service('walking/command', timeout=5)
            walk_cmd = rospy.ServiceProxy('walking/command', SetWalkingCommand)
            walk_cmd('stop')
            time.sleep(0.5)
            walk_cmd('disable')
            rospy.loginfo('[PoseMimic] walking module disabled')
        except Exception as e:
            rospy.logwarn('[PoseMimic] cannot connect walking service: %s' % str(e))

        # Go to standing position
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

        rospy.loginfo('[PoseMimic] Ready! Sliding window=%d, send every %d frames' % (
            WINDOW_SIZE, SEND_EVERY))
        rospy.loginfo('[PoseMimic] Cross arms -> stand. Uncross -> resume.')

    def shutdown(self, signum, frame):
        self.running = False

    def image_callback(self, ros_image):
        self.image = np.ndarray(
            shape=(ros_image.height, ros_image.width, 3),
            dtype=np.uint8, buffer=ros_image.data,
        )

    def _send_stand(self):
        """Send all servos to stand.d6a values."""
        cmds = [[SERVO_ID[j], STAND_PULSE[j]] for j in SERVO_ID]
        self.motion_manager.set_servos_position(800, cmds)
        self.last_pulse.clear()
        self.pulse_window.clear()
        self.frame_count = 0

    # ------------------------------------------------------------------
    # Gesture: crossed arms
    # ------------------------------------------------------------------
    def _arms_crossed(self, marks):
        l_sho, r_sho = marks[0], marks[1]
        l_wri, r_wri = marks[4], marks[5]

        sho_w = math.sqrt((l_sho[0] - r_sho[0])**2 + (l_sho[1] - r_sho[1])**2)
        wri_d = math.sqrt((l_wri[0] - r_wri[0])**2 + (l_wri[1] - r_wri[1])**2)

        if sho_w < 10:
            return False

        ratio = wri_d / sho_w
        crossed = ratio < CROSS_DIST_RATIO

        if int(time.time()) != getattr(self, '_cross_dbg_t', 0):
            self._cross_dbg_t = int(time.time())
            print('[Cross] wri=%.0f sho=%.0f ratio=%.2f -> %s' % (
                wri_d, sho_w, ratio, crossed), flush=True)
        return crossed

    # ------------------------------------------------------------------
    # 2D angle extraction (TonyPi style) with safety check
    # ------------------------------------------------------------------
    def compute_arm_angles(self, marks, width):
        """Compute 2D angles for arms. Returns (angles_dict, safety_dict) or (None, None).
        angles_dict: raw angles in degrees (clamped to ±90)
        safety_dict: {'left_ok': bool, 'right_ok': bool} based on x>0 check"""

        l_sho, r_sho = marks[0], marks[1]
        l_elb, r_elb = marks[2], marks[3]
        l_wri, r_wri = marks[4], marks[5]

        # Reference: horizontal from shoulder
        l_ref = [width, l_sho[1]]
        r_ref = [0, r_sho[1]]

        # Raw angles
        a_l_sho = vector_2d_angle(
            np.array(l_sho) - np.array(l_ref),
            np.array(l_sho) - np.array(l_elb))
        a_l_elb = vector_2d_angle(
            np.array(l_elb) - np.array(l_sho),
            np.array(l_wri) - np.array(l_elb))
        a_r_sho = vector_2d_angle(
            np.array(r_sho) - np.array(r_ref),
            np.array(r_sho) - np.array(r_elb))
        a_r_elb = vector_2d_angle(
            np.array(r_elb) - np.array(r_sho),
            np.array(r_wri) - np.array(r_elb))

        if None in (a_l_sho, a_l_elb, a_r_sho, a_r_elb):
            return None, None

        # Clamp angles to [-90, 90] BEFORE mapping (prevents out-of-range pulses)
        a_l_sho = clamp(a_l_sho, -90, 90)
        a_l_elb = clamp(a_l_elb, -90, 90)
        a_r_sho = clamp(a_r_sho, -90, 90)
        a_r_elb = clamp(a_r_elb, -90, 90)

        # Safety check: arm endpoint must be in front of body (like TonyPi)
        x_left = L1 * math.cos(math.radians(a_l_sho)) + L2 * math.cos(
            math.radians(a_l_elb) + math.radians(a_l_sho))
        x_right = L1 * math.cos(math.radians(a_r_sho)) + L2 * math.cos(
            math.radians(a_r_elb) + math.radians(a_r_sho))

        angles = {
            'l_sho_roll': a_l_sho,
            'r_sho_roll': a_r_sho,
            'l_el_pitch': a_l_elb,
            'r_el_pitch': a_r_elb,
        }
        safety = {'left_ok': x_left > 0, 'right_ok': x_right > 0}
        return angles, safety

    def angles_to_pulses(self, angles):
        """Convert clamped angles to servo pulses.
        sho_roll controls arm up/down (what camera sees).
        Both sho_roll use SAME mapping — opposite-sign angles handle mirroring.
          Left arm:  +90(down)→830, 0(horizontal)→500, -90(up)→170
          Right arm: -90(down)→170, 0(horizontal)→500, +90(up)→830"""
        # Shoulders → sho_roll (ID 15/16), range 170-830
        p_l_sho = int(clamp(val_map(angles['l_sho_roll'], -90, 90, 170, 830), 125, 875))
        p_r_sho = int(clamp(val_map(angles['r_sho_roll'], -90, 90, 170, 830), 125, 875))
        # Elbows (ID 17/18), both same direction
        p_l_elb = int(clamp(val_map(angles['l_el_pitch'], -90, 90, 125, 875), 125, 875))
        p_r_elb = int(clamp(val_map(angles['r_el_pitch'], -90, 90, 125, 875), 125, 875))

        return {
            'l_sho_roll': p_l_sho,
            'r_sho_roll': p_r_sho,
            'l_el_pitch': p_l_elb,
            'r_el_pitch': p_r_elb,
        }

    # ------------------------------------------------------------------
    # Sliding window average
    # ------------------------------------------------------------------
    def _window_average(self):
        """Average pulse values across the sliding window."""
        if len(self.pulse_window) < MIN_FRAMES_BEFORE_SEND:
            return None
        avg = {}
        for joint in ARM_JOINTS:
            vals = [p[joint] for p in self.pulse_window if joint in p]
            if vals:
                avg[joint] = int(sum(vals) / len(vals))
        return avg if len(avg) == len(ARM_JOINTS) else None

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

            # --- Pose detection (no CLAHE, raw image) ---
            results = self.pose.process(image_flip)
            bgr_image = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Draw full skeleton
                self.mp_drawing.draw_landmarks(
                    bgr_image, results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS)

                # Extract arm landmarks as pixel coords
                indices = [
                    self.PL.LEFT_SHOULDER.value,
                    self.PL.RIGHT_SHOULDER.value,
                    self.PL.LEFT_ELBOW.value,
                    self.PL.RIGHT_ELBOW.value,
                    self.PL.LEFT_WRIST.value,
                    self.PL.RIGHT_WRIST.value,
                ]
                marks = []
                all_visible = True
                for idx in indices:
                    p = lm[idx]
                    if p.visibility < 0.3:
                        all_visible = False
                    marks.append([int(p.x * width), int(p.y * height)])

                # --- GESTURE CHECK ---
                crossed = self._arms_crossed(marks) if all_visible else False

                if crossed:
                    self.gesture_count += 1
                    self.no_gesture_count = 0
                else:
                    self.no_gesture_count += 1
                    if not self.in_stand_mode:
                        self.gesture_count = 0

                if not self.in_stand_mode and self.gesture_count >= STAND_GESTURE_FRAMES:
                    rospy.loginfo('[PoseMimic] Arms crossed -> STAND!')
                    print('[GESTURE] Arms crossed! Standing.', flush=True)
                    self._send_stand()
                    self.in_stand_mode = True
                    self.gesture_count = 0

                if self.in_stand_mode and self.no_gesture_count >= RESUME_FRAMES:
                    rospy.loginfo('[PoseMimic] Resuming.')
                    print('[GESTURE] Resuming follow.', flush=True)
                    self.in_stand_mode = False
                    self.no_gesture_count = 0

                # --- Arm control ---
                if self.in_stand_mode:
                    cv2.putText(bgr_image, 'STANDING (uncross to resume)', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                elif all_visible:
                    # Highlight arm joints
                    for m in marks:
                        cv2.circle(bgr_image, tuple(m), 8, (0, 255, 255), -1)

                    angles, safety = self.compute_arm_angles(marks, width)
                    if angles is not None:
                        # Only include joints whose arm passes safety check
                        pulses = self.angles_to_pulses(angles)
                        safe_pulses = {}
                        if safety['left_ok']:
                            safe_pulses['l_sho_roll'] = pulses['l_sho_roll']
                            safe_pulses['l_el_pitch'] = pulses['l_el_pitch']
                        if safety['right_ok']:
                            safe_pulses['r_sho_roll'] = pulses['r_sho_roll']
                            safe_pulses['r_el_pitch'] = pulses['r_el_pitch']

                        if safe_pulses:
                            # Add to sliding window (fills missing joints with last known)
                            full_pulses = {}
                            for j in ARM_JOINTS:
                                if j in safe_pulses:
                                    full_pulses[j] = safe_pulses[j]
                                elif self.pulse_window:
                                    full_pulses[j] = self.pulse_window[-1].get(
                                        j, STAND_PULSE[j])
                                else:
                                    full_pulses[j] = STAND_PULSE[j]
                            self.pulse_window.append(full_pulses)
                            self.frame_count += 1

                        # Show buffer status
                        buf_len = len(self.pulse_window)
                        cv2.putText(bgr_image, 'Window [%d/%d] send in %d' % (
                            buf_len, WINDOW_SIZE,
                            max(0, SEND_EVERY - self.frame_count)),
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

                        # Send every SEND_EVERY frames
                        if self.frame_count >= SEND_EVERY and buf_len >= MIN_FRAMES_BEFORE_SEND:
                            avg_pulses = self._window_average()
                            if avg_pulses is not None:
                                servo_cmds = []
                                info_lines = []

                                for joint_name in ARM_JOINTS:
                                    pulse = avg_pulses[joint_name]
                                    # Deadzone
                                    last = self.last_pulse.get(joint_name, STAND_PULSE[joint_name])
                                    if abs(pulse - last) < DEADZONE_PULSE:
                                        pulse = last
                                    else:
                                        self.last_pulse[joint_name] = pulse

                                    servo_cmds.append([SERVO_ID[joint_name], pulse])
                                    info_lines.append('%s:%d' % (joint_name, pulse))

                                self.motion_manager.set_servos_position(600, servo_cmds)
                                # Debug: show angles and pulses
                                angle_str = 'Lroll=%.0f Rroll=%.0f Lelb=%.0f Relb=%.0f' % (
                                    angles.get('l_sho_roll', 0), angles.get('r_sho_roll', 0),
                                    angles.get('l_el_pitch', 0), angles.get('r_el_pitch', 0))
                                print('[SEND] %s | %s' % (angle_str, ' | '.join(info_lines)), flush=True)

                            self.frame_count = 0  # reset send counter (NOT clearing window!)

                        # Show angles and safety on image
                        if angles is not None:
                            y_off = 42
                            for j in ARM_JOINTS:
                                p = pulses.get(j, 0) if 'pulses' in dir() else 0
                                a = angles.get(j.replace('_pitch', '').replace('sho', 'sho_pitch').replace('el', 'el_pitch'), 0)
                                side = 'L' if j.startswith('l') else 'R'
                                ok = safety['left_ok'] if j.startswith('l') else safety['right_ok']
                                cv2.putText(bgr_image, '%s %s: %d %s' % (
                                    side, j.split('_')[2], pulses.get(j, 0),
                                    'OK' if ok else 'SKIP'),
                                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (0, 255, 0) if ok else (0, 0, 255), 1)
                                y_off += 18
                    else:
                        cv2.putText(bgr_image, 'Angle calc failed', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(bgr_image, 'Arms not fully visible', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
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

        rospy.loginfo('[PoseMimic] Returning to stand...')
        self._send_stand()
        self.pose.close()
        rospy.signal_shutdown('shutdown')


if __name__ == "__main__":
    import sys
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    print("[MAIN] Starting pose_mimic node v3...", flush=True)
    try:
        node = PoseMimicNode('pose_mimic')
        print("[MAIN] Node initialized, entering run loop", flush=True)
        node.run()
    except Exception as e:
        print("[MAIN] ERROR: %s" % e, flush=True)
        import traceback
        traceback.print_exc()
