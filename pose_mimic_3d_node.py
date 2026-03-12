#!/usr/bin/env python3
# encoding: utf-8
# Pose Mimic Node 3D — 2D angles + Z depth for full arm control (ID 13-22)
#
# Based on pose_mimic_node.py (2D only, 4 servos).
# Adds MediaPipe Z-coordinate for depth-dependent joints:
#   - sho_pitch (13/14): forward/backward arm swing via elbow Z vs shoulder Z
#   - el_yaw (19/20): forearm rotation via wrist Z vs elbow Z
#   - gripper (21/22): held at stand values (hand detection unreliable at distance)
#
# Sliding window smoothing. Crossed-arms gesture for standing.

import cv2
import math
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

# Servo config from servo_controller.yaml:
#   sho_pitch (13/14): init=875/125, min=1000, max=0 → REVERSED, coef=-238.73
#   sho_roll  (15/16): init=500,     min=0,    max=1000 → NORMAL
#   el_pitch  (17/18): init=500,     min=0,    max=1000 → NORMAL
#   el_yaw    (19/20): init=500,     min=0,    max=1000 → NORMAL
#   gripper   (21/22): init=500,     min=0,    max=1000 → NORMAL

# All arm joints we control (10 servos, 5 per arm)
ARM_JOINTS = [
    'l_sho_pitch', 'r_sho_pitch',  # Z-depth based (forward/backward)
    'l_sho_roll',  'r_sho_roll',   # 2D angle based (lateral raise)
    'l_el_pitch',  'r_el_pitch',   # 2D angle based (elbow bend)
    'l_el_yaw',    'r_el_yaw',     # Z-depth based (forearm rotation)
    'l_gripper',   'r_gripper',    # Held at stand (no reliable detection)
]

# --- Gesture settings ---
STAND_GESTURE_FRAMES = 5
RESUME_FRAMES = 12
CROSS_DIST_RATIO = 0.8

# --- Anti-twitch: sliding window ---
WINDOW_SIZE = 8
SEND_EVERY = 4
MIN_FRAMES_BEFORE_SEND = 4
DEADZONE_PULSE = 30

# Arm segment lengths (for safety check, like TonyPi)
L1 = 0.06
L2 = 0.11

# Z-depth mapping sensitivity
# MediaPipe Z is normalized roughly to the same scale as X.
# Typical elbow-shoulder Z difference: -0.3 (forward) to +0.3 (backward)
Z_PITCH_RANGE = 0.30   # Z difference that maps to full pitch range
Z_YAW_RANGE = 0.25     # Z difference for full yaw range


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


class PoseMimic3DNode:
    def __init__(self, name):
        rospy.init_node(name, anonymous=False)
        self.name = name
        self.running = True
        self.image = None
        self.fps = fps.FPS()

        signal.signal(signal.SIGINT, self.shutdown)

        # ---- MediaPipe Pose ----
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

        rospy.loginfo('[PoseMimic3D] Ready! Full arm control (ID 13-22) with Z-depth')
        rospy.loginfo('[PoseMimic3D] Cross arms -> stand. Uncross -> resume.')

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
    def _arms_crossed(self, marks_2d):
        l_sho, r_sho = marks_2d[0], marks_2d[1]
        l_wri, r_wri = marks_2d[4], marks_2d[5]

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
    # Compute all arm pulses: 2D angles + Z depth
    # ------------------------------------------------------------------
    def compute_all_arm_pulses(self, marks_2d, marks_z, width):
        """Compute servo pulses for all 10 arm joints.

        marks_2d: pixel coords [l_sho, r_sho, l_elb, r_elb, l_wri, r_wri]
        marks_z:  Z values      [l_sho_z, r_sho_z, l_elb_z, r_elb_z, l_wri_z, r_wri_z]

        Returns dict of {joint_name: pulse} or None."""

        l_sho, r_sho = marks_2d[0], marks_2d[1]
        l_elb, r_elb = marks_2d[2], marks_2d[3]
        l_wri, r_wri = marks_2d[4], marks_2d[5]

        l_sho_z, r_sho_z = marks_z[0], marks_z[1]
        l_elb_z, r_elb_z = marks_z[2], marks_z[3]
        l_wri_z, r_wri_z = marks_z[4], marks_z[5]

        # ==== 2D ANGLES (for sho_roll and el_pitch) ====
        l_ref = [width, l_sho[1]]
        r_ref = [0, r_sho[1]]

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
            return None

        a_l_sho = clamp(a_l_sho, -90, 90)
        a_l_elb = clamp(a_l_elb, -90, 90)
        a_r_sho = clamp(a_r_sho, -90, 90)
        a_r_elb = clamp(a_r_elb, -90, 90)

        # Safety check
        x_left = L1 * math.cos(math.radians(a_l_sho)) + L2 * math.cos(
            math.radians(a_l_elb) + math.radians(a_l_sho))
        x_right = L1 * math.cos(math.radians(a_r_sho)) + L2 * math.cos(
            math.radians(a_r_elb) + math.radians(a_r_sho))

        # --- sho_roll (ID 15/16): 2D lateral raise ---
        # Left: +90(down)→830, 0(horizontal)→500, -90(up)→170
        p_l_sho_roll = int(clamp(val_map(a_l_sho, -90, 90, 170, 830), 125, 875))
        p_r_sho_roll = int(clamp(val_map(a_r_sho, -90, 90, 170, 830), 125, 875))

        # --- el_pitch (ID 17/18): 2D elbow bend ---
        p_l_el_pitch = int(clamp(val_map(a_l_elb, -90, 90, 125, 875), 125, 875))
        p_r_el_pitch = int(clamp(val_map(a_r_elb, -90, 90, 125, 875), 125, 875))

        # ==== Z DEPTH (for sho_pitch and el_yaw) ====
        # MediaPipe Z: negative = closer to camera (forward), positive = away (backward)
        # Z values are relative to hip midpoint depth.

        # --- sho_pitch (ID 13/14): forward/backward arm swing ---
        # Use elbow Z relative to shoulder Z.
        # dz < 0 means elbow is forward of shoulder (arm reaching forward)
        # dz > 0 means elbow is behind shoulder (arm reaching backward)
        dz_l_pitch = l_elb_z - l_sho_z  # negative = forward
        dz_r_pitch = r_elb_z - r_sho_z

        # Clamp Z difference to ±Z_PITCH_RANGE, then map to pulse
        dz_l_pitch = clamp(dz_l_pitch, -Z_PITCH_RANGE, Z_PITCH_RANGE)
        dz_r_pitch = clamp(dz_r_pitch, -Z_PITCH_RANGE, Z_PITCH_RANGE)

        # l_sho_pitch: init=875, reversed servo. Stand=835 (arm at side, dz≈0)
        #   arm forward (dz negative) → pulse decreases from 835
        #   arm backward (dz positive) → pulse increases toward 875+
        # Map: dz -Z_PITCH_RANGE(forward) → 500, 0(neutral) → 835, +Z_PITCH_RANGE(back) → 875
        # Actually: full forward should go lower than 500, full backward capped at 875
        p_l_sho_pitch = int(clamp(val_map(dz_l_pitch, -Z_PITCH_RANGE, Z_PITCH_RANGE, 165, 875), 125, 875))

        # r_sho_pitch: init=125, reversed servo. Stand=165 (arm at side, dz≈0)
        #   arm forward (dz negative) → pulse increases from 165
        #   arm backward (dz positive) → pulse decreases toward 125
        # Mirrored: same mapping works because dz signs are opposite for same physical motion
        # Wait — actually both arms reaching forward have NEGATIVE dz. So we need opposite mapping for right.
        p_r_sho_pitch = int(clamp(val_map(dz_r_pitch, -Z_PITCH_RANGE, Z_PITCH_RANGE, 835, 125), 125, 875))

        # --- el_yaw (ID 19/20): forearm rotation ---
        # Use wrist Z relative to elbow Z.
        # dz < 0 means wrist forward of elbow (forearm rotated inward)
        # dz > 0 means wrist behind elbow (forearm rotated outward)
        dz_l_yaw = l_wri_z - l_elb_z
        dz_r_yaw = r_wri_z - r_elb_z

        dz_l_yaw = clamp(dz_l_yaw, -Z_YAW_RANGE, Z_YAW_RANGE)
        dz_r_yaw = clamp(dz_r_yaw, -Z_YAW_RANGE, Z_YAW_RANGE)

        # l_el_yaw: stand=150. Range 0-1000, init=500.
        # At rest (dz≈0), forearm is in neutral. Map dz to yaw rotation.
        p_l_el_yaw = int(clamp(val_map(dz_l_yaw, -Z_YAW_RANGE, Z_YAW_RANGE, 125, 875), 125, 875))
        # r_el_yaw: stand=850. Mirrored from left.
        p_r_el_yaw = int(clamp(val_map(dz_r_yaw, -Z_YAW_RANGE, Z_YAW_RANGE, 875, 125), 125, 875))

        # --- gripper (ID 21/22): held at stand ---
        p_l_gripper = STAND_PULSE['l_gripper']
        p_r_gripper = STAND_PULSE['r_gripper']

        # Build full pulse dict
        pulses = {
            'l_sho_pitch': p_l_sho_pitch,
            'r_sho_pitch': p_r_sho_pitch,
            'l_sho_roll':  p_l_sho_roll,
            'r_sho_roll':  p_r_sho_roll,
            'l_el_pitch':  p_l_el_pitch,
            'r_el_pitch':  p_r_el_pitch,
            'l_el_yaw':    p_l_el_yaw,
            'r_el_yaw':    p_r_el_yaw,
            'l_gripper':   p_l_gripper,
            'r_gripper':   p_r_gripper,
        }

        safety = {'left_ok': x_left > 0, 'right_ok': x_right > 0}

        # Debug Z values (once per second)
        if int(time.time()) != getattr(self, '_z_dbg_t', 0):
            self._z_dbg_t = int(time.time())
            print('[Z] L_pitch_dz=%.3f R_pitch_dz=%.3f | L_yaw_dz=%.3f R_yaw_dz=%.3f' % (
                dz_l_pitch, dz_r_pitch, dz_l_yaw, dz_r_yaw), flush=True)

        return pulses, safety

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

            results = self.pose.process(image_flip)
            bgr_image = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                # Draw full skeleton
                self.mp_drawing.draw_landmarks(
                    bgr_image, results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS)

                # Extract arm landmarks: pixel coords + Z values
                indices = [
                    self.PL.LEFT_SHOULDER.value,
                    self.PL.RIGHT_SHOULDER.value,
                    self.PL.LEFT_ELBOW.value,
                    self.PL.RIGHT_ELBOW.value,
                    self.PL.LEFT_WRIST.value,
                    self.PL.RIGHT_WRIST.value,
                ]
                marks_2d = []
                marks_z = []
                all_visible = True
                for idx in indices:
                    p = lm[idx]
                    if p.visibility < 0.3:
                        all_visible = False
                    marks_2d.append([int(p.x * width), int(p.y * height)])
                    marks_z.append(p.z)

                # --- GESTURE CHECK ---
                crossed = self._arms_crossed(marks_2d) if all_visible else False

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
                elif all_visible:
                    # Highlight arm joints
                    for m in marks_2d:
                        cv2.circle(bgr_image, tuple(m), 8, (0, 255, 255), -1)

                    result = self.compute_all_arm_pulses(marks_2d, marks_z, width)
                    if result is not None:
                        pulses, safety = result

                        # Apply safety: skip unsafe arm's joints
                        safe_pulses = {}
                        left_joints = ['l_sho_pitch', 'l_sho_roll', 'l_el_pitch', 'l_el_yaw', 'l_gripper']
                        right_joints = ['r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw', 'r_gripper']
                        if safety['left_ok']:
                            for j in left_joints:
                                safe_pulses[j] = pulses[j]
                        if safety['right_ok']:
                            for j in right_joints:
                                safe_pulses[j] = pulses[j]

                        if safe_pulses:
                            full_pulses = {}
                            for j in ARM_JOINTS:
                                if j in safe_pulses:
                                    full_pulses[j] = safe_pulses[j]
                                elif self.pulse_window:
                                    full_pulses[j] = self.pulse_window[-1].get(j, STAND_PULSE[j])
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
                                    last = self.last_pulse.get(joint_name, STAND_PULSE[joint_name])
                                    if abs(pulse - last) < DEADZONE_PULSE:
                                        pulse = last
                                    else:
                                        self.last_pulse[joint_name] = pulse
                                    servo_cmds.append([SERVO_ID[joint_name], pulse])
                                    info_lines.append('%s:%d' % (joint_name, pulse))

                                self.motion_manager.set_servos_position(600, servo_cmds)
                                # Compact debug output
                                print('[SEND] %s' % ' | '.join(info_lines), flush=True)

                            self.frame_count = 0

                        # Show pulse info on image
                        if pulses is not None:
                            y_off = 42
                            for j in ARM_JOINTS:
                                if j.endswith('gripper'):
                                    continue  # skip gripper display
                                side = 'L' if j.startswith('l') else 'R'
                                ok = safety['left_ok'] if j.startswith('l') else safety['right_ok']
                                jname = j.split('_', 1)[1]  # e.g. "sho_pitch"
                                cv2.putText(bgr_image, '%s %s: %d %s' % (
                                    side, jname, pulses.get(j, 0),
                                    'OK' if ok else 'SKIP'),
                                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                    (0, 255, 0) if ok else (0, 0, 255), 1)
                                y_off += 16
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

        rospy.loginfo('[PoseMimic3D] Returning to stand...')
        self._send_stand()
        self.pose.close()
        rospy.signal_shutdown('shutdown')


if __name__ == "__main__":
    import sys
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    print("[MAIN] Starting pose_mimic_3d node...", flush=True)
    try:
        node = PoseMimic3DNode('pose_mimic')
        print("[MAIN] Node initialized, entering run loop", flush=True)
        node.run()
    except Exception as e:
        print("[MAIN] ERROR: %s" % e, flush=True)
        import traceback
        traceback.print_exc()
