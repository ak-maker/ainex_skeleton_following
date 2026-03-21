#!/usr/bin/env python3
# encoding: utf-8
# Pose Mimic IK Node — Pinocchio IK-based upper body imitation following
#
# Architecture:
#   Main process (system Python 3.8): camera, MediaPipe, servo control, ROS
#   IK subprocess (conda Python 3.10): Pinocchio IK solving via stdin/stdout JSON
#
# This node uses MediaPipe to detect the user's pose, extracts shoulder→wrist
# direction vectors, maps them to robot body frame, scales to robot arm length,
# and uses Pinocchio IK to solve for joint angles. This replaces the manual
# atan2/arccos approach in pose_mimic_3d_node.py.
#
# Upper body only: sho_pitch, sho_roll, el_pitch, el_yaw (4 DOF per arm)
# Gripper and head held at stand.
# Legs not controlled (held at stand).

import os
import cv2
import json
import math
import time
import rospy
import signal
import subprocess
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

IK_SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'ik_server.py')
IK_PYTHON = '/home/ubuntu/miniforge3/envs/pin/bin/python3'

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

# Servo safety limits (pulse) — NEVER exceed these
SERVO_LIMITS = {
    'l_el_yaw':  (50, 600),   # servo 19: 0=bent, 600=straight
    'r_el_yaw':  (400, 950),  # servo 20: 950=bent, 400=straight (BURNED before at <360!)
    'l_sho_pitch': (50, 950),
    'r_sho_pitch': (50, 950),
    'l_sho_roll':  (70, 900),
    'r_sho_roll':  (100, 930),
    'l_el_pitch':  (125, 875),
    'r_el_pitch':  (125, 875),
}

# Arm joints controlled by IK
ARM_JOINTS = [
    'l_sho_pitch', 'r_sho_pitch',
    'l_sho_roll',  'r_sho_roll',
    'l_el_pitch',  'r_el_pitch',
    'l_el_yaw',    'r_el_yaw',
]

# All joints sent to servos (arm + gripper)
SEND_JOINTS = ARM_JOINTS + ['l_gripper', 'r_gripper']

# --- Gesture ---
STAND_GESTURE_FRAMES = 5
CROSS_DIST_RATIO = 0.5

# --- Anti-twitch (tuned for IK: less aggressive than 3D node) ---
WINDOW_SIZE = 8
SEND_EVERY = 12          # ~1.2 seconds at 10Hz
MIN_FRAMES_BEFORE_SEND = 5
DEADZONE_PULSE = 15

# --- Rotation mapping ---
# Human neutral (arms at sides) → direction (0,0,-1) in "human robot frame"
# Robot standing → direction d_stand (from FK)
# We precompute R that maps (0,0,-1) → d_stand, then for any human direction
# d_human, the robot target direction is R @ d_human.
# This guarantees targets are always on the arm-length sphere (always reachable).

def rotation_between_vectors(v1, v2):
    """Rotation matrix that maps unit vector v1 to unit vector v2 (Rodrigues)."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cross = np.cross(v1, v2)
    dot = np.dot(v1, v2)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-8:
        if dot > 0:
            return np.eye(3)
        # Opposite — 180° around any perpendicular axis
        perp = np.array([1, 0, 0]) if abs(v1[0]) < 0.9 else np.array([0, 1, 0])
        perp = perp - np.dot(perp, v1) * v1
        perp = perp / np.linalg.norm(perp)
        return 2 * np.outer(perp, perp) - np.eye(3)
    K = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + dot))


def rad_to_pulse(rad):
    """弧度 → 脉冲 (通用, 0 rad = 500 pulse)"""
    return int(500 + rad / 0.004189)


# URDF左右臂轴方向相同(没有镜像), 但实际舵机是镜像安装的.
# 只有axis=(0,-1,0)的关节需要取反: sho_pitch, el_pitch
# axis=(1,0,0)的sho_roll和axis=(0,0,1)的el_yaw不需要取反.
INVERT_JOINTS = {'r_sho_pitch', 'r_el_pitch'}


def rad_to_pulse_joint(jname, rad):
    """弧度 → 脉冲 (考虑舵机方向)"""
    if jname in INVERT_JOINTS:
        return int(500 - rad / 0.004189)
    return int(500 + rad / 0.004189)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class PoseMimicIKNode:
    def __init__(self, name):
        rospy.init_node(name, anonymous=False)
        self.name = name
        self.running = True
        self.image = None
        self.fps = fps.FPS()
        self.frame_ts = 0

        signal.signal(signal.SIGINT, self.shutdown)

        # ---- Start IK subprocess ----
        print("[IK] Starting IK server subprocess...", flush=True)
        self.ik_proc = subprocess.Popen(
            [IK_PYTHON, IK_SERVER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # let stderr pass through to our console
            bufsize=0,
        )

        # Get standing FK info from IK server
        self._ik_send({'cmd': 'fk_stand'})
        fk_info = self._ik_recv()
        if not fk_info or not fk_info.get('ok'):
            raise RuntimeError("Failed to get FK stand info from IK server: %s" % fk_info)

        self.shoulder_pos = {
            'left': np.array(fk_info['shoulder_left']),
            'right': np.array(fk_info['shoulder_right']),
        }
        self.arm_length = {
            'left': fk_info['arm_length_left'],
            'right': fk_info['arm_length_right'],
        }
        self.upper_arm_length = {
            'left': fk_info['upper_arm_length_left'],
            'right': fk_info['upper_arm_length_right'],
        }
        self.gripper_pos = {
            'left': np.array(fk_info['gripper_left']),
            'right': np.array(fk_info['gripper_right']),
        }
        self.elbow_pos = {
            'left': np.array(fk_info['elbow_left']),
            'right': np.array(fk_info['elbow_right']),
        }

        # Precompute rotation matrices:
        # Maps human-neutral direction (0,0,-1) to robot standing arm direction
        # So any human arm direction d_human → R @ d_human → always on the reachable sphere
        HUMAN_NEUTRAL = np.array([0.0, 0.0, -1.0])  # "arm hanging down"
        self.stand_rotation = {}
        self.stand_elbow_rotation = {}
        for side in ('left', 'right'):
            # Full arm (shoulder → gripper) standing direction
            d_arm = self.gripper_pos[side] - self.shoulder_pos[side]
            d_arm = d_arm / np.linalg.norm(d_arm)
            self.stand_rotation[side] = rotation_between_vectors(HUMAN_NEUTRAL, d_arm)

            # Upper arm (shoulder → elbow) standing direction
            d_upper = self.elbow_pos[side] - self.shoulder_pos[side]
            d_upper = d_upper / np.linalg.norm(d_upper)
            self.stand_elbow_rotation[side] = rotation_between_vectors(HUMAN_NEUTRAL, d_upper)

        print("[IK] Standing FK: L_sho=%s R_sho=%s arm=%.1fcm upper=%.1fcm" % (
            np.round(self.shoulder_pos['left'] * 100, 1),
            np.round(self.shoulder_pos['right'] * 100, 1),
            self.arm_length['left'] * 100,
            self.upper_arm_length['left'] * 100), flush=True)
        print("[IK] Stand dirs: L_grip=%s R_grip=%s" % (
            np.round((self.gripper_pos['left'] - self.shoulder_pos['left']) * 100, 1),
            np.round((self.gripper_pos['right'] - self.shoulder_pos['right']) * 100, 1)), flush=True)

        # ---- MediaPipe PoseLandmarker ----
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
            rospy.loginfo('[PoseMimicIK] walking module disabled')
        except Exception as e:
            rospy.logwarn('[PoseMimicIK] cannot connect walking service: %s' % str(e))

        self._send_stand()
        time.sleep(1.0)

        # Gesture state
        self.gesture_count = 0
        self.in_stand_mode = False

        # ---- ROS ----
        self.camera = rospy.get_param('/camera')
        rospy.Subscriber(
            '/{}/{}'.format(self.camera['camera_name'], self.camera['image_topic']),
            Image, self.image_callback,
        )
        self.result_pub = rospy.Publisher('~image_result', Image, queue_size=1)

        rospy.loginfo('[PoseMimicIK] Ready! IK-based upper body following')

    def shutdown(self, signum, frame):
        self.running = False

    def image_callback(self, ros_image):
        self.image = np.ndarray(
            shape=(ros_image.height, ros_image.width, 3),
            dtype=np.uint8, buffer=ros_image.data,
        )

    # ------------------------------------------------------------------
    # IK subprocess communication
    # ------------------------------------------------------------------
    def _ik_send(self, msg):
        """Send JSON message to IK subprocess."""
        line = json.dumps(msg) + '\n'
        self.ik_proc.stdin.write(line.encode())
        self.ik_proc.stdin.flush()

    def _ik_recv(self):
        """Receive JSON response from IK subprocess."""
        line = self.ik_proc.stdout.readline()
        if not line:
            return None
        return json.loads(line.decode())

    def _ik_solve(self, side, target, elbow_hint=None):
        """Solve IK for one arm. Returns dict of {joint: rad} or None."""
        msg = {'side': side, 'target': target.tolist()}
        if elbow_hint is not None:
            msg['elbow_hint'] = elbow_hint.tolist()
        self._ik_send(msg)
        resp = self._ik_recv()
        if resp and resp.get('joints'):
            return resp['joints'], resp.get('error_mm', 999)
        return None, 999

    # ------------------------------------------------------------------
    # Servo control helpers
    # ------------------------------------------------------------------
    def _send_stand(self):
        cmds = [[SERVO_ID[j], STAND_PULSE[j]] for j in SERVO_ID]
        self.motion_manager.set_servos_position(1600, cmds)
        self.last_pulse.clear()
        self.pulse_window.clear()
        self.frame_count = 0

    def _safe_pulse(self, joint, pulse):
        """Clamp pulse to servo safety limits."""
        lo, hi = SERVO_LIMITS.get(joint, (0, 1000))
        return int(clamp(pulse, lo, hi))

    # ------------------------------------------------------------------
    # Gesture: hands close together
    # ------------------------------------------------------------------
    def _hands_close(self, norm_lm):
        l_sho = norm_lm[11]
        r_sho = norm_lm[12]
        l_wri = norm_lm[15]
        r_wri = norm_lm[16]

        sho_w = math.sqrt((l_sho.x - r_sho.x)**2 + (l_sho.y - r_sho.y)**2)
        wri_d = math.sqrt((l_wri.x - r_wri.x)**2 + (l_wri.y - r_wri.y)**2)

        if sho_w < 0.02:
            return False

        ratio = wri_d / sho_w
        close = ratio < CROSS_DIST_RATIO

        if int(time.time()) != getattr(self, '_cross_dbg_t', 0):
            self._cross_dbg_t = int(time.time())
            print('[Hands] ratio=%.2f close=%s' % (ratio, close), flush=True)
        return close

    # ------------------------------------------------------------------
    # Compute IK targets from MediaPipe landmarks
    # ------------------------------------------------------------------
    def compute_ik_targets(self, world_lm, norm_lm, width, height):
        """Compute IK targets using rotation mapping:

        1. Compute human arm direction from pitch (world Y,Z) + roll (screen coords)
        2. Apply precomputed rotation R that maps human-neutral → robot-standing
        3. Target = shoulder + R(d_human) * arm_length → always on reachable sphere

        This avoids:
        - Unreliable world X (screen-based lateral instead)
        - Unreachable targets (rotation keeps targets on the arm sphere)
        """
        targets = {}

        for side, sho_idx, elb_idx, wri_idx in [
            ('left', 11, 13, 15),
            ('right', 12, 14, 16),
        ]:
            # World landmarks (Y, Z only for pitch)
            sho_w = np.array([world_lm[sho_idx].x, world_lm[sho_idx].y, world_lm[sho_idx].z])
            elb_w = np.array([world_lm[elb_idx].x, world_lm[elb_idx].y, world_lm[elb_idx].z])
            wri_w = np.array([world_lm[wri_idx].x, world_lm[wri_idx].y, world_lm[wri_idx].z])

            # Screen landmarks (for lateral direction)
            sho_px = np.array([norm_lm[sho_idx].x * width, norm_lm[sho_idx].y * height])
            elb_px = np.array([norm_lm[elb_idx].x * width, norm_lm[elb_idx].y * height])
            wri_px = np.array([norm_lm[wri_idx].x * width, norm_lm[wri_idx].y * height])

            # === UPPER ARM (shoulder → elbow) → elbow target ===
            upper_w = elb_w - sho_w
            if abs(upper_w[1]) + abs(upper_w[2]) < 0.01:
                return None
            upper_pitch = math.atan2(-upper_w[2], upper_w[1])

            upper_dx = elb_px[0] - sho_px[0]
            upper_dy = elb_px[1] - sho_px[1]
            if math.sqrt(upper_dx**2 + upper_dy**2) < 1:
                return None
            upper_roll = math.atan2(-upper_dx, upper_dy)

            # Human direction: (pitch=0,roll=0) → (0,0,-1) = arm down = neutral
            sp, cp = math.sin(upper_pitch), math.cos(upper_pitch)
            sr, cr = math.sin(upper_roll), math.cos(upper_roll)
            d_human_upper = np.array([sp, sr * cp, -cp * cr])
            n = np.linalg.norm(d_human_upper)
            if n > 0.01:
                d_human_upper /= n

            # Rotate: human neutral → robot standing direction
            d_robot_upper = self.stand_elbow_rotation[side] @ d_human_upper
            elbow_tgt = self.shoulder_pos[side] + d_robot_upper * self.upper_arm_length[side]
            targets[side + '_elbow'] = elbow_tgt

            # === FULL ARM (shoulder → wrist) → wrist target ===
            full_w = wri_w - sho_w
            if abs(full_w[1]) + abs(full_w[2]) < 0.01:
                return None
            full_pitch = math.atan2(-full_w[2], full_w[1])

            full_dx = wri_px[0] - sho_px[0]
            full_dy = wri_px[1] - sho_px[1]
            if math.sqrt(full_dx**2 + full_dy**2) < 1:
                return None
            full_roll = math.atan2(-full_dx, full_dy)

            sp, cp = math.sin(full_pitch), math.cos(full_pitch)
            sr, cr = math.sin(full_roll), math.cos(full_roll)
            d_human_full = np.array([sp, sr * cp, -cp * cr])
            n = np.linalg.norm(d_human_full)
            if n > 0.01:
                d_human_full /= n

            d_robot_full = self.stand_rotation[side] @ d_human_full
            wrist_tgt = self.shoulder_pos[side] + d_robot_full * self.arm_length[side]
            targets[side] = wrist_tgt

        # Debug (once per second)
        if int(time.time()) != getattr(self, '_tgt_dbg_t', 0):
            self._tgt_dbg_t = int(time.time())
            lt = targets['left'] * 100
            rt = targets['right'] * 100
            # Show distance from standing gripper
            dl = np.linalg.norm(targets['left'] - self.gripper_pos['left']) * 100
            dr = np.linalg.norm(targets['right'] - self.gripper_pos['right']) * 100
            print('[TGT] L=[%.1f,%.1f,%.1f] R=[%.1f,%.1f,%.1f]cm dist_from_stand L=%.1f R=%.1fcm' % (
                lt[0], lt[1], lt[2], rt[0], rt[1], rt[2], dl, dr), flush=True)

        return targets

    # ------------------------------------------------------------------
    # Convert IK joint angles (radians) to servo pulses
    # ------------------------------------------------------------------
    def ik_joints_to_pulses(self, left_joints, right_joints):
        """Convert IK solution (radians) to servo pulses with safety limits.
        Right arm pulses are inverted because URDF axes are NOT mirrored
        but physical servos ARE mirror-mounted.

        Returns dict of {joint_name: pulse} for all arm joints."""
        pulses = {}

        if left_joints:
            for jname, rad in left_joints.items():
                pulse = rad_to_pulse_joint(jname, rad)
                pulses[jname] = self._safe_pulse(jname, pulse)

        if right_joints:
            for jname, rad in right_joints.items():
                pulse = rad_to_pulse_joint(jname, rad)
                pulses[jname] = self._safe_pulse(jname, pulse)

        # Gripper held at stand
        pulses['l_gripper'] = STAND_PULSE['l_gripper']
        pulses['r_gripper'] = STAND_PULSE['r_gripper']

        return pulses

    # ------------------------------------------------------------------
    # Sliding window average
    # ------------------------------------------------------------------
    def _window_average(self):
        if len(self.pulse_window) < MIN_FRAMES_BEFORE_SEND:
            return None
        avg = {}
        for joint in SEND_JOINTS:
            vals = [p[joint] for p in self.pulse_window if joint in p]
            if vals:
                avg[joint] = int(sum(vals) / len(vals))
        return avg if len(avg) == len(SEND_JOINTS) else None

    # ------------------------------------------------------------------
    # Draw landmarks
    # ------------------------------------------------------------------
    def _draw_landmarks(self, bgr_image, norm_landmarks):
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

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_flip)
            self.frame_ts += 33
            result = self.detector.detect_for_video(mp_image, self.frame_ts)

            bgr_image = cv2.cvtColor(image_flip, cv2.COLOR_RGB2BGR)

            if result.pose_landmarks and result.pose_world_landmarks:
                norm_lm = result.pose_landmarks[0]
                world_lm = result.pose_world_landmarks[0]

                self._draw_landmarks(bgr_image, norm_lm)

                # --- GESTURE CHECK ---
                hands_close = self._hands_close(norm_lm)

                if hands_close:
                    self.gesture_count += 1
                else:
                    self.gesture_count = 0

                if self.gesture_count >= STAND_GESTURE_FRAMES:
                    if not self.in_stand_mode:
                        print('[GESTURE] Hands close -> STAND!', flush=True)
                        self._send_stand()
                        self.in_stand_mode = True
                    else:
                        print('[GESTURE] Hands close -> RESUME!', flush=True)
                        self.in_stand_mode = False
                    self.gesture_count = 0

                # --- IK control ---
                if self.in_stand_mode:
                    cv2.putText(bgr_image, 'STANDING (hands together to resume)', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    targets = self.compute_ik_targets(world_lm, norm_lm, width, height)

                    if targets is not None:
                        # Solve IK for both arms (with elbow hints)
                        l_joints, l_err = self._ik_solve('left', targets['left'],
                                                          elbow_hint=targets.get('left_elbow'))
                        r_joints, r_err = self._ik_solve('right', targets['right'],
                                                          elbow_hint=targets.get('right_elbow'))

                        if l_joints and r_joints:
                            pulses = self.ik_joints_to_pulses(l_joints, r_joints)

                            self.pulse_window.append(pulses)
                            self.frame_count += 1

                            # Debug
                            if int(time.time()) != getattr(self, '_dbg_t', 0):
                                self._dbg_t = int(time.time())
                                print('[IK] L_err=%.1fmm R_err=%.1fmm | sho_pitch L:%d R:%d | sho_roll L:%d R:%d | el L:%d R:%d' % (
                                    l_err, r_err,
                                    pulses.get('l_sho_pitch', 0), pulses.get('r_sho_pitch', 0),
                                    pulses.get('l_sho_roll', 0), pulses.get('r_sho_roll', 0),
                                    pulses.get('l_el_yaw', 0), pulses.get('r_el_yaw', 0)),
                                    flush=True)

                            # Show buffer status
                            buf_len = len(self.pulse_window)
                            cv2.putText(bgr_image, 'IK Window [%d/%d] send in %d' % (
                                buf_len, WINDOW_SIZE,
                                max(0, SEND_EVERY - self.frame_count)),
                                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

                            if self.frame_count >= SEND_EVERY and buf_len >= MIN_FRAMES_BEFORE_SEND:
                                avg_pulses = self._window_average()
                                if avg_pulses is not None:
                                    servo_cmds = []
                                    info_lines = []
                                    for joint_name in SEND_JOINTS:
                                        pulse = avg_pulses[joint_name]
                                        last = self.last_pulse.get(joint_name, STAND_PULSE.get(joint_name, 500))
                                        if abs(pulse - last) < DEADZONE_PULSE:
                                            pulse = last
                                        else:
                                            self.last_pulse[joint_name] = pulse
                                        servo_cmds.append([SERVO_ID[joint_name], pulse])
                                        info_lines.append('%s:%d' % (joint_name, pulse))

                                    self.motion_manager.set_servos_position(1200, servo_cmds)
                                    print('[SEND] %s' % ' | '.join(info_lines), flush=True)

                                self.frame_count = 0

                            # Show pulse values on image
                            y_off = 42
                            for j in ARM_JOINTS:
                                side = 'L' if j.startswith('l') else 'R'
                                jname = j.split('_', 1)[1]
                                cv2.putText(bgr_image, '%s %s: %d' % (side, jname, pulses.get(j, 0)),
                                            (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                            (0, 255, 0), 1)
                                y_off += 16
                        else:
                            cv2.putText(bgr_image, 'IK solve failed', (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(bgr_image, 'Bad landmarks', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(bgr_image, 'No person detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Gesture progress
            if self.gesture_count > 0 and not self.in_stand_mode:
                cv2.putText(bgr_image, 'HANDS CLOSE [%d/%d]' % (
                    min(self.gesture_count, STAND_GESTURE_FRAMES), STAND_GESTURE_FRAMES),
                    (10, bgr_image.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

            self.fps.update()
            bgr_image = self.fps.show_fps(bgr_image)
            self.result_pub.publish(cv2_image2ros(cv2.resize(bgr_image, (640, 480)), self.name))
            rate.sleep()

        # Cleanup
        rospy.loginfo('[PoseMimicIK] Returning to stand...')
        self._send_stand()
        self._ik_send({'cmd': 'quit'})
        self.ik_proc.wait(timeout=5)
        self.detector.close()
        rospy.signal_shutdown('shutdown')


if __name__ == "__main__":
    import sys
    sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
    print("[MAIN] Starting pose_mimic_ik node...", flush=True)
    try:
        node = PoseMimicIKNode('pose_mimic')
        print("[MAIN] Node initialized, entering run loop", flush=True)
        node.run()
    except Exception as e:
        print("[MAIN] ERROR: %s" % e, flush=True)
        import traceback
        traceback.print_exc()
