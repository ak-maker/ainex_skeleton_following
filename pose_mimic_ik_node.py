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

# --- Anti-twitch ---
WINDOW_SIZE = 15
SEND_EVERY = 30          # ~3 seconds at 10Hz
MIN_FRAMES_BEFORE_SEND = 10
DEADZONE_PULSE = 50

# --- Coordinate mapping ---
# MediaPipe world: X=person's left, Y=down, Z=toward camera (person's forward when facing cam)
# Robot URDF (ROS): X=forward, Y=left, Z=up
# Mapping: Robot_X = MP_Z, Robot_Y = MP_X, Robot_Z = -MP_Y


def mp_world_to_robot(vec):
    """Convert MediaPipe world coordinate vector to robot body frame."""
    return np.array([vec[2], vec[0], -vec[1]])


def rad_to_pulse(rad):
    """弧度 → 脉冲 (通用, 0 rad = 500 pulse)"""
    return int(500 + rad / 0.004189)


# URDF左右臂轴方向完全相同(没有镜像), 但实际舵机是镜像安装的.
# 所以右臂的IK角度转脉冲时需要取反: pulse = 500 - rad / 0.004189
# 对于sho_roll和el_yaw, 轴也是相同的, 同样需要取反.
RIGHT_ARM_INVERT = {'r_sho_pitch', 'r_sho_roll', 'r_el_pitch', 'r_el_yaw'}


def rad_to_pulse_joint(jname, rad):
    """弧度 → 脉冲 (考虑舵机方向)"""
    if jname in RIGHT_ARM_INVERT:
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
        print("[IK] Standing FK: L_sho=%s R_sho=%s arm_len=%.1fcm" % (
            np.round(self.shoulder_pos['left'] * 100, 1),
            np.round(self.shoulder_pos['right'] * 100, 1),
            self.arm_length['left'] * 100), flush=True)

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

    def _ik_solve(self, side, target):
        """Solve IK for one arm. Returns dict of {joint: rad} or None."""
        self._ik_send({'side': side, 'target': target.tolist()})
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
    def compute_ik_targets(self, world_lm):
        """Extract shoulder→wrist vectors, convert to robot frame, scale to robot arm length.

        Returns dict with 'left' and 'right' target positions in robot frame (meters).
        """
        # MediaPipe world landmarks (after image flip):
        # 11 = person's actual RIGHT shoulder → robot LEFT
        # 12 = person's actual LEFT shoulder → robot RIGHT
        l_sho = np.array([world_lm[11].x, world_lm[11].y, world_lm[11].z])
        r_sho = np.array([world_lm[12].x, world_lm[12].y, world_lm[12].z])
        l_wri = np.array([world_lm[15].x, world_lm[15].y, world_lm[15].z])
        r_wri = np.array([world_lm[16].x, world_lm[16].y, world_lm[16].z])

        targets = {}
        for side, sho, wri in [('left', l_sho, l_wri), ('right', r_sho, r_wri)]:
            # Direction from shoulder to wrist in MediaPipe world
            direction = wri - sho
            human_len = np.linalg.norm(direction)
            if human_len < 0.01:
                return None

            direction_unit = direction / human_len

            # Convert to robot body frame
            robot_dir = mp_world_to_robot(direction_unit)

            # Scale to robot arm length and add to robot shoulder position
            robot_target = self.shoulder_pos[side] + robot_dir * self.arm_length[side]
            targets[side] = robot_target

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
                    targets = self.compute_ik_targets(world_lm)

                    if targets is not None:
                        # Solve IK for both arms
                        l_joints, l_err = self._ik_solve('left', targets['left'])
                        r_joints, r_err = self._ik_solve('right', targets['right'])

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
