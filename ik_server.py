#!/home/ubuntu/miniforge3/envs/pin/bin/python3
"""
Pinocchio IK Server — runs as subprocess, communicates via stdin/stdout JSON.

Protocol:
  Input (JSON per line):  {"side": "left", "target": [x, y, z]}
  Output (JSON per line): {"side": "left", "joints": {"l_sho_pitch": 1.2, ...}, "ok": true}
                          or {"ok": false, "error": "..."}

  Special commands:
    {"cmd": "fk_stand"}  → returns standing FK positions of shoulders and grippers
    {"cmd": "quit"}      → exit

Run with: /home/ubuntu/miniforge3/envs/pin/bin/python3 ik_server.py
"""

import sys
import json
import numpy as np
import pinocchio as pin

URDF_PATH = "/tmp/ainex.urdf"

# Servo ID → joint name
SERVO_TO_JOINT = {
    1: "l_hip_yaw", 2: "r_hip_yaw",
    3: "l_hip_roll", 4: "r_hip_roll",
    5: "l_hip_pitch", 6: "r_hip_pitch",
    7: "l_knee", 8: "r_knee",
    9: "l_ank_pitch", 10: "r_ank_pitch",
    11: "l_ank_roll", 12: "r_ank_roll",
    13: "l_sho_pitch", 14: "r_sho_pitch",
    15: "l_sho_roll", 16: "r_sho_roll",
    17: "l_el_pitch", 18: "r_el_pitch",
    19: "l_el_yaw", 20: "r_el_yaw",
    21: "l_gripper", 22: "r_gripper",
    23: "head_pan", 24: "head_tilt",
}

# Standing pulse values (from real robot)
STAND_PULSE = {
    'l_ank_roll': 500, 'r_ank_roll': 500,
    'l_ank_pitch': 640, 'r_ank_pitch': 360,
    'l_knee': 500, 'r_knee': 500,
    'l_hip_pitch': 350, 'r_hip_pitch': 650,
    'l_hip_roll': 500, 'r_hip_roll': 500,
    'l_hip_yaw': 500, 'r_hip_yaw': 500,
    'l_sho_pitch': 835, 'r_sho_pitch': 165,
    'l_sho_roll': 830, 'r_sho_roll': 170,
    'l_el_pitch': 500, 'r_el_pitch': 500,
    'l_el_yaw': 150, 'r_el_yaw': 850,
    'l_gripper': 500, 'r_gripper': 500,
    'head_pan': 500, 'head_tilt': 500,
}

LEFT_ARM_JOINTS = ["l_sho_pitch", "l_sho_roll", "l_el_pitch", "l_el_yaw"]
RIGHT_ARM_JOINTS = ["r_sho_pitch", "r_sho_roll", "r_el_pitch", "r_el_yaw"]

def pulse_to_rad(pulse):
    return (pulse - 500) * 0.004189

def rad_to_pulse(rad):
    return int(500 + rad / 0.004189)


class IKServer:
    def __init__(self):
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data = self.model.createData()

        # Standing q
        self.q_stand = pin.neutral(self.model)
        for servo_id, jname in SERVO_TO_JOINT.items():
            jid = self.model.getJointId(jname)
            if jid < self.model.njoints:
                qi = self.model.joints[jid].idx_q
                self.q_stand[qi] = pulse_to_rad(STAND_PULSE.get(jname, 500))

        # Cache joint/frame info
        self.arm_info = {}
        for side, joints, frame in [
            ('left', LEFT_ARM_JOINTS, 'l_gripper'),
            ('right', RIGHT_ARM_JOINTS, 'r_gripper'),
        ]:
            joint_ids = []
            q_indices = []
            v_indices = []
            for jname in joints:
                jid = self.model.getJointId(jname)
                joint_ids.append(jid)
                q_indices.append(self.model.joints[jid].idx_q)
                v_indices.append(self.model.joints[jid].idx_v)
            self.arm_info[side] = {
                'joints': joints,
                'joint_ids': joint_ids,
                'q_indices': q_indices,
                'v_indices': v_indices,
                'frame_id': self.model.getFrameId(frame),
            }

        # Compute standing FK
        pin.forwardKinematics(self.model, self.data, self.q_stand)
        pin.updateFramePlacements(self.model, self.data)

        # Store shoulder and gripper positions at stand
        self.stand_fk = {}
        for name in ['l_sho_pitch', 'r_sho_pitch', 'l_gripper', 'r_gripper']:
            fid = self.model.getFrameId(name)
            self.stand_fk[name] = self.data.oMf[fid].translation.copy()

        # Robot arm length (shoulder to gripper at stand)
        self.arm_length = {
            'left': np.linalg.norm(self.stand_fk['l_gripper'] - self.stand_fk['l_sho_pitch']),
            'right': np.linalg.norm(self.stand_fk['r_gripper'] - self.stand_fk['r_sho_pitch']),
        }

        # Last solved q for warm-starting
        self.q_last = {
            'left': self.q_stand.copy(),
            'right': self.q_stand.copy(),
        }

        sys.stderr.write("[IK] Model loaded: %d DOF, arm_len L=%.1fcm R=%.1fcm\n" % (
            self.model.nq,
            self.arm_length['left'] * 100,
            self.arm_length['right'] * 100))
        sys.stderr.flush()

    def solve(self, side, target_pos, max_iter=100, eps=1e-3, damp=1e-6):
        """Jacobian pseudo-inverse IK for one arm."""
        info = self.arm_info[side]
        frame_id = info['frame_id']
        q_indices = info['q_indices']
        v_indices = info['v_indices']

        # Warm-start from last solution
        q = self.q_last[side].copy()
        target = np.array(target_pos)

        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            current_pos = self.data.oMf[frame_id].translation.copy()
            error = target - current_pos
            err_norm = np.linalg.norm(error)

            if err_norm < eps:
                # Save for warm-start
                self.q_last[side] = q.copy()
                # Extract arm joint angles
                joints = {}
                for jname, qi in zip(info['joints'], q_indices):
                    joints[jname] = float(q[qi])
                return True, joints, err_norm

            J_full = pin.computeFrameJacobian(
                self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)
            J_pos = J_full[:3, :]
            J_arm = J_pos[:, v_indices]

            JJT = J_arm @ J_arm.T + damp * np.eye(3)
            dq_arm = J_arm.T @ np.linalg.solve(JJT, error)

            # Limit step size
            max_step = 0.15
            dq_norm = np.linalg.norm(dq_arm)
            if dq_norm > max_step:
                dq_arm = dq_arm * max_step / dq_norm

            for k, qi in enumerate(q_indices):
                q[qi] += dq_arm[k]

            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)

        # Didn't converge — still return best result
        self.q_last[side] = q.copy()
        joints = {}
        for jname, qi in zip(info['joints'], q_indices):
            joints[jname] = float(q[qi])
        return False, joints, float(err_norm)

    def handle_fk_stand(self):
        """Return FK positions for standing pose."""
        result = {
            'ok': True,
            'shoulder_left': self.stand_fk['l_sho_pitch'].tolist(),
            'shoulder_right': self.stand_fk['r_sho_pitch'].tolist(),
            'gripper_left': self.stand_fk['l_gripper'].tolist(),
            'gripper_right': self.stand_fk['r_gripper'].tolist(),
            'arm_length_left': float(self.arm_length['left']),
            'arm_length_right': float(self.arm_length['right']),
        }
        return result

    def run(self):
        sys.stderr.write("[IK] Server ready, waiting for commands...\n")
        sys.stderr.flush()

        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue

            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                resp = {'ok': False, 'error': 'JSON parse error: %s' % str(e)}
                print(json.dumps(resp), flush=True)
                continue

            cmd = req.get('cmd', '')

            if cmd == 'quit':
                sys.stderr.write("[IK] Quit command received\n")
                break

            elif cmd == 'fk_stand':
                resp = self.handle_fk_stand()
                print(json.dumps(resp), flush=True)

            elif 'side' in req and 'target' in req:
                side = req['side']
                target = req['target']
                if side not in ('left', 'right'):
                    print(json.dumps({'ok': False, 'error': 'side must be left or right'}), flush=True)
                    continue

                ok, joints, err = self.solve(side, target)
                resp = {
                    'ok': ok,
                    'side': side,
                    'joints': joints,
                    'error_mm': round(err * 1000, 2),
                }
                print(json.dumps(resp), flush=True)

            else:
                print(json.dumps({'ok': False, 'error': 'unknown command'}), flush=True)


if __name__ == '__main__':
    server = IKServer()
    server.run()
