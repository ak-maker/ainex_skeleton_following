#!/usr/bin/env python3
# encoding: utf-8
# 逐个测试AiNex所有24个servo是否正常工作
# 每个servo从stand位置移动+50 units，然后回到stand
# 观察哪些servo没有动——可能已损坏
#
# 用法:
# bash -c 'source /opt/ros/noetic/setup.bash && source /home/ubuntu/ros_ws/devel/setup.bash && python3 /home/ubuntu/ros_ws/src/skeleton_follow/test_all_servos.py'

import rospy
import time
from ainex_kinematics.motion_manager import MotionManager

SERVOS = {
    1:  ("l_ank_roll",   500),
    2:  ("r_ank_roll",   500),
    3:  ("l_ank_pitch",  640),
    4:  ("r_ank_pitch",  360),
    5:  ("l_knee",       500),
    6:  ("r_knee",       500),
    7:  ("l_hip_pitch",  350),
    8:  ("r_hip_pitch",  650),
    9:  ("l_hip_roll",   500),
    10: ("r_hip_roll",   500),
    11: ("l_hip_yaw",    500),
    12: ("r_hip_yaw",    500),
    13: ("l_sho_pitch",  835),
    14: ("r_sho_pitch",  165),
    15: ("l_sho_roll",   830),
    16: ("r_sho_roll",   170),
    17: ("l_el_pitch",   500),
    18: ("r_el_pitch",   500),
    19: ("l_el_yaw",     150),
    20: ("r_el_yaw",     850),
    21: ("l_gripper",    500),
    22: ("r_gripper",    500),
    23: ("head_pan",     500),
    24: ("head_tilt",    500),
}

MOVE_OFFSET = 50  # units to move from stand position


def main():
    rospy.init_node("servo_test", anonymous=True)
    mm = MotionManager()

    print("Standing first...", flush=True)
    mm.run_action("stand")
    time.sleep(2)

    for sid in range(1, 25):
        name, stand_val = SERVOS[sid]
        test_val = min(875, max(125, stand_val + MOVE_OFFSET))
        print("Testing servo %d (%s): stand=%d -> test=%d ..." % (sid, name, stand_val, test_val), flush=True)
        mm.set_servos_position(500, [[sid, test_val]])
        time.sleep(1.0)
        mm.set_servos_position(500, [[sid, stand_val]])
        time.sleep(0.8)
        print("  Servo %d done." % sid, flush=True)

    print("\n=== ALL SERVOS TESTED ===", flush=True)
    print("Check which ones did NOT move — those may be broken.", flush=True)
    mm.run_action("stand")


if __name__ == "__main__":
    main()
