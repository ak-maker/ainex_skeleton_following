# AiNex Skeleton Pose Mimic v2

## Changes from v1
1. **Standing = ALL servos pulse 500** (confirmed correct for this robot)
2. **Removed head servos** (head_pan ID23, head_tilt ID24 not present on hardware)
3. **Fixed return-to-stand bug**: v1 used init_pose.yaml angles as standing base, which produced non-500 pulses (e.g. l_sho_roll->809, r_sho_roll->191, l_el_yaw->40) causing arms-open and tilt-back
4. **Simplified pulse formula**: `pulse = 500 + offset_rad * coef` (no init_pulse lookup needed)
5. **Added "hands above head" gesture**: raise both wrists above nose for ~0.8s to return to standing; lower hands below shoulders to resume skeleton following

## Hardware
- **Robot**: AiNex (Hiwonder), 22 bus servos (20 body + 2 grippers, NO head)
- **Controller**: Raspberry Pi 5, Docker container (Ubuntu 20.04 + ROS Noetic)
- **Shared dir**: host `/home/pi/docker/tmp` = container `/home/ubuntu/share/tmp`
- **Camera**: USB, 640x480

## 22 Servo Mapping

### Left Leg (6)
| Joint | ID | Note |
|-------|-----|------|
| l_ank_roll | 1 | ankle roll |
| l_ank_pitch | 3 | ankle pitch |
| l_knee | 5 | knee (hardware init=240, NOT 500) |
| l_hip_pitch | 7 | hip pitch |
| l_hip_roll | 9 | hip roll |
| l_hip_yaw | 11 | hip yaw |

### Right Leg (6)
| Joint | ID | Note |
|-------|-----|------|
| r_ank_roll | 2 | ankle roll |
| r_ank_pitch | 4 | ankle pitch |
| r_knee | 6 | knee (hardware init=760, NOT 500) |
| r_hip_pitch | 8 | hip pitch |
| r_hip_roll | 10 | hip roll |
| r_hip_yaw | 12 | hip yaw |

### Left Arm (4 + gripper)
| Joint | ID | Note |
|-------|-----|------|
| l_sho_pitch | 13 | FLIPPED (init=875, min=1000>max=0) |
| l_sho_roll | 15 | shoulder roll |
| l_el_pitch | 17 | elbow pitch |
| l_el_yaw | 19 | elbow yaw |
| l_gripper | 21 | gripper |

### Right Arm (4 + gripper)
| Joint | ID | Note |
|-------|-----|------|
| r_sho_pitch | 14 | FLIPPED (init=125, min=1000>max=0) |
| r_sho_roll | 16 | shoulder roll |
| r_el_pitch | 18 | elbow pitch |
| r_el_yaw | 20 | elbow yaw |
| r_gripper | 22 | gripper |

## Pulse conversion
```
ENCODER_TICKS_PER_RADIAN = 180 / pi / 240 * 1000 ~= 238.73
pulse = 500 + offset_from_standing * coef
```
Where `coef = +238.73` for normal servos, `-238.73` for flipped (l/r_sho_pitch).

## Key code paths (container)
- **Servo control**: `ainex_kinematics/src/ainex_kinematics/motion_manager.py`
  - `set_servos_position(duration_ms, [[id, pulse], ...])`
  - `run_action('action_name')` -> plays .d6a SQLite file
- **Servo config**: `ainex_kinematics/config/servo_controller.yaml`
- **Init pose**: `ainex_kinematics/config/init_pose.yaml`
- **Controller**: `ainex_kinematics/scripts/ainex_controller.py`
- **Action groups**: `/home/ubuntu/software/ainex_controller/ActionGroups/*.d6a`

## How to run
```bash
# In Docker container (ubuntu user)
# 1. Ensure bringup is running
#    sudo systemctl status start_app_node.service

# 2. Run skeleton follow
bash /home/ubuntu/ros_ws/src/skeleton_follow/start.sh

# 3. Browser: http://<raspberry-pi-ip>:8080
```

## Gesture control
- **Raise both hands above head** (~0.8s) -> robot returns to standing (all servos 500)
- **Lower both hands below shoulders** -> skeleton following resumes

## Adjustable parameters (in skeleton_follow_node.py)
| Param | Default | Description |
|-------|---------|-------------|
| smooth_factor | 0.35 | Smoothing coefficient (smaller = smoother but laggier) |
| servo_duration | 200 | Servo movement time (ms) |
| MIN_VIS | 0.4 | MediaPipe landmark minimum visibility |
| Rate | 12 Hz | Main loop frequency |
| STAND_GESTURE_FRAMES | 10 | Frames needed for stand gesture (~0.8s) |
