# AiNex Skeleton Following

Pose mimic system for the AiNex humanoid robot. Uses MediaPipe to detect human poses and drive arm servos in real-time.

## Files

| File | Description |
|------|-------------|
| `pose_mimic_node.py` | 2D approach — uses screen coordinates for sho_roll (15/16) and el_pitch (17/18). Proven working. |
| `pose_mimic_3d_node.py` | 3D approach — adds world coordinates for sho_pitch (13/14) forward/backward and elbow bend (19/20). Uses new MediaPipe Tasks API with VIDEO mode. |
| `skeleton_follow_node.py` | Original skeleton follow node (unmodified). |
| `servo_controller.yaml` | Servo configuration (IDs, init values, min/max). Copied from ainex_driver. |
| `skeleton_follow.launch` | ROS launch file. |
| `model/pose_landmarker_lite.task` | MediaPipe PoseLandmarker lite model (~5.5MB). |

## Running

```bash
# Start the 3D node
bash -c 'source /opt/ros/noetic/setup.bash && source /home/ubuntu/ros_ws/devel/setup.bash && python3 /home/ubuntu/ros_ws/src/skeleton_follow/pose_mimic_3d_node.py'

# Start web video server (in another terminal)
bash -c 'source /opt/ros/noetic/setup.bash && source /home/ubuntu/ros_ws/devel/setup.bash && rosrun web_video_server web_video_server _port:=8080'

# View stream at: http://192.168.8.219:8080/stream?topic=/pose_mimic/image_result

# Kill the node
pkill -9 -f pose_mimic_3d_node.py

# Stand the robot
bash -c 'source /opt/ros/noetic/setup.bash && source /home/ubuntu/ros_ws/devel/setup.bash && python3 -c "import rospy; from ainex_kinematics.motion_manager import MotionManager; rospy.init_node(\"s\",anonymous=True); MotionManager().run_action(\"stand\"); print(\"ok\")"'
```

## Servo Mapping

### Arm Servos (ID 13-22)

| ID | YAML Name | Actual Function | Direction | Stand |
|----|-----------|----------------|-----------|-------|
| 13 | l_sho_pitch | Left shoulder forward/backward | big=back, small=forward | 835 |
| 14 | r_sho_pitch | Right shoulder forward/backward | small=back, big=forward | 165 |
| 15 | l_sho_roll | Left shoulder lateral raise | small=raised, big=down | 830 |
| 16 | r_sho_roll | Right shoulder lateral raise | small=down, big=raised | 170 |
| 17 | l_el_pitch | **Actually** left forearm rotation | small=CCW, big=CW | 500 |
| 18 | r_el_pitch | **Actually** right forearm rotation | small=CCW, big=CW | 500 |
| 19 | l_el_yaw | **Actually** left elbow bend | small=bent, 600=straight, range 0-600 | 150 |
| 20 | r_el_yaw | **Actually** right elbow bend | big=bent, 450=straight, range 360-850 (BURNED) | 850 |
| 21 | l_gripper | Left gripper | — | 500 |
| 22 | r_gripper | Right gripper | — | 500 |

**NOTE**: YAML names for servo 17/18 and 19/20 are swapped vs actual physical function.

### Servo Unit System

- 0-1000 pulse units = 240 degrees of servo rotation
- 125-875 = 750 units = 180 degrees (TonyPi's 1:1 angle mapping: 4.17 units/degree)
- Left and right servos are mirror-mounted, so same physical movement requires opposite pulse directions

### Elbow Mapping (why NOT TonyPi 125-875)

TonyPi uses 125-875 = 750 units = 180° for a clean 1:1 mapping. This works when the servo's full useful range (straight to fully bent) fits within 125-875. However:

- **Servo 19 (left elbow bend)**: physical straight position is at **600**, not 875. Range is **0-600**.
  - If we used TonyPi mapping (125-875), straight(180°) would map to 875 — far beyond the physical limit of 600.
  - Correct mapping: 0°(bent)→0, 180°(straight)→600.
  - Actual servo rotation: 600 units × 0.24°/unit = **144° of servo rotation** for 180° elbow angle.
  - So the ratio is 0.8° servo rotation per 1° elbow angle (not 1:1).
- **Servo 20 (right elbow bend)**: physical straight at **450**, range **360-850** (BURNED, held at stand).
  - Similar issue: TonyPi's 125 would be below the physical minimum of 360.

**Lesson**: TonyPi's 125-875 mapping is an ideal that assumes all servos have symmetric, centered ranges. Real servos may have asymmetric physical limits. Always map to the **actual physical range** (bent position → straight position) rather than forcing a theoretical mapping.

### Unit conversion reference

- 1 unit = 0.24° (240° / 1000 units)
- 1° = ~4.17 units (1000 / 240)
- TonyPi range: 750 units × 0.24° = 180° servo rotation → 1:1 with 180° joint angle
- Servo 19 range: 600 units × 0.24° = 144° servo rotation → 0.8:1 with 180° elbow angle

## MediaPipe Coordinate Systems

### Normalized (screen) coordinates
- X = right, Y = down, Z = out of screen

### World coordinates (origin = center of hips)
- X = person's left
- Y = down
- Z = camera lens direction (away from camera, into scene)
- When person faces camera: Z > 0 = toward person's back, Z < 0 = forward

### Landmark Identity
- Landmark 11 = person's own LEFT shoulder (anatomical, per Google docs)
- Landmark 12 = person's own RIGHT shoulder
- **With flipped image**: MediaPipe doesn't know the image is flipped, so landmark 11 in the flipped image corresponds to the person's actual RIGHT side
- This gives correct mirror behavior: landmark 11 -> robot LEFT servos

## Mistakes Log

Detailed in `pose_mimic_3d_node.py` header comments. Summary:

1. **Burned servo 20** — sent pulse 202 when hardware minimum is 360. Used theoretical 0-1000 range instead of tested safe range. NEVER exceed physical servo limits.
2. **sho_pitch direction flipped multiple times** — confused world Z direction (Z > 0 = backward, not forward) with servo direction (mirror-mounted = opposite pulse for same movement).
3. **YAML elbow names are wrong** — el_pitch (17/18) is actually rotation, el_yaw (19/20) is actually bend.
4. **sho_roll world coordinates unreliable** — switched to 2D screen coordinates (same as working pose_mimic_node.py).
5. **World Y-axis assumed up** — actually Y-down, caused 180-degree angle offset.
6. **Landmark 11/12 identity confusion** — docs say person's left/right, but with flipped image the mapping reverses.
7. **World coordinate axes undocumented** — Google docs don't clearly specify X/Y/Z directions. Determined empirically: X=left, Y=down, Z=camera lens direction.
8. **Elbow mapping used TonyPi 125-875 blindly** — Servo 19's straight position is at 600, not 875. Mapping 180°→875 would command the servo far past its physical limit. Must map to actual physical range (0-600), not theoretical TonyPi range.

## Anti-Twitch

- MediaPipe Tasks API VIDEO mode provides built-in temporal smoothing
- Sliding window (6 frames, send every 3)
- Deadzone of 25 pulse units
- Crossed-arms gesture returns robot to standing position
