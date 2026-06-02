# Week Goals

Week 1-2: Get familiar with the system, in terms of the vision tracking, robot control, interface. Goal: Scott can pre-define some demos of the robot.

Also, determine why the image is so blurry in the livestream pipeline

# Weekly Agenda

## Tuesday 5/26

- Orientation events and workshops
- Orientation with Jiewen

## Wednesday 5/27

- Get set up with Slack, create plan and documents, set automations, outline concrete plan
- Review the mediapipe-based script—annotating the file
- Making progress on CITI research ethics certifications

## Thursday 5/28

- Attend workshops with library services and Dr Guo
- Progress on CITI research ethics certifications

## Friday 5/29 (Now I have lab access)

- Test robot servos and experiment with creating new scripts in the GUI
- Complete CITI research ethics
- Research ROS and seek to gain foundational understanding
- Continue annotating the mediapipe-based script
- Begin writing simple demo script to gain familiarity with AiNex / ROS

*Note: For most of the week, I could not yet access the lab, so most of my work involved exploring code without the robot and completing the necessary CITI certification*

# Future Tasks

- Finish annotating the mediapipe-based script
- Do more literature review (find articles, journals, papers addressing similar problems)
- Imagine some pre-defined demos of the robot (write my own script):
  - Easy: Robot performs a sequence (no vision model)
  - Medium: Robot raises whichever hand is higher in PoseLandmarker model
  - Hard: Robot moves head to track human subject and center them
- Check where the livestreaming blur is coming from. It likely isn't the PoseLandmarker, since that never touches the images. Try creating a webserver streaming script that shows just the raw camera footage, to see if it's a camera hardware issue.
- Configure to run Claude code in the GUI

# Questions
- Is there a reason for the assumption that the arm is pointed roughly along the x-axis?

```python
# Compute rotation angle using Y and Z components of projected vector
rotation_angle = math.degrees(math.atan2(palm_perp[2], palm_perp[1]))
```

AI suggestion (Claude): 

"""

The more rigorous approach would be:
- Pick an arbitrary reference vector in the perpendicular plane (e.g. world Y projected onto it)
- Use atan2 of the cross and dot product between palm_perp and that reference

"""

# Ideas

- **Adjusting confidence threshold**: I wonder if the confidence threshold for the MediaPipe images should be increased. During the demos, it felt like often it was trying to build the skeleton on a frame that wasn't ideal for accurate landmarks. This could prevent the robot from doing risky movements.
- **Shoulder servo control adjustment**: I noticed during the demo that the shoulder rotation seems to awkwardly match the image in a way that doesn't quite seem to align with how the human actually moves their arm. It might be worthwhile to further understand how the "angle calculation" works.
- **Reducing Model Calls (related to "most stable skeleton")**: I wonder if we could speed up our robot by reducing the number of calls to the landmark model. Perhaps a freeze frame approach might also keep us from being subject to noisy passing gestures. But the problem then becomes we become "sampling" at a lower rate, and we might capture a passing gesture.
- **Person Tracking with Head Servo**: It might be helpful to add a "tracking" feature, where the head moves to try to keep the center of the person centered in the image. This potentially can be expanded to include tracking using full body rotations (yaw) to keep the person centered, and also following a person using walking maneuvers. We could even expand this to include "search mode", where the robot spins in a circle ("yaw") searching for a person.

# Related Literature

- Computer vision-based hand gesture recognition for human-robot interaction: a review (https://link.springer.com/article/10.1007/s40747-023-01173-6)
