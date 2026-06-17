To run the full pipeline:

- SSH into the pi, and move into docker
- In four separate terminals, run:

pyrun pose_classifier/publisher.py
pyrun pose_classifier/head.py
pyrun pose_classifier/control.py
pyrun pose_classifier/webserver.py

Each of those processes, corresponds to a ROS node