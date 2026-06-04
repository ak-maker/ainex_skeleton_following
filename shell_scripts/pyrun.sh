#!/usr/bin/env bash
# =============================================================================
# pyrun — run a ROS python script from inside the ainex container
#
# Usage:
#   pyrun <path_under_ros_ws/src>
#
# Example:
#   pyrun demos/test_head_servo.py
#
# Install (from outside, using dump):
#   dump this file then: docker exec ainex chmod +x /usr/local/bin/pyrun
# =============================================================================

ROS_SCRIPT_BASE="/home/ubuntu/ros_ws/src"

if [[ $# -eq 0 ]]; then
  echo "Usage: pyrun <path_under_ros_ws/src>"
  echo "Example: pyrun demos/test_head_servo.py"
  exit 1
fi

REL_PATH="${1##/}"
SCRIPT_FULL="${ROS_SCRIPT_BASE}/${REL_PATH}"
shift  # remove the script path, leaving any extra flags in $@

if [[ ! -f "$SCRIPT_FULL" ]]; then
  echo "✗ File not found: ${SCRIPT_FULL}"
  exit 1
fi

echo "▶ pyrun  ${REL_PATH} $@"
echo ""

source /opt/ros/noetic/setup.bash
source /home/ubuntu/ros_ws/devel/setup.bash
python3 "$SCRIPT_FULL" "$@"
