#!/bin/bash
# Skeleton Pose Mimic v2 - startup script
# Usage: bash /home/ubuntu/ros_ws/src/skeleton_follow/start.sh
#
# Prerequisites: bringup must be running (usually auto-started on boot)
#   Check: sudo systemctl status start_app_node.service
#   Manual: roslaunch ainex_bringup bringup.launch

echo "=========================================="
echo "  Skeleton Pose Mimic v2"
echo "=========================================="
echo ""
echo "Controls: arms (sho pitch/roll + elbow) + legs (hip/knee/ankle)"
echo "NO head servos."
echo ""
echo "Stand in front of the camera and strike a pose!"
echo "Raise BOTH hands above head -> return to standing (all 500)"
echo ""
echo "Browser view: http://$(hostname -I | awk '{print $1}'):8080"
echo ""
echo "Press Ctrl+C to exit (robot returns to standing)"
echo ""

python3 /home/ubuntu/ros_ws/src/skeleton_follow/skeleton_follow_node.py
