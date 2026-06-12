#!/usr/bin/env python3
"""Compatibility wrapper for robot_vision_pipeline Layer 3 marker node.

The maintained implementation lives in
``robot_vision_pipeline.pose_estimation.vision_detection_marker_node``.

Subscribes to:
  /vision/wood_objects    WoodArray
  /vision/box_objects   BoxArray

Publishes:
  /vision/detection_markers   visualization_msgs/MarkerArray
"""

from robot_vision_pipeline.pose_estimation.vision_detection_marker_node import (  # noqa: F401
    VisionDetectionMarkerNode,
    main,
)


if __name__ == "__main__":
    main()
