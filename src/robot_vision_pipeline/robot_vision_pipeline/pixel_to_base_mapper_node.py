#!/usr/bin/env python3
"""Compatibility entry point for the pixel-to-base mapper node.

The maintained implementation lives in
``robot_vision_pipeline.pose_estimation.pixel_to_base_mapper_node``.

Inputs:
  /vision/wood_detection         BoxDetection
  /vision/box_detection         BoxDetection
  /camera/camera/color/camera_info
  /camera/camera/aligned_depth_to_color/image_raw
  /camera/camera/color/image_raw

Outputs:
  /vision/wood_objects          WoodArray
  /vision/box_objects          BoxArray
  /vision/debug_image_camera    sensor_msgs/Image
  /vision/detection_status     std_msgs/String

Note: coordinates are in camera_color_optical_frame. Camera -> base_link
transform is NOT implemented yet.
"""

from robot_vision_pipeline.pose_estimation.pixel_to_base_mapper_node import (  # noqa: F401
    PixelToBaseMapperNode,
    main,
)


if __name__ == "__main__":
    main()
