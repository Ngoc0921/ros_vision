#!/usr/bin/env python3
"""Compatibility entry point for the pixel-to-base mapper node.

The maintained implementation lives in
``robot_vision_pipeline.pose_estimation.pixel_to_base_mapper_node``. Keeping this
module lets old imports and direct runs use the same node contract.

Inputs:
  /vision/wood_detection
  /camera/camera/color/image_raw
  /camera/camera/aligned_depth_to_color/image_raw
  /camera/camera/color/camera_info

Outputs:
  /vision/objects                           robot_vision_pipeline_msgs/ObjectArray
  /vision/debug_image_camera                 sensor_msgs/Image
  /vision/detection_status                  std_msgs/String
"""

from robot_vision_pipeline.pose_estimation.pixel_to_base_mapper_node import (  # noqa: F401
    PixelToBaseMapperNode,
    main,
)


if __name__ == "__main__":
    main()
