"""Runtime ROS 2 nodes for the gp7_vision_pipeline package.

Architecture (3 layers):
  1. Detection  — yolo_box_detector_node:     YOLO 2D detection
  2. Mapping   — pixel_to_base_mapper_node:    pixel+depth → base_link 3D
  3. Viz       — vision_detection_marker_node:  RViz MarkerArray

Usage:
  ros2 run gp7_vision_pipeline yolo_box_detector_node
  ros2 run gp7_vision_pipeline pixel_to_base_mapper_node
  ros2 run gp7_vision_pipeline vision_detection_marker_node
"""

from gp7_vision_pipeline.yolo_box_detector_node import YoloBoxDetectorNode, main as yolo_main
from gp7_vision_pipeline.pixel_to_base_mapper_node import PixelToBaseMapperNode, main as mapper_main
from gp7_vision_pipeline.vision_detection_marker_node import VisionDetectionMarkerNode, main as marker_main

__all__ = [
    "YoloBoxDetectorNode",
    "PixelToBaseMapperNode",
    "VisionDetectionMarkerNode",
]
