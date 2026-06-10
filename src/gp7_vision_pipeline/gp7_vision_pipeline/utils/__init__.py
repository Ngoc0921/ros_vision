"""Utility modules for the gp7_vision_pipeline package.

Provides:
  depth_utils    — depth image reading, filtering, and robust center-depth estimation
  detection_visualizer — BGR image annotation (bbox, crosshair, distance overlay)
"""

from gp7_vision_pipeline.depth_utils import (
    bbox_clip_to_image,
    depth_at_pixel,
    median_depth_meters,
    robust_center_depth,
)
from gp7_vision_pipeline.detection_visualizer import (
    draw_all_detections,
    draw_no_detection,
)

__all__ = [
    "bbox_clip_to_image",
    "depth_at_pixel",
    "median_depth_meters",
    "robust_center_depth",
    "draw_all_detections",
    "draw_no_detection",
]
