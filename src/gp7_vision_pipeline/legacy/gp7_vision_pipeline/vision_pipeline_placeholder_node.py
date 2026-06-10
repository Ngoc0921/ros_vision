"""Placeholder node for future D435 perception orchestration.

TODO(M3+): YOLO box detection (2D). TODO(M4+): ROI + aligned depth + intrinsics → 3D center
and approximate L×W×H in ``base_link`` (not from camera semantic APIs). TODO(M5+): downstream
simulation via a separate Gazebo bridge package — not here.

This node must never command the robot; motion stays in MoveIt / ``gp7_task_executor``.
"""

from __future__ import annotations

import sys

import rclpy
from rclpy.node import Node


class VisionPipelinePlaceholderNode(Node):
    """Spins idle while exposing parameters reserved for the real pipeline."""

    def __init__(self) -> None:
        super().__init__("vision_pipeline_placeholder")

        # Eye-to-hand: static TF base_link → camera_link from gp7_vision_bringup.
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("camera_link_frame", "camera_link")
        self.declare_parameter("camera_color_optical_frame", "camera_color_optical_frame")
        self.declare_parameter("target_frame", "base_link")

        # Remap if ``realsense2_camera`` namespace differs.
        self.declare_parameter("rgb_image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_image_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

        self.get_logger().info(
            "gp7_vision_pipeline placeholder running; no image processing yet (see TODOs in source)."
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VisionPipelinePlaceholderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
