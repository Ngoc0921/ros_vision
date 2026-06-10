"""Start the vision pipeline placeholder (no RealSense driver; no detectors).

TODO(Milestone-2+): Add composable nodes or additional includes as perception grows.

Usage::

    ros2 launch gp7_vision_pipeline vision_pipeline.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory("gp7_vision_pipeline")
    params = os.path.join(pkg, "config", "vision_pipeline.yaml")

    return LaunchDescription(
        [
            Node(
                package="gp7_vision_pipeline",
                executable="vision_pipeline_placeholder",
                name="vision_pipeline_placeholder",
                output="screen",
                parameters=[params],
            ),
        ]
    )
