"""Start only the vision detection marker node (no YOLO or mapper).

Use this launch file when YOLO + mapper are already running and you only need
the RViz marker visualization. Useful for:
  - Starting markers on a separate machine or container
  - Restarting RViz markers without restarting the full pipeline
  - Testing marker parameters in isolation

Usage::

    ros2 launch gp7_vision_pipeline vision_markers_only.launch.py

Verify the marker node is running::

    ros2 topic info /vision/detection_markers
    # Publisher: /vision_detection_marker_node
"""

from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = "gp7_vision_pipeline"

    launch_args = [
        DeclareLaunchArgument(
            "marker_frame_id",
            default_value="base_link",
            description="Reference frame for marker headers.",
        ),
        DeclareLaunchArgument(
            "target_marker_diameter",
            default_value="0.04",
            description="Target cylinder diameter (m).",
        ),
        DeclareLaunchArgument(
            "target_marker_height",
            default_value="0.01",
            description="Target cylinder height (m).",
        ),
        DeclareLaunchArgument(
            "target_marker_z_is_bottom",
            default_value="false",
            description=(
                "false = target_position.z is object centre (use with homography mapper). "
                "true = target_position.z is bottom contact point (add height/2 offset)."
            ),
        ),
    ]

    share = get_package_share_directory(pkg)

    nodes = [
        LogInfo(
            msg="[vision_markers_only] Starting vision_detection_marker_node only."
        ),

        Node(
            package=pkg,
            executable="vision_detection_marker_node",
            name="vision_detection_marker_node",
            output="screen",
            parameters=[
                {
                    "marker_frame_id": LaunchConfiguration("marker_frame_id"),
                    "target_marker_diameter": float(LaunchConfiguration("target_marker_diameter")),
                    "target_marker_height": float(LaunchConfiguration("target_marker_height")),
                    "target_marker_z_is_bottom": LaunchConfiguration("target_marker_z_is_bottom"),
                },
                f"{share}/config/vision_markers.yaml",
            ],
        ),
    ]

    return LaunchDescription(launch_args + nodes)
