"""Start vision pipeline when RealSense topics already exist.

Use this launch file when RealSense is running separately or from another workspace.
All three runtime layers are started without the camera driver.

Usage::

    ros2 launch gp7_vision_pipeline vision_no_camera.launch.py

Or override camera topics::

    ros2 launch gp7_vision_pipeline vision_no_camera.launch.py \\
        color_topic:=/my_camera/color/image_raw \\
        depth_topic:=/my_camera/aligned_depth/image_raw \\
        camera_info_topic:=/my_camera/color/camera_info
"""

from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = "gp7_vision_pipeline"

    launch_args = [
        DeclareLaunchArgument(
            "model_path",
            default_value=(
                "/home/norman/pap_yaskawa_ws/src/gp7_vision_pipeline"
                "/model/box_target/weights/best.pt"
            ),
            description="Path to YOLO .pt model file.",
        ),
        DeclareLaunchArgument(
            "conf_threshold",
            default_value="0.8",
            description="YOLO confidence threshold.",
        ),
        DeclareLaunchArgument(
            "color_topic",
            default_value="/camera/camera/color/image_raw",
            description="RGB image topic from camera.",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw",
            description="Aligned depth image topic from camera.",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera/color/camera_info",
            description="Camera info topic.",
        ),
        DeclareLaunchArgument(
            "frame_id",
            default_value="base_link",
            description="Reference frame for markers.",
        ),
        DeclareLaunchArgument(
            "use_detection_markers",
            default_value="true",
            description="Start vision_detection_marker_node (publishes /vision/detection_markers).",
        ),
    ]

    share = get_package_share_directory(pkg)

    nodes = [
        LogInfo(
            msg="[vision_no_camera] Starting vision pipeline without camera driver."
        ),

        # Layer 1 — YOLO detector
        Node(
            package=pkg,
            executable="yolo_box_detector_node",
            name="yolo_box_detector_node",
            output="screen",
            parameters=[
                {
                    "model_path": LaunchConfiguration("model_path"),
                    "conf_threshold": LaunchConfiguration("conf_threshold"),
                    "color_topic": LaunchConfiguration("color_topic"),
                    "depth_topic": LaunchConfiguration("depth_topic"),
                    "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                },
            ],
        ),

        # Layer 2 — Pixel to base mapper
        Node(
            package=pkg,
            executable="pixel_to_base_mapper_node",
            name="pixel_to_base_mapper_node",
            output="screen",
            parameters=[
                {
                    "homography_yaml": "config/pixel_to_base_homography.yaml",
                    "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                    "base_debug_image_input_topic": LaunchConfiguration("color_topic"),
                    "depth_topic": LaunchConfiguration("depth_topic"),
                    # All other mapper params come from config/pixel_to_base_mapper.yaml
                },
                f"{share}/config/pixel_to_base_mapper.yaml",
            ],
        ),

        # Layer 3 — Detection markers (RViz)
        Node(
            package=pkg,
            executable="vision_detection_marker_node",
            name="vision_detection_marker_node",
            output="screen",
            parameters=[
                {
                    "marker_frame_id": LaunchConfiguration("frame_id"),
                },
                f"{share}/config/vision_markers.yaml",
            ],
            condition=IfCondition(LaunchConfiguration("use_detection_markers")),
        ),
    ]

    return LaunchDescription(launch_args + nodes)
