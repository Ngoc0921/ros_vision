"""Launch the GP7 vision pipeline.

Three-layer architecture:

  Layer 1 — Detection :  RealSense → YOLO box detector
  Layer 2 — Mapping  :  YOLO + depth → base_link (homography)
  Layer 3 — Viz     :  base_link → RViz MarkerArray

RealSense driver is started automatically unless use_camera:=false.

Environment::

    source /opt/ros/humble/setup.bash
    source ~/yolo_env/bin/activate    # provides ultralytics
    source ~/pap_yaskawa_ws/install/setup.bash

Usage::

    # Default: camera + YOLO + mapper + markers
    ros2 launch gp7_vision_pipeline vision_full_pipeline.launch.py

    # Skip camera (topics already exist)
    ros2 launch gp7_vision_pipeline vision_full_pipeline.launch.py use_camera:=false

    # With RViz
    ros2 launch gp7_vision_pipeline vision_full_pipeline.launch.py use_rviz:=true

    # Override YOLO confidence
    ros2 launch gp7_vision_pipeline vision_full_pipeline.launch.py conf_threshold:=0.5

For a simpler setup (no camera driver, no RViz)::

    ros2 launch gp7_vision_pipeline vision_no_camera.launch.py

To start only the marker node::

    ros2 launch gp7_vision_pipeline vision_markers_only.launch.py
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_pipeline = "gp7_vision_pipeline"
    pkg_bringup = "gp7_bringup"

    pipeline_share = get_package_share_directory(pkg_pipeline)
    bringup_share = get_package_share_directory(pkg_bringup)

    realsense_d435_launch = os.path.join(bringup_share, "launch", "realsense_d435.launch.py")

    # ── Default values ──────────────────────────────────────────────────────────
    DEFAULT_MODEL_PATH = (
        "/home/norman/pap_yaskawa_ws/src/gp7_vision_pipeline/model/box_target/weights/best.pt"
    )

    # ── Launch arguments ───────────────────────────────────────────────────────
    launch_args = [
        # Camera driver
        DeclareLaunchArgument(
            "use_camera",
            default_value="true",
            description="Start the RealSense D435 driver. Set false if already running.",
        ),
        # YOLO detector
        DeclareLaunchArgument(
            "model_path",
            default_value=DEFAULT_MODEL_PATH,
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
            description="RGB image topic from RealSense.",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw",
            description="Aligned depth image topic from RealSense.",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera/color/camera_info",
            description="Camera info topic from RealSense.",
        ),
        # Mapper and markers
        DeclareLaunchArgument(
            "frame_id",
            default_value="base_link",
            description="Reference frame for marker nodes.",
        ),
        # RViz
        DeclareLaunchArgument(
            "use_rviz",
            default_value="false",
            description="Launch RViz2.",
        ),
        DeclareLaunchArgument(
            "use_homography_mapper",
            default_value="true",
            description=(
                "Start pixel_to_base_mapper_node. Set false to disable base-frame "
                "mapping (/vision/target_position, /vision/box, /vision/debug_image_base)."
            ),
        ),
        DeclareLaunchArgument(
            "use_detection_markers",
            default_value="true",
            description=(
                "Start vision_detection_marker_node. Publishes /vision/detection_markers "
                "(blue CYLINDER for target, yellow CUBE for box) to RViz. "
                "Recommended: true in most cases."
            ),
        ),
    ]

    # ── RealSense driver arguments ────────────────────────────────────────────
    # These are the ONLY launch_arguments passed to realsense_d435.launch.py.
    # Non-RealSense params (model_path, conf_threshold, etc.) are NOT forwarded here.
    realsense_launch_args = [
        ("rgb_camera.color_profile", "848x480x30"),
        ("depth_module.depth_profile", "848x480x30"),
        ("enable_color", "true"),
        ("enable_depth", "true"),
        ("align_depth.enable", "true"),
        ("pointcloud.enable", "false"),
    ]

    # ── Nodes ─────────────────────────────────────────────────────────────────
    nodes: list = [
        LogInfo(
            msg="[vision_full_pipeline] Starting vision pipeline..."
        ),

        # A. RealSense D435 camera driver (only when use_camera:=true)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(realsense_d435_launch),
            condition=IfCondition(LaunchConfiguration("use_camera")),
            launch_arguments=realsense_launch_args,
        ),

        # B. YOLO box detector node — always started
        Node(
            package=pkg_pipeline,
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

        # C. Pixel-to-base mapper node — converts detections to base_link via homography
        Node(
            package=pkg_pipeline,
            executable="pixel_to_base_mapper_node",
            name="pixel_to_base_mapper_node",
            output="screen",
            parameters=[
                {
                    "homography_yaml": "config/pixel_to_base_homography.yaml",
                    "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                    "base_debug_image_input_topic": LaunchConfiguration("color_topic"),
                    "depth_topic": LaunchConfiguration("depth_topic"),
                },
                f"{pipeline_share}/config/pixel_to_base_mapper.yaml",
            ],
            condition=IfCondition(LaunchConfiguration("use_homography_mapper")),
        ),

        # D. Vision detection marker node — publishes /vision/detection_markers for RViz
        Node(
            package=pkg_pipeline,
            executable="vision_detection_marker_node",
            name="vision_detection_marker_node",
            output="screen",
            parameters=[
                {
                    "marker_frame_id": LaunchConfiguration("frame_id"),
                },
                f"{pipeline_share}/config/vision_markers.yaml",
            ],
            condition=IfCondition(LaunchConfiguration("use_detection_markers")),
        ),

        # E. RViz2
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(LaunchConfiguration("use_rviz")),
        ),
    ]

    return LaunchDescription(launch_args + nodes)
