"""Launch camera_projection_node.

Node nay nhan BoxDetection tu yolo_box_detector_node,
project pixel → 3D pose trong base_link su dung ma tran ngoai.

Thu tu khoi dong:
  1. RealSense driver (neu chua chay)
  2. yolo_box_detector_node (neu chua chay)
  3. camera_projection_node (node nay)

Usage::

    ros2 launch gp7_vision_pipeline camera_projection.launch.py

    # Override extrinsics
    ros2 launch gp7_vision_pipeline camera_projection.launch.py \\
        extr_tx:=-0.1 extr_ty:=-0.72 extr_tz:=0.98 \\
        extr_roll:=3.1416 extr_pitch:=0.0 extr_yaw:=0.0

Output topics:
  /vision/target_pose_in_base_link   (PoseStamped, base_link)
  /vision/box_pose_in_base_link      (PoseStamped, base_link)

Service:
  /vision/get_pixel_pose             (GetPixelPose)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_PI = "3.141592653589793"


def generate_launch_description() -> LaunchDescription:
    pkg = "gp7_vision_pipeline"
    pkg_share = get_package_share_directory(pkg)
    params_file = os.path.join(pkg_share, "config", "camera_projection.yaml")

    return LaunchDescription([
        LogInfo(msg="[camera_projection] Starting camera projection node..."),

        DeclareLaunchArgument(
            "params_file",
            default_value=params_file,
            description="Path to camera_projection.yaml",
        ),

        # Extrinsics overrides
        DeclareLaunchArgument("extr_tx", default_value="0.0"),
        DeclareLaunchArgument("extr_ty", default_value="-0.7"),
        DeclareLaunchArgument("extr_tz", default_value="1.0"),
        DeclareLaunchArgument("extr_roll", default_value=_PI),
        DeclareLaunchArgument("extr_pitch", default_value="0.0"),
        DeclareLaunchArgument("extr_yaw", default_value="0.0"),
        DeclareLaunchArgument("depth_roi_half_size", default_value="5"),

        Node(
            package=pkg,
            executable="camera_projection_node",
            name="camera_projection_node",
            output="screen",
            parameters=[
                LaunchConfiguration("params_file"),
                {
                    "extr_tx": LaunchConfiguration("extr_tx"),
                    "extr_ty": LaunchConfiguration("extr_ty"),
                    "extr_tz": LaunchConfiguration("extr_tz"),
                    "extr_roll": LaunchConfiguration("extr_roll"),
                    "extr_pitch": LaunchConfiguration("extr_pitch"),
                    "extr_yaw": LaunchConfiguration("extr_yaw"),
                    "depth_roi_half_size": LaunchConfiguration("depth_roi_half_size"),
                },
            ],
        ),
    ])
