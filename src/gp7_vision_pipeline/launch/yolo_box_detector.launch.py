"""Launch YOLO box detector with parameters from YAML.

RealSense image topics default in ``config/yolo_box_detector.yaml`` (e.g.
``/camera/camera/color/image_raw`` when the driver uses ``camera_frame_id:=camera``).
Override via that file or extra ``parameters:=`` on the command line.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("gp7_vision_pipeline")
    params_file = os.path.join(pkg_share, "config", "yolo_box_detector.yaml")

    return LaunchDescription(
        [
            Node(
                package="gp7_vision_pipeline",
                executable="yolo_box_detector_node",
                name="yolo_box_detector_node",
                output="screen",
                # Includes color_topic, depth_topic, camera_info_topic, model_path, thresholds, etc.
                parameters=[params_file],
            ),
        ]
    )
