"""Launch the vision pipeline for testing with a static image (no real camera).

Nodes launched:
  1. static_image_camera_node         — fake camera from image
  2. yolo_detect_node               — YOLO detection
  3. yolo_json_to_box_detection_node — YOLO JSON -> BoxDetection
  4. pixel_to_base_mapper_node      — pixel + depth -> camera-frame pose
  5. vision_detection_marker_node    — RViz markers (optional)

Usage:
  ros2 launch robot_vision_pipeline vision_image_test.launch.py \\
    image_path:=/path/to/image.jpg \\
    model_path:=/path/to/model.pt \\
    fake_depth_m:=0.55

Arguments:
  image_path   : Path to the static test image (required)
  model_path   : Path to YOLO .pt model file
  fake_depth_m : Simulated depth in metres (default 0.55)
  use_markers  : Launch vision_detection_marker_node (default true)

Default topics (simulate RealSense D435):
  /camera/camera/color/image_raw
  /camera/camera/aligned_depth_to_color/image_raw
  /camera/camera/color/camera_info
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    pkg_share = FindPackageShare("robot_vision_pipeline")

    default_model_path = (
        "/home/asus/ros_vision/src/robot_vision_pipeline/models/best_real.pt"
    )

    image_path_arg = DeclareLaunchArgument(
        "image_path",
        default_value="",
        description="Path to the static test image",
    )
    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value=default_model_path,
        description="Path to YOLO .pt model file",
    )
    fake_depth_arg = DeclareLaunchArgument(
        "fake_depth_m",
        default_value="0.55",
        description="Simulated depth in metres for static camera",
    )
    use_markers_arg = DeclareLaunchArgument(
        "use_markers",
        default_value="true",
        description="Launch vision_detection_marker_node",
    )

    yolo_param_file = PathJoinSubstitution([pkg_share, "config", "yolo_detect_real.yaml"])
    adapter_param_file = PathJoinSubstitution([pkg_share, "config", "yolo_json_adapter.yaml"])
    mapper_param_file = PathJoinSubstitution([pkg_share, "config", "pixel_to_base_mapper.yaml"])
    marker_param_file = PathJoinSubstitution([pkg_share, "config", "vision_markers.yaml"])

    static_camera_node = Node(
        package="robot_vision_pipeline",
        executable="static_image_camera_node",
        name="static_image_camera_node",
        output="screen",
        parameters=[
            {
                "image_path": LaunchConfiguration("image_path"),
                "fake_depth_m": LaunchConfiguration("fake_depth_m"),
                "publish_rate_hz": 1.0,
                "fx": 615.0,
                "fy": 615.0,
                "image_frame_id": "camera_color_optical_frame",
                "depth_frame_id": "camera_color_optical_frame",
            }
        ],
    )

    yolo_node = Node(
        package="robot_vision_pipeline",
        executable="yolo_detect_node",
        name="yolo_detect_node",
        output="screen",
        parameters=[
            yolo_param_file,
            {"model_path_override": LaunchConfiguration("model_path")},
        ],
    )

    adapter_node = Node(
        package="robot_vision_pipeline",
        executable="yolo_json_to_box_detection_node",
        name="yolo_json_to_box_detection_node",
        output="screen",
        parameters=[
            adapter_param_file,
            {
                "fake_depth_m": LaunchConfiguration("fake_depth_m"),
                "use_fake_depth": True,
            },
        ],
    )

    mapper_node = Node(
        package="robot_vision_pipeline",
        executable="pixel_to_base_mapper_node",
        name="pixel_to_base_mapper_node",
        output="screen",
        parameters=[
            mapper_param_file,
            {
                "fake_depth_m": LaunchConfiguration("fake_depth_m"),
                "use_fake_depth_if_missing": True,
            },
        ],
    )

    marker_node = Node(
        package="robot_vision_pipeline",
        executable="vision_detection_marker_node",
        name="vision_detection_marker_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_markers")),
        parameters=[marker_param_file],
    )

    return LaunchDescription([
        LogInfo(msg="[vision_image_test] Starting static-image vision pipeline..."),
        image_path_arg,
        model_path_arg,
        fake_depth_arg,
        use_markers_arg,
        static_camera_node,
        yolo_node,
        adapter_node,
        mapper_node,
        marker_node,
    ])
