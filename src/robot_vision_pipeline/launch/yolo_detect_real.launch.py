from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    pkg_share = FindPackageShare("robot_vision_pipeline")
    realsense_pkg_share = FindPackageShare("realsense2_camera")

    default_param_file = PathJoinSubstitution([
        pkg_share,
        "config",
        "yolo_detect_real.yaml",
    ])

    default_rs_config_file = PathJoinSubstitution([
        pkg_share,
        "config",
        "rs_camera_yolo.yaml",
    ])

    param_file_arg = DeclareLaunchArgument(
        "param_file",
        default_value=default_param_file,
        description="Path to YOLO detection parameter file",
    )

    rs_config_file_arg = DeclareLaunchArgument(
        "rs_config_file",
        default_value=default_rs_config_file,
        description="Path to RealSense camera parameter file",
    )

    model_path_arg = DeclareLaunchArgument(
        "model_path",
        default_value="",
        description="Override YOLO model path. Empty means use YAML config.",
    )

    image_topic_arg = DeclareLaunchArgument(
        "image_topic",
        default_value="/camera/camera/color/image_raw",
        description="Override input image topic.",
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                realsense_pkg_share,
                "launch",
                "rs_launch.py",
            ])
        ),
        launch_arguments={
            "config_file": LaunchConfiguration("rs_config_file"),
        }.items(),
    )

    node = Node(
        package="robot_vision_pipeline",
        executable="yolo_detect_node",
        name="yolo_detect_node",
        output="screen",
        # Ép node chạy bằng Python trong venv ros_env
        prefix="/home/minhquang/venvs/ros_yolo/bin/python3",
        parameters=[
            LaunchConfiguration("param_file"),
            {
                "model_path_override": LaunchConfiguration("model_path"),
                "image_topic_override": LaunchConfiguration("image_topic"),
            },
        ],
    )

    return LaunchDescription([
        param_file_arg,
        rs_config_file_arg,
        model_path_arg,
        image_topic_arg,
        realsense_launch,
        node,
    ])
