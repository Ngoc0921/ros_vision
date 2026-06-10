from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    pkg_share = FindPackageShare("robot_vision_pipeline")

    default_param_file = PathJoinSubstitution([
        pkg_share,
        "config",
        "aruco_detect.yaml",
    ])

    param_file_arg = DeclareLaunchArgument(
        "param_file",
        default_value=default_param_file,
        description="Path to ArUco detection parameter file",
    )

    node = Node(
        package="robot_vision_pipeline",
        executable="aruco_detect_node",
        name="aruco_detect_node",
        output="screen",
        prefix="/home/minhquang/venvs/ros_env/bin/python3",
        parameters=[LaunchConfiguration("param_file")],
    )

    return LaunchDescription([
        param_file_arg,
        node,
    ])
