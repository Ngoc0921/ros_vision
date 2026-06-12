from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    pkg_share = FindPackageShare("robot_vision_pipeline")
    realsense_pkg_share = FindPackageShare("realsense2_camera")

    yolo_param_file = PathJoinSubstitution([
        pkg_share, "config", "yolo_detect_real.yaml"
    ])
    rs_config_file = PathJoinSubstitution([
        pkg_share, "config", "rs_camera.yaml"
    ])
    adapter_param_file = PathJoinSubstitution([
        pkg_share, "config", "yolo_json_adapter.yaml"
    ])
    mapper_param_file = PathJoinSubstitution([
        pkg_share, "config", "pixel_to_base_mapper.yaml"
    ])
    marker_param_file = PathJoinSubstitution([
        pkg_share, "config", "vision_markers.yaml"
    ])

    use_camera_arg = DeclareLaunchArgument("use_camera", default_value="true")
    use_mapper_arg = DeclareLaunchArgument("use_mapper", default_value="true")
    use_markers_arg = DeclareLaunchArgument("use_markers", default_value="true")
    model_path_arg = DeclareLaunchArgument("model_path", default_value="")
    image_topic_arg = DeclareLaunchArgument(
        "image_topic", default_value="/camera/camera/color/image_raw"
    )

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([realsense_pkg_share, "launch", "rs_launch.py"])
        ),
        condition=IfCondition(LaunchConfiguration("use_camera")),
        launch_arguments={"config_file": rs_config_file}.items(),
    )

    yolo_node = Node(
        package="robot_vision_pipeline",
        executable="yolo_detect_node",
        name="yolo_detect_node",
        output="screen",
        parameters=[
            yolo_param_file,
            {
                "model_path_override": LaunchConfiguration("model_path"),
                "image_topic_override": LaunchConfiguration("image_topic"),
                "image_qos": "best_effort",
            },
        ],
    )

    adapter_node = Node(
        package="robot_vision_pipeline",
        executable="yolo_json_to_box_detection_node",
        name="yolo_json_to_box_detection_node",
        output="screen",
        parameters=[
            adapter_param_file,
            {"use_fake_depth": False},
        ],
    )

    mapper_node = Node(
        package="robot_vision_pipeline",
        executable="pixel_to_base_mapper_node",
        name="pixel_to_base_mapper_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_mapper")),
        parameters=[
            mapper_param_file,
            {"use_fake_depth_if_missing": False},
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
        LogInfo(msg="[vision_full_pipeline] Starting full vision pipeline..."),
        use_camera_arg,
        use_mapper_arg,
        use_markers_arg,
        model_path_arg,
        image_topic_arg,
        realsense_launch,
        yolo_node,
        adapter_node,
        mapper_node,
        marker_node,
    ])
