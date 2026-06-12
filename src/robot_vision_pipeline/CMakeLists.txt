cmake_minimum_required(VERSION 3.8)
project(robot_vision_pipeline)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArucoPose.msg"
  "msg/ArucoPoseArray.msg"
  "msg/BoxDetection.msg"
  "msg/Box.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)

# Do not use ament_python_install_package() here: it conflicts with
# rosidl_generate_interfaces() for the same project name. Install Python
# modules next to the generated msg package, same style as gp7_vision_pipeline.
ament_get_python_install_dir(python_install_dir)
install(
  DIRECTORY robot_vision_pipeline/
  DESTINATION "${python_install_dir}/robot_vision_pipeline"
  FILES_MATCHING PATTERN "*.py"
)

install(
  PROGRAMS
  scripts/yolo_detect_node
  scripts/yolo_detect_node_v1
  scripts/yolo_json_to_box_detection_node
  scripts/pixel_to_base_mapper_node
  scripts/vision_detection_marker_node
  scripts/realsense_depth_debug
  scripts/aruco_detect_node
  scripts/vision_gui
  scripts/vision_gui_astra
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config models
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
