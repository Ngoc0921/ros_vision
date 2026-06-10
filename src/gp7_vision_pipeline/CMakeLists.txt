cmake_minimum_required(VERSION 3.8)
project(gp7_vision_pipeline)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoxDetection.msg"
  "msg/Box.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# Do not use ament_python_install_package() here: it conflicts with
# rosidl_generate_interfaces() for the same project name. Install our
# Python modules next to the generated msg package.
ament_get_python_install_dir(python_install_dir)
install(
  DIRECTORY gp7_vision_pipeline/
  DESTINATION "${python_install_dir}/gp7_vision_pipeline"
  FILES_MATCHING PATTERN "*.py"
)

install(
  PROGRAMS
  # ── Runtime nodes ────────────────────────────────────────────────
  scripts/yolo_box_detector_node
  scripts/pixel_to_base_mapper_node
  scripts/vision_detection_marker_node
  # ── Calibration tools ────────────────────────────────────────────
  scripts/tools/compute_pixel_to_base_homography
  scripts/tools/test_pixel_to_base_homography
  # ── Optional debug tool ──────────────────────────────────────────
  scripts/tools/realsense_depth_debug
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config models model
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
