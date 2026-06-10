cmake_minimum_required(VERSION 3.8)
project(robot_vision_pipeline_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ArucoPose.msg"
  "msg/ArucoPoseArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_package()
