std_msgs/Header header
string class_name
float32 confidence
int32 object_id

# Refined object center in image coordinates.
float32 center_x_px
float32 center_y_px

# Orientation from post-YOLO image processing.
float32 yaw_img_deg
float32 yaw_robot_deg

# 3-D point from camera intrinsics/depth and mapped robot/base point.
geometry_msgs/Point camera_position
geometry_msgs/Point robot_position

# Robot/base pose used for pick/visualization.
geometry_msgs/Pose pose
geometry_msgs/Vector3 size

bool refined
