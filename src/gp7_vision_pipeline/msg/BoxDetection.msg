std_msgs/Header header
string class_name
float32 confidence
int32 x_min
int32 y_min
int32 x_max
int32 y_max
int32 center_x
int32 center_y
int32 width_px
int32 height_px
# Depth fields used by pixel_to_base_mapper_node for height computation.
# For encoding "16UC1": values are millimetres as-is.
# For encoding "32FC1": values are stored as millimetres (original_m * 1000).
uint16 center_raw_depth
uint16 roi_median_raw_depth
string depth_encoding
float32 distance_m
