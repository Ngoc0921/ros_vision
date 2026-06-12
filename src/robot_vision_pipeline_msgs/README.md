# robot_vision_pipeline_msgs

Custom message definitions for `robot_vision_pipeline`.

## Package Structure

```
robot_vision_pipeline_msgs/
├── msg/
│   ├── BoxDetection.msg   # Raw YOLO bounding box + depth
│   ├── Wood.msg          # Wood object with 3D pose
│   ├── WoodArray.msg     # Array of Wood objects
│   ├── Box.msg          # Box/obstacle with 3D pose and size
│   └── BoxArray.msg     # Array of Box objects
├── CMakeLists.txt
└── package.xml
```

## Messages

### `BoxDetection.msg`

Raw bounding-box detection from YOLO (used for both wood and box).

| Field | Type | Description |
|-------|------|-------------|
| `header` | `std_msgs/Header` | Timestamp and frame |
| `class_name` | `string` | `"wood"` or `"box"` |
| `confidence` | `float32` | YOLO confidence score |
| `object_id` | `int32` | Detection ID in current frame |
| `x_min, y_min, x_max, y_max` | `int32` | Bounding box pixel coordinates |
| `center_x, center_y` | `int32` | Bounding box center |
| `width_px, height_px` | `int32` | Bounding box dimensions |
| `center_raw_depth` | `uint16` | Raw depth at bbox center (mm) |
| `roi_median_raw_depth` | `uint16` | Median raw depth in ROI (mm) |
| `depth_encoding` | `string` | Depth encoding (`16UC1` or `32FC1`) |
| `distance_m` | `float32` | Distance in metres |

### `Wood.msg`

Wood object (target to pick) with 3D pose in camera frame.

| Field | Type | Description |
|-------|------|-------------|
| `header` | `std_msgs/Header` | Timestamp and frame |
| `wood_id` | `int32` | Wood object ID |
| `class_name` | `string` | Always `"wood"` |
| `confidence` | `float32` | Detection confidence |
| `pose` | `geometry_msgs/Pose` | 3D pose (camera frame) |

### `WoodArray.msg`

```
std_msgs/Header header
robot_vision_pipeline_msgs/Wood[] woods
```

### `Box.msg`

Box/obstacle with 3D pose and estimated physical size.

| Field | Type | Description |
|-------|------|-------------|
| `header` | `std_msgs/Header` | Timestamp and frame |
| `box_id` | `int32` | Box object ID |
| `class_name` | `string` | Always `"box"` |
| `confidence` | `float32` | Detection confidence |
| `pose` | `geometry_msgs/Pose` | 3D pose (camera frame) |
| `size` | `geometry_msgs/Vector3` | Estimated size (x, y, z) in metres |

### `BoxArray.msg`

```
std_msgs/Header header
robot_vision_pipeline_msgs/Box[] boxes
```

## Build

```bash
colcon build --packages-select robot_vision_pipeline_msgs
source install/setup.bash
```

## Usage

```bash
# Inspect messages
ros2 msg show robot_vision_pipeline_msgs/BoxDetection
ros2 msg show robot_vision_pipeline_msgs/Wood
ros2 msg show robot_vision_pipeline_msgs/Box
```
