# robot_vision_pipeline_msgs

Custom message definitions for `robot_vision_pipeline`.

## Package Structure

```
robot_vision_pipeline_msgs/
├── msg/
│   ├── ArucoPose.msg         # Single ArUco marker pose
│   └── ArucoPoseArray.msg    # Array of ArUco marker poses
├── CMakeLists.txt
└── package.xml
```

## Messages

### `ArucoPose.msg`

```
uint16 marker_id
geometry_msgs/Pose pose
float64 distance
---
```

| Field | Type | Description |
|-------|------|-------------|
| `marker_id` | `uint16` | ArUco marker ID |
| `pose` | `geometry_msgs/Pose` | Marker pose in camera frame |
| `distance` | `float64` | Distance from camera to marker (meters) |

### `ArucoPoseArray.msg`

```
std_msgs/Header header
ArucoPose[] markers
```

## Build

```bash
cd ~/ros2
colcon build --packages-select robot_vision_pipeline_msgs
source install/setup.bash
```

## Usage

```bash
# Inspect message
ros2 msg show robot_vision_pipeline_msgs/ArucoPose
ros2 msg show robot_vision_pipeline_msgs/ArucoPoseArray

# Echo poses
ros2 topic echo /aruco_pose
```

## Dependencies

- `std_msgs` — Header and built-in types
- `geometry_msgs` — Pose type
- `rosidl_default_generators` / `rosidl_default_runtime`
