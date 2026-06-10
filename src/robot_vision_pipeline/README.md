# robot_vision_pipeline

Robot vision pipeline using YOLO for real-time object detection and ArUco marker detection for pose estimation in ROS 2.

## Package Structure

```
robot_vision_pipeline/
├── robot_vision_pipeline/
│   ├── aruco/                   # ArUco detection module
│   ├── vision_gui/              # Vision GUI module
│   └── yolo/                    # YOLO detection module
├── scripts/
│   ├── yolo_detect_node         # YOLO detection executable
│   ├── aruco_detect_node         # ArUco detection executable
│   ├── vision_gui               # Qt vision GUI executable
│   └── vision_gui_astra         # Astra camera vision GUI
├── launch/
│   ├── yolo_detect_real.launch.py
│   ├── yolo_detect_sim.launch.py
│   └── aruco_detect.launch.py
├── config/
│   ├── rs_camera.yaml           # RealSense config (depth + aligned depth)
│   ├── rs_camera_yolo.yaml      # RealSense config (color only, for YOLO)
│   ├── yolo_detect_real.yaml     # YOLO parameters
│   └── aruco_detect.yaml        # ArUco parameters
├── models/
│   └── best_real.pt             # Trained YOLO model
├── msg/
│   ├── ArucoPose.msg
│   └── ArucoPoseArray.msg
├── test/
├── CMakeLists.txt
└── package.xml
```

## Build

```bash
cd ~/ros2
colcon build --packages-select robot_vision_pipeline
source install/setup.bash

# If using virtualenv for YOLO/ArUco
source ~/venvs/ros_env/bin/activate
```

## YOLO Detection

### Run with RealSense D435

```bash
ros2 launch robot_vision_pipeline yolo_detect_real.launch.py
```

### Override at Runtime

```bash
# Different model
ros2 launch robot_vision_pipeline yolo_detect_real.launch.py \
  model_path:=/path/to/model.pt

# Different image topic
ros2 launch robot_vision_pipeline yolo_detect_real.launch.py \
  image_topic:=/camera/camera/color/image_raw
```

### Run Node Directly

If camera is already running:

```bash
ros2 run robot_vision_pipeline yolo_detect_node \
  --ros-args --params-file ~/ros2/src/robot_vision_pipeline/config/yolo_detect_real.yaml
```

### YOLO Topics

| Topic | Direction | Description |
|-------|-----------|-------------|
| `/camera/camera/color/image_raw` | Subscribe | Input image |
| `/vision/yolo/image_annotated` | Publish | Annotated image with bounding boxes |
| `/vision/yolo/detections_json` | Publish | Detection results as JSON |

### Check Results

```bash
ros2 topic echo /vision/yolo/detections_json
ros2 run rqt_image_view rqt_image_view /vision/yolo/image_annotated
```

## ArUco Detection

```bash
ros2 launch robot_vision_pipeline aruco_detect.launch.py
```

### ArUco Topics

| Topic | Direction | Description |
|-------|-----------|-------------|
| `/aruco/image_annotated` | Publish | Annotated image with marker corners |
| `/aruco/detections_json` | Publish | Detection results as JSON |
| `/aruco_pose` | Publish | Marker poses (`robot_vision_pipeline_msgs/ArucoPoseArray`) |

### Check Results

```bash
ros2 topic echo /aruco/detections_json
ros2 topic echo /aruco_pose
ros2 run rqt_image_view rqt_image_view /aruco/image_annotated
```

## GUI

```bash
# Standard GUI
ros2 run robot_vision_pipeline vision_gui

# Astra camera GUI
ros2 run robot_vision_pipeline vision_gui_astra

# Fix Qt platform issues if needed
QT_QPA_PLATFORM=xcb ros2 run robot_vision_pipeline vision_gui
```

## Camera Configuration

| File | Use Case |
|------|----------|
| `config/rs_camera.yaml` | Depth + aligned depth (for ArUco 3D pose) |
| `config/rs_camera_yolo.yaml` | Color only (for YOLO) |

Common RealSense topics:
```
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/depth/image_rect_raw
/camera/camera/aligned_depth_to_color/image_raw
```

## Troubleshooting

**No YOLO detections:**
- Verify camera topic has images: `ros2 topic hz /camera/camera/color/image_raw`
- Check model exists: `ls ~/ros2/src/robot_vision_pipeline/models/`
- Lower `conf_threshold` in config
- Check `class_filter` (default: `wood`)

**No ArUco poses:**
- Verify correct dictionary (default: `DICT_4X4_50`)
- Check `camera_info_topic` is set
- Verify depth aligned topic for 3D coordinates

**Executables not found after build:**
```bash
ros2 pkg executables robot_vision_pipeline
```
