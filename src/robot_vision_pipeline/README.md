# robot_vision_pipeline

Robot vision pipeline using YOLO for real-time object detection and ArUco marker detection for pose estimation in ROS 2.

## Package Structure

```
robot_vision_pipeline/
├── CMakeLists.txt                         # ament_cmake build, ROS 2 msg generation, install rules
├── package.xml                            # ROS 2 package manifest and dependencies
├── setup.py                               # Python package metadata and console entry points
├── setup.cfg                              # Python install/script configuration
├── README.md                              # Current package documentation
├── README_old.md                          # Legacy package documentation
├── resource/
│   └── robot_vision_pipeline              # ament resource marker
├── config/
│   ├── aruco_detect.yaml                  # ArUco detector parameters
│   ├── pixel_to_base_homography.yaml      # Saved pixel-to-base homography matrix/config
│   ├── pixel_to_base_mapper.yaml          # Pixel-to-base mapper node parameters
│   ├── rs_camera.yaml                     # RealSense config with depth/aligned depth
│   ├── rs_camera_yolo.yaml                # RealSense color-only config for YOLO
│   ├── vision_markers.yaml                # RViz/marker visualization parameters
│   ├── yolo_detect_real.yaml              # Real-camera YOLO parameters
│   └── yolo_json_adapter.yaml             # YOLO JSON-to-message adapter parameters
├── launch/
│   ├── aruco_detect.launch.py             # ArUco detection launch file
│   ├── vision_full_pipeline.launch.py     # Full vision pipeline launch file
│   ├── yolo_detect_real.launch.py         # Real-camera YOLO launch file
│   └── yolo_detect_sim.launch.py          # Simulation YOLO launch file
├── models/
│   ├── best_real.pt                       # YOLO model for real setup
│   ├── hbb.pt                             # YOLO model variant
│   ├── obb_v3.pt                          # Oriented bounding box model variant
│   ├── sim.pt                             # YOLO model for simulation
│   ├── tray.pt                            # Tray detection model
│   └── wood.pt                            # Wood detection model
├── msg/
│   ├── ArucoPose.msg                      # Single ArUco pose message
│   ├── ArucoPoseArray.msg                 # Array of ArUco pose messages
│   ├── Box.msg                            # Single detected box message
│   └── BoxDetection.msg                   # Box detection batch/result message
├── robot_vision_pipeline/
│   ├── __init__.py
│   ├── compute_pixel_to_base_homography.py # Homography calibration helper
│   ├── depth_utils.py                     # Shared depth/deprojection utilities
│   ├── detection_visualizer.py            # Detection overlay/visualization helpers
│   ├── pixel_to_base_mapper_node.py       # Converts image detections to base-frame targets
│   ├── realsense_depth_debug_node.py      # RealSense depth debug node
│   ├── test_pixel_to_base_homography.py   # Homography test/helper node
│   ├── vision_detection_marker_node.py    # Publishes visualization markers for detections
│   ├── yolo_json_to_box_detection_node.py # Converts YOLO JSON output to BoxDetection messages
│   ├── aruco/
│   │   ├── __init__.py
│   │   ├── aruco_calib_size_check.py      # ArUco calibration size checker
│   │   ├── aruco_detect_node.py           # ArUco detector node
│   │   ├── aruco_distance.py              # ArUco distance utility
│   │   ├── aruco_pose_node.py             # ArUco pose estimation node
│   │   └── aruco_size_check.py            # ArUco marker size checker
│   ├── depth/
│   │   └── __init__.py                    # Depth subpackage placeholder
│   ├── pose_estimation/
│   │   ├── __init__.py
│   │   ├── pick_pose_estimator_node.py    # Pick-pose estimation node
│   │   ├── target_filter_node.py          # Target filtering node
│   │   └── tf_utils.py                    # TF transform helpers
│   ├── vision_gui/
│   │   ├── __init__.py
│   │   ├── README.md                      # GUI-specific notes
│   │   ├── vision_gui_astra.py            # Astra camera GUI
│   │   └── vision_gui_main.py             # Main Qt vision GUI
│   └── yolo/
│       ├── __init__.py
│       ├── yolo_detect_node.py            # Main YOLO detector node
│       ├── yolo_detect_node_v1.py         # Legacy/alternate YOLO detector node
│       └── yolo_utils.py                  # YOLO helper functions
├── scripts/
│   ├── aruco_detect_node                  # Executable wrapper
│   ├── pixel_to_base_mapper_node          # Executable wrapper
│   ├── realsense_depth_debug              # Executable wrapper
│   ├── vision_detection_marker_node       # Executable wrapper
│   ├── vision_gui                         # Executable wrapper
│   ├── vision_gui_astra                   # Executable wrapper
│   ├── vision_gui_launcher.py             # GUI launcher helper
│   ├── yolo_detect_node                   # Executable wrapper
│   ├── yolo_detect_node_v1                # Executable wrapper
│   ├── yolo_json_to_box_detection_node    # Executable wrapper
│   └── tools/
│       ├── compute_pixel_to_base_homography # Homography tool wrapper
│       └── test_pixel_to_base_homography    # Homography test tool wrapper
└── test/                                  # Package tests
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
