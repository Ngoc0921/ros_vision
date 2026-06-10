# `gp7_vision_pipeline`

**Purpose:** Perception pipeline for the GP7 robot — transforms RealSense D435 RGB-D images into robot-usable 3D object detections in the `base_link` frame via YOLO detection + homography mapping.

**Role in the Project:** Provides 3-layer perception: (1) YOLO object detection, (2) pixel-to-base mapping via homography, (3) RViz marker visualization. Feeds `gp7_drl_inference` and `gp7_task_executor`.

---

## Quick Start

```bash
# 1. Build
cd ~/pap_yaskawa_ws && colcon build --packages-select gp7_vision_pipeline --symlink-install

# 2. Environment
source /opt/ros/humble/setup.bash
source ~/yolo_env/bin/activate       # provides ultralytics + opencv-python
source ~/pap_yaskawa_ws/install/setup.bash

# 3. Launch
ros2 launch gp7_vision_pipeline vision_full_pipeline.launch.py
```

For YOLO + mapper + markers only (RealSense already running):

```bash
ros2 launch gp7_vision_pipeline vision_no_camera.launch.py
```

To start only the marker node (rest of pipeline already running):

```bash
ros2 launch gp7_vision_pipeline vision_markers_only.launch.py
```

---

## Architecture

```
gp7_bringup/realsense_d435.launch.py        (RealSense driver)
    │                                       ← ros2 launch gp7_bringup realsense_d435.launch.py
    ├── /camera/camera/color/image_raw ──────────────────────┐
    ├── /camera/camera/aligned_depth_to_color/image_raw ────┤
    └── /camera/camera/color/camera_info ────────────────────┘
                                                         │
                                                         ▼
                           ┌─────────────────────────────┐
                           │  Layer 1 — YOLO Detector     │
                           │  yolo_box_detector_node      │
                           │                              │
                           │  Inputs:  color + depth      │
                           │  Output:  /vision/box_detection  │
                           │          /vision/target_detection │
                           │          /vision/debug_image      │
                           └──────────────┬──────────────────┘
                                          │
                                          ▼
                           ┌─────────────────────────────┐
                           │  Layer 2 — Pixel→Base Mapper │
                           │  pixel_to_base_mapper_node   │
                           │                               │
                           │  Inputs:  detections + depth  │
                           │          + homography H       │
                           │          + camera intrinsics  │
                           │                               │
                           │  Outputs: /vision/target_position │  ← frame_id=base_link
                           │           /vision/box          │  ← frame_id=base_link
                           │           /vision/debug_image_base │
                           │           /vision/target_detected  │
                           │           /vision/box_detected     │
                           └──────────────┬──────────────────┘
                                          │
                                          ▼
                           ┌─────────────────────────────┐
                           │  Layer 3 — Detection Markers │
                           │  vision_detection_marker_node │
                           │                               │
                           │  Inputs:  target_position     │
                           │          + box (pose + size)  │
                           │                               │
                           │  Output:  /vision/detection_markers │
                           │           (blue CYLINDER = target) │
                           │           (yellow CUBE   = box)    │
                           └────────────────────────────────────┘
```

**Coordinate conversion** is purely homography-based: `H @ [u, v, 1]` maps pixel coordinates to `base_link` (x, y) with fixed Z=55mm for the target. No tf2, no camera extrinsics TF lookup.

---

## Launch Files

### `vision_full_pipeline.launch.py` — Default

Starts the RealSense driver (via `gp7_bringup/realsense_d435.launch.py`) + all 3 layers.

```bash
ros2 launch gp7_vision_pipeline vision_full_pipeline.launch.py
```

| Argument | Default | Description |
|----------|---------|-------------|
| `use_camera` | `true` | Start RealSense D435 driver |
| `use_homography_mapper` | `true` | Start `pixel_to_base_mapper_node` |
| `use_detection_markers` | `true` | Start `vision_detection_marker_node` |
| `use_rviz` | `false` | Launch RViz2 |
| `model_path` | `.../model/box_target/weights/best.pt` | YOLO model path |
| `conf_threshold` | `0.8` | YOLO confidence threshold |
| `color_topic` | `/camera/camera/color/image_raw` | RGB image topic |
| `depth_topic` | `/camera/camera/aligned_depth_to_color/image_raw` | Depth topic |
| `camera_info_topic` | `/camera/camera/color/camera_info` | Camera intrinsics topic |
| `frame_id` | `base_link` | Reference frame for markers |

### `vision_no_camera.launch.py`

Starts YOLO + mapper + markers only (camera topics already exist).

### `vision_markers_only.launch.py`

Starts only `vision_detection_marker_node` — useful for restarting RViz markers without rebuilding the pipeline.

---

## Topics

### Published

| Topic | Type | Node | Description |
|-------|------|------|-------------|
| `/vision/box_detection` | `BoxDetection` | `yolo_box_detector_node` | YOLO "box" class detection (confidence, bbox, depth) |
| `/vision/target_detection` | `BoxDetection` | `yolo_box_detector_node` | YOLO "target" class detection |
| `/vision/debug_image` | `Image` | `yolo_box_detector_node` | BGR image with YOLO bboxes, crosshair, distance overlay |
| `/vision/target_position` | `PointStamped` | `pixel_to_base_mapper_node` | Target center x,y,z in `base_link` (metres) |
| `/vision/box` | `Box` | `pixel_to_base_mapper_node` | Box pose + size in `base_link` |
| `/vision/debug_image_base` | `Image` | `pixel_to_base_mapper_node` | BGR image with base-frame crosshair overlay |
| `/vision/target_detected` | `Bool` | `pixel_to_base_mapper_node` | Target detection status |
| `/vision/box_detected` | `Bool` | `pixel_to_base_mapper_node` | Box detection status |
| `/vision/detection_markers` | `MarkerArray` | `vision_detection_marker_node` | **Official RViz marker topic** |
| `/vision/detection_status` | `String` | `vision_detection_marker_node` | Combined detection status |

### Subscribed

| Topic | Subscribed By |
|-------|--------------|
| `/camera/camera/color/image_raw` | `yolo_box_detector_node` |
| `/camera/camera/aligned_depth_to_color/image_raw` | `yolo_box_detector_node`, `pixel_to_base_mapper_node` |
| `/camera/camera/color/camera_info` | `pixel_to_base_mapper_node` |
| `/vision/target_detection` | `pixel_to_base_mapper_node` |
| `/vision/box_detection` | `pixel_to_base_mapper_node` |
| `/vision/target_position` | `vision_detection_marker_node` |
| `/vision/box` | `vision_detection_marker_node` |
| `/vision/target_detected` | `vision_detection_marker_node` |
| `/vision/box_detected` | `vision_detection_marker_node` |

---

## Calibration Model

### Homography — Pixel to Base Frame

A 3×3 homography matrix `H` maps pixel coordinates to base frame (x, y) in millimetres:

```
H @ [u, v, 1]^T = [X, Y, Z]^T
x_mm = X / Z
y_mm = Y / Z
```

The matrix is stored in `config/pixel_to_base_homography.yaml`.

To recompute after camera mount or table position changes:

```bash
ros2 run gp7_vision_pipeline compute_pixel_to_base_homography
ros2 run gp7_vision_pipeline test_pixel_to_base_homography
```

This requires `calibration_points.csv` in the package root (point correspondences between pixel (u, v) and base frame (x_mm, y_mm)).

### Intrinsics

Camera focal lengths `fx`, `fy` and principal point `cx`, `cy` are read at runtime from `/camera/camera/color/camera_info` (CameraInfo K matrix). The `pixel_to_base_mapper_node` subscribes to this topic automatically.

### Target Z

Target Z in `base_link` is **fixed at 55 mm** (height of calibration sticker center above the table). Box Z is the table surface (50 mm).

---

## Box Size Formula

Box width and length are derived from the bounding-box pixel size, depth, and camera focal length:

```
width_m  = bbox_pixel_width  * depth_m  / fx
length_m = bbox_pixel_height * depth_m  / fy
height_m = table_z - box_bottom_z
        = (table_z_base_mm - roi_median_raw_depth_mm) / 1000.0
```

Box depth comes from the **median raw depth in the ROI** (`roi_median_raw_depth` in `BoxDetection.msg`).

---

## Debug Images

| Topic | Content |
|-------|---------|
| `/vision/debug_image` | YOLO detections drawn on RGB image (bbox, label, confidence, distance) |
| `/vision/debug_image_base` | YOLO detections with base-frame crosshair overlay (origin = box target, crosshair = base-frame x/y axes) |

Use `rqt_image_view` to inspect:

```bash
ros2 run rqt_image_view rqt_image_view
```

---

## RViz Visualization

```bash
ros2 launch gp7_vision_pipeline vision_full_pipeline.launch.py use_rviz:=true
```

In RViz:
- **Fixed Frame:** `base_link`
- **Marker topic:** `/vision/detection_markers`

Markers:
- **Blue CYLINDER** — target sticker (centered at `/vision/target_position`)
- **Yellow CUBE** — box (at `/vision/box` position, sized from `/vision/box` dimensions)

---

## Message Types

### `BoxDetection.msg`

```
std_msgs/Header header
string class_name             # "box" or "target"
float32 confidence
float32 center_x, center_y   # pixel coords of bbox center
float32 x_min, x_max, y_min, y_max  # bbox pixel bounds
float32 distance_m           # median depth in metres
float32 center_raw_depth     # raw depth at centre pixel (mm, 16UC1)
float32 roi_median_raw_depth # median raw depth in ROI (mm, 16UC1)
string depth_encoding        # "16UC1" or "32FC1"
```

### `Box.msg`

```
std_msgs/Header header
string class_name
float32 confidence
geometry_msgs/Pose pose       # position + orientation in base_link
geometry_msgs/Vector3 size    # x, y, z in metres
```

---

## Package Structure

```
gp7_vision_pipeline/
├── gp7_vision_pipeline/          # Python source
│   ├── yolo_box_detector_node.py
│   ├── pixel_to_base_mapper_node.py
│   ├── vision_detection_marker_node.py
│   ├── depth_utils.py             # depth image reading & filtering
│   ├── detection_visualizer.py    # BGR image annotation
│   ├── realsense_depth_debug_node.py
│   └── utils/
├── msg/
│   ├── BoxDetection.msg
│   └── Box.msg
├── launch/
│   ├── vision_full_pipeline.launch.py    # DEFAULT launch
│   ├── vision_no_camera.launch.py       # YOLO + mapper + markers only
│   ├── vision_markers_only.launch.py    # markers only
│   └── yolo_box_detector.launch.py
├── config/
│   ├── pixel_to_base_homography.yaml    # homography calibration matrix
│   ├── pixel_to_base_mapper.yaml        # mapper parameters (z heights)
│   ├── vision_markers.yaml              # marker colors/sizes
│   ├── yolo_box_detector.yaml           # detector defaults
│   └── realsense_depth_debug.yaml       # depth debug defaults
├── model/box_target/weights/
│   ├── best.pt                         # active YOLO model
│   └── last.pt
├── scripts/                       # executable wrappers
├── calibration_points.csv         # homography calibration data
└── legacy/                        # superseded files (preserved, not built)
    ├── gp7_vision_pipeline/
    │   ├── camera_projection_node.py
    │   ├── vision_marker_adapter_node.py
    │   ├── target_marker_node.py
    │   ├── vision_pipeline_placeholder_node.py
    │   ├── compute_camera_extrinsics.py
    │   └── test_camera_extrinsics.py
    ├── launch/
    │   ├── vision_pipeline.launch.py
    │   └── camera_projection.launch.py
    ├── config/
    │   ├── camera_extrinsics.yaml
    │   ├── camera_projection.yaml
    │   ├── vision_pipeline.yaml
    │   ├── yolo_model.yaml
    │   └── vision_sim_rgbd.yaml
    └── srv/
        └── GetPixelPose.srv
```

---

## Troubleshooting

### No detections appearing

```bash
# Lower confidence threshold
ros2 param set /yolo_box_detector_node conf_threshold 0.3

# Check camera topics are publishing
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw

# Check YOLO model is loading
ros2 topic echo /vision/box_detection --once
```

### Markers not visible in RViz

1. Set **Fixed Frame** to `base_link`
2. Add `/vision/detection_markers` in Displays panel
3. Verify publisher is alive:

```bash
ros2 topic info /vision/detection_markers -v
ros2 topic echo /vision/detection_markers --once
```

### RealSense warnings about unsupported parameters

The RealSense driver accepts only RGB/depth profile arguments. All other pipeline arguments (`use_projection`, `model_path`, etc.) are NOT forwarded to the driver and will not generate warnings.

```bash
# List supported RealSense arguments
ros2 launch realsense2_camera rs_launch.py --show-args
```

### Box size is wrong

1. Verify camera intrinsics match your hardware (check `/camera/camera/color/camera_info`):
   - D435: fx ≈ fy ≈ 609, cx ≈ 424, cy ≈ 236 at 848×480
2. Verify `fx`, `fy` in `pixel_to_base_mapper_node.py` match the camera info K matrix
3. Check `config/pixel_to_base_mapper.yaml` for correct `table_z_base_mm`

### Homography is inaccurate after camera move

Recalibrate:
```bash
ros2 run gp7_vision_pipeline compute_pixel_to_base_homography
ros2 run gp7_vision_pipeline test_pixel_to_base_homography
```
Edit `config/pixel_to_base_homography.yaml` with the new `h_matrix` values, then rebuild.

---

## Legacy Files

The following files are superseded by the homography-based pipeline and are preserved in the `legacy/` directory for reference:

- `legacy/gp7_vision_pipeline/camera_projection_node.py` — extrinsics-based pixel→base projection
- `legacy/gp7_vision_pipeline/vision_marker_adapter_node.py` — adapter for old marker topic names
- `legacy/gp7_vision_pipeline/target_marker_node.py` — alternative marker publisher
- `legacy/gp7_vision_pipeline/vision_pipeline_placeholder_node.py` — placeholder node
- `legacy/gp7_vision_pipeline/compute_camera_extrinsics.py` — extrinsics calibration tool
- `legacy/gp7_vision_pipeline/test_camera_extrinsics.py` — extrinsics test tool
- `legacy/launch/vision_pipeline.launch.py` — old placeholder launch
- `legacy/launch/camera_projection.launch.py` — extrinsics pipeline launch
- `legacy/config/camera_extrinsics.yaml` — extrinsics calibration data
- `legacy/config/camera_projection.yaml` — projection node parameters
- `legacy/config/vision_pipeline.yaml` — placeholder parameters
- `legacy/config/yolo_model.yaml` — YOLO model hint (superseded by runtime model_path)
- `legacy/config/vision_sim_rgbd.yaml` — Gazebo simulation topics
- `legacy/srv/GetPixelPose.srv` — pixel→pose service (superseded by homography pipeline)
