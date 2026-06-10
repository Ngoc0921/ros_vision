# Legacy Vision Pipeline (Archived)

This directory contains older versions of the vision pipeline code that have been superseded by the active `gp7_vision_pipeline` package.

## Status: Archived — Not Part of Active Pipeline

These files are **not installed as executables** and are **not launched** by any active launch file. They are kept for reference only.

## Contents

| File/Directory | Description |
|---|---|
| `gp7_vision_pipeline/target_marker_node.py` | Legacy single-marker publisher (superseded by `vision_detection_marker_node.py`) |
| `gp7_vision_pipeline/vision_marker_adapter_node.py` | Legacy marker adapter |
| `gp7_vision_pipeline/camera_projection_node.py` | Legacy camera-to-robot projection |
| `gp7_vision_pipeline/compute_camera_extrinsics.py` | Legacy extrinsic calibration tool |
| `gp7_vision_pipeline/vision_pipeline_placeholder_node.py` | Placeholder for the old pipeline |
| `scripts/` | Standalone shell scripts for camera tools |
| `config/` | Old configuration files |
| `launch/` | Old launch files |
| `srv/` | Old service definitions |

## Active Pipeline

The active vision pipeline is in `../gp7_vision_pipeline/`:

- **`vision_detection_marker_node.py`** — publishes `/vision/detection_markers` (MarkerArray) aggregating target and box
- **`/vision/detection_status`** — String status topic
- Data topics: `/vision/target_position`, `/vision/box`, `/vision/target_detected`, `/vision/box_detected`

## Why These Are Archived

These files used legacy visualization topics that have been removed:

- `/visualization/target_marker` — replaced by `/vision/detection_markers`
- `/visualization/box_marker` — replaced by `/vision/detection_markers`
- `/visualization/target_marker_array` — was dead code, never published

The new pipeline consolidates all vision markers into a single `MarkerArray` on `/vision/detection_markers`, which is then aggregated by `gp7_visualization` into `/scene/markers` for RViz.

## Do Not Use

Do not launch any file in this directory. Use `gp7_vision_pipeline` launch files instead:

```bash
ros2 launch gp7_vision_pipeline vision_full_pipeline.launch.py
ros2 launch gp7_vision_pipeline vision_markers_only.launch.py
```
