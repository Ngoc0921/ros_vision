# gp7_vision_pipeline Tasks

## Purpose

This task file tracks work specific to the `gp7_vision_pipeline` package. This package provides the perception pipeline for the GP7 robot with Intel RealSense D435: YOLO-based box detection, pixel-to-base mapping using homography + camera intrinsics, and RViz marker visualization.

## High Priority

- [ ] **TODO: verify** — `yolo_box_detector_node.py` loads the YOLO model from the configured path and runs inference on color frames.
- [ ] **TODO: verify** — `yolo_box_detector_node.py` publishes `/vision/target_detection` and `/vision/box_detection` topics with `BoxDetection` messages.
- [ ] **TODO: verify** — `pixel_to_base_mapper_node.py` subscribes to `/vision/target_detection` and `/vision/box_detection` and publishes `/vision/target_position` (PointStamped), `/vision/box` (Box), `/vision/target_detected`, `/vision/box_detected`, and `/vision/debug_image_base`.
- [ ] **TODO: verify** — `pixel_to_base_homography.yaml` exists with a valid 3x3 homography matrix. This must be calibrated for the physical camera mount.
- [ ] **TODO: verify** — The `best.pt` YOLO model file exists at `model/box_target/weights/best.pt` and matches the trained configuration (class names: "box", "target").
- [ ] **TODO: verify** — `vision_detection_marker_node.py` publishes `/vision/detection_markers` (MarkerArray) and legacy `/visualization/*` topics for RViz.
- [ ] **TODO: verify** — The robust center depth estimation (`median_depth_meters`, `robust_center_depth`) works correctly with both RealSense depth encodings (`32FC1` and `mm` formats).

## Medium Priority

- [ ] **TODO: verify** — `camera_extrinsics.yaml` contains the camera mount calibration (parent: `world`, child: `camera_link`, translation: `(0.0, -0.7, 1.0)`, roll: `π`).
- [ ] **TODO: verify** — All launch files work: `vision_full_pipeline.launch.py`, `vision_no_camera.launch.py`, `vision_markers_only.launch.py`, `yolo_box_detector.launch.py`.
- [ ] **TODO: verify** — `yolo_box_detector.yaml` config is loaded correctly by the YOLO node.
- [ ] **TODO: verify** — `pixel_to_base_mapper.yaml` and `vision_markers.yaml` configs are loaded correctly.
- [ ] **TODO: verify** — The stale detection timer in `pixel_to_base_mapper_node` correctly publishes `False` for detection status after timeout.

## Low Priority

- [ ] **TODO: verify** — `tools/compute_pixel_to_base_homography.py` and `test_pixel_to_base_homography.py` scripts are documented and functional.
- [ ] **TODO: verify** — `realsense_depth_debug.py` is functional for troubleshooting depth camera issues.
- [ ] **TODO: verify** — Camera intrinsics (`fx`, `fy`, `cx`, `cy`) are correctly extracted from the CameraInfo message and used for box size computation.

## Debugging Tasks

- [ ] If YOLO model fails to load: verify the model path is correct and the file exists. Check `ultralytics` is installed in the Python environment.
- [ ] If no detections are published: verify the RealSense camera is running and `/camera/camera/color/image_raw` is being published.
- [ ] If `/vision/detection_markers` is not visible in RViz: set Fixed Frame to `base_link` and verify the topic is published.
- [ ] If depth values are incorrect: verify the depth encoding (`32FC1` vs `mm`) and the conversion formula.
- [ ] If homography mapping produces wrong positions: recalibrate using `compute_pixel_to_base_homography.py`.

## Documentation Tasks

- [ ] Document how to recalibrate the homography matrix for a new camera mount.
- [ ] Document the depth encoding handling (mm vs meters) for both RealSense formats.
- [ ] Document the box size estimation from pixel dimensions and camera intrinsics.
- [ ] Document the stale detection mechanism and how to adjust the timeout.

## TODO Verify

- [ ] **TODO: verify** — `msg/BoxDetection.msg` and `msg/Box.msg` definitions match what the nodes actually publish.
- [ ] **TODO: verify** — `srv/GetPixelPose.srv` is functional or can be removed if unused.
- [ ] **TODO: verify** — All `*.py` files in `tools/` are documented and in the correct state for production use.
- [ ] **TODO: verify** — The `models/` directory contains the YOLO weights; `model/` (singular) and `models/` (plural) directories are not confused.
- [ ] **TODO: verify** — `package.xml` license (`TODO`) is updated to Apache-2.0.
