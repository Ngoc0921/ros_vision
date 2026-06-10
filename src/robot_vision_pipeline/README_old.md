# robot_vision_pipeline

Package ROS 2 cho pipeline vision của robot:

- Chạy RealSense D435.
- Detect vật bằng YOLO real-time.
- Detect ArUco marker và publish pose.
- Publish ảnh annotated và JSON detection.
- Có GUI quan sát camera/ArUco.

Package hiện được tổ chức theo kiểu `gp7_vision_pipeline`: dùng `ament_cmake`, generate custom message trong cùng package, install Python node bằng wrapper trong `scripts/`.

## Cấu Trúc Chính

```text
robot_vision_pipeline/
├── config/
│   ├── rs_camera.yaml
│   ├── rs_camera_yolo.yaml
│   ├── yolo_detect_real.yaml
│   └── aruco_detect.yaml
├── launch/
│   ├── yolo_detect_real.launch.py
│   ├── yolo_detect_sim.launch.py
│   └── aruco_detect.launch.py
├── models/
│   └── best_real.pt
├── msg/
│   ├── ArucoPose.msg
│   └── ArucoPoseArray.msg
├── scripts/
│   ├── yolo_detect_node
│   ├── aruco_detect_node
│   ├── vision_gui
│   └── vision_gui_astra
└── robot_vision_pipeline/
    ├── aruco/
    ├── vision_gui/
    └── yolo/
```

`gp7_vision_pipeline` đã có `COLCON_IGNORE`, nên `colcon build` sẽ bỏ qua package đó.

## Build

```bash
cd ~/ros2
colcon build --packages-select robot_vision_pipeline
source install/setup.bash
```

Nếu muốn build toàn workspace:

```bash
cd ~/ros2
colcon build --symlink-install
source install/setup.bash
```

Nếu dùng virtualenv cho YOLO/ArUco:

```bash
source ~/venvs/ros_env/bin/activate
```

YOLO launch hiện dùng Python:

```text
/home/minhquang/venvs/ros_yolo/bin/python3
```

ArUco launch hiện dùng Python:

```text
/home/minhquang/venvs/ros_env/bin/python3
```

Nếu đổi venv, sửa `prefix` trong file launch tương ứng.

## Chạy RealSense + YOLO

Lệnh chính:

```bash
ros2 launch robot_vision_pipeline yolo_detect_real.launch.py
```

Launch này sẽ chạy:

- `realsense2_camera/rs_launch.py`
- `robot_vision_pipeline/yolo_detect_node`

File cấu hình camera:

```text
config/rs_camera_yolo.yaml
```

Nội dung chính:

```yaml
device_type: d435
enable_color: true
enable_depth: false
enable_sync: true
align_depth.enable: false
pointcloud.enable: false
rgb_camera.color_profile: 640x480x15
```

`config/rs_camera.yaml` vẫn được giữ cho workflow cần depth/aligned depth, ví dụ ArUco 3D pose.

File cấu hình YOLO:

```text
config/yolo_detect_real.yaml
```

Topic input mặc định:

```text
/camera/camera/color/image_raw
```

Model mặc định:

```text
/home/minhquang/ros2/src/robot_vision_pipeline/models/best_real.pt
```

YOLO đang chạy real-time vì:

```yaml
detect_period_sec: 0.0
```

Nếu muốn giảm tải CPU/GPU, đặt ví dụ:

```yaml
detect_period_sec: 0.2
```

nghĩa là detect khoảng 5 FPS.

## Override Khi Chạy YOLO

Đổi model:

```bash
ros2 launch robot_vision_pipeline yolo_detect_real.launch.py \
  model_path:=/home/minhquang/ros2/src/robot_vision_pipeline/models/wood.pt
```

Đổi topic ảnh:

```bash
ros2 launch robot_vision_pipeline yolo_detect_real.launch.py \
  image_topic:=/camera/camera/color/image_raw
```

Đổi file config RealSense:

```bash
ros2 launch robot_vision_pipeline yolo_detect_real.launch.py \
  rs_config_file:=/home/minhquang/ros2/src/robot_vision_pipeline/config/rs_camera_yolo.yaml
```

Chạy node YOLO trực tiếp nếu camera đã chạy sẵn:

```bash
ros2 run robot_vision_pipeline yolo_detect_node --ros-args \
  --params-file ~/ros2/src/robot_vision_pipeline/config/yolo_detect_real.yaml
```

## Topic YOLO

Input:

```text
/camera/camera/color/image_raw
```

Output:

```text
/vision/yolo/image_annotated
/vision/yolo/detections_json
```

Xem kết quả:

```bash
ros2 topic echo /vision/yolo/detections_json
ros2 run rqt_image_view rqt_image_view /vision/yolo/image_annotated
```

## Chạy ArUco

```bash
ros2 launch robot_vision_pipeline aruco_detect.launch.py
```

File cấu hình:

```text
config/aruco_detect.yaml
```

Các topic RealSense thường dùng cho ArUco:

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/aligned_depth_to_color/camera_info
```

Output ArUco:

```text
/aruco/image_annotated
/aruco/detections_json
/aruco_pose
```

Xem kết quả:

```bash
ros2 topic echo /aruco/detections_json
ros2 topic echo /aruco_pose
ros2 run rqt_image_view rqt_image_view /aruco/image_annotated
```

`/aruco_pose` dùng custom message nội bộ:

```text
robot_vision_pipeline/msg/ArucoPoseArray
```

## GUI

GUI chính:

```bash
ros2 run robot_vision_pipeline vision_gui
```

GUI Astra:

```bash
ros2 run robot_vision_pipeline vision_gui_astra
```

Nếu Qt lỗi platform:

```bash
QT_QPA_PLATFORM=xcb ros2 run robot_vision_pipeline vision_gui
```

## Kiểm Tra RealSense Topic

Sau khi chạy RealSense:

```bash
ros2 topic list | grep camera
```

Các topic hay dùng:

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/depth/image_rect_raw
/camera/camera/depth/camera_info
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/aligned_depth_to_color/camera_info
```

Kiểm tra FPS:

```bash
ros2 topic hz /camera/camera/color/image_raw
```

Xem ảnh:

```bash
ros2 run rqt_image_view rqt_image_view /camera/camera/color/image_raw
```

## Debug Nhanh

Không thấy YOLO detection:

- Kiểm tra camera topic có ảnh:
  ```bash
  ros2 topic hz /camera/camera/color/image_raw
  ```
- Kiểm tra model tồn tại:
  ```bash
  ls ~/ros2/src/robot_vision_pipeline/models
  ```
- Giảm `conf_threshold` trong `config/yolo_detect_real.yaml`.
- Kiểm tra `class_filter`, hiện mặc định là `wood`.

Không thấy ArUco pose:

- Kiểm tra đúng dictionary, hiện mặc định `DICT_4X4_50`.
- Kiểm tra `camera_info_topic`.
- Kiểm tra depth aligned topic nếu cần tọa độ 3D.
- Kiểm tra TF từ camera frame về `base_link`.

Kiểm tra package đã install executable:

```bash
ros2 pkg executables robot_vision_pipeline
```

Kết quả nên có:

```text
robot_vision_pipeline aruco_detect_node
robot_vision_pipeline vision_gui
robot_vision_pipeline vision_gui_astra
robot_vision_pipeline yolo_detect_node
```
