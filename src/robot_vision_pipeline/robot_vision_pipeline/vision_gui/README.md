# Vision GUI Module

Vision GUI là một ứng dụng PyQt6 để hiển thị và xử lý các hình ảnh từ camera trong robot vision pipeline.

## Tính năng

### Tab 1: Input Image (Hình ảnh đầu vào)
- Hiển thị hình ảnh từ camera ROS2 theo thời gian thực
- Tải hình ảnh từ tệp
- Zoom và fit to window
- Hiển thị thông tin hình ảnh (kích thước, số channels)

### Tab 2: ArUco Detection (Phát hiện ArUco)
- Phát hiện các marker ArUco trong hình ảnh
- Chọn từ điển ArUco (DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_5X5_50)
- Vẽ trục 3D cho các marker được phát hiện
- Điều chỉnh kích thước marker
- Hiển thị số lượng marker được phát hiện

## Cài đặt

### Yêu cầu
- Python 3.8+
- ROS2 (Humble hoặc mới hơn)
- PyQt6
- OpenCV
- numpy

### Cài đặt gói

```bash
cd ~/ros2/src/robot_vision_pipeline
pip install PyQt6 opencv-python

# Build the package
colcon build --packages-select robot_vision_pipeline
```

## Sử dụng

### Cách 1: Chạy từ command line

```bash
# Sau khi source ROS2
source ~/ros2/install/setup.bash

# Chạy GUI
ros2 run robot_vision_pipeline vision_gui
```

### Cách 2: Chạy script launcher

```bash
python3 ~/ros2/src/robot_vision_pipeline/scripts/vision_gui_launcher.py
```

## Cấu trúc

```
vision_gui/
├── __init__.py                 # Package initialization
├── vision_gui_main.py          # Main GUI application
├── README.md                   # Documentation
```

## Thành phần chính

### VisionGUI (Main Class)
- Cửa sổ chính với các tab
- Quản lý ROS2 subscribers
- Xử lý các sự kiện của người dùng

### ImageDisplayWidget
- Widget hiển thị hình ảnh
- Hỗ trợ zoom và fit-to-window
- Hiển thị thông tin hình ảnh

### RosImageSubscriber (Thread)
- Chạy trong thread riêng
- Subscribe đến ROS2 Image topic
- Chuyển đổi ROS2 Image message sang OpenCV format

## Chủ đề ROS2

Mặc định, GUI subscribe đến:
- `/astra/rgb/image_raw` - Hình ảnh RGB từ camera Astra

Bạn có thể thay đổi topic bằng cách sửa đổi dòng:
```python
self.subscriber = RosImageSubscriber("/your/custom/image/topic")
```

## Phím tắt

- **Fit to Window**: Auto-fit image vào cửa sổ
- **Zoom**: Điều chỉnh tỷ lệ zoom (10% - 300%)

## Ghi chú

- Nếu không có ROS2 image stream, bạn có thể tải hình ảnh từ tệp bằng nút "Load Image File"
- ArUco detection sử dụng camera matrix được khởi tạo với các giá trị mặc định (focal length = 500)
- Để độ chính xác cao hơn, hãy cung cấp camera calibration file

## Lỗi thường gặp

### Module 'robot_vision_pipeline' not found
Đảm bảo rằng bạn đã build package:
```bash
colcon build --packages-select robot_vision_pipeline
source ~/ros2/install/setup.bash
```

### PyQt6 not found
Cài đặt PyQt6:
```bash
pip install PyQt6
```

### ROS not initialized
Đảm bảo ROS2 đã được source:
```bash
source /opt/ros/humble/setup.bash
source ~/ros2/install/setup.bash
```

## Phát triển tiếp theo

Có thể mở rộng GUI để hỗ trợ:
- Hiển thị YOLO detections
- Xem nhiều camera cùng lúc
- Lưu các frame đã xử lý
- Thêm các filter và xử lý hình ảnh
- Cấu hình camera dynamically

## Liên lạc

Tác giả: minhquang
Email: minhquang@example.com
