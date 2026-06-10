#!/usr/bin/env python3

import sys
import cv2
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QFileDialog,
    QPlainTextEdit,
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from robot_vision_pipeline.msg import ArucoPoseArray # type: ignore
from cv_bridge import CvBridge


class ImageDisplayWidget(QWidget):
    """Widget for displaying images with optional zoom controls."""

    def __init__(self, title="Image Display"):
        super().__init__()
        self.title = title
        self.image = None
        self.scale_factor = 100.0
        self.setup_ui()

    def setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout()

        # Title
        title_label = QLabel(self.title)
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 1px solid black; background-color: gray;")
        layout.addWidget(self.image_label)

        # Controls
        control_layout = QHBoxLayout()

        # Zoom controls
        control_layout.addWidget(QLabel("Zoom:"))
        self.zoom_spin = QSpinBox()
        self.zoom_spin.setMinimum(10)
        self.zoom_spin.setMaximum(300)
        self.zoom_spin.setValue(100)
        self.zoom_spin.setSuffix("%")
        self.zoom_spin.valueChanged.connect(self.on_zoom_changed)
        control_layout.addWidget(self.zoom_spin)

        # Fit to window button
        fit_button = QPushButton("Fit to Window")
        fit_button.clicked.connect(self.fit_to_window)
        control_layout.addWidget(fit_button)

        control_layout.addStretch()

        # Image info label
        self.info_label = QLabel("No image")
        self.info_label.setStyleSheet("color: gray;")
        control_layout.addWidget(self.info_label)

        layout.addLayout(control_layout)
        self.setLayout(layout)

    def set_image(self, cv_image):
        """Set image from OpenCV format."""
        if cv_image is None:
            self.image_label.setText("No image available")
            return

        self.image = cv_image
        self.update_display()

        # Update info
        h, w = cv_image.shape[:2]
        channels = cv_image.shape[2] if len(cv_image.shape) > 2 else 1
        self.info_label.setText(f"Size: {w}x{h} | Channels: {channels}")

    def update_display(self):
        """Update the displayed image based on zoom factor."""
        if self.image is None:
            return

        # Convert BGR to RGB for display
        if len(self.image.shape) == 3 and self.image.shape[2] == 3:
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = self.image

        # Apply zoom
        h, w = rgb_image.shape[:2]
        new_w = int(w * self.scale_factor / 100)
        new_h = int(h * self.scale_factor / 100)
        zoomed = cv2.resize(rgb_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        zoomed = np.ascontiguousarray(zoomed)

        # Convert to QImage. copy() detaches the Qt image from the numpy buffer.
        if len(zoomed.shape) == 3:
            h, w, ch = zoomed.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                zoomed.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
            ).copy()
        else:
            h, w = zoomed.shape
            bytes_per_line = w
            qt_image = QImage(
                zoomed.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8
            ).copy()

        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

    def on_zoom_changed(self, value):
        """Handle zoom change."""
        self.scale_factor = value
        self.update_display()

    def fit_to_window(self):
        """Auto-fit image to window."""
        if self.image is None:
            return

        label_w = self.image_label.width()
        label_h = self.image_label.height()
        img_h, img_w = self.image.shape[:2]

        if img_w == 0 or img_h == 0:
            return

        scale_w = (label_w / img_w) * 100
        scale_h = (label_h / img_h) * 100
        self.zoom_spin.setValue(min(int(scale_w), int(scale_h)))


class RosImageSubscriber(QThread):
    """ROS2 Image subscriber running in a separate thread."""

    image_received = pyqtSignal(np.ndarray)
    aruco_image_received = pyqtSignal(np.ndarray)
    aruco_pose_received = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        image_topic="/camera/color/image_raw", # /webcam/image_raw, /camera/color/image_raw
        aruco_image_topic="/aruco/image_annotated",
        aruco_pose_topic="/aruco_pose",
    ):
        super().__init__()
        self.image_topic = image_topic
        self.aruco_image_topic = aruco_image_topic
        self.aruco_pose_topic = aruco_pose_topic
        self.node = None
        self.bridge = CvBridge()
        self.running = True

    def run(self):
        """Run the ROS2 subscriber."""
        try:
            if not rclpy.ok():
                rclpy.init()
            self.node = Node("vision_gui_subscriber")
            self.node.create_subscription(
                RosImage,
                self.image_topic,
                self.image_callback,
                qos_profile_sensor_data,
            )
            self.node.create_subscription(
                RosImage,
                self.aruco_image_topic,
                self.aruco_image_callback,
                qos_profile_sensor_data,
            )
            self.node.create_subscription(
                ArucoPoseArray,
                self.aruco_pose_topic,
                self.aruco_pose_callback,
                qos_profile_sensor_data,
            )

            while self.running:
                rclpy.spin_once(self.node, timeout_sec=0.1)

        except Exception as e:
            self.error_occurred.emit(f"ROS Error: {str(e)}")
        finally:
            if self.node:
                self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    def image_callback(self, msg):
        """Callback for image subscription."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.image_received.emit(cv_image)
        except Exception as e:
            self.error_occurred.emit(f"Image conversion error: {str(e)}")

    def aruco_image_callback(self, msg):
        """Callback for ArUco annotated image subscription."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.aruco_image_received.emit(cv_image)
        except Exception as e:
            self.error_occurred.emit(f"ArUco image conversion error: {str(e)}")

    def aruco_pose_callback(self, msg):
        """Callback for ArUco pose array subscription."""
        try:
            detections = []
            for pose_msg in msg.poses:
                pose_info = {
                    "id": pose_msg.id,
                    "frame_cam": pose_msg.frame_cam,
                    "frame_base": pose_msg.frame_base,
                    "has_pose_base": pose_msg.has_pose_base,
                    "yaw_deg": pose_msg.yaw_deg,
                    "pose_base": {
                        "x": pose_msg.pose_base.position.x,
                        "y": pose_msg.pose_base.position.y,
                        "z": pose_msg.pose_base.position.z,
                    },
                }
                detections.append(pose_info)
            self.aruco_pose_received.emit(detections)
        except Exception as e:
            self.error_occurred.emit(f"ArUco pose conversion error: {str(e)}")

    def stop(self):
        """Stop the subscriber."""
        self.running = False


class VisionGUI(QMainWindow):
    """Main Vision GUI Application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Robot Vision Pipeline - GUI")
        self.setGeometry(100, 100, 1400, 800)

        self.input_image = None
        self.aruco_image = None

        self.setup_ui()
        self.start_ros_subscriber()

    def setup_ui(self):
        """Setup the main UI with tabs."""
        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Tab 1: Input Image
        self.input_tab = QWidget()
        self.input_layout = QVBoxLayout()
        self.input_display = ImageDisplayWidget("Input Image")
        self.input_layout.addWidget(self.input_display)

        # Controls for input tab
        input_control_layout = QHBoxLayout()
        self.input_topic_label = QLabel("Topic: /camera/color/image_raw")
        input_control_layout.addWidget(self.input_topic_label)
        input_control_layout.addStretch()
        load_image_btn = QPushButton("Load Image File")
        load_image_btn.clicked.connect(self.load_image_file)
        input_control_layout.addWidget(load_image_btn)
        self.input_layout.addLayout(input_control_layout)

        self.input_tab.setLayout(self.input_layout)
        self.tabs.addTab(self.input_tab, "Input Image")

        # Tab 2: ArUco Detection
        self.aruco_tab = QWidget()
        self.aruco_layout = QVBoxLayout()
        self.detected_display = ImageDisplayWidget("ArUco Annotated Image")
        self.aruco_layout.addWidget(self.detected_display)

        # Topic info for ArUco annotated image generated by aruco_detect_node.
        aruco_control_layout = QHBoxLayout()
        self.aruco_topic_label = QLabel("Topic: /aruco/image_annotated")
        aruco_control_layout.addWidget(self.aruco_topic_label)
        aruco_control_layout.addStretch()

        self.aruco_layout.addLayout(aruco_control_layout)
        self.aruco_tab.setLayout(self.aruco_layout)
        self.tabs.addTab(self.aruco_tab, "ArUco Detection")

        # Tab 3: ArUco Pose
        self.pose_tab = QWidget()
        self.pose_layout = QVBoxLayout()
        pose_topic_layout = QHBoxLayout()
        self.pose_topic_label = QLabel("Topic: /aruco_pose")
        pose_topic_layout.addWidget(self.pose_topic_label)
        pose_topic_layout.addStretch()
        self.pose_count_label = QLabel("Markers: 0")
        pose_topic_layout.addWidget(self.pose_count_label)
        self.pose_layout.addLayout(pose_topic_layout)

        self.pose_display = QPlainTextEdit()
        self.pose_display.setReadOnly(True)
        self.pose_display.setPlaceholderText("Waiting for ArUco pose messages on /aruco_pose...")
        self.pose_layout.addWidget(self.pose_display)
        self.pose_tab.setLayout(self.pose_layout)
        self.tabs.addTab(self.pose_tab, "ArUco Pose")

        # Status bar
        self.statusBar().showMessage("Ready")

    def start_ros_subscriber(self):
        """Start ROS2 image subscriber in a thread."""
        self.subscriber = RosImageSubscriber(
            image_topic="/camera/color/image_raw",
            aruco_image_topic="/aruco/image_annotated",
            aruco_pose_topic="/aruco_pose",
        )
        self.subscriber.image_received.connect(self.on_image_received)
        self.subscriber.aruco_image_received.connect(self.on_aruco_image_received)
        self.subscriber.aruco_pose_received.connect(self.on_aruco_pose_received)
        self.subscriber.error_occurred.connect(self.on_ros_error)
        self.subscriber.start()

    def on_image_received(self, cv_image):
        """Handle received image from ROS."""
        self.input_image = cv_image
        self.input_display.set_image(cv_image)
        self.statusBar().showMessage(f"Image received: {cv_image.shape}")

    def on_aruco_image_received(self, cv_image):
        """Handle annotated ArUco image from aruco_detect_node."""
        self.aruco_image = cv_image
        self.detected_display.set_image(cv_image)
        self.statusBar().showMessage(f"ArUco annotated image received: {cv_image.shape}")

    def on_aruco_pose_received(self, pose_list):
        """Handle ArUco pose messages and display them."""
        if not pose_list:
            self.pose_count_label.setText("Markers: 0")
            self.pose_display.setPlainText("No ArUco poses received yet.")
            return

        self.pose_count_label.setText(f"Markers: {len(pose_list)}")
        lines = []
        for pose in pose_list:
            lines.append(f"ID: {pose['id']}")
            lines.append(f"  frame_cam: {pose['frame_cam']}")
            lines.append(f"  frame_base: {pose['frame_base']}")
            if pose["has_pose_base"]:
                lines.append(
                    f"  base pose: x={pose['pose_base']['x']:.3f}, y={pose['pose_base']['y']:.3f}, z={pose['pose_base']['z']:.3f}, yaw={pose['yaw_deg']:.1f}"
                )
            else:
                lines.append("  pose not available")
            lines.append("")

        self.pose_display.setPlainText("\n".join(lines))
        self.statusBar().showMessage(f"Aruco pose message received: {len(pose_list)} markers")

    def on_ros_error(self, error_msg):
        """Handle ROS errors."""
        self.statusBar().showMessage(f"Error: {error_msg}")

    def load_image_file(self):
        """Load image from file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.bmp *.tiff)"
        )
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                self.input_image = image
                self.input_display.set_image(image)
                self.statusBar().showMessage(f"Loaded: {Path(file_path).name}")

    def closeEvent(self, event):
        """Handle window close event."""
        self.subscriber.stop()
        self.subscriber.wait()
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    gui = VisionGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
