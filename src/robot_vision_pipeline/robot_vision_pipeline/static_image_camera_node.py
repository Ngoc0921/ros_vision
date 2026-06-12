#!/usr/bin/env python3
"""Publish fake camera topics from a static image for testing without real hardware.

Simulates RealSense D435-style topics:
  /camera/camera/color/image_raw
  /camera/camera/aligned_depth_to_color/image_raw
  /camera/camera/color/camera_info

Parameters:
  image_path:       path to the test image (BGR format)
  publish_rate_hz:  publishing rate in Hz (default 1.0)
  fake_depth_m:     simulated depth in metres (default 0.55)
  image_frame_id:   frame_id for color image (default camera_color_optical_frame)
  depth_frame_id:   frame_id for depth image (default camera_color_optical_frame)
  fx, fy, cx, cy:  camera intrinsics (default fx=fy=615, cx=width/2, cy=height/2)
"""

import os
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header


class StaticImageCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("static_image_camera_node")

        self.declare_parameter("image_path", "")
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("fake_depth_m", 0.55)
        self.declare_parameter("image_frame_id", "camera_color_optical_frame")
        self.declare_parameter("depth_frame_id", "camera_color_optical_frame")
        self.declare_parameter("fx", 615.0)
        self.declare_parameter("fy", 615.0)
        self.declare_parameter("cx", -1.0)
        self.declare_parameter("cy", -1.0)

        image_path = str(self.get_parameter("image_path").value).strip()
        self._publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self._fake_depth_m = float(self.get_parameter("fake_depth_m").value)
        self._image_frame_id = str(self.get_parameter("image_frame_id").value)
        self._depth_frame_id = str(self.get_parameter("depth_frame_id").value)
        self._fx = float(self.get_parameter("fx").value)
        self._fy = float(self.get_parameter("fy").value)
        self._cx = float(self.get_parameter("cx").value)
        self._cy = float(self.get_parameter("cy").value)

        self._bridge = self._get_bridge()
        self._image_bgr: np.ndarray = self._load_image(image_path)

        if self._image_bgr is None:
            self.get_logger().fatal(
                "image_path is empty or image file does not exist. "
                "Please provide a valid image path via the 'image_path' launch argument."
            )
            return

        self._height, self._width = self._image_bgr.shape[:2]

        if self._cx <= 0.0:
            self._cx = float(self._width) * 0.5
        if self._cy <= 0.0:
            self._cy = float(self._height) * 0.5

        self._color_pub = self.create_publisher(
            Image, "/camera/camera/color/image_raw", 10
        )
        self._depth_pub = self.create_publisher(
            Image, "/camera/camera/aligned_depth_to_color/image_raw", 10
        )
        self._info_pub = self.create_publisher(
            CameraInfo, "/camera/camera/color/camera_info", 10
        )

        period_sec = 1.0 / self._publish_rate_hz if self._publish_rate_hz > 0 else 1.0
        self._timer = self.create_timer(period_sec, self._timer_callback)

        self.get_logger().info(
            f"StaticImageCamera started | "
            f"image={image_path} ({self._width}x{self._height}), "
            f"rate={self._publish_rate_hz} Hz, "
            f"fake_depth={self._fake_depth_m} m"
        )
        self.get_logger().info(
            f"fx={self._fx}, fy={self._fy}, cx={self._cx}, cy={self._cy}"
        )

    def _get_bridge(self):
        from cv_bridge import CvBridge
        return CvBridge()

    def _load_image(self, image_path: str) -> np.ndarray:
        if not image_path:
            self.get_logger().error("image_path is empty. Pass image_path:=/absolute/path/to/image.jpg")
            return None
        if not os.path.isfile(image_path):
            self.get_logger().error(f"image_path does not exist or is not a file: {image_path}")
            return None
        img = cv2.imread(image_path)
        if img is None:
            self.get_logger().error(f"OpenCV failed to read image_path: {image_path}")
            return None
        self.get_logger().info(f"Loaded image: {image_path} ({img.shape[1]}x{img.shape[0]})")
        return img

    def _make_header(self, frame_id: str) -> Header:
        now = self.get_clock().now().to_msg()
        h = Header()
        h.stamp = now
        h.frame_id = frame_id
        return h

    def _make_camera_info(self) -> CameraInfo:
        info = CameraInfo()
        info.header = self._make_header(self._image_frame_id)
        info.height = self._height
        info.width = self._width
        info.k = [
            self._fx, 0.0, self._cx,
            0.0, self._fy, self._cy,
            0.0, 0.0, 1.0,
        ]
        info.p = [
            self._fx, 0.0, self._cx, 0.0,
            0.0, self._fy, self._cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        return info

    def _timer_callback(self) -> None:
        if self._image_bgr is None:
            return

        header = self._make_header(self._image_frame_id)

        color_msg = self._bridge.cv2_to_imgmsg(self._image_bgr, encoding="bgr8")
        color_msg.header = header
        self._color_pub.publish(color_msg)

        depth_mm = int(round(self._fake_depth_m * 1000.0))
        depth_data = np.full(
            (self._height, self._width), depth_mm, dtype=np.uint16
        )
        depth_msg = self._bridge.cv2_to_imgmsg(
            depth_data, encoding="16UC1"
        )
        depth_header = self._make_header(self._depth_frame_id)
        depth_msg.header = depth_header
        self._depth_pub.publish(depth_msg)

        info_msg = self._make_camera_info()
        self._info_pub.publish(info_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StaticImageCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
