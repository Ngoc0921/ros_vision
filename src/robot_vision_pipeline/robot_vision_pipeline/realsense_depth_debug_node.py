"""Lightweight RealSense debug: verify RGB + aligned depth + CameraInfo; center-pixel depth.

Milestone 2 only — no YOLO, no ``base_link`` geometry output yet. Depth values are
device-reported and **not** claimed as metrology-grade.
"""

from __future__ import annotations

import struct
import sys
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


def _depth_m_at_center(msg: Image, u: int, v: int) -> Optional[float]:
    """Return depth in meters at pixel (u, v).

    Supports RealSense-style ``16UC1`` (millimeters) and Gazebo ``32FC1`` (meters).
    """
    if u < 0 or v < 0 or u >= msg.width or v >= msg.height:
        return None
    row_off = v * msg.step

    if msg.encoding in ("16UC1", "mono16"):
        pix_off = row_off + u * 2
        if pix_off + 2 > len(msg.data):
            return None
        raw = struct.unpack_from("<H", msg.data, pix_off)[0]
        if raw == 0:
            return None
        return raw / 1000.0

    if msg.encoding == "32FC1":
        pix_off = row_off + u * 4
        if pix_off + 4 > len(msg.data):
            return None
        z = struct.unpack_from("<f", msg.data, pix_off)[0]
        if z <= 0.0 or z != z:  # nan
            return None
        return z

    return None


class RealsenseDepthDebugNode(Node):
    """Subscribes to color, aligned depth, and camera info; logs sanity checks."""

    def __init__(self) -> None:
        super().__init__("realsense_depth_debug")

        self.declare_parameter("output_frame_reference", "base_link")
        self.declare_parameter("rgb_image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_image_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("log_period_sec", 2.0)
        self.declare_parameter("warn_no_data_sec", 5.0)

        self._rgb: Optional[Image] = None
        self._depth: Optional[Image] = None
        self._info: Optional[CameraInfo] = None
        self._rgb_logged = False
        self._depth_encoding_warned = False
        self._start = time.monotonic()

        rgb_topic = self.get_parameter("rgb_image_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_image_topic").get_parameter_value().string_value
        info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value

        self.create_subscription(Image, rgb_topic, self._on_rgb, qos_profile_sensor_data)
        self.create_subscription(Image, depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, info_topic, self._on_info, qos_profile_sensor_data)

        period = float(self.get_parameter("log_period_sec").value)
        self.create_timer(max(period, 0.5), self._on_timer)

        out_frame = self.get_parameter("output_frame_reference").get_parameter_value().string_value
        self.get_logger().info(
            f"Subscribed rgb='{rgb_topic}' depth='{depth_topic}' info='{info_topic}'. "
            f"(Future box outputs will use frame '{out_frame}'.)"
        )

    def _on_rgb(self, msg: Image) -> None:
        self._rgb = msg
        if not self._rgb_logged:
            self._rgb_logged = True
            self.get_logger().info(
                f"RGB OK: {msg.width}x{msg.height} encoding={msg.encoding} "
                f"step={msg.step} frame_id={msg.header.frame_id}"
            )

    def _on_depth(self, msg: Image) -> None:
        self._depth = msg

    def _on_info(self, msg: CameraInfo) -> None:
        self._info = msg

    def _on_timer(self) -> None:
        now = time.monotonic()
        elapsed = now - self._start
        warn_after = float(self.get_parameter("warn_no_data_sec").value)

        if self._rgb is None and elapsed > warn_after:
            self.get_logger().warning("No RGB image yet — check camera topics / driver.")
        if self._depth is None and elapsed > warn_after:
            self.get_logger().warning("No aligned depth image yet — enable align_depth in launch args.")
        if self._info is None and elapsed > warn_after:
            self.get_logger().warning("No CameraInfo yet — check color camera_info topic.")

        if self._depth is None or self._info is None:
            return

        d = self._depth
        u, v = d.width // 2, d.height // 2
        z = _depth_m_at_center(d, u, v)

        k = self._info.k
        fx = k[0] if len(k) > 0 else float("nan")

        if z is None:
            if d.encoding not in ("16UC1", "mono16", "32FC1") and not self._depth_encoding_warned:
                self.get_logger().warning(
                    f"Depth encoding '{d.encoding}' not handled (expected 16UC1 or 32FC1); "
                    "center depth skipped — extend realsense_depth_debug_node if needed."
                )
                self._depth_encoding_warned = True
            self.get_logger().info(
                f"Aligned depth frame {d.width}x{d.height} encoding={d.encoding} "
                f"center=({u},{v}) depth=invalid_or_zero (fx={fx:.5f})"
            )
        else:
            # Plausible range check for ~0.5 m workspace (wide tolerance)
            if z < 0.15 or z > 3.0:
                self.get_logger().warning(
                    f"Center depth z={z:.3f} m is outside typical 0.15–3.0 m sanity band "
                    f"(check scene, exposure, or alignment)."
                )
            else:
                self.get_logger().info(
                    f"Center depth OK: z≈{z:.3f} m at ({u},{v}) "
                    f"(aligned_depth frame_id={d.header.frame_id}, fx={fx:.5f})"
                )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RealsenseDepthDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
