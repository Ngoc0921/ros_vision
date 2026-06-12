"""Publish RViz MarkerArray for wood and box detections.

Subscribes to:
  /vision/wood_objects    robot_vision_pipeline_msgs/WoodArray
  /vision/box_objects   robot_vision_pipeline_msgs/BoxArray

Publishes:
  /vision/detection_markers   visualization_msgs/MarkerArray

Marker design:
  wood: green CUBE, default size from parameters (no size in Wood.msg)
  box:  yellow CUBE, scale from box.size
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from visualization_msgs.msg import Marker, MarkerArray

from robot_vision_pipeline_msgs.msg import WoodArray, BoxArray


class VisionDetectionMarkerNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_detection_marker_node")

        self._data_lock = threading.Lock()
        self._latest_wood_arr: Optional[WoodArray] = None
        self._latest_box_arr: Optional[BoxArray] = None

        self.declare_parameter("marker_frame_id", "camera_color_optical_frame")
        self.declare_parameter("marker_publish_rate_hz", 5.0)

        self.declare_parameter("wood_marker_size_x_m", 0.04)
        self.declare_parameter("wood_marker_size_y_m", 0.04)
        self.declare_parameter("wood_marker_size_z_m", 0.04)

        self.declare_parameter("box_marker_size_x_m", 0.08)
        self.declare_parameter("box_marker_size_y_m", 0.08)
        self.declare_parameter("box_marker_size_z_m", 0.05)

        self.declare_parameter("wood_detection_timeout_sec", 1.0)
        self.declare_parameter("wood_confidence_threshold", 0.5)

        self._marker_frame_id = str(self.get_parameter("marker_frame_id").value)
        rate_hz = float(self.get_parameter("marker_publish_rate_hz").value)

        self._wood_size_x = float(self.get_parameter("wood_marker_size_x_m").value)
        self._wood_size_y = float(self.get_parameter("wood_marker_size_y_m").value)
        self._wood_size_z = float(self.get_parameter("wood_marker_size_z_m").value)

        self._default_box_x = float(self.get_parameter("box_marker_size_x_m").value)
        self._default_box_y = float(self.get_parameter("box_marker_size_y_m").value)
        self._default_box_z = float(self.get_parameter("box_marker_size_z_m").value)

        self._wood_timeout = float(self.get_parameter("wood_detection_timeout_sec").value)
        self._wood_conf_thresh = float(self.get_parameter("wood_confidence_threshold").value)

        det_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pub_markers = self.create_publisher(MarkerArray, "/vision/detection_markers", 10)

        self.create_subscription(
            WoodArray, "/vision/wood_objects", self._on_wood_array, det_qos
        )
        self.create_subscription(
            BoxArray, "/vision/box_objects", self._on_box_array, det_qos
        )

        period_sec = 1.0 / rate_hz if rate_hz > 0 else 0.2
        self._publish_timer = self.create_timer(period_sec, self._publish_markers)

        self._marker_id_counter = 0

        self.get_logger().info(
            f"vision_detection_marker_node started | "
            f"frame={self._marker_frame_id}, "
            f"wood_size=({self._wood_size_x},{self._wood_size_y},{self._wood_size_z}), "
            f"default_box=({self._default_box_x},{self._default_box_y},{self._default_box_z})"
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_wood_array(self, msg: WoodArray) -> None:
        with self._data_lock:
            self._latest_wood_arr = msg

    def _on_box_array(self, msg: BoxArray) -> None:
        with self._data_lock:
            self._latest_box_arr = msg

    def _next_id(self) -> int:
        self._marker_id_counter += 1
        return self._marker_id_counter

    def _make_delete_all(self) -> Marker:
        m = Marker()
        m.action = Marker.DELETEALL
        return m

    def _make_wood_marker(
        self, wood_id: int, frame_id: str, stamp_sec: float,
        px: float, py: float, pz: float,
    ) -> Marker:
        m = Marker()
        m.header.stamp.sec = int(stamp_sec)
        m.header.stamp.nanosec = int((stamp_sec % 1.0) * 1e9)
        m.header.frame_id = frame_id
        m.ns = "wood"
        m.id = self._next_id()
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = px
        m.pose.position.y = py
        m.pose.position.z = pz
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = self._wood_size_x
        m.scale.y = self._wood_size_y
        m.scale.z = self._wood_size_z
        m.color.r = 0.0
        m.color.g = 0.8
        m.color.b = 0.0
        m.color.a = 0.85
        return m

    def _make_box_marker(
        self, box_id: int, frame_id: str, stamp_sec: float,
        px: float, py: float, pz: float,
        sx: float, sy: float, sz: float,
    ) -> Marker:
        if sx <= 0.0:
            sx = self._default_box_x
        if sy <= 0.0:
            sy = self._default_box_y
        if sz <= 0.0:
            sz = self._default_box_z

        m = Marker()
        m.header.stamp.sec = int(stamp_sec)
        m.header.stamp.nanosec = int((stamp_sec % 1.0) * 1e9)
        m.header.frame_id = frame_id
        m.ns = "box"
        m.id = self._next_id()
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = px
        m.pose.position.y = py
        m.pose.position.z = pz
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = 1.0
        m.scale.x = sx
        m.scale.y = sy
        m.scale.z = sz
        m.color.r = 0.0
        m.color.g = 0.8
        m.color.b = 0.8
        m.color.a = 0.85
        return m

    def _publish_markers(self) -> None:
        now = self._now_seconds()

        with self._data_lock:
            wood_arr = self._latest_wood_arr
            box_arr = self._latest_box_arr

        arr = MarkerArray()
        arr.markers.append(self._make_delete_all())

        if wood_arr is not None:
            frame = wood_arr.header.frame_id or self._marker_frame_id
            stamp = float(wood_arr.header.stamp.sec) + float(
                wood_arr.header.stamp.nanosec
            ) * 1e-9
            for wood in wood_arr.woods:
                if wood.confidence < self._wood_conf_thresh:
                    continue
                if (now - stamp) > self._wood_timeout:
                    continue
                arr.markers.append(
                    self._make_wood_marker(
                        wood.wood_id, frame, stamp,
                        wood.pose.position.x,
                        wood.pose.position.y,
                        wood.pose.position.z,
                    )
                )

        if box_arr is not None:
            frame = box_arr.header.frame_id or self._marker_frame_id
            stamp = float(box_arr.header.stamp.sec) + float(
                box_arr.header.stamp.nanosec
            ) * 1e-9
            for box in box_arr.boxes:
                arr.markers.append(
                    self._make_box_marker(
                        box.box_id, frame, stamp,
                        box.pose.position.x,
                        box.pose.position.y,
                        box.pose.position.z,
                        box.size.x,
                        box.size.y,
                        box.size.z,
                    )
                )

        if arr.markers:
            self._pub_markers.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionDetectionMarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
