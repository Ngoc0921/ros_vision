"""Layer 3 marker visualization for robot_vision_pipeline.

Receives ObjectArray from pixel_to_base_mapper_node and publishes RViz markers.

Inputs:
  /vision/wood_objects              robot_vision_pipeline_msgs/ObjectArray
  /vision/target_position          geometry_msgs/PointStamped  (optional)
  /vision/target_detected          std_msgs/Bool               (optional)

Outputs:
  /vision/detection_markers        visualization_msgs/MarkerArray
  /vision/detection_status         std_msgs/String
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

from robot_vision_pipeline_msgs.msg import ObjectArray


class VisionDetectionMarkerNode(Node):
    """Publish RViz markers from ObjectArray and target position topics."""

    def __init__(self) -> None:
        super().__init__("vision_detection_marker_node")

        self.declare_parameter("marker_frame_id", "camera_color_optical_frame")
        self.declare_parameter("marker_publish_rate_hz", 5.0)
        self.declare_parameter("target_marker_diameter", 0.04)
        self.declare_parameter("target_marker_height", 0.01)
        self.declare_parameter("target_marker_z_is_bottom", False)
        self.declare_parameter("target_detection_timeout_sec", 1.0)
        self.declare_parameter("wood_marker_length", 0.08)
        self.declare_parameter("wood_marker_width", 0.03)
        self.declare_parameter("wood_marker_height", 0.02)

        self._frame_id = str(self.get_parameter("marker_frame_id").value)
        self._rate_hz = float(self.get_parameter("marker_publish_rate_hz").value)
        self._target_diameter = float(self.get_parameter("target_marker_diameter").value)
        self._target_height = float(self.get_parameter("target_marker_height").value)
        self._target_z_is_bottom = bool(self.get_parameter("target_marker_z_is_bottom").value)
        self._target_timeout = float(self.get_parameter("target_detection_timeout_sec").value)
        self._wood_length = float(self.get_parameter("wood_marker_length").value)
        self._wood_width = float(self.get_parameter("wood_marker_width").value)
        self._wood_height = float(self.get_parameter("wood_marker_height").value)

        self._target_detected = False
        self._target_pos: Optional[PointStamped] = None
        self._last_target_msg_time: Optional[float] = None

        self._latest_wood_array: Optional[ObjectArray] = None

        marker_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, "/vision/detection_markers", marker_qos
        )
        self._status_pub = self.create_publisher(String, "/vision/detection_status", 10)

        self.create_subscription(
            ObjectArray,
            "/vision/wood_objects",
            self._on_wood_objects,
            10,
        )
        self.create_subscription(
            PointStamped, "/vision/target_position", self._on_target_position, 10
        )
        self.create_subscription(Bool, "/vision/target_detected", self._on_target_detected, 10)

        self._timer = self.create_timer(1.0 / self._rate_hz, self._on_timer)

        self.get_logger().info(
            f"vision_detection_marker_node started | frame={self._frame_id}, "
            f"target_timeout={self._target_timeout}s, "
            f"wood_marker=({self._wood_length:.3f}, {self._wood_width:.3f}, "
            f"{self._wood_height:.3f})m"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _on_wood_objects(self, msg: ObjectArray) -> None:
        self._latest_wood_array = msg

    def _on_target_position(self, msg: PointStamped) -> None:
        self._target_pos = msg
        self._last_target_msg_time = self._now_sec()

    def _on_target_detected(self, msg: Bool) -> None:
        was = self._target_detected
        self._target_detected = bool(msg.data)
        if msg.data and not was:
            self.get_logger().info("Target detected")
        elif not msg.data and was:
            self.get_logger().warn("Target lost / not detected")

    def _on_timer(self) -> None:
        now = self._now_sec()

        target_active = self._target_detected
        if self._last_target_msg_time is not None:
            if (now - self._last_target_msg_time) > self._target_timeout:
                target_active = False

        target_pos = self._target_pos

        n_woods = 0
        wood_objects = []
        if self._latest_wood_array is not None:
            wood_objects = list(self._latest_wood_array.objects)
            n_woods = len(wood_objects)

        wood_active = n_woods > 0
        wood_detected = wood_active

        self._publish_markers(target_active, target_pos, wood_active, wood_objects)

        status = String()
        status.data = (
            f"target_detected={target_active}, "
            f"wood_detected={wood_detected}, n_woods={n_woods}"
        )
        self._status_pub.publish(status)

    def _is_valid_wood_object(self, obj) -> bool:
        values = (
            obj.pose.position.x,
            obj.pose.position.y,
            obj.pose.position.z,
            obj.pose.orientation.x,
            obj.pose.orientation.y,
            obj.pose.orientation.z,
            obj.pose.orientation.w,
        )
        return all(math.isfinite(float(v)) for v in values)

    def _publish_markers(
        self,
        target_active: bool,
        target_pos: Optional[PointStamped],
        wood_active: bool,
        wood_objects,
    ) -> None:
        now = self.get_clock().now().to_msg()
        markers: list[Marker] = []

        clear = Marker()
        clear.header.stamp = now
        clear.header.frame_id = self._frame_id
        clear.action = Marker.DELETEALL
        markers.append(clear)

        if target_active and target_pos is not None:
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = self._frame_id
            marker.ns = "vision_target"
            marker.id = 0
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(target_pos.point.x)
            marker.pose.position.y = float(target_pos.point.y)
            marker.pose.position.z = float(target_pos.point.z)
            if self._target_z_is_bottom:
                marker.pose.position.z += self._target_height / 2.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = self._target_diameter
            marker.scale.y = self._target_diameter
            marker.scale.z = self._target_height
            marker.color.r = 0.0
            marker.color.g = 0.2
            marker.color.b = 1.0
            marker.color.a = 1.0
            markers.append(marker)

        if wood_active:
            for idx, obj in enumerate(wood_objects):
                if not self._is_valid_wood_object(obj):
                    continue
                marker = Marker()
                marker.header.stamp = now
                marker.header.frame_id = self._frame_id
                marker.ns = "vision_wood"
                marker.id = 1 + idx
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.pose = obj.pose
                marker.scale.x = self._wood_length
                marker.scale.y = self._wood_width
                marker.scale.z = self._wood_height
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.75
                markers.append(marker)

        self._marker_pub.publish(MarkerArray(markers=markers))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionDetectionMarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
