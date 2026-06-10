"""Vision detection marker node — publishes RViz MarkerArray for detected target and box.

Subscribes to vision pipeline outputs and publishes a combined MarkerArray to
/vision/detection_markers. This node owns all vision-related RViz visualization.

Topic ownership:
  gp7_vision_pipeline publishes:
    /vision/target_position
    /vision/box
    /vision/target_detected
    /vision/box_detected
    /vision/detection_status
    /vision/detection_markers   ← this node

  gp7_drl_inference subscribes to vision topics and plans trajectories.
"""

from __future__ import annotations

import threading
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray

from gp7_vision_pipeline.msg import Box


class VisionDetectionMarkerNode(Node):
    """Publishes RViz MarkerArray for vision detections.

    Marker design:
      Target: blue CYLINDER, 4 cm diameter, 1 cm tall, at target_position.
      Box:    yellow CUBE with exact detected size and orientation.

    DELETEALL is published first every cycle so stale markers are removed.
    """

    def __init__(self) -> None:
        super().__init__("vision_detection_marker_node")

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter("marker_frame_id", "base_link")
        self.declare_parameter("marker_publish_rate_hz", 5.0)
        self.declare_parameter("target_marker_diameter", 0.04)
        self.declare_parameter("target_marker_height", 0.01)
        self.declare_parameter("target_marker_z_is_bottom", False)
        self.declare_parameter("target_detection_timeout_sec", 1.0)
        self.declare_parameter("box_detection_timeout_sec", 1.0)
        self.declare_parameter("box_confidence_threshold", 0.5)

        self._frame_id = str(self.get_parameter("marker_frame_id").value)
        self._rate_hz = float(self.get_parameter("marker_publish_rate_hz").value)
        self._target_diameter = float(self.get_parameter("target_marker_diameter").value)
        self._target_height = float(self.get_parameter("target_marker_height").value)
        self._target_z_is_bottom = bool(self.get_parameter("target_marker_z_is_bottom").value)
        self._target_timeout = float(self.get_parameter("target_detection_timeout_sec").value)
        self._box_timeout = float(self.get_parameter("box_detection_timeout_sec").value)
        self._box_conf_thresh = float(self.get_parameter("box_confidence_threshold").value)

        # ── State (guarded by _lock) ──────────────────────────────────────
        self._lock = threading.RLock()
        self._target_detected: bool = False
        self._target_pos: Optional[PointStamped] = None
        self._last_target_msg_time: Optional[float] = None
        self._box_detected: bool = False
        self._box: Optional[Box] = None
        self._last_box_msg_time: Optional[float] = None
        self._box_confidence: float = 0.0

        # ── Publishers ────────────────────────────────────────────────────
        marker_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, "/vision/detection_markers", marker_qos
        )
        self._status_pub = self.create_publisher(
            String, "/vision/detection_status", 10
        )

        # ── Subscriptions ─────────────────────────────────────────────────
        self.create_subscription(
            PointStamped,
            "/vision/target_position",
            self._on_target_position,
            10,
        )
        self.create_subscription(
            Bool,
            "/vision/target_detected",
            self._on_target_detected,
            10,
        )
        self.create_subscription(
            Box,
            "/vision/box",
            self._on_box,
            10,
        )
        self.create_subscription(
            Bool,
            "/vision/box_detected",
            self._on_box_detected,
            10,
        )

        # ── Timer: publish markers at configured rate ─────────────────────
        period_sec = 1.0 / self._rate_hz
        self._timer = self.create_timer(period_sec, self._on_timer)

        self.get_logger().info(
            f"vision_detection_marker_node started | frame={self._frame_id} "
            f"rate={self._rate_hz} Hz | "
            f"target_diameter={self._target_diameter}m "
            f"height={self._target_height}m "
            f"z_is_bottom={self._target_z_is_bottom} | "
            f"target_timeout={self._target_timeout}s "
            f"box_timeout={self._box_timeout}s "
            f"box_conf_thresh={self._box_conf_thresh}"
        )

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_target_position(self, msg: PointStamped) -> None:
        with self._lock:
            self._target_pos = msg
            self._last_target_msg_time = self.get_clock().now().seconds_nanoseconds()[0]

    def _on_target_detected(self, msg: Bool) -> None:
        was_detected = self._target_detected
        self._target_detected = msg.data
        if msg.data and not was_detected:
            self.get_logger().info("Target detected")
        elif not msg.data and was_detected:
            self.get_logger().warn("Target lost / not detected")

    def _on_box(self, msg: Box) -> None:
        confidence = float(getattr(msg, "confidence", 0.0))
        if confidence < self._box_conf_thresh:
            with self._lock:
                if self._box_detected:
                    self.get_logger().warn(
                        f"Vision box ignored in marker node: "
                        f"confidence={confidence:.3f} < {self._box_conf_thresh}"
                    )
                self._box_detected = False
            return
        with self._lock:
            self._box = msg
            self._box_confidence = confidence
            self._last_box_msg_time = self.get_clock().now().seconds_nanoseconds()[0]
            self._box_detected = True

    def _on_box_detected(self, msg: Bool) -> None:
        was_detected = self._box_detected
        self._box_detected = msg.data
        if msg.data and not was_detected:
            self.get_logger().info("Box detected")
        elif not msg.data and was_detected:
            self.get_logger().warn("Box lost / not detected")

    # ── Timer: check staleness and publish ─────────────────────────────────────

    def _on_timer(self) -> None:
        now = self.get_clock().now().seconds_nanoseconds()[0]

        with self._lock:
            target_active = self._target_detected
            box_active = self._box_detected

            # Check target staleness
            if self._last_target_msg_time is not None:
                age = now - self._last_target_msg_time
                if age > self._target_timeout:
                    if target_active:
                        self.get_logger().warn(
                            f"Target marker stale: last seen {age:.1f}s ago "
                            f"(timeout={self._target_timeout}s)"
                        )
                    target_active = False

            # Check box staleness
            if self._last_box_msg_time is not None:
                age = now - self._last_box_msg_time
                if age > self._box_timeout:
                    if box_active:
                        self.get_logger().warn(
                            f"Box marker stale: last seen {age:.1f}s ago "
                            f"(timeout={self._box_timeout}s)"
                        )
                    box_active = False

            target_pos = self._target_pos
            box = self._box

        # Build and publish markers
        self._publish_markers(target_active, target_pos, box_active, box)

        # Publish combined status string
        status = String()
        status.data = (
            f"target_detected={target_active}, box_detected={box_active}"
        )
        self._status_pub.publish(status)

    # ── Marker assembly ────────────────────────────────────────────────────────

    def _publish_markers(
        self,
        target_active: bool,
        target_pos: Optional[PointStamped],
        box_active: bool,
        box: Optional[Box],
    ) -> None:
        now = self.get_clock().now().to_msg()
        markers: list[Marker] = []

        # DELETEALL first — clears all previous markers so stale ones disappear
        da = Marker()
        da.header.stamp = now
        da.header.frame_id = self._frame_id
        da.action = Marker.DELETEALL
        markers.append(da)

        # ── Target: blue CYLINDER ───────────────────────────────────────
        target_marker: Optional[Marker] = None
        if target_active and target_pos is not None:
            z = float(target_pos.point.z)
            if self._target_z_is_bottom:
                z += self._target_height / 2.0

            m = Marker()
            m.header.stamp = now
            m.header.frame_id = self._frame_id
            m.ns = "vision_target"
            m.id = 0
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(target_pos.point.x)
            m.pose.position.y = float(target_pos.point.y)
            m.pose.position.z = z
            m.pose.orientation.w = 1.0
            m.scale.x = self._target_diameter
            m.scale.y = self._target_diameter
            m.scale.z = self._target_height
            m.color.r = 0.0
            m.color.g = 0.2
            m.color.b = 1.0
            m.color.a = 1.0
            markers.append(m)
            target_marker = m
            self.get_logger().info(
                f"Vision target marker ADD cylinder blue: "
                f"center=({target_pos.point.x:.4f}, {target_pos.point.y:.4f}, "
                f"{target_pos.point.z:.4f}), diameter={self._target_diameter}, "
                f"height={self._target_height}, marker_z={z:.4f}"
            )

        # ── Box: yellow CUBE ─────────────────────────────────────────
        box_marker: Optional[Marker] = None
        if box_active and box is not None:
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = self._frame_id
            m.ns = "vision_box"
            m.id = 1
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = float(box.pose.position.x)
            m.pose.position.y = float(box.pose.position.y)
            m.pose.position.z = float(box.pose.position.z)
            m.pose.orientation.x = float(box.pose.orientation.x)
            m.pose.orientation.y = float(box.pose.orientation.y)
            m.pose.orientation.z = float(box.pose.orientation.z)
            m.pose.orientation.w = float(box.pose.orientation.w)
            m.scale.x = float(box.size.x)
            m.scale.y = float(box.size.y)
            m.scale.z = float(box.size.z)
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.7
            markers.append(m)
            box_marker = m
            self.get_logger().info(
                f"Vision box marker ADD cube yellow: "
                f"center=({box.pose.position.x:.4f}, {box.pose.position.y:.4f}, "
                f"{box.pose.position.z:.4f}), "
                f"size=({box.size.x:.4f}, {box.size.y:.4f}, {box.size.z:.4f})"
            )

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
