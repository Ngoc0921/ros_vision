"""Visualize a detected target and box in RViz2 using visualization_msgs/Marker.

Subscriber 1 — target marker (blue cylinder):
  Subscribes to a geometry_msgs/PointStamped (x, y in base_link) and publishes a
  Marker on every incoming message so the marker is continuously updated.

Subscriber 2 — box marker (yellow cube):
  Subscribes to /vision/box (gp7_vision_pipeline/Box) and publishes a cube Marker
  using the detected pose, size, and orientation from the message.

Parameters (all with sensible defaults):
  input_topic        -- subscription topic for geometry_msgs/PointStamped
  output_topic       -- publication topic for visualization_msgs/Marker
  frame_id           -- reference frame for the marker (default: base_link)
  target_diameter    -- cylinder diameter in metres (default: 0.04)
  target_height      -- cylinder height in metres (default: 0.01)
  fixed_z            -- z coordinate of the marker centre (default: 0.03)
  alpha              -- marker opacity 0-1 (default: 0.7)
  use_bottom_z       -- if true, fixed_z is the bottom of the cylinder, not the centre
  box_input_topic    -- subscription topic for box detection (default: /vision/box)
  box_output_topic   -- publication topic for box marker (default: /visualization/box_marker)
  box_alpha          -- box marker opacity 0-1 (default: 0.6)
  default_box_width  -- fallback width when size is unavailable (default: 0.1)
  default_box_depth  -- fallback depth when size is unavailable (default: 0.1)
  default_box_height -- fallback height when size is unavailable (default: 0.1)

Usage::

    # Build & run
    cd ~/pap_yaskawa_ws
    colcon build --packages-select gp7_vision_pipeline
    source install/setup.bash
    ros2 run gp7_vision_pipeline target_marker_node

    # Override any parameter at the command line
    ros2 run gp7_vision_pipeline target_marker_node --ros-args \\
        -p input_topic:=/vision/target_position \\
        -p target_diameter:=0.05

    # Test target marker
    ros2 topic pub /vision/target_position geometry_msgs/PointStamped \\
        '{header: {stamp: {sec: 0, nanosec: 0}, frame_id: "base_link"},
          point: {x: 0.3, y: -0.2, z: 0.0}}' -r 2

    # Test box marker
    ros2 topic pub /vision/box gp7_vision_pipeline/msg/Box \\
        '{header: {frame_id: "base_link"},
          pose: {position: {x: 0.4, y: 0.1, z: 0.08},
                 orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}},
          size: {x: 0.1, y: 0.1, z: 0.2},
          class_name: "box",
          confidence: 0.9}' -r 2

    # In RViz2:
    #   Fixed Frame: base_link
    #   Add → Marker and subscribe to /visualization/target_marker and /visualization/box_marker
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

from gp7_vision_pipeline.msg import Box


class TargetMarkerNode(Node):
    """Subscribe to a PointStamped target position and publish a cylinder Marker for RViz2."""

    def __init__(self) -> None:
        super().__init__("target_marker_node")

        # --- Declare target marker parameters ------------------------------------
        self.declare_parameter("input_topic", "/vision/target_position")
        self.declare_parameter("output_topic", "/visualization/target_marker")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("target_diameter", 0.04)
        self.declare_parameter("target_height", 0.01)
        self.declare_parameter("fixed_z", 0.03)
        self.declare_parameter("alpha", 0.7)
        self.declare_parameter("use_bottom_z", False)

        # --- Declare box marker parameters --------------------------------------
        self.declare_parameter("box_input_topic", "/vision/box")
        self.declare_parameter("box_output_topic", "/visualization/box_marker")
        self.declare_parameter("box_alpha", 0.6)
        self.declare_parameter("default_box_width", 0.1)
        self.declare_parameter("default_box_depth", 0.1)
        self.declare_parameter("default_box_height", 0.1)

        self._input_topic: str = self.get_parameter("input_topic").value
        self._output_topic: str = self.get_parameter("output_topic").value
        self._frame_id: str = self.get_parameter("frame_id").value
        self._diameter: float = self.get_parameter("target_diameter").value
        self._height: float = self.get_parameter("target_height").value
        self._fixed_z: float = self.get_parameter("fixed_z").value
        self._alpha: float = self.get_parameter("alpha").value
        self._use_bottom_z: bool = self.get_parameter("use_bottom_z").value

        self._box_input_topic: str = self.get_parameter("box_input_topic").value
        self._box_output_topic: str = self.get_parameter("box_output_topic").value
        self._box_alpha: float = self.get_parameter("box_alpha").value
        self._default_box_width: float = self.get_parameter("default_box_width").value
        self._default_box_depth: float = self.get_parameter("default_box_depth").value
        self._default_box_height: float = self.get_parameter("default_box_height").value

        # --- QoS profile --------------------------------------------------------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # --- Target marker publisher & subscriber ------------------------------
        self._publisher = self.create_publisher(Marker, self._output_topic, qos)
        self._subscriber = self.create_subscription(
            PointStamped,
            self._input_topic,
            self._on_target_position,
            qos,
        )

        # --- Box marker publisher & subscriber ---------------------------------
        self._box_publisher = self.create_publisher(Marker, self._box_output_topic, qos)
        self._box_subscriber = self.create_subscription(
            Box,
            self._box_input_topic,
            self._on_box_detection,
            qos,
        )

        self.get_logger().info(
            f"[TargetMarkerNode] started — target: '{self._input_topic}' → '{self._output_topic}', "
            f"box: '{self._box_input_topic}' → '{self._box_output_topic}'"
        )
        self.get_logger().info(
            f"[TargetMarkerNode] frame={self._frame_id}, "
            f"target: diameter={self._diameter:.3f}m, height={self._height:.3f}m, "
            f"z={self._fixed_z:.3f}m, alpha={self._alpha}, use_bottom_z={self._use_bottom_z}, "
            f"box: alpha={self._box_alpha}, "
            f"default_size=({self._default_box_width:.3f}, {self._default_box_depth:.3f}, "
            f"{self._default_box_height:.3f})"
        )

    # -------------------------------------------------------------------------

    def _on_target_position(self, msg: PointStamped) -> None:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._frame_id

        marker.ns = "target"
        marker.id = 0

        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        # Scale — diameter shared across x and y
        marker.scale.x = self._diameter
        marker.scale.y = self._diameter
        marker.scale.z = self._height

        # Position — x and y from the vision topic, z is fixed
        marker.pose.position.x = msg.point.x
        marker.pose.position.y = msg.point.y

        if self._use_bottom_z:
            marker.pose.position.z = self._fixed_z + self._height / 2.0
        else:
            marker.pose.position.z = self._fixed_z

        # Orientation — no rotation (identity quaternion)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Color — blue
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = self._alpha

        # Lifetime — 0 means the marker persists until explicitly deleted or updated
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()

        self._publisher.publish(marker)

        self.get_logger().debug(
            f"[TargetMarkerNode] marker updated at "
            f"({marker.pose.position.x:.4f}, {marker.pose.position.y:.4f}, "
            f"{marker.pose.position.z:.4f})"
        )

    # -------------------------------------------------------------------------

    def _on_box_detection(self, msg: Box) -> None:
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self._frame_id

        marker.ns = "box"
        marker.id = 0

        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Scale — use detected size, fall back to defaults if not set
        marker.scale.x = msg.size.x if msg.size.x > 0 else self._default_box_width
        marker.scale.y = msg.size.y if msg.size.y > 0 else self._default_box_depth
        marker.scale.z = msg.size.z if msg.size.z > 0 else self._default_box_height

        # Position — directly from detected pose
        marker.pose.position.x = msg.pose.position.x
        marker.pose.position.y = msg.pose.position.y
        marker.pose.position.z = msg.pose.position.z

        # Orientation — from detected pose; fall back to identity if all zeros
        orient = msg.pose.orientation
        if orient.x == 0.0 and orient.y == 0.0 and orient.z == 0.0 and orient.w == 0.0:
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
        else:
            marker.pose.orientation.x = orient.x
            marker.pose.orientation.y = orient.y
            marker.pose.orientation.z = orient.z
            marker.pose.orientation.w = orient.w

        # Color — light yellow
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.4
        marker.color.a = self._box_alpha

        # Lifetime — 0 means the marker persists until explicitly deleted or updated
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()

        self._box_publisher.publish(marker)

        self.get_logger().debug(
            f"[TargetMarkerNode] box marker updated at "
            f"({marker.pose.position.x:.4f}, {marker.pose.position.y:.4f}, "
            f"{marker.pose.position.z:.4f}) "
            f"size=({marker.scale.x:.4f}, {marker.scale.y:.4f}, {marker.scale.z:.4f})"
        )


# -----------------------------------------------------------------------------

def main(argv=None) -> None:
    rclpy.init(args=argv)
    try:
        node = TargetMarkerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except NameError:
            pass
        rclpy.shutdown()


if __name__ == "__main__":
    main()
