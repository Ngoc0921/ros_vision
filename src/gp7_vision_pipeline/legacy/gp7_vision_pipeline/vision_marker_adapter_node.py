"""Bridge existing pose topics to the marker nodes' expected topics.

Subscribes to:
  /vision/target_pose_in_base_link  → PoseStamped → /vision/target_position (PointStamped)
  /vision/box_pose_in_base_link    → PoseStamped → /vision/box            (gp7_vision_pipeline/Box)

Parameters:
  target_pose_input_topic       default "/vision/target_pose_in_base_link"
  target_position_output_topic default "/vision/target_position"
  box_pose_input_topic          default "/vision/box_pose_in_base_link"
  box_output_topic              default "/vision/box"
  default_box_width            default 0.1
  default_box_depth            default 0.1
  default_box_height           default 0.1

Usage::

    # Build & run
    cd ~/pap_yaskawa_ws
    colcon build --packages-select gp7_vision_pipeline
    source install/setup.bash
    ros2 run gp7_vision_pipeline vision_marker_adapter_node

    # Override topics at the command line
    ros2 run gp7_vision_pipeline vision_marker_adapter_node --ros-args \\
        -p target_pose_input_topic:=/vision/target_pose_in_base_link
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PointStamped, PoseStamped
from gp7_vision_pipeline.msg import Box


class VisionMarkerAdapterNode(Node):
    """Bridge PoseStamped topics to PointStamped and Box for the marker nodes."""

    def __init__(self) -> None:
        super().__init__("vision_marker_adapter_node")

        self.declare_parameter("target_pose_input_topic", "/vision/target_pose_in_base_link")
        self.declare_parameter("target_position_output_topic", "/vision/target_position")
        self.declare_parameter("box_pose_input_topic", "/vision/box_pose_in_base_link")
        self.declare_parameter("box_output_topic", "/vision/box")
        self.declare_parameter("default_box_width", 0.1)
        self.declare_parameter("default_box_depth", 0.1)
        self.declare_parameter("default_box_height", 0.1)

        self._target_pose_topic: str = self.get_parameter("target_pose_input_topic").value
        self._target_position_topic: str = self.get_parameter("target_position_output_topic").value
        self._box_pose_topic: str = self.get_parameter("box_pose_input_topic").value
        self._box_out_topic: str = self.get_parameter("box_output_topic").value
        self._default_width: float = self.get_parameter("default_box_width").value
        self._default_depth: float = self.get_parameter("default_box_depth").value
        self._default_height: float = self.get_parameter("default_box_height").value

        # Subscriptions: BEST_EFFORT to match camera_projection_node publisher QoS
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        # Publications: RELIABLE (marker subscribers expect reliable delivery)
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # Bridge A: /vision/target_pose_in_base_link → /vision/target_position
        self._target_pub = self.create_publisher(PointStamped, self._target_position_topic, qos_pub)
        self.create_subscription(
            PoseStamped,
            self._target_pose_topic,
            self._on_target_pose,
            qos_sub,
        )

        # Bridge B: /vision/box_pose_in_base_link → /vision/box
        self._box_pub = self.create_publisher(Box, self._box_out_topic, qos_pub)
        self.create_subscription(
            PoseStamped,
            self._box_pose_topic,
            self._on_box_pose,
            qos_sub,
        )

        self.get_logger().info(
            f"[VisionMarkerAdapter] target: '{self._target_pose_topic}' → '{self._target_position_topic}', "
            f"box: '{self._box_pose_topic}' → '{self._box_out_topic}'"
        )
        self.get_logger().info(
            f"[VisionMarkerAdapter] default box size: "
            f"({self._default_width}, {self._default_depth}, {self._default_height})"
        )

    def _on_target_pose(self, msg: PoseStamped) -> None:
        out = PointStamped()
        out.header = msg.header
        out.point.x = msg.pose.position.x
        out.point.y = msg.pose.position.y
        out.point.z = msg.pose.position.z
        self._target_pub.publish(out)

    def _on_box_pose(self, msg: PoseStamped) -> None:
        out = Box()
        out.header = msg.header
        out.pose = msg.pose
        out.size.x = self._default_width
        out.size.y = self._default_depth
        out.size.z = self._default_height
        out.class_name = "box"
        out.confidence = 1.0
        self._box_pub.publish(out)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    try:
        node = VisionMarkerAdapterNode()
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
