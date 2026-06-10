"""Project 2D detections to 3D PoseStamped in base_link using camera extrinsics.

Luong xu ly:
  1. Nhan BoxDetection tu /vision/target_detection va /vision/box_detection
  2. Lay camera intrinsics tu /camera/camera/color/camera_info
  3. Lay depth tu aligned depth ROI quanh detection center
  4. Chuyen (u, v, depth) → point camera_optical_frame (pinhole)
  5. Transform: camera_optical_frame → base_link (tf2_ros)
  6. Publish: /vision/target_pose_in_base_link, /vision/box_pose_in_base_link
  7. Service: /vision/get_pixel_pose — goi truc tiep pixel → base_link

Ma tran ngoai (camera_optical_frame → base_link):
  Doc tu ROS 2 parameters hoac mac dinh:
    tx=0, ty=-0.7, tz=1.0
    roll=pi, pitch=0, yaw=0  (lens down)

Usage::

    ros2 launch gp7_vision_pipeline camera_projection.launch.py

    # Override extrinsics (metres, radians)
    ros2 launch gp7_vision_pipeline camera_projection.launch.py \\
        extr_tx:=-0.1 extr_ty:=-0.72 extr_tz:=0.98 \\
        extr_roll:=3.1416 extr_pitch:=0.0 extr_yaw:=0.0

    # Kiem tra pose output
    ros2 topic echo /vision/target_pose_in_base_link
    ros2 topic echo /vision/box_pose_in_base_link

    # Goi service truc tiep
    ros2 service call /vision/get_pixel_pose gp7_vision_pipeline/GetPixelPose "{use_detection_center: true}"
"""

from __future__ import annotations

import math
import threading
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import tf_transformations

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from gp7_vision_pipeline.msg import BoxDetection
from gp7_vision_pipeline.srv import GetPixelPose


class CameraProjectionNode(Node):
    """Project 2D detections to 3D pose in base_link via camera extrinsics."""

    def __init__(self) -> None:
        super().__init__("camera_projection_node")
        self._bridge = CvBridge()
        self._depth_lock = threading.Lock()
        self._latest_depth: Optional[Image] = None
        self._latest_depth_encoding = ""
        self._camera_info: Optional[CameraInfo] = None

        # Camera info lock not strictly needed (written/read on same node, GIL)
        self._latest_cam_info: Optional[CameraInfo] = None

        # Extrinsics: camera_optical_frame → base_link
        self.declare_parameter("extr_tx", 0.0)
        self.declare_parameter("extr_ty", -0.7)
        self.declare_parameter("extr_tz", 1.0)
        self.declare_parameter("extr_roll", math.pi)
        self.declare_parameter("extr_pitch", 0.0)
        self.declare_parameter("extr_yaw", 0.0)

        tx = self.get_parameter("extr_tx").value
        ty = self.get_parameter("extr_ty").value
        tz = self.get_parameter("extr_tz").value
        roll = self.get_parameter("extr_roll").value
        pitch = self.get_parameter("extr_pitch").value
        yaw = self.get_parameter("extr_yaw").value

        # Pre-compute rotation matrix from Euler ZYX
        q = tf_transformations.quaternion_from_euler(roll, pitch, yaw)
        self._R_extr = np.array(
            tf_transformations.quaternion_matrix(q)[:3, :3], dtype=np.float64
        )
        self._t_extr = np.array([tx, ty, tz], dtype=np.float64)

        self.get_logger().info(
            f"Camera extrinsics (camera_optical → base_link):  "
            f"t=({tx:.4f}, {ty:.4f}, {tz:.4f})  "
            f"R=({roll:.4f}, {pitch:.4f}, {yaw:.4f})"
        )

        # Fixed rotation: camera_link → camera_color_optical_frame
        # ROS optical convention: X right, Y down, Z forward
        self._R_optical_to_link = np.array([
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)

        # Depth ROI half-size
        self.declare_parameter("depth_roi_half_size", 5)
        self._roi_half = int(self.get_parameter("depth_roi_half_size").value)

        # Subscriptions
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )

        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self._on_camera_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self._on_depth,
            qos_profile_sensor_data,
        )
        self._sub_target = self.create_subscription(
            BoxDetection,
            "/vision/target_detection",
            self._on_target_detection,
            qos,
        )
        self._sub_box = self.create_subscription(
            BoxDetection,
            "/vision/box_detection",
            self._on_box_detection,
            qos,
        )

        # Publications
        self._pub_target = self.create_publisher(
            PoseStamped, "/vision/target_pose_in_base_link", qos
        )
        self._pub_box = self.create_publisher(
            PoseStamped, "/vision/box_pose_in_base_link", qos
        )

        # Service
        self._svc = self.create_service(
            GetPixelPose,
            "/vision/get_pixel_pose",
            self._on_get_pixel_pose,
        )

        self.get_logger().info(
            "Camera projection node ready.  "
            "Projects target/box detections to base_link using camera extrinsics."
        )
        self.get_logger().info(
            "Output topics: /vision/target_pose_in_base_link, /vision/box_pose_in_base_link"
        )
        self.get_logger().info(
            "Service: /vision/get_pixel_pose"
        )

    # -------------------------------------------------------------------------
    # Camera intrinsics
    # -------------------------------------------------------------------------

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._latest_cam_info = msg

    def _get_intrinsics(self) -> Optional[tuple]:
        """Return (fx, fy, cx, cy) from latest CameraInfo."""
        if self._latest_cam_info is None:
            return None
        info = self._latest_cam_info
        return float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5])

    # -------------------------------------------------------------------------
    # Depth cache
    # -------------------------------------------------------------------------

    def _on_depth(self, msg: Image) -> None:
        with self._depth_lock:
            self._latest_depth = msg
            self._latest_depth_encoding = msg.encoding

    # -------------------------------------------------------------------------
    # Detection callbacks
    # -------------------------------------------------------------------------

    def _on_target_detection(self, msg: BoxDetection) -> None:
        if msg.confidence <= 0.0:
            return
        pose, depth = self._project_to_base_link(
            msg.center_x, msg.center_y,
            msg.distance_m if msg.distance_m > 0 else None,
        )
        if pose is not None:
            pose.header.frame_id = "base_link"
            pose.header.stamp = self.get_clock().now().to_msg()
            self._pub_target.publish(pose)
            self.get_logger().debug(
                f"target in base_link: ({pose.pose.position.x:.4f}, "
                f"{pose.pose.position.y:.4f}, {pose.pose.position.z:.4f}) "
                f"depth={depth:.4f}m"
            )

    def _on_box_detection(self, msg: BoxDetection) -> None:
        if msg.confidence <= 0.0:
            return
        pose, depth = self._project_to_base_link(
            msg.center_x, msg.center_y,
            msg.distance_m if msg.distance_m > 0 else None,
        )
        if pose is not None:
            pose.header.frame_id = "base_link"
            pose.header.stamp = self.get_clock().now().to_msg()
            self._pub_box.publish(pose)
            self.get_logger().debug(
                f"box in base_link: ({pose.pose.position.x:.4f}, "
                f"{pose.pose.position.y:.4f}, {pose.pose.position.z:.4f}) "
                f"depth={depth:.4f}m"
            )

    # -------------------------------------------------------------------------
    # Service handler
    # -------------------------------------------------------------------------

    def _on_get_pixel_pose(
        self,
        request: GetPixelPose.Request,
        response: GetPixelPose.Response,
    ) -> GetPixelPose.Response:
        intrinsics = self._get_intrinsics()
        if intrinsics is None:
            response.success = False
            response.message = "camera_info chua co san"
            return response

        if request.use_detection_center:
            # Phuong an: lay tu detection moi nhat (khong co trong request,
            # nen lay tu target moi nhat)
            response.success = False
            response.message = (
                "use_detection_center=true chua ho tro "
                "(can truyen detection cu the qua service)"
            )
            return response

        # Chi can center_x/y tu BoxDetection la du
        pose, depth = self._project_to_base_link(
            0, 0, None  # placeholder; override below
        )
        response.success = False
        response.message = "Chua implement pixel request"
        return response

    # -------------------------------------------------------------------------
    # Core projection logic
    # -------------------------------------------------------------------------

    def _project_to_base_link(
        self,
        u: int,
        v: int,
        override_depth: Optional[float] = None,
    ) -> tuple[Optional[PoseStamped], float]:
        """Project pixel (u,v) to PoseStamped in base_link.

        Returns (pose, depth_used) or (None, -1.0) if failed.
        """
        intrinsics = self._get_intrinsics()
        if intrinsics is None:
            self.get_logger().warn("camera_info chua co — skip projection", throttle_duration_sec=5.0)
            return None, -1.0

        fx, fy, cx, cy = intrinsics

        # Lay depth
        if override_depth is not None and override_depth > 0:
            depth_m = float(override_depth)
        else:
            depth_m = self._get_depth_at(u, v)
            if depth_m <= 0:
                self.get_logger().debug(
                    f"Depth invalid at ({u}, {v}) — skip projection", throttle_duration_sec=5.0
                )
                return None, -1.0

        # Step 1: pixel → camera_optical_frame (pinhole, depth from aligned depth)
        x_c = (u - cx) * depth_m / fx
        y_c = (v - cy) * depth_m / fy
        z_c = depth_m
        p_cam = np.array([x_c, y_c, z_c], dtype=np.float64)

        # Step 2: camera_optical_frame → camera_link
        p_link = self._R_optical_to_link @ p_cam

        # Step 3: camera_link → base_link (extrinsics)
        p_base = self._R_extr @ p_link + self._t_extr

        # Build PoseStamped (orientation = identity)
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = float(p_base[0])
        pose.pose.position.y = float(p_base[1])
        pose.pose.position.z = float(p_base[2])
        pose.pose.orientation.w = 1.0

        return pose, depth_m

    def _get_depth_at(self, u: int, v: int) -> float:
        """Sample median depth in a small ROI around (u, v)."""
        with self._depth_lock:
            depth_msg = self._latest_depth
            depth_enc = self._latest_depth_encoding

        if depth_msg is None:
            return -1.0

        try:
            depth_cv = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError:
            return -1.0

        if depth_cv.ndim != 2:
            return -1.0

        h, w = depth_cv.shape[:2]
        x_min = max(0, u - self._roi_half)
        x_max = min(w, u + self._roi_half)
        y_min = max(0, v - self._roi_half)
        y_max = min(h, v + self._roi_half)

        if x_min >= x_max or y_min >= y_max:
            return -1.0

        roi = depth_cv[y_min:y_max, x_min:x_max].astype(np.float32)

        if depth_enc == "16UC1":
            roi = roi * 0.001

        valid = roi[roi > 0.0]
        if valid.size == 0:
            return -1.0

        return float(np.median(valid))


def main(argv=None):
    rclpy.init(args=argv)
    node = CameraProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
