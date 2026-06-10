"""Map YOLO 2-D detections to 3-D base-frame positions using homography + depth-based height.

Subscribes to:
  /vision/target_detection  → BoxDetection
  /vision/box_detection     → BoxDetection
  /camera/camera/color/image_raw  → sensor_msgs/Image  (raw colour, for base debug image)
  /camera/camera/color/camera_info → CameraInfo  (intrinsics for box size)

Publishes:
  /vision/target_position   → geometry_msgs/PointStamped  (x_b, y_b, z_b=0.055 m, fixed)
  /vision/box              → gp7_vision_pipeline/Box     (Pose + size)
  /vision/debug_image_base → sensor_msgs/Image  (raw image with base-frame overlay)

Geometry model:
  Known calibration target: height=10 mm, center_z=55 mm, top_depth=556 mm
    table_z_base    = target_center_z - target_height/2       = 50 mm
    table_depth_raw = target_top_depth  + target_height       = 566 mm

  TARGET:
    x,y  = homography(center_u, center_v)
    z    = target_center_z_base_mm  (FIXED, not from depth)
    height = target_height_mm  (FIXED, not from depth)

  BOX:
    x,y = homography(bbox_center_u, bbox_center_v)
    box_top_depth_mm  = roi_median_raw_depth  (or center_raw_depth if unavailable)
    height_mm        = table_depth_raw - box_top_depth_mm
    center_z_mm      = table_z_base + height_mm / 2
    width_mm         = bbox_width_px  * box_top_depth_mm / fx   (camera intrinsics)
    length_mm        = bbox_height_px * box_top_depth_mm / fy  (camera intrinsics)
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Header, String

from gp7_vision_pipeline.depth_utils import bbox_clip_to_image, robust_center_depth
from gp7_vision_pipeline.msg import BoxDetection, Box


# ─────────────────────────────────────────────────────────────────────────────
# Homography helper
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_base_xy(u: float, v: float, H: np.ndarray) -> tuple[float, float]:
    """Apply homography H to map pixel (u, v) → base (x_mm, y_mm)."""
    src = np.array([u, v, 1.0], dtype=np.float64)
    dst_h = H @ src
    return float(dst_h[0] / dst_h[2]), float(dst_h[1] / dst_h[2])


# ─────────────────────────────────────────────────────────────────────────────
# Storage for latest detection data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TargetData:
    confidence: float
    x_mm: float
    y_mm: float
    z_mm: float
    stamp_sec: float
    # Pixel bbox for drawing
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    center_x: int
    center_y: int


@dataclass
class BoxData:
    confidence: float
    x_mm: float
    y_mm: float
    z_mm: float
    width_mm: float
    length_mm: float
    height_mm: float
    stamp_sec: float
    # Pixel bbox for drawing
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    center_x: int
    center_y: int


# ─────────────────────────────────────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────────────────────────────────────

class PixelToBaseMapperNode(Node):
    """Homography + intrinsics-based box size + fixed target height."""

    def __init__(self) -> None:
        super().__init__("pixel_to_base_mapper_node")

        self._homography: Optional[np.ndarray] = None

        # Camera intrinsics (set by camera_info callback)
        self._cam_info_lock = threading.Lock()
        self._cam_info_received = False
        self._fx: float = 0.0
        self._fy: float = 0.0
        self._cx: float = 0.0
        self._cy: float = 0.0

        # Depth cache (for robust center depth computation)
        self._depth_lock = threading.Lock()
        self._latest_depth_msg: Optional[Image] = None
        self._latest_depth_encoding: str = ""

        # Image bridge
        self._bridge = CvBridge()

        # Latest detection data (protected by _data_lock)
        self._data_lock = threading.Lock()
        self._latest_target: Optional[TargetData] = None
        self._latest_box: Optional[BoxData] = None

        # Tracking for explicit detection status (staleness-based)
        self._last_target_detected_time: Optional[float] = None
        self._last_box_detected_time: Optional[float] = None
        self._stale_timeout_sec = 2.0  # seconds before publishing False for a detection
        self._publish_timer = self.create_timer(
            0.5, self._on_publish_stale_status
        )

        # ── Parameters ─────────────────────────────────────────────────────
        self.declare_parameter("homography_yaml", "config/pixel_to_base_homography.yaml")

        # Target (fixed geometry)
        self.declare_parameter("target_height_mm", 10.0)
        self.declare_parameter("target_center_z_base_mm", 55.0)
        self.declare_parameter("target_top_depth_raw_mm", 556.0)

        # Table model (derived from target calibration)
        self.declare_parameter("table_z_base_mm", 50.0)
        self.declare_parameter("table_depth_raw_mm", 566.0)

        # Box height
        self.declare_parameter("default_box_height_mm", 10.0)
        self.declare_parameter("min_valid_height_mm", 1.0)
        self.declare_parameter("max_valid_height_mm", 300.0)

        # Box size from intrinsics
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("box_width_scale", 1.0)
        self.declare_parameter("box_length_scale", 1.0)

        # Debug image
        self.declare_parameter("base_debug_image_input_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("overlay_timeout_sec", 1.0)
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")

        # Box robust depth estimation (5×5 window with outlier filtering)
        self.declare_parameter("box_center_depth_window_radius", 2)
        self.declare_parameter("box_min_valid_depth_m", 0.1)
        self.declare_parameter("box_max_valid_depth_m", 2.0)
        self.declare_parameter("box_depth_outlier_threshold_m", 0.02)
        self.declare_parameter("box_min_valid_depth_samples", 5)

        # Read parameters
        homography_yaml = self.get_parameter("homography_yaml").get_parameter_value().string_value
        self._target_height_mm = self.get_parameter("target_height_mm").get_parameter_value().double_value
        self._target_center_z_base_mm = self.get_parameter("target_center_z_base_mm").get_parameter_value().double_value
        self._target_top_depth_raw_mm = self.get_parameter("target_top_depth_raw_mm").get_parameter_value().double_value
        self._table_z_base_mm = self.get_parameter("table_z_base_mm").get_parameter_value().double_value
        self._table_depth_raw_mm = self.get_parameter("table_depth_raw_mm").get_parameter_value().double_value
        self._default_box_height_mm = self.get_parameter("default_box_height_mm").get_parameter_value().double_value
        self._min_height_mm = self.get_parameter("min_valid_height_mm").get_parameter_value().double_value
        self._max_height_mm = self.get_parameter("max_valid_height_mm").get_parameter_value().double_value
        self._camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self._box_width_scale = self.get_parameter("box_width_scale").get_parameter_value().double_value
        self._box_length_scale = self.get_parameter("box_length_scale").get_parameter_value().double_value
        self._base_debug_input_topic = self.get_parameter("base_debug_image_input_topic").get_parameter_value().string_value
        self._overlay_timeout_sec = self.get_parameter("overlay_timeout_sec").get_parameter_value().double_value
        self._depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self._box_depth_radius = int(self.get_parameter("box_center_depth_window_radius").value)
        self._box_min_depth_m = float(self.get_parameter("box_min_valid_depth_m").value)
        self._box_max_depth_m = float(self.get_parameter("box_max_valid_depth_m").value)
        self._box_depth_outlier_m = float(self.get_parameter("box_depth_outlier_threshold_m").value)
        self._box_min_samples = int(self.get_parameter("box_min_valid_depth_samples").value)

        # ── Load homography ─────────────────────────────────────────────────
        self._load_homography(homography_yaml)

        self.get_logger().info(
            f"Calibration model:\n"
            f"  target_height={self._target_height_mm} mm (FIXED)\n"
            f"  target_center_z={self._target_center_z_base_mm} mm (FIXED)\n"
            f"  table_z_base={self._table_z_base_mm} mm\n"
            f"  table_depth_raw={self._table_depth_raw_mm} mm\n"
            f"  default_box_height={self._default_box_height_mm} mm\n"
            f"  box_width_scale={self._box_width_scale}\n"
            f"  box_length_scale={self._box_length_scale}\n"
            f"  camera_info_topic={self._camera_info_topic}\n"
            f"  base_debug_image_input_topic={self._base_debug_input_topic}\n"
            f"  overlay_timeout_sec={self._overlay_timeout_sec} s\n"
            f"  robust depth: radius={self._box_depth_radius} "
            f"[{2*self._box_depth_radius+1}x{2*self._box_depth_radius+1} window], "
            f"depth=[{self._box_min_depth_m:.2f}, {self._box_max_depth_m:.2f}] m, "
            f"outlier_thresh={self._box_depth_outlier_m*1000:.0f} mm, "
            f"min_samples={self._box_min_samples}"
        )

        # ── QoS profiles ───────────────────────────────────────────────────
        detection_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Publishers ──────────────────────────────────────────────────────
        self._pub_target_pos = self.create_publisher(
            PointStamped, "/vision/target_position", detection_qos)
        self._pub_box = self.create_publisher(
            Box, "/vision/box", detection_qos)
        # Explicit detection status publishers
        self._pub_target_detected = self.create_publisher(
            Bool, "/vision/target_detected", detection_qos)
        self._pub_box_detected = self.create_publisher(
            Bool, "/vision/box_detected", detection_qos)
        self._pub_detection_status = self.create_publisher(
            String, "/vision/detection_status", detection_qos)

        debug_base_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub_debug_base = self.create_publisher(
            Image, "/vision/debug_image_base", debug_base_qos)

        # ── Subscriptions ───────────────────────────────────────────────────
        self.create_subscription(
            BoxDetection,
            "/vision/target_detection",
            self._on_target_detection,
            detection_qos,
        )
        self.create_subscription(
            BoxDetection,
            "/vision/box_detection",
            self._on_box_detection,
            detection_qos,
        )
        self.create_subscription(
            CameraInfo,
            self._camera_info_topic,
            self._on_camera_info,
            image_qos,
        )
        self.create_subscription(
            Image,
            self._base_debug_input_topic,
            self._on_raw_image,
            image_qos,
        )
        self.create_subscription(
            Image,
            self._depth_topic,
            self._on_depth_image,
            image_qos,
        )

        self.get_logger().info(
            f"pixel_to_base_mapper_node started.\n"
            f"  base_debug_image_input_topic={self._base_debug_input_topic}\n"
            f"  debug_image_base_topic=/vision/debug_image_base\n"
            f"  Publishing /vision/debug_image_base with RELIABLE QoS for RViz compatibility."
        )

    def _load_homography(self, yaml_path: str) -> None:
        resolved = None
        attempted: List[str] = []

        if os.path.isabs(yaml_path):
            if os.path.isfile(yaml_path):
                resolved = yaml_path
            else:
                attempted.append(yaml_path)
        else:
            cwd_path = os.path.join(os.getcwd(), yaml_path)
            if os.path.isfile(cwd_path):
                resolved = cwd_path
            else:
                attempted.append(cwd_path)

            if resolved is None:
                try:
                    pkg_share = get_package_share_directory("gp7_vision_pipeline")
                    pkg_share_path = os.path.join(pkg_share, yaml_path)
                    if os.path.isfile(pkg_share_path):
                        resolved = pkg_share_path
                    else:
                        attempted.append(pkg_share_path)
                except Exception:  # noqa: BLE001
                    attempted.append(f"<package_share>/{yaml_path}")

        if resolved is None:
            self.get_logger().error(
                f"Homography YAML not found. Tried:\n  " + "\n  ".join(attempted)
            )
            return

        with open(resolved, encoding="utf-8") as f:
            doc = yaml.safe_load(f)

        data = doc["homography"]["data"]
        if len(data) != 9:
            self.get_logger().error(f"homography.data must have 9 entries; got {len(data)}")
            return

        self._homography = np.array(data, dtype=np.float64).reshape(3, 3)
        self.get_logger().info(f"Loaded homography from {resolved}")

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    # ── Camera info ────────────────────────────────────────────────────────

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._cam_info_lock:
            if not self._cam_info_received:
                self._cam_info_received = True
            self._fx = float(msg.k[0])
            self._fy = float(msg.k[4])
            self._cx = float(msg.k[2])
            self._cy = float(msg.k[5])
        self.get_logger().info(
            f"Camera intrinsics: fx={self._fx:.2f} fy={self._fy:.2f} "
            f"cx={self._cx:.2f} cy={self._cy:.2f}"
        )

    # ── Target detection ────────────────────────────────────────────────────

    def _on_target_detection(self, msg: BoxDetection) -> None:
        if self._homography is None:
            return

        # XY from homography
        x_mm, y_mm = pixel_to_base_xy(
            float(msg.center_x), float(msg.center_y), self._homography
        )

        # Target z is FIXED from calibration
        z_center_mm = self._target_center_z_base_mm

        # Log depth for debug only (target z is not recomputed from depth)
        roi_depth_raw = int(msg.roi_median_raw_depth)
        self.get_logger().info(
            f"target u={msg.center_x} v={msg.center_y} "
            f"top_depth_raw={roi_depth_raw}mm "
            f"z={z_center_mm:.1f}mm [FIXED] "
            f"x={x_mm:.1f}mm y={y_mm:.1f}mm"
        )

        # Store latest target data
        now = self._now_sec()
        self._last_target_detected_time = now
        with self._data_lock:
            self._latest_target = TargetData(
                confidence=float(msg.confidence),
                x_mm=x_mm,
                y_mm=y_mm,
                z_mm=z_center_mm,
                stamp_sec=now,
                x_min=int(msg.x_min),
                y_min=int(msg.y_min),
                x_max=int(msg.x_max),
                y_max=int(msg.y_max),
                center_x=int(msg.center_x),
                center_y=int(msg.center_y),
            )

        # Publish PointStamped
        pt = PointStamped()
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.header.frame_id = "base_link"
        pt.point.x = x_mm / 1000.0
        pt.point.y = y_mm / 1000.0
        pt.point.z = z_center_mm / 1000.0
        self._pub_target_pos.publish(pt)

        # Publish explicit True detection status
        self._pub_target_detected.publish(Bool(data=True))

    # ── Box detection ───────────────────────────────────────────────────────

    # Throttle the detailed depth debug log to one per N frames
    _BOX_LOG_EVERY = 30
    _box_frame_count = 0

    def _on_box_detection(self, msg: BoxDetection) -> None:
        if self._homography is None:
            return

        # Centre pixel coords
        center_u = float(msg.center_x)
        center_v = float(msg.center_y)

        # XY from homography (stable — only depends on bbox center in pixels)
        cx_mm, cy_mm = pixel_to_base_xy(center_u, center_v, self._homography)

        # ── Robust center depth ────────────────────────────────────────────────
        # Try to compute robust depth from the cached raw depth image first.
        # If depth cache is unavailable, fall back to the pre-computed values in msg.
        with self._depth_lock:
            depth_cv = self._latest_depth_msg
            depth_enc = self._latest_depth_encoding

        robust_depth_m: Optional[float] = None
        raw_center_depth_m: Optional[float] = None
        median_depth_m: Optional[float] = None
        valid_sample_count = 0

        if depth_cv is not None and depth_enc:
            try:
                depth_arr = self._bridge.imgmsg_to_cv2(
                    depth_cv, desired_encoding="passthrough"
                )
            except CvBridgeError:
                depth_arr = None

            if depth_arr is not None:
                (
                    robust_depth_m,
                    valid_sample_count,
                    raw_center_depth_m,
                    median_depth_m,
                    _,
                ) = robust_center_depth(
                    depth_arr,
                    int(center_u),
                    int(center_v),
                    self._box_depth_radius,
                    depth_enc,
                    min_depth_m=self._box_min_depth_m,
                    max_depth_m=self._box_max_depth_m,
                    outlier_threshold_m=self._box_depth_outlier_m,
                    min_valid_samples=self._box_min_samples,
                )
        else:
            self.get_logger().warn(
                "Depth cache not available for robust depth; using msg fallback. "
                "Ensure aligned depth topic is being published."
            )

        # ── Fall back to msg-derived depth if robust failed ────────────────────
        if robust_depth_m is None:
            # Try the pre-computed ROI median from the YOLO node
            roi_raw = int(msg.roi_median_raw_depth)
            if roi_raw > 0:
                if depth_enc in ("32FC1",):
                    robust_depth_m = float(roi_raw)
                else:
                    robust_depth_m = float(roi_raw) * 0.001
            else:
                # Last resort: single-point center depth
                center_raw = int(msg.center_raw_depth)
                if center_raw > 0:
                    if depth_enc in ("32FC1",):
                        robust_depth_m = float(center_raw) * 0.001
                    else:
                        robust_depth_m = float(center_raw) * 0.001
                else:
                    robust_depth_m = None

        # ── Height from depth model ───────────────────────────────────────────
        if robust_depth_m is not None and robust_depth_m > 0.0:
            # Convert robust depth (metres) to mm for the table model
            depth_mm = robust_depth_m * 1000.0
            height_mm = self._table_depth_raw_mm - depth_mm
            if not (self._min_height_mm <= height_mm <= self._max_height_mm):
                self.get_logger().warn(
                    f"box height {height_mm:.1f}mm outside valid range "
                    f"[{self._min_height_mm}, {self._max_height_mm}mm]; using default "
                    f"{self._default_box_height_mm}mm"
                )
                height_mm = self._default_box_height_mm
        else:
            height_mm = self._default_box_height_mm
            self.get_logger().warn(
                f"Box depth unavailable (valid_sample_count={valid_sample_count}, "
                f"robust_depth={robust_depth_m}); using default height "
                f"{height_mm:.1f}mm"
            )

        center_z_mm = self._table_z_base_mm + height_mm / 2.0

        # ── Box size from camera intrinsics + robust depth ─────────────────────
        bw_px = float(msg.x_max) - float(msg.x_min)
        bh_px = float(msg.y_max) - float(msg.y_min)

        use_robust = robust_depth_m is not None and robust_depth_m > 0.0
        depth_for_size_m = robust_depth_m if use_robust else None

        with self._cam_info_lock:
            if (
                self._cam_info_received
                and self._fx > 0.0
                and self._fy > 0.0
                and depth_for_size_m is not None
            ):
                width_mm = bw_px * depth_for_size_m * 1000.0 / self._fx
                length_mm = bh_px * depth_for_size_m * 1000.0 / self._fy
                width_mm *= self._box_width_scale
                length_mm *= self._box_length_scale
            else:
                width_mm = self._default_box_height_mm
                length_mm = self._default_box_height_mm
                if not self._cam_info_received:
                    if not hasattr(self, "_warned_no_cam_info"):
                        self._warned_no_cam_info = True
                        self.get_logger().warn(
                            "camera_info not received yet, cannot compute "
                            "intrinsics-based box size; using default size"
                        )

        # ── Throttled debug log ───────────────────────────────────────────────
        __class__._box_frame_count += 1
        if __class__._box_frame_count % __class__._BOX_LOG_EVERY == 0:
            raw_str = f"{raw_center_depth_m:.4f}" if raw_center_depth_m else "N/A"
            med_str = f"{median_depth_m:.4f}" if median_depth_m else "N/A"
            rob_str = f"{robust_depth_m:.4f}" if robust_depth_m else "N/A"
            self.get_logger().info(
                f"[box depth frame {__class__._box_frame_count}] "
                f"pixel_center=({center_u:.0f}, {center_v:.0f}), "
                f"valid_samples={valid_sample_count}, "
                f"raw_center={raw_str}m, median={med_str}m, "
                f"robust={rob_str}m, "
                f"depth_used={depth_mm:.1f}mm, "
                f"height={height_mm:.1f}mm, center_z={center_z_mm:.1f}mm, "
                f"xy=({cx_mm:.1f}, {cy_mm:.1f})mm, "
                f"size=({width_mm:.1f}, {length_mm:.1f}, {height_mm:.1f})mm"
            )

        # Store latest box data
        now = self._now_sec()
        self._last_box_detected_time = now
        with self._data_lock:
            self._latest_box = BoxData(
                confidence=float(msg.confidence),
                x_mm=cx_mm,
                y_mm=cy_mm,
                z_mm=center_z_mm,
                width_mm=width_mm,
                length_mm=length_mm,
                height_mm=height_mm,
                stamp_sec=now,
                x_min=int(msg.x_min),
                y_min=int(msg.y_min),
                x_max=int(msg.x_max),
                y_max=int(msg.y_max),
                center_x=int(msg.center_x),
                center_y=int(msg.center_y),
            )

        # Publish Box
        box_msg = Box()
        box_msg.header.stamp = self.get_clock().now().to_msg()
        box_msg.header.frame_id = "base_link"
        box_msg.class_name = msg.class_name
        box_msg.confidence = msg.confidence
        box_msg.pose.position.x = cx_mm / 1000.0
        box_msg.pose.position.y = cy_mm / 1000.0
        box_msg.pose.position.z = center_z_mm / 1000.0
        box_msg.pose.orientation.w = 1.0
        box_msg.size.x = width_mm / 1000.0
        box_msg.size.y = length_mm / 1000.0
        box_msg.size.z = height_mm / 1000.0
        self._pub_box.publish(box_msg)

        # Publish explicit True detection status
        self._pub_box_detected.publish(Bool(data=True))

    # ── Stale-status publisher (timer-driven) ─────────────────────────────────

    def _on_publish_stale_status(self) -> None:
        """Publish False for any detection that has gone stale.

        Runs on a 0.5 s timer so that stale detections are explicitly cleared
        even when no new detection message arrives.
        """
        now = self._now_sec()
        target_stale = (
            self._last_target_detected_time is None
            or (now - self._last_target_detected_time) > self._stale_timeout_sec
        )
        box_stale = (
            self._last_box_detected_time is None
            or (now - self._last_box_detected_time) > self._stale_timeout_sec
        )

        if target_stale:
            self._pub_target_detected.publish(Bool(data=False))
        if box_stale:
            self._pub_box_detected.publish(Bool(data=False))

        # Combined String status
        target_state = "False" if target_stale else "True"
        box_state = "False" if box_stale else "True"
        status_msg = String()
        status_msg.data = (
            f"target_detected={target_state}, box_detected={box_state}"
        )
        self._pub_detection_status.publish(status_msg)

    # ── Depth cache ───────────────────────────────────────────────────────────

    def _on_depth_image(self, msg: Image) -> None:
        """Cache the latest aligned depth image for robust center depth computation."""
        with self._depth_lock:
            self._latest_depth_msg = msg
            self._latest_depth_encoding = msg.encoding

    # ── Debug image overlay ────────────────────────────────────────────────

    def _on_raw_image(self, msg: Image) -> None:
        """On each raw camera frame, draw base-frame overlay and publish."""
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().warn(f"cv_bridge error on raw image: {e}")
            return

        now = self._now_sec()

        with self._data_lock:
            target = self._latest_target
            box = self._latest_box

        # Draw target bbox + crosshair if recent
        if target is not None and (now - target.stamp_sec) <= self._overlay_timeout_sec:
            bgr = self._draw_target_overlay(bgr, target)

        # Draw box bbox + crosshair if recent
        if box is not None and (now - box.stamp_sec) <= self._overlay_timeout_sec:
            bgr = self._draw_box_overlay(bgr, box)

        # Build text overlay lines (top-left)
        lines: List[str] = []

        if target is not None and (now - target.stamp_sec) <= self._overlay_timeout_sec:
            lines.append(f"target_pose conf={target.confidence:.2f}")
            lines.append(
                f"x={target.x_mm:.1f} y={target.y_mm:.1f} z={target.z_mm:.1f} mm"
            )

        if box is not None and (now - box.stamp_sec) <= self._overlay_timeout_sec:
            if lines:
                lines.append("")
            lines.append(f"box_center conf={box.confidence:.2f}")
            lines.append(
                f"x={box.x_mm:.1f} y={box.y_mm:.1f} z={box.z_mm:.1f} mm"
            )
            lines.append(
                f"size: w={box.width_mm:.1f} l={box.length_mm:.1f} h={box.height_mm:.1f} mm"
            )

        if lines:
            bgr = self._overlay_text_block(bgr, lines, origin_x=8, origin_y=8)

        try:
            out_msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            out_msg.header = msg.header
            self._pub_debug_base.publish(out_msg)
        except CvBridgeError as e:
            self.get_logger().warn(f"Debug image base publish error: {e}")

    def _draw_target_overlay(
        self, bgr: np.ndarray, target: TargetData
    ) -> np.ndarray:
        """Draw blue bbox, crosshairs, and centre marker for target."""
        import cv2
        out = bgr.copy()
        h, w = out.shape[:2]

        x_min = max(0, min(target.x_min, w - 1))
        y_min = max(0, min(target.y_min, h - 1))
        x_max = max(0, min(target.x_max, w - 1))
        y_max = max(0, min(target.y_max, h - 1))
        cx = max(0, min(target.center_x, w - 1))
        cy = max(0, min(target.center_y, h - 1))

        color = (255, 128, 0)  # blue-orange in BGR

        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), color, 2, cv2.LINE_AA)
        cv2.line(out, (0, cy), (w - 1, cy), color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, 0), (cx, h - 1), color, 1, cv2.LINE_AA)

        r = max(4, min(w, h) // 120)
        r = min(r, 14)
        cv2.circle(out, (cx, cy), r, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), r, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def _draw_box_overlay(
        self, bgr: np.ndarray, box: BoxData
    ) -> np.ndarray:
        """Draw yellow bbox, crosshairs, and centre marker for box."""
        import cv2
        out = bgr.copy()
        h, w = out.shape[:2]

        x_min = max(0, min(box.x_min, w - 1))
        y_min = max(0, min(box.y_min, h - 1))
        x_max = max(0, min(box.x_max, w - 1))
        y_max = max(0, min(box.y_max, h - 1))
        cx = max(0, min(box.center_x, w - 1))
        cy = max(0, min(box.center_y, h - 1))

        color = (0, 255, 255)  # yellow in BGR

        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), color, 2, cv2.LINE_AA)
        cv2.line(out, (0, cy), (w - 1, cy), color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, 0), (cx, h - 1), color, 1, cv2.LINE_AA)

        r = max(4, min(w, h) // 120)
        r = min(r, 14)
        cv2.circle(out, (cx, cy), r, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), r, (255, 255, 255), 1, cv2.LINE_AA)

        return out

    def _overlay_text_block(
        self,
        img: np.ndarray,
        lines: List[str],
        origin_x: int,
        origin_y: int,
        font_scale: float = 0.6,
        thickness: int = 1,
        line_gap: int = 3,
        padding: int = 6,
        bg_color: tuple = (0, 0, 0),
    ) -> np.ndarray:
        """Draw a stacked text block on img.

        Per-line colours:
          target_pose  → blue (255, 128, 0) in BGR
          box_center   → yellow (0, 255, 255) in BGR
          x/y/z lines → light cyan (128, 255, 255) in BGR
          size lines  → light green (0, 255, 128) in BGR
        """
        import cv2
        out = img.copy()
        ih, iw = out.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX

        def _line_color(line: str) -> tuple:
            if "target_pose" in line:
                return (255, 128, 0)
            if "box_center" in line:
                return (0, 255, 255)
            if line.startswith("size"):
                return (0, 255, 128)
            if line == "":
                return (0, 0, 0)  # invisible separator
            return (128, 255, 255)  # x= y= z= line

        cursor_y = origin_y
        for line in lines:
            if line == "":
                cursor_y += 8
                continue
            (tw, th), bl = cv2.getTextSize(line, font, font_scale, thickness)
            block_h = th + bl + line_gap + 2 * padding
            x0 = max(0, min(origin_x, iw - tw - 2 * padding))
            y0 = cursor_y
            y1 = min(ih - 1, y0 + block_h)
            x1 = min(iw - 1, x0 + tw + 2 * padding)

            cv2.rectangle(out, (x0, y0), (x1, y1), bg_color, -1)
            cv2.rectangle(out, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)
            baseline_y = min(y0 + padding + th, ih - 1)
            cv2.putText(
                out, line,
                (x0 + padding, baseline_y),
                font, font_scale,
                _line_color(line),
                thickness,
                cv2.LINE_AA,
            )
            cursor_y = baseline_y + bl + line_gap

        return out


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = PixelToBaseMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
