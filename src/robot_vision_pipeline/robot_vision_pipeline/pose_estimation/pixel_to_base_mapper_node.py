"""Map YOLO 2-D detections to 3-D base-frame positions using homography + depth-based height.

Subscribes to:
  /vision/target_detection  → ObjectDetection
  /vision/box_detection     → ObjectDetection
  /camera/camera/color/image_raw  → sensor_msgs/Image  (raw colour, for base debug image)
  /camera/camera/color/camera_info → CameraInfo  (intrinsics for box size)

Publishes:
  /vision/target_position   → geometry_msgs/PointStamped  (x_b, y_b, z_b=0.055 m, fixed)
  /vision/box              → robot_vision_pipeline_msgs/Object (Pose + size)
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

import math
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from robot_vision_pipeline.depth_utils import robust_center_depth
from robot_vision_pipeline_msgs.msg import ObjectDetection, Object


# ─────────────────────────────────────────────────────────────────────────────
# Homography helper
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_base_xy(u: float, v: float, H: np.ndarray) -> tuple[float, float]:
    """Apply homography H to map pixel (u, v) → base (x_mm, y_mm)."""
    src = np.array([u, v, 1.0], dtype=np.float64)
    dst_h = H @ src
    return float(dst_h[0] / dst_h[2]), float(dst_h[1] / dst_h[2])


def normalize_axis_deg(angle_deg: float) -> float:
    """Normalize an object-axis angle to [-90, 90)."""
    return float((angle_deg + 90.0) % 180.0 - 90.0)


def order_points_clockwise(pts4: np.ndarray) -> np.ndarray:
    """Order four polygon points clockwise, starting near top-left."""
    pts = np.asarray(pts4, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def yaw_from_poly4_longest_edge(poly4: np.ndarray) -> float:
    """Return image yaw from the longest edge of a four-point polygon."""
    pts = order_points_clockwise(poly4)
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.size == 0 or float(np.max(lengths)) < 1e-6:
        return 0.0

    edge = edges[int(np.argmax(lengths))]
    yaw = math.degrees(math.atan2(float(edge[1]), float(edge[0])))
    return normalize_axis_deg(yaw)


def yaw_to_quaternion(yaw_deg: float) -> Tuple[float, float, float, float]:
    """Create a base-frame yaw-only quaternion."""
    half = math.radians(float(yaw_deg)) * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def image_yaw_to_base_yaw(
    u_px: float,
    v_px: float,
    yaw_img_deg: float,
    H: np.ndarray,
    step_px: float = 60.0,
) -> float:
    """Convert image-axis yaw to base-frame yaw by mapping two pixels with H."""
    yaw_rad = math.radians(float(yaw_img_deg))
    x1_mm, y1_mm = pixel_to_base_xy(float(u_px), float(v_px), H)
    x2_mm, y2_mm = pixel_to_base_xy(
        float(u_px) + float(step_px) * math.cos(yaw_rad),
        float(v_px) + float(step_px) * math.sin(yaw_rad),
        H,
    )
    dx = x2_mm - x1_mm
    dy = y2_mm - y1_mm
    if math.hypot(dx, dy) < 1e-9:
        return normalize_axis_deg(yaw_img_deg)
    return normalize_axis_deg(math.degrees(math.atan2(dy, dx)))


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
    object_id: int
    confidence: float
    x_mm: float
    y_mm: float
    z_mm: float
    width_mm: float
    length_mm: float
    height_mm: float
    yaw_deg: float
    yaw_robot_deg: float
    refined: bool
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

        # Colour image cache for post-YOLO ROI contour refinement
        self._color_lock = threading.Lock()
        self._latest_color_bgr: Optional[np.ndarray] = None

        # Image bridge
        self._bridge = CvBridge()

        # Latest detection data (protected by _data_lock)
        self._data_lock = threading.Lock()
        self._latest_target: Optional[TargetData] = None
        self._latest_boxes: Dict[int, BoxData] = {}

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

        # Object height
        self.declare_parameter("default_box_height_mm", 10.0)
        self.declare_parameter("min_valid_height_mm", 1.0)
        self.declare_parameter("max_valid_height_mm", 300.0)

        # Object size from intrinsics
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("box_width_scale", 1.0)
        self.declare_parameter("box_length_scale", 1.0)

        # Debug image
        self.declare_parameter("base_debug_image_input_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("overlay_timeout_sec", 1.0)
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")

        # Object robust depth estimation (5×5 window with outlier filtering)
        self.declare_parameter("box_center_depth_window_radius", 2)
        self.declare_parameter("box_min_valid_depth_m", 0.1)
        self.declare_parameter("box_max_valid_depth_m", 2.0)
        self.declare_parameter("box_depth_outlier_threshold_m", 0.02)
        self.declare_parameter("box_min_valid_depth_samples", 5)

        # Post-YOLO contour refinement, adapted from gui_pick_and_place.py.
        self.declare_parameter("enable_roi_refinement", True)
        self.declare_parameter("use_refined_center", True)
        self.declare_parameter("use_refined_yaw_orientation", True)
        self.declare_parameter("roi_mask_shrink", 0.92)
        self.declare_parameter("roi_morph_kernel", 5)
        self.declare_parameter("roi_min_contour_area", 300.0)
        self.declare_parameter("roi_hsv_s_max", 80)
        self.declare_parameter("roi_hsv_v_min", 35)
        self.declare_parameter("roi_hsv_v_max", 245)

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
        self._enable_roi_refinement = bool(self.get_parameter("enable_roi_refinement").value)
        self._use_refined_center = bool(self.get_parameter("use_refined_center").value)
        self._use_refined_yaw_orientation = bool(
            self.get_parameter("use_refined_yaw_orientation").value
        )
        self._roi_mask_shrink = float(self.get_parameter("roi_mask_shrink").value)
        self._roi_morph_kernel = int(self.get_parameter("roi_morph_kernel").value)
        self._roi_min_contour_area = float(self.get_parameter("roi_min_contour_area").value)
        self._roi_hsv_s_max = int(self.get_parameter("roi_hsv_s_max").value)
        self._roi_hsv_v_min = int(self.get_parameter("roi_hsv_v_min").value)
        self._roi_hsv_v_max = int(self.get_parameter("roi_hsv_v_max").value)

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
            f"min_samples={self._box_min_samples}\n"
            f"  ROI refinement: enabled={self._enable_roi_refinement}, "
            f"use_center={self._use_refined_center}, "
            f"use_yaw={self._use_refined_yaw_orientation}, "
            f"mask_shrink={self._roi_mask_shrink:.2f}, "
            f"morph_kernel={self._roi_morph_kernel}, "
            f"min_area={self._roi_min_contour_area:.1f}, "
            f"HSV S<= {self._roi_hsv_s_max}, "
            f"V=[{self._roi_hsv_v_min}, {self._roi_hsv_v_max}]"
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
            Object, "/vision/box", detection_qos)
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
            ObjectDetection,
            "/vision/target_detection",
            self._on_target_detection,
            detection_qos,
        )
        self.create_subscription(
            ObjectDetection,
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
                    pkg_share = get_package_share_directory("robot_vision_pipeline")
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

    def _on_target_detection(self, msg: ObjectDetection) -> None:
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

    # ── Object detection ───────────────────────────────────────────────────────

    def _shrink_poly(self, poly4: np.ndarray, scale: float) -> np.ndarray:
        pts = np.asarray(poly4, dtype=np.float32).reshape(4, 2)
        center = pts.mean(axis=0, keepdims=True)
        return (center + (pts - center) * float(scale)).astype(np.float32)

    def _refine_box_roi(
        self,
        msg: ObjectDetection,
    ) -> Tuple[float, float, float, bool, Optional[np.ndarray]]:
        """Refine bbox center and image yaw using post-YOLO ROI image processing.

        The method mirrors gui_pick_and_place.py: use YOLO bbox as an ROI, shrink
        its polygon slightly, segment low-saturation gray/white object pixels,
        clean with morphology, then compute center and yaw from the largest
        contour's minAreaRect.
        """
        bbox_poly = np.array(
            [
                [float(msg.x_min), float(msg.y_min)],
                [float(msg.x_max), float(msg.y_min)],
                [float(msg.x_max), float(msg.y_max)],
                [float(msg.x_min), float(msg.y_max)],
            ],
            dtype=np.float32,
        )
        bbox_poly = order_points_clockwise(bbox_poly)

        fallback_u = float(msg.center_x)
        fallback_v = float(msg.center_y)
        fallback_yaw = yaw_from_poly4_longest_edge(bbox_poly)

        if not self._enable_roi_refinement:
            return fallback_u, fallback_v, fallback_yaw, False, bbox_poly

        with self._color_lock:
            color_bgr = None if self._latest_color_bgr is None else self._latest_color_bgr.copy()

        if color_bgr is None or color_bgr.size == 0:
            return fallback_u, fallback_v, fallback_yaw, False, bbox_poly

        img_h, img_w = color_bgr.shape[:2]
        if img_h < 5 or img_w < 5:
            return fallback_u, fallback_v, fallback_yaw, False, bbox_poly

        shrunk = self._shrink_poly(bbox_poly, self._roi_mask_shrink)
        x1 = max(0, int(np.floor(np.min(shrunk[:, 0]))))
        y1 = max(0, int(np.floor(np.min(shrunk[:, 1]))))
        x2 = min(img_w - 1, int(np.ceil(np.max(shrunk[:, 0]))))
        y2 = min(img_h - 1, int(np.ceil(np.max(shrunk[:, 1]))))

        if x2 <= x1 or y2 <= y1:
            return fallback_u, fallback_v, fallback_yaw, False, bbox_poly

        roi = color_bgr[y1:y2, x1:x2].copy()
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return fallback_u, fallback_v, fallback_yaw, False, bbox_poly

        mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        local_poly = (shrunk - np.array([x1, y1], dtype=np.float32)).astype(np.int32)

        import cv2

        cv2.fillPoly(mask, [local_poly], 255)

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, sat, val = cv2.split(hsv)
        obj_mask = (
            (sat <= self._roi_hsv_s_max)
            & (val >= self._roi_hsv_v_min)
            & (val <= self._roi_hsv_v_max)
        ).astype(np.uint8) * 255
        obj_mask = cv2.bitwise_and(obj_mask, obj_mask, mask=mask)

        kernel_size = max(3, int(self._roi_morph_kernel))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        found = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        if not contours:
            return fallback_u, fallback_v, fallback_yaw, False, bbox_poly

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self._roi_min_contour_area:
            return fallback_u, fallback_v, fallback_yaw, False, bbox_poly

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            return fallback_u, fallback_v, fallback_yaw, False, bbox_poly

        refined_u = float(moments["m10"] / moments["m00"]) + float(x1)
        refined_v = float(moments["m01"] / moments["m00"]) + float(y1)

        rect = cv2.minAreaRect(contour)
        rect_poly = cv2.boxPoints(rect).astype(np.float32)
        rect_poly_global = rect_poly + np.array([x1, y1], dtype=np.float32)
        refined_yaw = yaw_from_poly4_longest_edge(rect_poly_global)

        out_u = refined_u if self._use_refined_center else fallback_u
        out_v = refined_v if self._use_refined_center else fallback_v
        out_yaw = refined_yaw if self._use_refined_yaw_orientation else fallback_yaw
        return out_u, out_v, normalize_axis_deg(out_yaw), True, rect_poly_global

    # Throttle the detailed depth debug log to one per N frames
    _BOX_LOG_EVERY = 30
    _box_frame_count = 0

    def _on_box_detection(self, msg: ObjectDetection) -> None:
        if self._homography is None:
            return

        object_id = int(getattr(msg, "object_id", 0))

        # Refine YOLO bbox with colour ROI processing from gui_pick_and_place.py.
        center_u, center_v, yaw_img_deg, roi_refined, _ = self._refine_box_roi(msg)

        # XY from homography (stable — only depends on bbox center in pixels)
        cx_mm, cy_mm = pixel_to_base_xy(center_u, center_v, self._homography)
        yaw_robot_deg = image_yaw_to_base_yaw(
            center_u, center_v, yaw_img_deg, self._homography
        )

        # ── Robust center depth ────────────────────────────────────────────────
        # Try to compute robust depth from the cached raw depth image first.
        # If depth cache is unavailable, fall back to the pre-computed values in msg.
        with self._depth_lock:
            depth_cv = self._latest_depth_msg
            depth_enc = self._latest_depth_encoding or str(msg.depth_encoding)

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
                robust_depth_m = float(roi_raw) * 0.001
            else:
                # Last resort: single-point center depth
                center_raw = int(msg.center_raw_depth)
                if center_raw > 0:
                    robust_depth_m = float(center_raw) * 0.001
                else:
                    robust_depth_m = None

        # ── Height from depth model ───────────────────────────────────────────
        depth_mm = 0.0
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
                f"Object depth unavailable (valid_sample_count={valid_sample_count}, "
                f"robust_depth={robust_depth_m}); using default height "
                f"{height_mm:.1f}mm"
            )

        center_z_mm = self._table_z_base_mm + height_mm / 2.0

        # ── Object size from camera intrinsics + robust depth ─────────────────────
        bw_px = float(msg.x_max) - float(msg.x_min)
        bh_px = float(msg.y_max) - float(msg.y_min)

        use_robust = robust_depth_m is not None and robust_depth_m > 0.0
        depth_for_size_m = robust_depth_m if use_robust else None

        camera_x_m = 0.0
        camera_y_m = 0.0
        camera_z_m = float(depth_for_size_m) if depth_for_size_m is not None else 0.0

        with self._cam_info_lock:
            if (
                self._cam_info_received
                and self._fx > 0.0
                and self._fy > 0.0
                and depth_for_size_m is not None
            ):
                camera_x_m = (center_u - self._cx) * depth_for_size_m / self._fx
                camera_y_m = (center_v - self._cy) * depth_for_size_m / self._fy
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
                f"size=({width_mm:.1f}, {length_mm:.1f}, {height_mm:.1f})mm, "
                f"yaw_img={yaw_img_deg:.1f}deg, "
                f"yaw_robot={yaw_robot_deg:.1f}deg, refined={roi_refined}"
            )

        # Store latest box data
        now = self._now_sec()
        self._last_box_detected_time = now
        with self._data_lock:
            self._latest_boxes[object_id] = BoxData(
                object_id=object_id,
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
                center_x=int(round(center_u)),
                center_y=int(round(center_v)),
                yaw_deg=yaw_img_deg,
                yaw_robot_deg=yaw_robot_deg,
                refined=roi_refined,
            )

        # Publish Object
        box_msg = Object()
        box_msg.header.stamp = self.get_clock().now().to_msg()
        box_msg.header.frame_id = "base_link"
        box_msg.class_name = msg.class_name
        box_msg.confidence = msg.confidence
        box_msg.object_id = object_id
        box_msg.center_x_px = float(center_u)
        box_msg.center_y_px = float(center_v)
        box_msg.yaw_img_deg = float(yaw_img_deg)
        box_msg.yaw_robot_deg = float(yaw_robot_deg)
        box_msg.camera_position.x = float(camera_x_m)
        box_msg.camera_position.y = float(camera_y_m)
        box_msg.camera_position.z = float(camera_z_m)
        box_msg.robot_position.x = cx_mm / 1000.0
        box_msg.robot_position.y = cy_mm / 1000.0
        box_msg.robot_position.z = center_z_mm / 1000.0
        box_msg.pose.position.x = cx_mm / 1000.0
        box_msg.pose.position.y = cy_mm / 1000.0
        box_msg.pose.position.z = center_z_mm / 1000.0
        qx, qy, qz, qw = yaw_to_quaternion(yaw_robot_deg)
        box_msg.pose.orientation.x = qx
        box_msg.pose.orientation.y = qy
        box_msg.pose.orientation.z = qz
        box_msg.pose.orientation.w = qw
        box_msg.size.x = width_mm / 1000.0
        box_msg.size.y = length_mm / 1000.0
        box_msg.size.z = height_mm / 1000.0
        box_msg.refined = bool(roi_refined)
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
        with self._data_lock:
            self._latest_boxes = {
                object_id: box
                for object_id, box in self._latest_boxes.items()
                if (now - box.stamp_sec) <= self._stale_timeout_sec
            }
            has_recent_box = bool(self._latest_boxes)

        target_stale = (
            self._last_target_detected_time is None
            or (now - self._last_target_detected_time) > self._stale_timeout_sec
        )
        box_stale = (
            not has_recent_box
            and (
                self._last_box_detected_time is None
                or (now - self._last_box_detected_time) > self._stale_timeout_sec
            )
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

        with self._color_lock:
            self._latest_color_bgr = bgr.copy()

        now = self._now_sec()

        with self._data_lock:
            target = self._latest_target
            boxes = [
                box
                for box in self._latest_boxes.values()
                if (now - box.stamp_sec) <= self._overlay_timeout_sec
            ]
            boxes.sort(key=lambda item: item.object_id)

        # Draw target bbox + crosshair if recent
        if target is not None and (now - target.stamp_sec) <= self._overlay_timeout_sec:
            bgr = self._draw_target_overlay(bgr, target)

        # Draw every recent object detection.
        for box in boxes:
            bgr = self._draw_box_overlay(bgr, box)

        # Build text overlay lines (top-left)
        lines: List[str] = []

        if target is not None and (now - target.stamp_sec) <= self._overlay_timeout_sec:
            lines.append(f"target_pose conf={target.confidence:.2f}")
            lines.append(
                f"x={target.x_mm:.1f} y={target.y_mm:.1f} z={target.z_mm:.1f} mm"
            )

        if boxes:
            if lines:
                lines.append("")
            lines.append(f"objects={len(boxes)}")
            for box in boxes[:4]:
                lines.append(
                    f"#{box.object_id} conf={box.confidence:.2f} "
                    f"x={box.x_mm:.0f} y={box.y_mm:.0f} z={box.z_mm:.0f}mm "
                    f"yaw={box.yaw_robot_deg:.1f}"
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
        cv2.putText(
            out,
            f"#{box.object_id}",
            (max(0, x_min), max(18, y_min - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

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

        arrow_len = max(28, min(w, h) // 8)
        ex = int(round(cx + arrow_len * math.cos(math.radians(box.yaw_deg))))
        ey = int(round(cy + arrow_len * math.sin(math.radians(box.yaw_deg))))
        ex = max(0, min(ex, w - 1))
        ey = max(0, min(ey, h - 1))
        cv2.arrowedLine(
            out,
            (cx, cy),
            (ex, ey),
            (255, 0, 0),
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )

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
