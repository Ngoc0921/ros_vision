"""Map wood detections from image pixels + depth + camera intrinsics to camera-frame poses.

Current stage:
  pixel (u, v) + depth (Z) + camera intrinsics (fx, fy, cx, cy)
  → X_C, Y_C, Z_C in camera frame.

Camera → base_link (B_T_C):
  NOT implemented yet because camera-to-base extrinsic calibration is not available.
  The camera → base_link transform will be added later when the extrinsic matrix or TF is available.
  All published poses have header.frame_id = camera_color_optical_frame.
  DO NOT set frame_id = "base_link" until the actual transform is applied.

Inputs:
  /vision/wood_detection                    robot_vision_pipeline_msgs/ObjectDetection
  /camera/camera/color/image_raw            sensor_msgs/Image
  /camera/camera/aligned_depth_to_color/image_raw  sensor_msgs/Image
  /camera/camera/color/camera_info         sensor_msgs/CameraInfo

Outputs:
  /vision/wood_objects                     robot_vision_pipeline_msgs/ObjectArray  (frame_id: camera_color_optical_frame)
  /vision/objects                          robot_vision_pipeline_msgs/ObjectArray  (compatibility alias)
  /vision/detection_status                 std_msgs/String
  /vision/debug_image_camera               sensor_msgs/Image
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String

from robot_vision_pipeline.depth_utils import robust_center_depth
from robot_vision_pipeline_msgs.msg import Object, ObjectArray, ObjectDetection


def pixel_to_camera_xyz(
    u: float,
    v: float,
    z_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[float, float, float]:
    """Convert pixel (u, v) + depth Z to 3D point in camera frame.

    Formula:
        X_C = (u - cx) * Z / fx
        Y_C = (v - cy) * Z / fy
        Z_C = Z

    Args:
        u:     Pixel column (x).
        v:     Pixel row (y).
        z_m:   Depth in metres.
        fx:    Focal length x (pixels).
        fy:    Focal length y (pixels).
        cx:    Principal point x (pixels).
        cy:    Principal point y (pixels).

    Returns:
        (X_C, Y_C, Z_C) in metres.

    Raises:
        ValueError: If fx or fy is zero/negative.
    """
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid camera intrinsics: fx={fx}, fy={fy}")
    x_c = (float(u) - float(cx)) * float(z_m) / float(fx)
    y_c = (float(v) - float(cy)) * float(z_m) / float(fy)
    z_c = float(z_m)
    return float(x_c), float(y_c), float(z_c)


def normalize_axis_deg(angle_deg: float) -> float:
    """Normalize an object-axis angle to [-90, 90)."""
    return float((angle_deg + 90.0) % 180.0 - 90.0)


def order_points_clockwise(pts4: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts4, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def yaw_from_poly4_longest_edge(poly4: np.ndarray) -> float:
    pts = order_points_clockwise(poly4)
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    if lengths.size == 0 or float(np.max(lengths)) < 1e-6:
        return 0.0
    edge = edges[int(np.argmax(lengths))]
    return normalize_axis_deg(math.degrees(math.atan2(float(edge[1]), float(edge[0]))))


def yaw_to_quaternion(yaw_deg: float) -> Tuple[float, float, float, float]:
    half = math.radians(float(yaw_deg)) * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


@dataclass
class WoodData:
    object_id: int
    class_name: str
    confidence: float
    x_c_m: float
    y_c_m: float
    z_c_m: float
    yaw_img_deg: float
    stamp_sec: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    center_x: int
    center_y: int


class PixelToBaseMapperNode(Node):
    """Map wood detections from pixel+depth+camera-intrinsics to camera-frame poses.

    Step 1 — Camera coordinate:
        X_C = (u - cx) * Z / fx
        Y_C = (v - cy) * Z / fy
        Z_C = Z

    Step 2 — Camera → base_link (B_T_C):
        NOT implemented yet. Camera-to-base extrinsic calibration is not available.
        Pose is published in the camera optical frame.
        When extrinsic calibration is available, this node will be updated to apply
        tf2 or the B_T_C matrix to transform coordinates to base_link.

    Inputs:
        /vision/wood_detection                    robot_vision_pipeline_msgs/ObjectDetection
        /camera/camera/color/image_raw            sensor_msgs/Image
        /camera/camera/aligned_depth_to_color/image_raw  sensor_msgs/Image
        /camera/camera/color/camera_info          sensor_msgs/CameraInfo

    Outputs:
        /vision/wood_objects                  robot_vision_pipeline_msgs/ObjectArray
                                                   (header.frame_id = camera_color_optical_frame)
        /vision/objects                       robot_vision_pipeline_msgs/ObjectArray  (compatibility alias)
        /vision/debug_image_camera            sensor_msgs/Image
        /vision/detection_status              std_msgs/String
    """

    def __init__(self) -> None:
        super().__init__("pixel_to_base_mapper_node")

        self._bridge = CvBridge()

        self._color_lock = threading.Lock()
        self._latest_color_bgr: Optional[np.ndarray] = None

        self._depth_lock = threading.Lock()
        self._latest_depth_msg: Optional[Image] = None
        self._latest_depth_encoding = ""

        self._camera_info_lock = threading.Lock()
        self._camera_intrinsics: Optional[Tuple[float, float, float, float]] = None

        self._data_lock = threading.Lock()
        self._latest_woods: Dict[int, WoodData] = {}
        self._last_wood_detected_time: Optional[float] = None

        # Camera intrinsics subscription
        self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        )
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("debug_image_input_topic", "/camera/camera/color/image_raw")

        # Output frame ID: all published poses are in this frame.
        # Currently this is the camera optical frame because camera-to-base
        # extrinsic calibration is not available yet.
        self.declare_parameter(
            "output_frame_id", "camera_color_optical_frame"
        )

        # Depth parameters
        self.declare_parameter("wood_depth_window_radius", 2)
        self.declare_parameter("wood_min_depth_m", 0.1)
        self.declare_parameter("wood_max_depth_m", 2.0)
        self.declare_parameter("wood_depth_outlier_threshold_m", 0.02)
        self.declare_parameter("wood_min_depth_samples", 5)

        # Post-YOLO ROI refinement (HSV segmentation on color image)
        self.declare_parameter("enable_roi_refinement", True)
        self.declare_parameter("use_refined_center", True)
        self.declare_parameter("use_refined_yaw_orientation", True)
        self.declare_parameter("roi_mask_shrink", 0.92)
        self.declare_parameter("roi_morph_kernel", 5)
        self.declare_parameter("roi_min_contour_area", 300.0)
        self.declare_parameter("roi_hsv_s_max", 80)
        self.declare_parameter("roi_hsv_v_min", 35)
        self.declare_parameter("roi_hsv_v_max", 245)

        self.declare_parameter("overlay_timeout_sec", 1.0)
        self.declare_parameter("stale_timeout_sec", 2.0)

        self._camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self._depth_topic = str(self.get_parameter("depth_topic").value)
        self._debug_image_input_topic = str(self.get_parameter("debug_image_input_topic").value)
        self._output_frame_id = str(self.get_parameter("output_frame_id").value)

        self._wood_depth_radius = int(self.get_parameter("wood_depth_window_radius").value)
        self._wood_min_depth_m = float(self.get_parameter("wood_min_depth_m").value)
        self._wood_max_depth_m = float(self.get_parameter("wood_max_depth_m").value)
        self._wood_depth_outlier_m = float(
            self.get_parameter("wood_depth_outlier_threshold_m").value
        )
        self._wood_min_depth_samples = int(
            self.get_parameter("wood_min_depth_samples").value
        )

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

        self._overlay_timeout_sec = float(self.get_parameter("overlay_timeout_sec").value)
        self._stale_timeout_sec = float(self.get_parameter("stale_timeout_sec").value)

        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        detection_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Publishers
        self._pub_wood_objects = self.create_publisher(
            ObjectArray, "/vision/wood_objects", detection_qos
        )
        self._pub_objects = self.create_publisher(
            ObjectArray, "/vision/objects", detection_qos
        )
        self._pub_detection_status = self.create_publisher(
            String, "/vision/detection_status", detection_qos
        )
        self._pub_debug_camera = self.create_publisher(
            Image,
            "/vision/debug_image_camera",
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

        # Subscriptions
        self.create_subscription(
            ObjectDetection,
            "/vision/wood_detection",
            self._on_wood_detection,
            detection_qos,
        )
        self.create_subscription(
            Image, self._debug_image_input_topic, self._on_raw_image, image_qos
        )
        self.create_subscription(
            Image, self._depth_topic, self._on_depth_image, image_qos
        )
        self.create_subscription(
            CameraInfo,
            self._camera_info_topic,
            self._on_camera_info,
            QoSProfile(depth=1),
        )

        self._publish_timer = self.create_timer(0.5, self._on_publish_stale_status)

        self.get_logger().info(
            "pixel_to_base_mapper_node started (camera-frame mode) | "
            f"camera_info={self._camera_info_topic}, "
            f"depth={self._depth_topic}, "
            f"wood_in=/vision/wood_detection, "
            f"wood_objects_out=/vision/wood_objects, "
            f"objects_compat_out=/vision/objects, "
            f"output_frame={self._output_frame_id}"
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        """Cache camera intrinsics from CameraInfo message.

        K matrix layout (sensor_msgs/CameraInfo):
            [fx  0  cx]
            [ 0 fy  cy]
            [ 0  0   1]

        k[0]=fx, k[1]=0,  k[2]=cx
        k[3]=0,  k[4]=fy, k[5]=cy
        k[6]=0,  k[7]=0,  k[8]=1
        """
        with self._camera_info_lock:
            if self._camera_intrinsics is None:
                k = msg.k
                fx = float(k[0])
                fy = float(k[4])
                cx = float(k[2])
                cy = float(k[5])
                self._camera_intrinsics = (fx, fy, cx, cy)
                self.get_logger().info(
                    f"Camera intrinsics loaded: fx={fx:.3f}, fy={fy:.3f}, "
                    f"cx={cx:.3f}, cy={cy:.3f}"
                )

    def _get_intrinsics(self) -> Optional[Tuple[float, float, float, float]]:
        with self._camera_info_lock:
            return self._camera_intrinsics

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _shrink_poly(self, poly4: np.ndarray, scale: float) -> np.ndarray:
        pts = np.asarray(poly4, dtype=np.float32).reshape(4, 2)
        center = pts.mean(axis=0, keepdims=True)
        return (center + (pts - center) * float(scale)).astype(np.float32)

    def _refine_wood_roi(
        self,
        msg: ObjectDetection,
    ) -> Tuple[float, float, float, bool, Optional[np.ndarray]]:
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

        import cv2

        img_h, img_w = color_bgr.shape[:2]
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
        if float(cv2.contourArea(contour)) < self._roi_min_contour_area:
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

    def _on_depth_image(self, msg: Image) -> None:
        with self._depth_lock:
            self._latest_depth_msg = msg
            self._latest_depth_encoding = msg.encoding

    def _wood_depth_m(self, center_u: float, center_v: float) -> Optional[float]:
        """Read robust depth at wood center using aligned depth image."""
        with self._depth_lock:
            depth_msg = self._latest_depth_msg
            depth_encoding = self._latest_depth_encoding

        if depth_msg is None:
            return None

        try:
            depth_arr = self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except CvBridgeError as exc:
            self.get_logger().warn(f"Depth cv_bridge error: {exc}")
            return None

        if depth_arr is None or not isinstance(depth_arr, np.ndarray) or depth_arr.ndim != 2:
            return None

        depth_m, valid_count, _, _, _ = robust_center_depth(
            depth_arr,
            int(round(center_u)),
            int(round(center_v)),
            self._wood_depth_radius,
            depth_encoding,
            min_depth_m=self._wood_min_depth_m,
            max_depth_m=self._wood_max_depth_m,
            outlier_threshold_m=self._wood_depth_outlier_m,
            min_valid_samples=self._wood_min_depth_samples,
        )

        if depth_m is not None and math.isfinite(depth_m) and depth_m > 0.0:
            return float(depth_m)

        self.get_logger().warn(
            f"Wood depth invalid at ({center_u:.0f},{center_v:.0f}); "
            f"valid_samples={valid_count}"
        )
        return None

    _WOOD_LOG_EVERY = 30
    _wood_frame_count = 0

    def _on_wood_detection(self, msg: ObjectDetection) -> None:
        """Process wood detection: pixel+depth+camera_info → X_C,Y_C,Z_C in camera frame."""
        intrinsics = self._get_intrinsics()
        if intrinsics is None:
            self.get_logger().warn_once(
                "No camera intrinsics yet; skipping wood detection. "
                f"Waiting for {self._camera_info_topic}"
            )
            return

        fx, fy, cx, cy = intrinsics

        object_id = int(getattr(msg, "object_id", 0))
        center_u, center_v, yaw_img_deg, _, _ = self._refine_wood_roi(msg)

        depth_m = self._wood_depth_m(center_u, center_v)

        if depth_m is None:
            self.get_logger().warn(
                f"Wood depth unavailable at pixel ({center_u:.0f},{center_v:.0f}); "
                "skipping this detection."
            )
            return

        try:
            x_c_m, y_c_m, z_c_m = pixel_to_camera_xyz(
                center_u, center_v, depth_m, fx, fy, cx, cy
            )
        except ValueError as exc:
            self.get_logger().warn(f"pixel_to_camera_xyz failed: {exc}")
            return

        # TODO:
        # Hiện tại chưa có ma trận ngoại chuẩn camera -> robot.
        # Vì vậy chưa chuyển tọa độ từ camera frame sang base_link.
        # Khi có extrinsic calibration, sẽ thêm bước:
        #     p_base = T_base_camera @ p_camera
        # và khi đó mới đổi header.frame_id = "base_link".

        yaw_deg = yaw_img_deg
        del yaw_img_deg  # stored in WoodData.yaw_img_deg; quaternion computed at publish time

        __class__._wood_frame_count += 1
        if __class__._wood_frame_count % __class__._WOOD_LOG_EVERY == 0:
            self.get_logger().info(
                f"[wood frame {__class__._wood_frame_count}] "
                f"class={msg.class_name} conf={float(msg.confidence):.2f} "
                f"pixel=({center_u:.0f},{center_v:.0f}) "
                f"camera=({x_c_m:.4f},{y_c_m:.4f},{z_c_m:.4f})m "
                f"depth={depth_m:.3f}m yaw={yaw_deg:.1f}deg"
            )

        now = self._now_sec()
        self._last_wood_detected_time = now

        with self._data_lock:
            self._latest_woods[object_id] = WoodData(
                object_id=object_id,
                class_name=str(msg.class_name),
                confidence=float(msg.confidence),
                x_c_m=x_c_m,
                y_c_m=y_c_m,
                z_c_m=z_c_m,
                yaw_img_deg=float(yaw_deg),
                stamp_sec=now,
                x_min=int(msg.x_min),
                y_min=int(msg.y_min),
                x_max=int(msg.x_max),
                y_max=int(msg.y_max),
                center_x=int(round(center_u)),
                center_y=int(round(center_v)),
            )

    def _on_publish_stale_status(self) -> None:
        now = self._now_sec()
        with self._data_lock:
            # Prune stale objects
            self._latest_woods = {
                object_id: wood
                for object_id, wood in self._latest_woods.items()
                if (now - wood.stamp_sec) <= self._stale_timeout_sec
            }

            # Convert WoodData cache → list of Object messages
            now_msg = self.get_clock().now().to_msg()
            frame_id = self._output_frame_id
            object_list: List[Object] = []
            for wood in sorted(self._latest_woods.values(), key=lambda w: w.object_id):
                obj = Object()
                obj.header.stamp = now_msg
                obj.header.frame_id = frame_id
                obj.object_id = wood.object_id
                obj.class_name = wood.class_name
                obj.confidence = wood.confidence
                qx, qy, qz, qw = yaw_to_quaternion(wood.yaw_img_deg)
                obj.pose.position.x = wood.x_c_m
                obj.pose.position.y = wood.y_c_m
                obj.pose.position.z = wood.z_c_m
                obj.pose.orientation.x = qx
                obj.pose.orientation.y = qy
                obj.pose.orientation.z = qz
                obj.pose.orientation.w = qw
                object_list.append(obj)

        # Publish ObjectArray on the primary topic /vision/wood_objects
        objects_msg = ObjectArray()
        objects_msg.header.stamp = now_msg
        objects_msg.header.frame_id = frame_id
        objects_msg.objects = object_list
        self._pub_wood_objects.publish(objects_msg)

        # Also publish on /vision/objects for backward compatibility
        compat_msg = ObjectArray()
        compat_msg.header.stamp = now_msg
        compat_msg.header.frame_id = frame_id
        compat_msg.objects = object_list
        self._pub_objects.publish(compat_msg)

        wood_stale = (
            self._last_wood_detected_time is None
            or (now - self._last_wood_detected_time) > self._stale_timeout_sec
        )

        status_msg = String()
        status_msg.data = f"wood_detected={not wood_stale}, n_woods={len(self._latest_woods)}"
        self._pub_detection_status.publish(status_msg)

    def _on_raw_image(self, msg: Image) -> None:
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"cv_bridge error on raw image: {exc}")
            return

        with self._color_lock:
            self._latest_color_bgr = bgr.copy()

        now = self._now_sec()
        with self._data_lock:
            woods = [
                wood
                for wood in self._latest_woods.values()
                if (now - wood.stamp_sec) <= self._overlay_timeout_sec
            ]
            woods.sort(key=lambda item: item.object_id)

        for wood in woods:
            bgr = self._draw_wood_overlay(bgr, wood)

        lines: List[str] = []
        for wood in woods[:4]:
            if lines:
                lines.append("")
            lines.append(f"{wood.class_name} conf={wood.confidence:.2f}")
            lines.append(f"cam=({wood.x_c_m:.4f},{wood.y_c_m:.4f},{wood.z_c_m:.4f})m")
            lines.append(f"yaw={wood.yaw_img_deg:.1f}deg")

        if lines:
            bgr = self._overlay_text_block(bgr, lines, origin_x=8, origin_y=8)

        try:
            out_msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            out_msg.header = msg.header
            self._pub_debug_camera.publish(out_msg)
        except CvBridgeError as exc:
            self.get_logger().warn(f"Debug image publish error: {exc}")

    def _draw_wood_overlay(self, bgr: np.ndarray, wood: WoodData) -> np.ndarray:
        import cv2

        out = bgr.copy()
        h, w = out.shape[:2]
        x_min = max(0, min(wood.x_min, w - 1))
        y_min = max(0, min(wood.y_min, h - 1))
        x_max = max(0, min(wood.x_max, w - 1))
        y_max = max(0, min(wood.y_max, h - 1))
        cx = max(0, min(wood.center_x, w - 1))
        cy = max(0, min(wood.center_y, h - 1))
        color = (0, 255, 255)

        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), color, 2, cv2.LINE_AA)
        cv2.putText(
            out,
            wood.class_name,
            (max(0, x_min), max(18, y_min - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

        arrow_len = max(28, min(w, h) // 8)
        ex = int(round(cx + arrow_len * math.cos(math.radians(wood.yaw_img_deg))))
        ey = int(round(cy + arrow_len * math.sin(math.radians(wood.yaw_img_deg))))
        ex = max(0, min(ex, w - 1))
        ey = max(0, min(ey, h - 1))
        cv2.arrowedLine(out, (cx, cy), (ex, ey), (255, 0, 0), 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)
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
    ) -> np.ndarray:
        import cv2

        out = img.copy()
        ih, iw = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        def line_color(line: str) -> tuple[int, int, int]:
            if line.startswith("cam="):
                return (128, 255, 255)
            if line.startswith("yaw="):
                return (200, 200, 0)
            return (0, 255, 255)

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
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 0), -1)
            cv2.rectangle(out, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)
            baseline_y = min(y0 + padding + th, ih - 1)
            cv2.putText(
                out,
                line,
                (x0 + padding, baseline_y),
                font,
                font_scale,
                line_color(line),
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
