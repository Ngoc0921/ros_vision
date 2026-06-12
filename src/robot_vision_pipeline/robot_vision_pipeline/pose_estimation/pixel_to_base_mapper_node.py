"""Map BoxDetection (wood / box) from pixel + depth + camera intrinsics to camera-frame poses.

IMPORTANT — current coordinate frame:
  All published poses are in camera_color_optical_frame.
  Camera → base_link transform is NOT implemented yet.
  This node does NOT use homography.

Inputs:
  /vision/wood_detection               robot_vision_pipeline_msgs/BoxDetection
  /vision/box_detection               robot_vision_pipeline_msgs/BoxDetection
  /camera/camera/color/camera_info    sensor_msgs/CameraInfo
  /camera/camera/aligned_depth_to_color/image_raw  sensor_msgs/Image
  /camera/camera/color/image_raw      sensor_msgs/Image

Outputs:
  /vision/wood_objects                robot_vision_pipeline_msgs/WoodArray
  /vision/box_objects                robot_vision_pipeline_msgs/BoxArray
  /vision/debug_image_camera          sensor_msgs/Image
  /vision/detection_status           std_msgs/String
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
from std_msgs.msg import String

from robot_vision_pipeline.depth_utils import robust_center_depth
from robot_vision_pipeline_msgs.msg import BoxDetection, Wood, WoodArray, Box, BoxArray


def pixel_to_camera_xyz(
    u: float,
    v: float,
    z_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[float, float, float]:
    """Convert pixel (u, v) + depth Z to 3D point in camera optical frame.

    Formula:
        Xc = (u - cx) * Z / fx
        Yc = (v - cy) * Z / fy
        Zc = Z

    Args:
        u:   Pixel column (x).
        v:   Pixel row (y).
        z_m: Depth in metres.
        fx:  Focal length x (pixels).
        fy:  Focal length y (pixels).
        cx:  Principal point x (pixels).
        cy:  Principal point y (pixels).

    Returns:
        (Xc, Yc, Zc) in metres, relative to camera optical frame.
    """
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"Invalid camera intrinsics: fx={fx}, fy={fy}")
    x_c = (float(u) - float(cx)) * float(z_m) / float(fx)
    y_c = (float(v) - float(cy)) * float(z_m) / float(fy)
    z_c = float(z_m)
    return float(x_c), float(y_c), float(z_c)


def _resolve_depth(det: BoxDetection, depth_m: Optional[float]) -> float:
    """Resolve effective depth for a detection.

    Priority:
      1. detection.distance_m if valid (> 0)
      2. depth_m from aligned depth image if valid
      3. detection.roi_median_raw_depth converted to metres
      4. detection.center_raw_depth converted to metres
      5. 0.0 (invalid)
    """
    if det.distance_m > 0.0:
        return float(det.distance_m)
    if depth_m is not None and depth_m > 0.0:
        return float(depth_m)
    if det.roi_median_raw_depth > 0:
        return float(det.roi_median_raw_depth) * 0.001
    if det.center_raw_depth > 0:
        return float(det.center_raw_depth) * 0.001
    return 0.0


def _compute_box_size(
    bbox_w_px: float,
    bbox_h_px: float,
    depth_m: float,
    fx: float,
    fy: float,
    default_x: float,
    default_y: float,
    default_z: float,
) -> Tuple[float, float, float]:
    """Estimate box size from bbox pixels + depth + intrinsics.

    size.x = bbox_w_px * depth_m / fx
    size.y = bbox_h_px * depth_m / fy
    size.z = default_z
    """
    if depth_m > 0.0 and fx > 0.0 and fy > 0.0:
        sx = float(bbox_w_px) * float(depth_m) / float(fx)
        sy = float(bbox_h_px) * float(depth_m) / float(fy)
        if sx <= 0.0:
            sx = default_x
        if sy <= 0.0:
            sy = default_y
        return sx, sy, default_z
    return default_x, default_y, default_z


@dataclass
class WoodData:
    wood_id: int
    class_name: str
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    center_x: int
    center_y: int
    x_c_m: float
    y_c_m: float
    z_c_m: float
    stamp_sec: float


@dataclass
class BoxData:
    box_id: int
    class_name: str
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    center_x: int
    center_y: int
    x_c_m: float
    y_c_m: float
    z_c_m: float
    size_x_m: float
    size_y_m: float
    size_z_m: float
    stamp_sec: float


class PixelToBaseMapperNode(Node):
    """Map wood and box detections to camera-frame 3D poses (no homography, no base_link)."""

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
        self._latest_boxes: Dict[int, BoxData] = {}
        self._last_wood_detected_time: Optional[float] = None
        self._last_box_detected_time: Optional[float] = None
        self._warned_no_intrinsics_wood = False
        self._warned_no_intrinsics_box = False

        # Parameters
        self.declare_parameter("output_frame_id", "camera_color_optical_frame")

        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("color_image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_image_topic", "/camera/camera/aligned_depth_to_color/image_raw")

        self.declare_parameter("wood_detection_topic", "/vision/wood_detection")
        self.declare_parameter("box_detection_topic", "/vision/box_detection")

        self.declare_parameter("wood_objects_topic", "/vision/wood_objects")
        self.declare_parameter("box_objects_topic", "/vision/box_objects")

        self.declare_parameter("default_box_size_x_m", 0.08)
        self.declare_parameter("default_box_size_y_m", 0.08)
        self.declare_parameter("default_box_size_z_m", 0.05)

        self.declare_parameter("fake_depth_m", 0.55)
        self.declare_parameter("use_fake_depth_if_missing", True)

        self.declare_parameter("overlay_timeout_sec", 1.0)
        self.declare_parameter("stale_timeout_sec", 2.0)

        self._output_frame_id = str(self.get_parameter("output_frame_id").value)

        self._camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self._color_image_topic = str(self.get_parameter("color_image_topic").value)
        self._depth_image_topic = str(self.get_parameter("depth_image_topic").value)

        self._wood_detection_topic = str(self.get_parameter("wood_detection_topic").value)
        self._box_detection_topic = str(self.get_parameter("box_detection_topic").value)

        self._wood_objects_topic = str(self.get_parameter("wood_objects_topic").value)
        self._box_objects_topic = str(self.get_parameter("box_objects_topic").value)

        self._default_box_x = float(self.get_parameter("default_box_size_x_m").value)
        self._default_box_y = float(self.get_parameter("default_box_size_y_m").value)
        self._default_box_z = float(self.get_parameter("default_box_size_z_m").value)

        self._fake_depth_m = float(self.get_parameter("fake_depth_m").value)
        self._use_fake_depth = bool(self.get_parameter("use_fake_depth_if_missing").value)

        self._overlay_timeout_sec = float(self.get_parameter("overlay_timeout_sec").value)
        self._stale_timeout_sec = float(self.get_parameter("stale_timeout_sec").value)

        det_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publishers
        self._pub_wood_objects = self.create_publisher(WoodArray, self._wood_objects_topic, det_qos)
        self._pub_box_objects = self.create_publisher(BoxArray, self._box_objects_topic, det_qos)
        self._pub_status = self.create_publisher(String, "/vision/detection_status", det_qos)
        self._pub_debug = self.create_publisher(
            Image, "/vision/debug_image_camera",
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

        # Subscriptions
        self.create_subscription(
            BoxDetection, self._wood_detection_topic, self._on_wood_detection, det_qos
        )
        self.create_subscription(
            BoxDetection, self._box_detection_topic, self._on_box_detection, det_qos
        )
        self.create_subscription(
            Image, self._color_image_topic, self._on_color_image, img_qos
        )
        self.create_subscription(
            Image, self._depth_image_topic, self._on_depth_image, img_qos
        )
        self.create_subscription(
            CameraInfo, self._camera_info_topic, self._on_camera_info,
            QoSProfile(depth=1),
        )

        self._publish_timer = self.create_timer(0.5, self._on_publish_stale_status)

        self.get_logger().info(
            f"pixel_to_base_mapper_node started (camera-frame, no homography) | "
            f"camera_info={self._camera_info_topic}, "
            f"depth={self._depth_image_topic}, "
            f"wood_in={self._wood_detection_topic}, "
            f"box_in={self._box_detection_topic}, "
            f"wood_out={self._wood_objects_topic}, "
            f"box_out={self._box_objects_topic}, "
            f"output_frame={self._output_frame_id}"
        )

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _publish_status(self, text: str) -> None:
        status_msg = String()
        status_msg.data = text
        self._pub_status.publish(status_msg)

    # ------------------------------------------------------------------ camera info
    def _on_camera_info(self, msg: CameraInfo) -> None:
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

    # ------------------------------------------------------------------ depth
    def _on_depth_image(self, msg: Image) -> None:
        with self._depth_lock:
            self._latest_depth_msg = msg
            self._latest_depth_encoding = msg.encoding

    def _depth_at_center(self, cx: int, cy: int) -> Optional[float]:
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

        if (
            depth_arr is None
            or not isinstance(depth_arr, np.ndarray)
            or depth_arr.ndim != 2
        ):
            return None

        depth_m, _, _, _, _ = robust_center_depth(
            depth_arr, cx, cy,
            radius=2,
            encoding=depth_encoding,
            min_depth_m=0.05,
            max_depth_m=3.0,
            outlier_threshold_m=0.02,
            min_valid_samples=3,
        )
        if depth_m is not None and math.isfinite(depth_m) and depth_m > 0.0:
            return float(depth_m)
        return None

    def _resolve_depth_for_detection(
        self, det: BoxDetection
    ) -> float:
        effective = _resolve_depth(det, None)
        if effective > 0.0:
            return effective
        if self._use_fake_depth:
            return self._fake_depth_m
        return 0.0

    # ------------------------------------------------------------------ colour image
    def _on_color_image(self, msg: Image) -> None:
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError:
            return

        with self._color_lock:
            self._latest_color_bgr = bgr.copy()

        now = self._now_seconds()

        with self._data_lock:
            woods = [
                w for w in self._latest_woods.values()
                if (now - w.stamp_sec) <= self._overlay_timeout_sec
            ]
            boxes = [
                b for b in self._latest_boxes.values()
                if (now - b.stamp_sec) <= self._overlay_timeout_sec
            ]

        intrinsics = self._get_intrinsics()
        if intrinsics is not None:
            fx, fy, cx_i, cy_i = intrinsics
        else:
            fx = fy = cx_i = cy_i = None

        import cv2

        for w in woods:
            h, ww = bgr.shape[:2]
            x1 = max(0, min(int(w.x_min), ww - 1))
            y1 = max(0, min(int(w.y_min), h - 1))
            x2 = max(0, min(int(w.x_max), ww - 1))
            y2 = max(0, min(int(w.y_max), h - 1))
            cx_px = max(0, min(int(w.center_x), ww - 1))
            cy_px = max(0, min(int(w.center_y), h - 1))
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.circle(bgr, (cx_px, cy_px), 6, (0, 200, 0), -1)
            cv2.circle(bgr, (cx_px, cy_px), 6, (255, 255, 255), 1)
            cv2.putText(
                bgr,
                f"{w.class_name} {w.confidence:.2f} ({w.x_c_m:.3f},{w.y_c_m:.3f},{w.z_c_m:.3f})",
                (x1 + 4, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1, cv2.LINE_AA
            )

        for b in boxes:
            h, ww = bgr.shape[:2]
            x1 = max(0, min(int(b.x_min), ww - 1))
            y1 = max(0, min(int(b.y_min), h - 1))
            x2 = max(0, min(int(b.x_max), ww - 1))
            y2 = max(0, min(int(b.y_max), h - 1))
            cx_px = max(0, min(int(b.center_x), ww - 1))
            cy_px = max(0, min(int(b.center_y), h - 1))
            cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(bgr, (cx_px, cy_px), 6, (0, 255, 255), -1)
            cv2.circle(bgr, (cx_px, cy_px), 6, (255, 255, 255), 1)
            cv2.putText(
                bgr,
                f"{b.class_name} {b.confidence:.2f} ({b.x_c_m:.3f},{b.y_c_m:.3f},{b.z_c_m:.3f})",
                (x1 + 4, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA
            )

        try:
            out_msg = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            out_msg.header = msg.header
            self._pub_debug.publish(out_msg)
        except CvBridgeError:
            pass

    # ------------------------------------------------------------------ detection callbacks
    def _on_wood_detection(self, msg: BoxDetection) -> None:
        intrinsics = self._get_intrinsics()
        if intrinsics is None:
            if not self._warned_no_intrinsics_wood:
                self.get_logger().warn(
                    "No camera intrinsics yet; skipping wood detection. "
                    f"Waiting for {self._camera_info_topic}"
                )
                self._warned_no_intrinsics_wood = True
            self._publish_status(f"waiting_for_camera_info topic={self._camera_info_topic}")
            return

        fx, fy, cx, cy = intrinsics

        cx_px = int(msg.center_x)
        cy_px = int(msg.center_y)
        depth_m = self._resolve_depth_for_detection(msg)

        if depth_m <= 0.0:
            self.get_logger().warn(
                f"Wood depth invalid for detection {msg.object_id}; skipping."
            )
            self._publish_status(
                f"invalid_wood_depth id={msg.object_id} center=({msg.center_x},{msg.center_y})"
            )
            return

        try:
            x_c, y_c, z_c = pixel_to_camera_xyz(
                float(cx_px), float(cy_px), depth_m, fx, fy, cx, cy
            )
        except ValueError as exc:
            self.get_logger().warn(f"pixel_to_camera_xyz failed: {exc}")
            return

        now = self._now_seconds()
        self._last_wood_detected_time = now

        with self._data_lock:
            self._latest_woods[msg.object_id] = WoodData(
                wood_id=int(msg.object_id),
                class_name="wood",
                confidence=float(msg.confidence),
                x_min=int(msg.x_min),
                y_min=int(msg.y_min),
                x_max=int(msg.x_max),
                y_max=int(msg.y_max),
                center_x=int(msg.center_x),
                center_y=int(msg.center_y),
                x_c_m=x_c,
                y_c_m=y_c,
                z_c_m=z_c,
                stamp_sec=now,
            )

    def _on_box_detection(self, msg: BoxDetection) -> None:
        intrinsics = self._get_intrinsics()
        if intrinsics is None:
            if not self._warned_no_intrinsics_box:
                self.get_logger().warn(
                    "No camera intrinsics yet; skipping box detection. "
                    f"Waiting for {self._camera_info_topic}"
                )
                self._warned_no_intrinsics_box = True
            self._publish_status(f"waiting_for_camera_info topic={self._camera_info_topic}")
            return

        fx, fy, cx, cy = intrinsics

        cx_px = int(msg.center_x)
        cy_px = int(msg.center_y)
        depth_m = self._resolve_depth_for_detection(msg)

        if depth_m <= 0.0:
            self.get_logger().warn(
                f"Box depth invalid for detection {msg.object_id}; skipping."
            )
            self._publish_status(
                f"invalid_box_depth id={msg.object_id} center=({msg.center_x},{msg.center_y})"
            )
            return

        try:
            x_c, y_c, z_c = pixel_to_camera_xyz(
                float(cx_px), float(cy_px), depth_m, fx, fy, cx, cy
            )
        except ValueError as exc:
            self.get_logger().warn(f"pixel_to_camera_xyz failed: {exc}")
            return

        size_x, size_y, size_z = _compute_box_size(
            float(msg.width_px) if msg.width_px > 0 else float(msg.x_max - msg.x_min),
            float(msg.height_px) if msg.height_px > 0 else float(msg.y_max - msg.y_min),
            depth_m,
            fx, fy,
            self._default_box_x,
            self._default_box_y,
            self._default_box_z,
        )

        now = self._now_seconds()
        self._last_box_detected_time = now

        with self._data_lock:
            self._latest_boxes[msg.object_id] = BoxData(
                box_id=int(msg.object_id),
                class_name="box",
                confidence=float(msg.confidence),
                x_min=int(msg.x_min),
                y_min=int(msg.y_min),
                x_max=int(msg.x_max),
                y_max=int(msg.y_max),
                center_x=int(msg.center_x),
                center_y=int(msg.center_y),
                x_c_m=x_c,
                y_c_m=y_c,
                z_c_m=z_c,
                size_x_m=size_x,
                size_y_m=size_y,
                size_z_m=size_z,
                stamp_sec=now,
            )

    # ------------------------------------------------------------------ publish
    def _on_publish_stale_status(self) -> None:
        now = self._now_seconds()

        with self._data_lock:
            # Prune stale
            self._latest_woods = {
                wid: w for wid, w in self._latest_woods.items()
                if (now - w.stamp_sec) <= self._stale_timeout_sec
            }
            self._latest_boxes = {
                bid: b for bid, b in self._latest_boxes.items()
                if (now - b.stamp_sec) <= self._stale_timeout_sec
            }

            now_msg = self.get_clock().now().to_msg()
            frame_id = self._output_frame_id

            # Build WoodArray
            wood_list: List[Wood] = []
            for w in sorted(self._latest_woods.values(), key=lambda x: x.wood_id):
                wood = Wood()
                wood.header.stamp = now_msg
                wood.header.frame_id = frame_id
                wood.wood_id = w.wood_id
                wood.class_name = w.class_name
                wood.confidence = w.confidence
                wood.pose.position.x = w.x_c_m
                wood.pose.position.y = w.y_c_m
                wood.pose.position.z = w.z_c_m
                wood.pose.orientation.x = 0.0
                wood.pose.orientation.y = 0.0
                wood.pose.orientation.z = 0.0
                wood.pose.orientation.w = 1.0
                wood_list.append(wood)

            wood_arr_msg = WoodArray()
            wood_arr_msg.header.stamp = now_msg
            wood_arr_msg.header.frame_id = frame_id
            wood_arr_msg.woods = wood_list

            # Build BoxArray
            box_list: List[Box] = []
            for b in sorted(self._latest_boxes.values(), key=lambda x: x.box_id):
                box = Box()
                box.header.stamp = now_msg
                box.header.frame_id = frame_id
                box.box_id = b.box_id
                box.class_name = b.class_name
                box.confidence = b.confidence
                box.pose.position.x = b.x_c_m
                box.pose.position.y = b.y_c_m
                box.pose.position.z = b.z_c_m
                box.pose.orientation.x = 0.0
                box.pose.orientation.y = 0.0
                box.pose.orientation.z = 0.0
                box.pose.orientation.w = 1.0
                box.size.x = b.size_x_m
                box.size.y = b.size_y_m
                box.size.z = b.size_z_m
                box_list.append(box)

            box_arr_msg = BoxArray()
            box_arr_msg.header.stamp = now_msg
            box_arr_msg.header.frame_id = frame_id
            box_arr_msg.boxes = box_list

        self._pub_wood_objects.publish(wood_arr_msg)
        self._pub_box_objects.publish(box_arr_msg)

        n_wood = len(self._latest_woods)
        n_box = len(self._latest_boxes)
        status_msg = String()
        missing = []
        if self._get_intrinsics() is None:
            missing.append("camera_info")
        with self._depth_lock:
            has_depth = self._latest_depth_msg is not None
        if not has_depth:
            missing.append("depth")
        if n_wood == 0 and n_box == 0:
            missing.append("detections")
        missing_text = ",".join(missing) if missing else "none"
        status_msg.data = f"wood={n_wood}, box={n_box}, frame={frame_id}, missing={missing_text}"
        self._pub_status.publish(status_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PixelToBaseMapperNode()
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
