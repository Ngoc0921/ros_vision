"""YOLO box detection on RealSense color frames with cached aligned depth.

Publishes detections per class:
  /vision/target_detection  → class "target"
  /vision/box_detection    → class "box"

Draws all detections on a single /vision/debug_image with per-class colours:
  box    → yellow  (0, 255, 255)
  target → blue   (255, 128, 0)
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from gp7_vision_pipeline.depth_utils import bbox_clip_to_image, depth_at_pixel, median_depth_meters
from gp7_vision_pipeline.detection_visualizer import (
    draw_all_detections,
    draw_no_detection,
)
from gp7_vision_pipeline.msg import BoxDetection


def _try_import_yolo():
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required. Install with: pip install ultralytics"
        ) from exc


# Per-class colours for the debug image (BGR, OpenCV convention).
CLASS_COLORS: Dict[str, tuple] = {
    "box": (0, 255, 255),     # yellow
    "target": (255, 128, 0),  # blue-orange
}


def _class_color(class_name: str) -> tuple:
    return CLASS_COLORS.get(class_name.lower(), (0, 255, 0))


class YoloBoxDetectorNode(Node):
    """Color-driven YOLO + latest cached depth for distance; optional CameraInfo."""

    # Throttle detection logs to one per N frames
    _LOG_EVERY = 30

    def __init__(self) -> None:
        super().__init__("yolo_box_detector_node")
        self._bridge = CvBridge()
        self._info_lock = threading.Lock()
        self._depth_lock = threading.Lock()
        self._latest_info: Optional[CameraInfo] = None
        self._latest_depth_msg: Optional[Image] = None
        self._warned_bad_depth = False
        self._model = None
        self._model_names: Dict[int, str] = {}
        self._model_load_error: Optional[str] = None
        self._logged_skip_model = False
        self._device = "cpu"
        self._frame_count = 0

        self._logged_color_callback = False
        self._logged_depth_cached = False
        self._logged_first_camera_info = False
        self._warned_camera_info_missing = False

        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("sync_queue_size", 10)
        self.declare_parameter("sync_slop", 0.1)
        self.declare_parameter(
            "model_path",
            "/home/norman/pap_yaskawa_ws/src/gp7_vision_pipeline/model/box_target/weights/best.pt",
        )
        self.declare_parameter("conf_threshold", 0.8)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("depth_roi_half_size", 5)

        color_topic = self.get_parameter("color_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value

        self._pub_target = self.create_publisher(BoxDetection, "/vision/target_detection", 10)
        self._pub_box = self.create_publisher(BoxDetection, "/vision/box_detection", 10)
        self._pub_debug = self.create_publisher(Image, "/vision/debug_image", 10)

        self._info_sub = self.create_subscription(
            CameraInfo,
            info_topic,
            self._on_camera_info,
            qos_profile_sensor_data,
        )

        self._color_sub = self.create_subscription(
            Image,
            color_topic,
            self._on_color_image,
            qos_profile_sensor_data,
        )
        self._depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self._on_depth_image,
            qos_profile_sensor_data,
        )

        self.get_logger().info("YOLO box detector (latest-depth-on-color).")
        self.get_logger().info(f"Subscribing to color topic: {color_topic}")
        self.get_logger().info(f"Subscribing to depth topic: {depth_topic}")
        self.get_logger().info(f"Subscribing to camera_info topic: {info_topic}")
        self._try_load_model()

    def _on_depth_image(self, msg: Image) -> None:
        with self._depth_lock:
            self._latest_depth_msg = msg
        if not self._logged_depth_cached:
            self._logged_depth_cached = True
            self.get_logger().info("DEPTH FRAME CACHED")

    def _try_load_model(self) -> None:
        path = self.get_parameter("model_path").get_parameter_value().string_value
        try:
            yolo_cls = _try_import_yolo()
        except RuntimeError as e:
            self._model_load_error = str(e)
            self.get_logger().error(self._model_load_error)
            return

        p = Path(path).expanduser()
        if not p.is_file():
            self._model_load_error = f"Model file not found: {p}"
            self.get_logger().error(self._model_load_error)
            return

        try:
            self._model = yolo_cls(str(p))
            self._model_names = {int(k): v for k, v in self._model.names.items()}
            self._device = self.get_parameter("device").get_parameter_value().string_value or "cpu"
            if self._device:
                self._model.to(self._device)
            self._model_load_error = None
            self.get_logger().info(
                f"Loaded YOLO model from {p}  "
                f"names={self._model.names}  device={self._device!r}"
            )
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._model_load_error = f"Failed to load YOLO model: {exc}"
            self.get_logger().error(self._model_load_error)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._info_lock:
            self._latest_info = msg
        self._warned_camera_info_missing = False
        if not self._logged_first_camera_info:
            self._logged_first_camera_info = True
            self.get_logger().info("First camera_info received.")

    def _get_camera_info(self) -> Optional[CameraInfo]:
        with self._info_lock:
            return self._latest_info

    def _maybe_warn_missing_camera_info(self) -> None:
        if self._get_camera_info() is not None:
            return
        if self._warned_camera_info_missing:
            return
        self._warned_camera_info_missing = True
        self.get_logger().warning(
            "CameraInfo not available yet; continuing without it "
            "(detection and debug image still run)."
        )

    def _on_color_image(self, color_msg: Image) -> None:
        if not self._logged_color_callback:
            self._logged_color_callback = True
            self.get_logger().info("COLOR CALLBACK TRIGGERED")

        self._frame_count += 1
        publish_debug = self.get_parameter("publish_debug_image").get_parameter_value().bool_value
        conf_thr = float(self.get_parameter("conf_threshold").value)
        iou_thr = float(self.get_parameter("iou_threshold").value)
        roi_half = int(self.get_parameter("depth_roi_half_size").value)

        self._maybe_warn_missing_camera_info()

        with self._depth_lock:
            depth_msg = self._latest_depth_msg

        depth_cv: Optional[np.ndarray] = None
        depth_enc = ""
        if depth_msg is not None:
            try:
                depth_cv = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                depth_enc = depth_msg.encoding
            except CvBridgeError as e:
                self.get_logger().warning(f"Cached depth cv_bridge error: {e}")
                depth_cv = None

        if self._model is None:
            if self._model_load_error and not self._logged_skip_model:
                self.get_logger().warning("Skipping inference until model loads successfully.")
                self._logged_skip_model = True
            if publish_debug:
                try:
                    bgr = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
                    dbg = draw_no_detection(bgr, "model not loaded")
                    self._publish_debug_image(color_msg.header, dbg)
                except CvBridgeError as e:
                    self.get_logger().warning(f"Debug image skipped (color cv_bridge): {e}")
            return

        try:
            bgr = self._bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().warning(f"Color cv_bridge error: {e}")
            return

        if depth_cv is not None and depth_cv.ndim != 2:
            if not self._warned_bad_depth:
                self.get_logger().warning(
                    f"Expected 2D depth image, got shape {depth_cv.shape}; "
                    "distance may be invalid."
                )
                self._warned_bad_depth = True
            depth_cv = None

        h, w = bgr.shape[:2]
        results = self._model.predict(
            source=bgr,
            verbose=False,
            conf=conf_thr,
            iou=iou_thr,
            device=self._device,
        )

        all_detections: List[BoxDetection] = []

        r0 = results[0] if results else None
        boxes = getattr(r0, "boxes", None)
        raw_boxes = boxes.cpu().numpy() if boxes is not None and len(boxes) > 0 else []

        if boxes is None or len(boxes) == 0:
            if publish_debug:
                self._publish_debug_image(
                    color_msg.header,
                    draw_no_detection(bgr, "No detection"),
                )
            return

        # Parallel depth debug info list — kept in sync with all_detections by index.
        # Each dict holds raw-sensor values and encoding for overlay/logging.
        # BoxDetection messages themselves are NOT modified (ROS msg classes reject
        # dynamic attributes).
        depth_debug_infos: List[dict] = []

        # Collect all detections across all classes.
        # For each detection we compute:
        #   - center_pixel_raw : raw depth value at the bbox center pixel
        #   - roi_median_raw   : raw median depth in the half_size ROI (mm for 16UC1)
        #   - dist_m           : converted distance in metres
        # These are kept separate so the debug overlay and logs show both the raw
        # sensor value and the human-readable metres value.
        for box in boxes:
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            cf = float(box.conf[0].item()) if box.conf is not None else 0.0
            class_name = str(self._model_names.get(cls_id, str(cls_id)))
            xyxy = box.xyxy[0].cpu().numpy()

            x_min_f, y_min_f, x_max_f, y_max_f = xyxy
            x_min, y_min, x_max, y_max = bbox_clip_to_image(
                int(round(x_min_f)),
                int(round(y_min_f)),
                int(round(x_max_f)),
                int(round(y_max_f)),
                w,
                h,
            )
            cx = int(round((x_min + x_max) * 0.5))
            cy = int(round((y_min + y_max) * 0.5))
            width_px = int(x_max - x_min)
            height_px = int(y_max - y_min)

            # --- distance: (dist_m, roi_median_raw) ---
            dist_m = -1.0
            roi_median_raw: Optional[int] = None
            if depth_cv is not None and depth_cv.ndim == 2:
                dist_m, roi_median_raw = median_depth_meters(
                    depth_cv, cx, cy, roi_half, depth_enc
                )

            # --- center-pixel raw depth ---
            center_raw, center_dist, _ = depth_at_pixel(depth_cv, cx, cy, depth_enc)

            # --- normalise raw depth to millimetres for the ROS message ---
            # For 16UC1: already mm (int)
            # For 32FC1: convert metres → mm
            if center_raw is not None:
                if depth_enc in ("32FC1",):
                    center_raw_mm = int(round(float(center_raw) * 1000.0))
                else:
                    center_raw_mm = int(center_raw)
            else:
                center_raw_mm = 0

            if roi_median_raw is not None:
                if depth_enc in ("32FC1",):
                    roi_median_raw_mm = int(round(float(roi_median_raw) * 1000.0))
                else:
                    roi_median_raw_mm = int(roi_median_raw)
            else:
                roi_median_raw_mm = 0

            det = BoxDetection()
            det.header = color_msg.header
            det.class_name = class_name
            det.confidence = cf
            det.x_min = x_min
            det.y_min = y_min
            det.x_max = x_max
            det.y_max = y_max
            det.center_x = cx
            det.center_y = cy
            det.width_px = width_px
            det.height_px = height_px
            det.center_raw_depth = center_raw_mm
            det.roi_median_raw_depth = roi_median_raw_mm
            det.depth_encoding = depth_enc
            det.distance_m = float(dist_m)
            all_detections.append(det)

            # Parallel dict: raw-sensor depth info for overlay / logging.
            # Raw values match the normalisation applied to the BoxDetection message.
            depth_debug_infos.append({
                "depth_encoding": depth_enc,
                "center_raw": center_raw,
                "center_raw_mm": center_raw_mm,
                "center_dist_m": center_dist,
                "roi_median_raw": roi_median_raw,
                "roi_median_raw_mm": roi_median_raw_mm,
                "roi_median_dist_m": dist_m,
            })

        if not all_detections:
            if publish_debug:
                self._publish_debug_image(
                    color_msg.header,
                    draw_no_detection(bgr, "No detection"),
                )
            return

        # Publish per-class detection topics
        for det in all_detections:
            if det.class_name.strip().lower() == "target":
                self._pub_target.publish(det)
            else:
                # "box" and any other class go to box_detection
                self._pub_box.publish(det)

        # Throttled detection summary log — every _LOG_EVERY frames
        if self._frame_count % self._LOG_EVERY == 0:
            names_seen = [d.class_name for d in all_detections]
            counts: Dict[str, int] = {}
            for n in names_seen:
                counts[n] = counts.get(n, 0) + 1
            self.get_logger().info(
                f"Frame {self._frame_count}: {len(all_detections)} detection(s) — "
                f"{counts}  conf_thr={conf_thr}"
            )
            for i, det in enumerate(all_detections):
                di = depth_debug_infos[i]
                self.get_logger().info(
                    f"  [{det.class_name}] u={det.center_x} v={det.center_y} "
                    f"encoding={di['depth_encoding']} "
                    f"center_raw={di['center_raw_mm']} "
                    f"roi_median_raw={di['roi_median_raw_mm']} "
                    f"dist_m={di['roi_median_dist_m']:.4f}"
                )

        # Debug image: draw all detections with per-class colours
        if publish_debug:
            debug_bgr = draw_all_detections(
                bgr,
                all_detections,
                CLASS_COLORS,
            )
            self._publish_debug_image(color_msg.header, debug_bgr)

    def _publish_debug_image(self, header: Header, bgr: np.ndarray) -> None:
        try:
            out = self._bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            out.header = header
            self._pub_debug.publish(out)
        except CvBridgeError as e:
            self.get_logger().warning(f"Debug image publish failed: {e}")


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = YoloBoxDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
