#!/home/minhquang/venvs/ros_env/bin/python3

import json
import time
from typing import Any, Dict, List, Optional, Tuple
import os
import threading

import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


class YoloDetectNode(Node):
    def __init__(self):
        super().__init__("yolo_detect_node")

        # =========================
        # Declare parameters
        # =========================
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("annotated_image_topic", "/vision/yolo/image_annotated")
        self.declare_parameter("detections_topic", "/vision/yolo/detections_json")

        self.declare_parameter(
            "model_path",
            os.path.expanduser("~/ros2/src/robot_vision_pipeline/model/best.pt"),
        )
        self.declare_parameter("device", "cpu")

        self.declare_parameter("conf_threshold", 0.35)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("max_det", 20)

        # <= 0: detect realtime trên mỗi frame nhận được.
        # > 0 : throttle inference theo chu kỳ cấu hình.
        self.declare_parameter("detect_period_sec", 0.0)

        self.declare_parameter("show_gui", False)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("image_qos", "reliable")
        self.declare_parameter("class_filter", "")
        self.declare_parameter("verbose_log", False)

        # Override from launch / command line
        self.declare_parameter("model_path_override", "")
        self.declare_parameter("image_topic_override", "")

        # ROI crop parameters
        self.declare_parameter("enable_roi_crop", True)
        self.declare_parameter("roi_x", 100)
        self.declare_parameter("roi_y", 80)
        self.declare_parameter("roi_width", 440)
        self.declare_parameter("roi_height", 320)
        self.declare_parameter("publish_roi_debug_image", True)

        # =========================
        # Get parameters
        # =========================
        image_topic = self.get_parameter("image_topic").value
        annotated_image_topic = self.get_parameter("annotated_image_topic").value
        detections_topic = self.get_parameter("detections_topic").value

        model_path = self.get_parameter("model_path").value
        model_path_override = self.get_parameter("model_path_override").value
        image_topic_override = self.get_parameter("image_topic_override").value

        if isinstance(model_path_override, str) and model_path_override.strip():
            model_path = model_path_override.strip()

        if isinstance(image_topic_override, str) and image_topic_override.strip():
            image_topic = image_topic_override.strip()

        self.image_topic = str(image_topic)
        self.device = str(self.get_parameter("device").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.detect_period_sec = float(self.get_parameter("detect_period_sec").value)

        self.show_gui = bool(self.get_parameter("show_gui").value)
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self.image_qos = str(self.get_parameter("image_qos").value).strip().lower()
        self.verbose_log = bool(self.get_parameter("verbose_log").value)

        class_filter_param = str(self.get_parameter("class_filter").value).strip()
        self.class_filter = set()
        if class_filter_param:
            self.class_filter = set(
                x.strip() for x in class_filter_param.split(",") if x.strip()
            )

        # ROI parameters
        self.enable_roi_crop = bool(self.get_parameter("enable_roi_crop").value)
        self.roi_x = int(self.get_parameter("roi_x").value)
        self.roi_y = int(self.get_parameter("roi_y").value)
        self.roi_width = int(self.get_parameter("roi_width").value)
        self.roi_height = int(self.get_parameter("roi_height").value)
        self.publish_roi_debug_image = bool(
            self.get_parameter("publish_roi_debug_image").value
        )

        self.bridge = CvBridge()
        self.model = None
        self.names = {}

        # =========================
        # Runtime state
        # =========================
        self.window_name = "YOLO Detect - realtime"
        self.latest_frame = None
        self.latest_header = None
        self.last_result_image = None
        self.last_payload = None
        self.latest_frame_seq = 0
        self.processed_frame_seq = -1
        self.total_rx_frames = 0

        # ROI state (updated each frame after validation against image size)
        self._current_roi_valid = False
        self._current_roi: Optional[Tuple[int, int, int, int]] = None
        self._roi_logged_warning = False

        self.last_detect_time = 0.0
        self.is_detecting = False
        self.frame_lock = threading.Lock()
        self.new_frame_event = threading.Event()
        self.stop_event = threading.Event()

        self.frame_count = 0
        self.last_fps_time = time.time()
        self.preview_fps = 0.0

        self.detect_count = 0
        self.detect_fps = 0.0
        self.detect_fps_count = 0
        self.last_detect_fps_time = time.time()

        if self.show_gui:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 960, 720)

        # =========================
        # Load YOLO only once
        # =========================
        try:
            from ultralytics import YOLO  # pyright: ignore[reportMissingImports]

            self.get_logger().info(f"Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)

            if hasattr(self.model, "names"):
                self.names = self.model.names
            else:
                self.names = {}

            self.get_logger().info("YOLO model loaded successfully.")
            self.get_logger().info(f"Class names: {self.names}")

        except Exception as e:
            self.get_logger().error("Failed to load YOLO model.")
            self.get_logger().error(str(e))
            self.get_logger().error(
                "Kiểm tra ultralytics:\n"
                "  python3 -c \"from ultralytics import YOLO; print('YOLO OK')\""
            )
            raise e

        # =========================
        # ROS publishers / subscribers
        # =========================
        if self.image_qos == "best_effort":
            image_reliability = ReliabilityPolicy.BEST_EFFORT
        else:
            image_reliability = ReliabilityPolicy.RELIABLE

        image_qos = QoSProfile(
            reliability=image_reliability,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            image_qos,
        )

        self.annotated_pub = self.create_publisher(
            Image,
            annotated_image_topic,
            10,
        )

        self.detections_pub = self.create_publisher(
            String,
            detections_topic,
            10,
        )

        self.roi_debug_pub = None
        if self.publish_roi_debug_image:
            self.roi_debug_pub = self.create_publisher(
                Image,
                "/vision/yolo/roi_debug_image",
                10,
            )

        self.status_timer = self.create_timer(5.0, self.status_timer_callback)
        self.detect_thread = threading.Thread(target=self.detect_worker, daemon=True)
        self.detect_thread.start()

        self.get_logger().info("========================================")
        self.get_logger().info("YOLO Detect Node started")
        if self.detect_period_sec <= 0.0:
            self.get_logger().info("Mode                  : REALTIME")
            self.get_logger().info("YOLO worker always processes the newest frame.")
        else:
            self.get_logger().info("Mode                  : THROTTLED")
            self.get_logger().info(f"Detect period         : {self.detect_period_sec:.3f} s")
        self.get_logger().info(f"Input image topic     : {image_topic}")
        self.get_logger().info(f"Input image QoS       : {self.image_qos}")
        self.get_logger().info(f"Annotated topic       : {annotated_image_topic}")
        self.get_logger().info(f"Detections JSON topic : {detections_topic}")
        self.get_logger().info(f"Device                : {self.device}")
        self.get_logger().info(f"Confidence threshold  : {self.conf_threshold}")
        self.get_logger().info(f"IoU threshold         : {self.iou_threshold}")
        self.get_logger().info(f"Image size            : {self.imgsz}")
        self.get_logger().info(f"Max detections        : {self.max_det}")
        self.get_logger().info(f"Publish annotated     : {self.publish_annotated}")
        self.get_logger().info(f"Show GUI              : {self.show_gui}")
        self.get_logger().info(f"Class filter          : {list(self.class_filter)}")
        self.get_logger().info(
            f"ROI crop enabled      : {self.enable_roi_crop}"
        )
        if self.enable_roi_crop:
            self.get_logger().info(
                f"ROI params            : x={self.roi_x}, y={self.roi_y}, "
                f"w={self.roi_width}, h={self.roi_height}"
            )
        self.get_logger().info(f"ROI debug image       : {self.publish_roi_debug_image}")
        self.get_logger().info("========================================")

    # ============================================================
    # ROI clamp and validate
    # ============================================================
    def _clamp_roi(
        self,
        roi_x: int,
        roi_y: int,
        roi_w: int,
        roi_h: int,
        img_w: int,
        img_h: int,
    ) -> Tuple[int, int, int, int, bool]:
        """Clamp ROI to image bounds.

        Returns:
            (clamped_x, clamped_y, clamped_w, clamped_h, is_valid)
            is_valid is True only when w > 0 and h > 0 after clamping.
        """
        x = max(0, int(roi_x))
        y = max(0, int(roi_y))
        w = max(0, int(roi_w))
        h = max(0, int(roi_h))

        x = min(x, img_w)
        y = min(y, img_h)

        w = min(w, img_w - x)
        h = min(h, img_h - y)

        is_valid = w > 0 and h > 0

        if not is_valid and not self._roi_logged_warning:
            self.get_logger().warn(
                f"ROI is invalid after clamping (w={w}, h={h}). "
                "Crop will be skipped; using full image."
            )
            self._roi_logged_warning = True

        return x, y, w, h, is_valid

    # ============================================================
    # Camera callback:
    # - callback chỉ giữ frame mới nhất để không nghẽn ROS subscriber
    # - detect_period_sec <= 0: worker chạy YOLO realtime trên frame mới nhất
    # - detect_period_sec > 0 : chỉ chạy theo chu kỳ cấu hình
    # ============================================================
    def image_callback(self, msg: Image):
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge convert error: {e}")
            return

        if frame_bgr is None:
            self.get_logger().warn("Received empty image.")
            return

        img_h, img_w = frame_bgr.shape[:2]

        # Validate and clamp ROI against this frame's size
        if self.enable_roi_crop:
            rx, ry, rw, rh, valid = self._clamp_roi(
                self.roi_x,
                self.roi_y,
                self.roi_width,
                self.roi_height,
                img_w,
                img_h,
            )
            self._current_roi = (rx, ry, rw, rh) if valid else None
            self._current_roi_valid = valid
        else:
            self._current_roi = None
            self._current_roi_valid = False

        with self.frame_lock:
            self.latest_frame = frame_bgr.copy()
            self.latest_header = msg.header
            self.latest_frame_seq += 1
            self.total_rx_frames += 1
            current_seq = self.latest_frame_seq

        self.new_frame_event.set()
        self.update_preview_fps()

        if self.show_gui:
            if self.last_result_image is not None:
                display = self.last_result_image.copy()
            else:
                display = frame_bgr.copy()

            self.draw_gui_overlay(display)
            cv2.imshow(self.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyWindow(self.window_name)
                self.show_gui = False
                self.get_logger().info(
                    "GUI closed. Press Ctrl+C in terminal to stop node."
                )

        if self.verbose_log:
            self.get_logger().info(f"Received frame seq={current_seq}")

    # ============================================================
    # Worker chạy YOLO ngoài image callback.
    # Luôn lấy frame mới nhất, bỏ qua frame cũ nếu inference chậm hơn camera.
    # ============================================================
    def detect_worker(self):
        while not self.stop_event.is_set():
            self.new_frame_event.wait(timeout=0.5)

            if self.stop_event.is_set():
                break

            with self.frame_lock:
                if self.latest_frame is None or self.latest_header is None:
                    self.new_frame_event.clear()
                    continue

                frame_seq = self.latest_frame_seq
                if frame_seq == self.processed_frame_seq:
                    self.new_frame_event.clear()
                    continue

                frame_bgr = self.latest_frame.copy()
                header = self.latest_header
                roi = self._current_roi

            if self.detect_period_sec > 0.0:
                now = time.time()
                remaining = self.detect_period_sec - (now - self.last_detect_time)
                if remaining > 0.0:
                    time.sleep(min(remaining, 0.05))
                    continue

            self.processed_frame_seq = frame_seq
            self.last_detect_time = time.time()
            self.run_detect_once(frame_bgr, header, roi)

    # ============================================================
    # Run YOLO once
    # ============================================================
    def run_detect_once(self, frame_bgr, header, roi: Optional[Tuple[int, int, int, int]]):
        self.is_detecting = True
        start = time.time()

        try:
            # --- Step 1: ROI crop ---
            if self.enable_roi_crop and roi is not None:
                roi_x, roi_y, roi_w, roi_h = roi
                roi_image = frame_bgr[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
                full_h, full_w = frame_bgr.shape[:2]
            else:
                roi_image = frame_bgr
                roi_x, roi_y, roi_w, roi_h = 0, 0, 0, 0
                full_h, full_w = frame_bgr.shape[:2]

            # --- Step 2: YOLO inference on ROI (or full image) ---
            result = self.model.predict(
                source=roi_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                max_det=self.max_det,
                verbose=False,
            )[0]

            inference_ms = (time.time() - start) * 1000.0
            self.update_detect_fps()

            detections = self.parse_yolo_result(result, roi_x, roi_y)

            mode_name = "realtime" if self.detect_period_sec <= 0.0 else "throttled"

            # --- Step 3: Build annotated image on full image ---
            annotated_full = frame_bgr.copy()

            # Draw ROI rectangle on full image
            if self.enable_roi_crop and roi is not None:
                cv2.rectangle(
                    annotated_full,
                    (roi_x, roi_y),
                    (roi_x + roi_w, roi_y + roi_h),
                    (0, 255, 0),
                    2,
                )
                roi_label = (
                    f"ROI [{roi_x},{roi_y},{roi_w},{roi_h}]"
                )
                cv2.putText(
                    annotated_full,
                    roi_label,
                    (roi_x + 4, roi_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Draw YOLO bboxes (already in full-image coordinates from parse_yolo_result)
            annotated_with_boxes = result.plot()
            if self.enable_roi_crop and roi is not None:
                annotated_full[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = (
                    annotated_with_boxes
                )
            else:
                annotated_full = annotated_with_boxes

            # Draw detection info bar
            cv2.putText(
                annotated_full,
                (
                    f"YOLO {mode_name} | Objects: {len(detections)} | "
                    f"infer: {inference_ms:.1f} ms | count: {self.detect_count + 1}"
                ),
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # --- Step 4: Publish annotated image ---
            if self.publish_annotated:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_full, encoding="bgr8")
                annotated_msg.header = header
                self.annotated_pub.publish(annotated_msg)

            # --- Step 5: Publish ROI debug image ---
            if self.publish_roi_debug_image and self.roi_debug_pub is not None:
                roi_debug_msg = self.bridge.cv2_to_imgmsg(roi_image, encoding="bgr8")
                roi_debug_msg.header = header
                self.roi_debug_pub.publish(roi_debug_msg)

            # --- Step 6: Build and publish JSON payload ---
            payload = {
                "stamp": {
                    "sec": int(header.stamp.sec),
                    "nanosec": int(header.stamp.nanosec),
                },
                "frame_id": header.frame_id,
                "image_width": int(full_w),
                "image_height": int(full_h),
                "mode": mode_name,
                "detect_period_sec": float(self.detect_period_sec),
                "rx_frames": int(self.total_rx_frames),
                "detect_count": int(self.detect_count + 1),
                "preview_fps": float(self.preview_fps),
                "detect_fps": float(self.detect_fps),
                "inference_time_ms": float(inference_ms),
                "num_detections": len(detections),
                "detections": detections,
                "roi_enabled": bool(self.enable_roi_crop and roi is not None),
                "roi_x": int(roi_x),
                "roi_y": int(roi_y),
                "roi_width": int(roi_w),
                "roi_height": int(roi_h),
            }

            json_msg = String()
            json_msg.data = json.dumps(payload, ensure_ascii=False)
            self.detections_pub.publish(json_msg)

            if self.verbose_log:
                self.get_logger().info(json_msg.data)

            self.detect_count += 1
            self.last_result_image = annotated_full.copy()
            self.last_payload = payload

            self.get_logger().info(
                f"Detect #{self.detect_count}: {len(detections)} object(s), "
                f"inference={inference_ms:.1f} ms"
            )

        except Exception as e:
            self.get_logger().error(f"YOLO detect error: {repr(e)}")

        finally:
            self.is_detecting = False

    # ============================================================
    # Parse YOLO result and offset bbox to full image coordinates
    # ============================================================
    def parse_yolo_result(
        self, result: Any, roi_x: int, roi_y: int
    ) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for i in range(len(boxes)):
            try:
                xyxy_roi = boxes.xyxy[i].detach().cpu().numpy().astype(float)
                conf = float(boxes.conf[i].detach().cpu().numpy())
                cls_id = int(boxes.cls[i].detach().cpu().numpy())

                class_name = self.get_class_name(cls_id)

                if len(self.class_filter) > 0:
                    if class_name not in self.class_filter:
                        continue

                # Convert ROI coordinates back to full-image coordinates
                x1_roi, y1_roi, x2_roi, y2_roi = xyxy_roi.tolist()

                x1_full = x1_roi + float(roi_x)
                y1_full = y1_roi + float(roi_y)
                x2_full = x2_roi + float(roi_x)
                y2_full = y2_roi + float(roi_y)

                cx_full = 0.5 * (x1_full + x2_full)
                cy_full = 0.5 * (y1_full + y2_full)
                w_full = x2_full - x1_full
                h_full = y2_full - y1_full

                det = {
                    "id": len(detections),
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_xyxy": {
                        "x1": x1_full,
                        "y1": y1_full,
                        "x2": x2_full,
                        "y2": y2_full,
                    },
                    "bbox_xywh": {
                        "cx": cx_full,
                        "cy": cy_full,
                        "w": w_full,
                        "h": h_full,
                    },
                }

                detections.append(det)

            except Exception as e:
                self.get_logger().warn(f"Parse detection error: {e}")
                continue

        return detections

    def get_class_name(self, cls_id: int) -> str:
        if isinstance(self.names, dict):
            return str(self.names.get(cls_id, cls_id))

        if isinstance(self.names, list):
            if 0 <= cls_id < len(self.names):
                return str(self.names[cls_id])

        return str(cls_id)

    # ============================================================
    # GUI overlay
    # ============================================================
    def draw_gui_overlay(self, img):
        cv2.rectangle(img, (0, 0), (img.shape[1], 80), (30, 30, 30), -1)

        if self.detect_period_sec <= 0.0:
            mode_text = "YOLO REALTIME DETECT"
        else:
            mode_text = f"YOLO THROTTLED DETECT | every {self.detect_period_sec:.1f}s"

        cv2.putText(
            img,
            (
                f"{mode_text} | Preview FPS: {self.preview_fps:.1f} | "
                "press q to close GUI"
            ),
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def update_preview_fps(self):
        self.frame_count += 1
        now = time.time()
        dt = now - self.last_fps_time

        if dt >= 1.0:
            self.preview_fps = self.frame_count / dt
            self.frame_count = 0
            self.last_fps_time = now

    def update_detect_fps(self):
        self.detect_fps_count += 1
        now = time.time()
        dt = now - self.last_detect_fps_time

        if dt >= 1.0:
            self.detect_fps = self.detect_fps_count / dt
            self.detect_fps_count = 0
            self.last_detect_fps_time = now

    def status_timer_callback(self):
        self.get_logger().info(
            "Status | "
            f"rx_frames={self.total_rx_frames}, "
            f"detect_count={self.detect_count}, "
            f"preview_fps={self.preview_fps:.1f}, "
            f"detect_fps={self.detect_fps:.1f}, "
            f"is_detecting={self.is_detecting}, "
            f"image_topic={self.image_topic}, "
            f"roi_enabled={self.enable_roi_crop}, "
            f"roi=({self.roi_x},{self.roi_y},{self.roi_width},{self.roi_height})"
        )

        if self.total_rx_frames == 0:
            image_topics = []
            for topic_name, topic_types in self.get_topic_names_and_types():
                if "sensor_msgs/msg/Image" in topic_types:
                    image_topics.append(topic_name)
            self.get_logger().warn(
                "No image received yet. Available Image topics: "
                f"{sorted(image_topics)}"
            )

    def destroy_node(self):
        self.stop_event.set()
        self.new_frame_event.set()
        if hasattr(self, "detect_thread") and self.detect_thread.is_alive():
            self.detect_thread.join(timeout=2.0)
        if self.show_gui:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = YoloDetectNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(f"[YoloDetectNode] Fatal error: {e}")

    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
