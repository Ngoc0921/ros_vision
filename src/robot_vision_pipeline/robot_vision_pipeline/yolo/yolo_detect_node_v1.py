#!/home/minhquang/venvs/ros_env/bin/python3

import json
import time
from typing import Any, Dict, List
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
        self.get_logger().info("========================================")

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

        with self.frame_lock:
            self.latest_frame = frame_bgr.copy()
            self.latest_header = msg.header
            self.latest_frame_seq += 1
            self.total_rx_frames += 1
            current_seq = self.latest_frame_seq

        self.new_frame_event.set()
        self.update_preview_fps()

        if self.show_gui:
            # GUI hiển thị kết quả detect gần nhất. Nếu chưa có kết quả thì hiển thị frame live.
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
                self.get_logger().info("GUI closed. Press Ctrl+C in terminal to stop node.")

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

            if self.detect_period_sec > 0.0:
                now = time.time()
                remaining = self.detect_period_sec - (now - self.last_detect_time)
                if remaining > 0.0:
                    time.sleep(min(remaining, 0.05))
                    continue

            self.processed_frame_seq = frame_seq
            self.last_detect_time = time.time()
            self.run_detect_once(frame_bgr, header)

    # ============================================================
    # Run YOLO once
    # ============================================================
    def run_detect_once(self, frame_bgr, header):
        self.is_detecting = True
        start = time.time()

        try:
            result = self.model.predict(
                source=frame_bgr,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                max_det=self.max_det,
                verbose=False,
            )[0]

            inference_ms = (time.time() - start) * 1000.0
            self.update_detect_fps()

            detections = self.parse_yolo_result(result)

            mode_name = "realtime" if self.detect_period_sec <= 0.0 else "throttled"

            # Ảnh đã vẽ bbox từ YOLO
            annotated = result.plot()

            # Vẽ thông tin detection
            cv2.putText(
                annotated,
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

            if self.publish_annotated:
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                annotated_msg.header = header
                self.annotated_pub.publish(annotated_msg)

            payload = {
                "stamp": {
                    "sec": int(header.stamp.sec),
                    "nanosec": int(header.stamp.nanosec),
                },
                "frame_id": header.frame_id,
                "image_width": int(frame_bgr.shape[1]),
                "image_height": int(frame_bgr.shape[0]),
                "mode": mode_name,
                "detect_period_sec": float(self.detect_period_sec),
                "rx_frames": int(self.total_rx_frames),
                "detect_count": int(self.detect_count + 1),
                "preview_fps": float(self.preview_fps),
                "detect_fps": float(self.detect_fps),
                "inference_time_ms": float(inference_ms),
                "num_detections": len(detections),
                "detections": detections,
            }

            json_msg = String()
            json_msg.data = json.dumps(payload, ensure_ascii=False)
            self.detections_pub.publish(json_msg)

            if self.verbose_log:
                self.get_logger().info(json_msg.data)

            self.detect_count += 1
            self.last_result_image = annotated.copy()
            self.last_payload = payload

            self.get_logger().info(
                f"Detect #{self.detect_count}: {len(detections)} object(s), inference={inference_ms:.1f} ms"
            )

        except Exception as e:
            self.get_logger().error(f"YOLO detect error: {repr(e)}")

        finally:
            self.is_detecting = False

    # ============================================================
    # Parse YOLO result
    # ============================================================
    def parse_yolo_result(self, result: Any) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []

        if result.boxes is None:
            return detections

        boxes = result.boxes

        for i in range(len(boxes)):
            try:
                xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(float)
                conf = float(boxes.conf[i].detach().cpu().numpy())
                cls_id = int(boxes.cls[i].detach().cpu().numpy())

                class_name = self.get_class_name(cls_id)

                # Nếu có class_filter thì chỉ lấy class nằm trong filter
                if len(self.class_filter) > 0:
                    if class_name not in self.class_filter:
                        continue

                x1, y1, x2, y2 = xyxy.tolist()

                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                w = x2 - x1
                h = y2 - y1

                det = {
                    "id": len(detections),
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_xyxy": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },
                    "bbox_xywh": {
                        "cx": cx,
                        "cy": cy,
                        "w": w,
                        "h": h,
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
        # Nền mờ phía trên để đọc chữ dễ hơn
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
            f"publishers={self.count_publishers(self.image_topic)}"
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
