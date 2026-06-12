#!/usr/bin/env python3
"""Adapt YOLO JSON detections into BoxDetection messages by class.

Subscribes to /vision/yolo/detections_json (YOLO JSON output).
Reads aligned depth image to enrich each detection with distance info.
Publishes BoxDetection messages:
  /vision/wood_detection  → class_name == "wood"
  /vision/box_detection  → class_name == "box"
"""

from __future__ import annotations

import json
import threading
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String

from robot_vision_pipeline.depth_utils import depth_at_pixel, median_depth_meters
from robot_vision_pipeline_msgs.msg import BoxDetection


class YoloJsonToBoxDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_json_to_box_detection_node")

        self.declare_parameter("detections_json_topic", "/vision/yolo/detections_json")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("depth_roi_half_size", 5)
        self.declare_parameter("fake_depth_m", 0.55)
        self.declare_parameter("use_fake_depth", False)

        self._bridge = CvBridge()
        self._depth_lock = threading.Lock()
        self._latest_depth_msg: Optional[Image] = None
        self._warned_missing_depth = False

        detections_topic = str(self.get_parameter("detections_json_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        self._depth_roi_half_size = int(self.get_parameter("depth_roi_half_size").value)
        self._fake_depth_m = float(self.get_parameter("fake_depth_m").value)
        self._use_fake_depth = bool(self.get_parameter("use_fake_depth").value)

        det_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._pub_wood = self.create_publisher(BoxDetection, "/vision/wood_detection", det_qos)
        self._pub_box = self.create_publisher(BoxDetection, "/vision/box_detection", det_qos)

        self.create_subscription(
            String,
            detections_topic,
            self._on_yolo_json,
            det_qos,
        )
        self.create_subscription(
            Image,
            depth_topic,
            self._on_depth_image,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            "YOLO JSON adapter started | "
            f"json={detections_topic}, depth={depth_topic}, "
            f"fake_depth={self._fake_depth_m} m, use_fake_depth={self._use_fake_depth}"
        )

    def _on_depth_image(self, msg: Image) -> None:
        with self._depth_lock:
            self._latest_depth_msg = msg

    def _depth_for_center(
        self, cx: int, cy: int
    ) -> tuple[int, int, str, float]:
        with self._depth_lock:
            depth_msg = self._latest_depth_msg

        if depth_msg is None:
            if self._use_fake_depth:
                if not self._warned_missing_depth:
                    self.get_logger().warn(
                        "No depth image received yet; using fake_depth_m because static-image mode is enabled."
                    )
                    self._warned_missing_depth = True
            elif not self._warned_missing_depth:
                self.get_logger().warn(
                    "No depth image received yet; detections will publish distance_m=-1.0 until depth arrives."
                )
                self._warned_missing_depth = True
            return self._fake_depth_values()

        try:
            depth_cv = self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except CvBridgeError as exc:
            self.get_logger().warn(f"Depth cv_bridge error: {exc}")
            return self._fake_depth_values()

        if (
            depth_cv is None
            or not isinstance(depth_cv, np.ndarray)
            or depth_cv.ndim != 2
        ):
            self.get_logger().warn(
                f"Invalid depth image shape/type for encoding={depth_msg.encoding}; cannot read depth."
            )
            return self._fake_depth_values()

        center_raw, center_dist, _ = depth_at_pixel(
            depth_cv, cx, cy, depth_msg.encoding
        )
        dist_m, roi_raw = median_depth_meters(
            depth_cv,
            cx,
            cy,
            self._depth_roi_half_size,
            depth_msg.encoding,
        )

        if center_raw is None:
            center_raw_mm = 0
        elif depth_msg.encoding == "32FC1":
            center_raw_mm = int(round(float(center_raw) * 1000.0))
        else:
            center_raw_mm = int(center_raw)

        if roi_raw is None:
            roi_raw_mm = 0
        elif depth_msg.encoding == "32FC1":
            roi_raw_mm = int(round(float(roi_raw) * 1000.0))
        else:
            roi_raw_mm = int(roi_raw)

        if dist_m < 0.0 and center_dist is not None:
            dist_m = float(center_dist)

        if dist_m <= 0.0:
            self.get_logger().warn(
                f"Invalid depth near bbox center ({cx}, {cy}); encoding={depth_msg.encoding}."
            )

        return (
            max(0, min(center_raw_mm, 65535)),
            max(0, min(roi_raw_mm, 65535)),
            depth_msg.encoding,
            float(dist_m),
        )

    def _fake_depth_values(self) -> tuple[int, int, str, float]:
        if self._use_fake_depth:
            mm = int(round(self._fake_depth_m * 1000.0))
            return mm, mm, "16UC1", float(self._fake_depth_m)
        return 0, 0, "16UC1", -1.0

    def _on_yolo_json(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Invalid YOLO JSON: {exc}")
            return

        header = Header()
        stamp = payload.get("stamp", {})
        header.stamp.sec = int(stamp.get("sec", 0))
        header.stamp.nanosec = int(stamp.get("nanosec", 0))
        header.frame_id = str(payload.get("frame_id", ""))

        detections = payload.get("detections", [])
        if not isinstance(detections, list):
            self.get_logger().warn("YOLO JSON field 'detections' is not a list; skipping payload.")
            return
        if not detections:
            self.get_logger().info("YOLO JSON contains no detections.")
            return

        for det_data in detections:
            try:
                class_name = str(det_data.get("class_name", "")).strip().lower()
                object_id = int(det_data.get("id", 0))
                confidence = float(det_data.get("confidence", 0.0))
                bbox = det_data.get("bbox_xyxy", {})
                xywh = det_data.get("bbox_xywh", {})

                x_min = int(round(float(bbox.get("x1", 0.0))))
                y_min = int(round(float(bbox.get("y1", 0.0))))
                x_max = int(round(float(bbox.get("x2", 0.0))))
                y_max = int(round(float(bbox.get("y2", 0.0))))

                center_x = int(round(float(xywh.get("cx", (x_min + x_max) * 0.5))))
                center_y = int(round(float(xywh.get("cy", (y_min + y_max) * 0.5))))
                width_px = int(round(float(xywh.get("w", x_max - x_min))))
                height_px = int(round(float(xywh.get("h", y_max - y_min))))
            except Exception as exc:
                self.get_logger().warn(f"Skip malformed detection: {exc}")
                continue

            center_raw, roi_raw, depth_encoding, dist_m = self._depth_for_center(
                center_x, center_y
            )

            det = BoxDetection()
            det.header = header
            det.class_name = class_name
            det.confidence = confidence
            det.object_id = object_id
            det.x_min = x_min
            det.y_min = y_min
            det.x_max = x_max
            det.y_max = y_max
            det.center_x = center_x
            det.center_y = center_y
            det.width_px = width_px
            det.height_px = height_px
            det.center_raw_depth = center_raw
            det.roi_median_raw_depth = roi_raw
            det.depth_encoding = depth_encoding
            det.distance_m = dist_m

            if class_name == "wood":
                self._pub_wood.publish(det)
            elif class_name == "box":
                self._pub_box.publish(det)
            else:
                self.get_logger().debug(
                    f"Skipping class '{class_name}' because only wood/box outputs are published."
                )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloJsonToBoxDetectionNode()
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
