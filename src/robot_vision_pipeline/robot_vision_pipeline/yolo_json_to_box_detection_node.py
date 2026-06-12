#!/usr/bin/env python3
"""Adapt YOLO JSON detections into ObjectDetection topics.

This keeps yolo_detect_node_v1.py unchanged. The stable YOLO node publishes
std_msgs/String JSON on /vision/yolo/detections_json; this adapter converts each
detection into robot_vision_pipeline_msgs/ObjectDetection messages for the GP7-style
mapping layer.
"""

from __future__ import annotations

import json
import threading
from typing import Optional

import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String

from robot_vision_pipeline.depth_utils import depth_at_pixel, median_depth_meters
from robot_vision_pipeline_msgs.msg import ObjectDetection


class YoloJsonToBoxDetectionNode(Node):
    """Convert YOLO JSON payloads into target/box detection messages."""

    def __init__(self) -> None:
        super().__init__("yolo_json_to_box_detection_node")

        self.declare_parameter("detections_json_topic", "/vision/yolo/detections_json")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("target_classes", "target")
        self.declare_parameter("depth_roi_half_size", 5)

        self._bridge = CvBridge()
        self._depth_lock = threading.Lock()
        self._latest_depth_msg: Optional[Image] = None

        detections_topic = str(self.get_parameter("detections_json_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        target_classes_raw = str(self.get_parameter("target_classes").value)
        self._target_classes = {
            x.strip().lower() for x in target_classes_raw.split(",") if x.strip()
        }
        self._depth_roi_half_size = int(self.get_parameter("depth_roi_half_size").value)

        self._pub_target = self.create_publisher(ObjectDetection, "/vision/target_detection", 10)
        self._pub_wood = self.create_publisher(ObjectDetection, "/vision/wood_detection", 10)

        self.create_subscription(
            String,
            detections_topic,
            self._on_yolo_json,
            10,
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
            f"target_classes={sorted(self._target_classes)}"
        )

    def _on_depth_image(self, msg: Image) -> None:
        with self._depth_lock:
            self._latest_depth_msg = msg

    def _depth_for_center(self, cx: int, cy: int) -> tuple[int, int, str, float]:
        with self._depth_lock:
            depth_msg = self._latest_depth_msg

        if depth_msg is None:
            return 0, 0, "", -1.0

        try:
            depth_cv = self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
        except CvBridgeError as exc:
            self.get_logger().warn(f"Depth cv_bridge error: {exc}")
            return 0, 0, depth_msg.encoding, -1.0

        if depth_cv is None or not isinstance(depth_cv, np.ndarray) or depth_cv.ndim != 2:
            return 0, 0, depth_msg.encoding, -1.0

        center_raw, center_dist, _ = depth_at_pixel(depth_cv, cx, cy, depth_msg.encoding)
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

        return (
            max(0, min(center_raw_mm, 65535)),
            max(0, min(roi_raw_mm, 65535)),
            depth_msg.encoding,
            float(dist_m),
        )

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
            return

        for det_data in detections:
            try:
                class_name = str(det_data.get("class_name", "")).strip()
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
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Skip malformed detection: {exc}")
                continue

            center_raw, roi_raw, depth_encoding, dist_m = self._depth_for_center(
                center_x, center_y
            )

            det = ObjectDetection()
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

            if class_name.lower() in self._target_classes:
                self._pub_target.publish(det)
            else:
                self._pub_wood.publish(det)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloJsonToBoxDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
