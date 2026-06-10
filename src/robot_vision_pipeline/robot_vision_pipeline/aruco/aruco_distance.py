#!/usr/bin/env python3

import json
import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge


class ArucoDistanceToId0Node(Node):
    def __init__(self):
        super().__init__("aruco_distance_to_id0_node")

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_size", 0.028)
        self.declare_parameter("reference_id", 0)
        self.declare_parameter("target_id", -1)
        self.declare_parameter("draw_debug", True)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.dictionary_name = self.get_parameter("dictionary").value
        self.marker_size = float(self.get_parameter("marker_size").value)
        self.reference_id = int(self.get_parameter("reference_id").value)
        self.target_id = int(self.get_parameter("target_id").value)
        self.draw_debug = bool(self.get_parameter("draw_debug").value)

        self.bridge = CvBridge()

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = ""

        self.aruco_dict = self.get_aruco_dictionary(self.dictionary_name)

        try:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters()

        self.sub_info = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.sub_img = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.pub_json = self.create_publisher(
            String,
            "/aruco/distance_to_id0_json",
            10,
        )

        self.pub_debug = self.create_publisher(
            Image,
            "/aruco/image_annotated",
            10,
        )

        self.get_logger().info("ArucoDistanceToId0Node started")
        self.get_logger().info(f"Image topic      : {self.image_topic}")
        self.get_logger().info(f"CameraInfo topic : {self.camera_info_topic}")
        self.get_logger().info(f"Dictionary       : {self.dictionary_name}")
        self.get_logger().info(f"Marker size      : {self.marker_size} m")
        self.get_logger().info(f"Reference ID     : {self.reference_id}")

        if self.target_id < 0:
            self.get_logger().info("Target ID        : all markers")
        else:
            self.get_logger().info(f"Target ID        : {self.target_id}")

    def get_aruco_dictionary(self, dictionary_name):
        if not hasattr(cv2.aruco, dictionary_name):
            self.get_logger().warn(
                f"Unknown dictionary {dictionary_name}, fallback to DICT_4X4_50"
            )
            dictionary_name = "DICT_4X4_50"

        return cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, dictionary_name)
        )

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        if len(msg.d) > 0:
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        else:
            self.dist_coeffs = np.zeros((5,), dtype=np.float64)

        self.camera_frame = msg.header.frame_id

    def estimate_pose_solvepnp(self, corners):
        s = self.marker_size / 2.0

        object_points = np.array(
            [
                [-s,  s, 0.0],
                [ s,  s, 0.0],
                [ s, -s, 0.0],
                [-s, -s, 0.0],
            ],
            dtype=np.float32,
        )

        image_points = corners.reshape(4, 2).astype(np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not ok:
            return None, None

        return rvec, tvec

    @staticmethod
    def calc_distance(pos, ref_pos):
        dx = float(pos[0] - ref_pos[0])
        dy = float(pos[1] - ref_pos[1])
        dz = float(pos[2] - ref_pos[2])

        distance_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        distance_xy = math.sqrt(dx * dx + dy * dy)

        return dx, dy, dz, distance_3d, distance_xy

    def image_callback(self, msg):
        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn("Waiting for camera_info...")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated = frame.copy()

        try:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params,
            )
        except Exception as e:
            self.get_logger().error(f"aruco detect error: {e}")
            return

        output = {
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "image_frame_id": msg.header.frame_id,
            "camera_info_frame_id": self.camera_frame,
            "reference_id": self.reference_id,
            "target_id": self.target_id,
            "marker_size_m": self.marker_size,
            "status": "no_markers_detected",
            "detected_ids": [],
            "reference_position_camera_m": None,
            "distances": [],
        }

        if ids is None or len(ids) == 0:
            self.publish_json(output)
            self.publish_debug(msg, annotated)
            return

        ids_flat = ids.flatten().astype(int)
        output["detected_ids"] = [int(x) for x in ids_flat]

        if self.draw_debug:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        marker_positions = {}
        marker_centers_px = {}

        for i, marker_id in enumerate(ids_flat):
            marker_id = int(marker_id)

            pts = corners[i].reshape(4, 2)
            center_px = np.mean(pts, axis=0)
            marker_centers_px[marker_id] = center_px

            rvec, tvec = self.estimate_pose_solvepnp(corners[i])

            if rvec is None or tvec is None:
                continue

            pos = tvec.reshape(3).astype(np.float64)
            marker_positions[marker_id] = pos

            if self.draw_debug:
                cv2.drawFrameAxes(
                    annotated,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    self.marker_size * 0.5,
                )

        if self.reference_id not in marker_positions:
            output["status"] = f"reference_marker_id_{self.reference_id}_not_found"
            self.get_logger().warn(
                f"Không thấy marker mốc ID {self.reference_id}. "
                f"Detected IDs: {output['detected_ids']}"
            )
            self.publish_json(output)
            self.publish_debug(msg, annotated)
            return

        ref_pos = marker_positions[self.reference_id]

        output["reference_position_camera_m"] = {
            "x": float(ref_pos[0]),
            "y": float(ref_pos[1]),
            "z": float(ref_pos[2]),
        }

        if self.target_id >= 0:
            target_ids = [self.target_id]
        else:
            target_ids = sorted(
                [mid for mid in marker_positions.keys() if mid != self.reference_id]
            )

        valid_target_found = False

        for marker_id in target_ids:
            if marker_id == self.reference_id:
                continue

            if marker_id not in marker_positions:
                continue

            valid_target_found = True

            pos = marker_positions[marker_id]
            dx, dy, dz, distance_3d, distance_xy = self.calc_distance(pos, ref_pos)

            item = {
                "id": int(marker_id),
                "position_camera_m": {
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                },
                "relative_to_id0_m": {
                    "dx": dx,
                    "dy": dy,
                    "dz": dz,
                },
                "distance_3d_m": float(distance_3d),
                "distance_xy_m": float(distance_xy),
                "distance_3d_mm": float(distance_3d * 1000.0),
                "distance_xy_mm": float(distance_xy * 1000.0),
            }

            output["distances"].append(item)

            self.get_logger().info(
                f"ID {marker_id} so với ID {self.reference_id}: "
                f"d3D = {distance_3d * 1000.0:.1f} mm, "
                f"dXY = {distance_xy * 1000.0:.1f} mm, "
                f"dx = {dx:.4f} m, dy = {dy:.4f} m, dz = {dz:.4f} m"
            )

            if self.draw_debug:
                c0 = marker_centers_px[self.reference_id].astype(int)
                c1 = marker_centers_px[marker_id].astype(int)

                cv2.line(
                    annotated,
                    tuple(c0),
                    tuple(c1),
                    (0, 255, 255),
                    2,
                )

                mid = ((c0 + c1) // 2).astype(int)

                cv2.putText(
                    annotated,
                    f"{distance_3d * 1000.0:.1f} mm",
                    (int(mid[0]) + 5, int(mid[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if not valid_target_found:
            if self.target_id >= 0:
                output["status"] = f"target_marker_id_{self.target_id}_not_found"
            else:
                output["status"] = "only_reference_marker_found"
        else:
            output["status"] = "ok"

        if self.draw_debug:
            for marker_id, center_px in marker_centers_px.items():
                x = int(center_px[0])
                y = int(center_px[1])

                if marker_id == self.reference_id:
                    label = f"REF ID {marker_id}"
                    color = (0, 255, 0)
                else:
                    label = f"ID {marker_id}"
                    color = (0, 255, 255)

                cv2.putText(
                    annotated,
                    label,
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        self.publish_json(output)
        self.publish_debug(msg, annotated)

    def publish_json(self, output):
        msg = String()
        msg.data = json.dumps(output, ensure_ascii=False)
        self.pub_json.publish(msg)

    def publish_debug(self, img_msg, annotated):
        if not self.draw_debug:
            return

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            debug_msg.header = img_msg.header
            self.pub_debug.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"publish debug image error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDistanceToId0Node()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()