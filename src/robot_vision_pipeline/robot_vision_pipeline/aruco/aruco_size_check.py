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


class ArucoSizeCheckNode(Node):
    """
    Detect ArUco markers and estimate marker side length in millimeters.

    Purpose:
    - Check whether webcam CameraInfo / calibration is reasonable.
    - Compare measured side length with the real printed marker size.

    Note:
    - With a single monocular camera, absolute millimeter scale needs a known
      real marker size. Therefore marker_size_mm must be set to the actual
      black-square side length of the printed ArUco marker.
    - If camera_info K is wrong, pose/depth/size will be unreliable.
    """

    def __init__(self):
        super().__init__("aruco_size_check_node")

        self.declare_parameter("image_topic", "/webcam/image_raw")
        self.declare_parameter("camera_info_topic", "/webcam/camera_info")
        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_size_mm", 28.0)
        self.declare_parameter("draw_debug", True)
        self.declare_parameter("log_every_n_frames", 15)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.dictionary_name = self.get_parameter("dictionary").value
        self.marker_size_mm = float(self.get_parameter("marker_size_mm").value)
        self.marker_size_m = self.marker_size_mm / 1000.0
        self.draw_debug = bool(self.get_parameter("draw_debug").value)
        self.log_every_n_frames = int(self.get_parameter("log_every_n_frames").value)

        self.bridge = CvBridge()

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = ""
        self.camera_info_valid = False
        self.printed_camera_info = False
        self.frame_count = 0

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
            "/aruco/size_check_json",
            10,
        )

        self.pub_debug = self.create_publisher(
            Image,
            "/aruco/image_annotated",
            10,
        )

        self.get_logger().info("ArucoSizeCheckNode started")
        self.get_logger().info(f"Image topic      : {self.image_topic}")
        self.get_logger().info(f"CameraInfo topic : {self.camera_info_topic}")
        self.get_logger().info(f"Dictionary       : {self.dictionary_name}")
        self.get_logger().info(f"Real marker size : {self.marker_size_mm:.3f} mm")

    def get_aruco_dictionary(self, dictionary_name):
        if not hasattr(cv2.aruco, dictionary_name):
            self.get_logger().warn(
                f"Unknown dictionary {dictionary_name}, fallback to DICT_4X4_50"
            )
            dictionary_name = "DICT_4X4_50"

        return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        if len(msg.d) > 0:
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        else:
            self.dist_coeffs = np.zeros((5,), dtype=np.float64)

        self.camera_frame = msg.header.frame_id

        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])

        self.camera_info_valid = (
            msg.width > 0
            and msg.height > 0
            and fx > 1.0
            and fy > 1.0
            and abs(float(self.camera_matrix[2, 2]) - 1.0) < 1e-6
        )

        if not self.printed_camera_info:
            self.printed_camera_info = True
            self.get_logger().info(
                "CameraInfo received | "
                f"frame={self.camera_frame}, width={msg.width}, height={msg.height}, "
                f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}"
            )

            if not self.camera_info_valid:
                self.get_logger().error(
                    "CameraInfo is invalid. K may be all zero or incomplete. "
                    "You must calibrate the webcam before checking ArUco size."
                )

    def estimate_pose_solvepnp(self, corners):
        s = self.marker_size_m / 2.0

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
            return None, None, object_points

        return rvec, tvec, object_points

    @staticmethod
    def side_lengths_px(pts):
        pts = pts.reshape(4, 2).astype(np.float64)
        return [
            float(np.linalg.norm(pts[1] - pts[0])),
            float(np.linalg.norm(pts[2] - pts[1])),
            float(np.linalg.norm(pts[3] - pts[2])),
            float(np.linalg.norm(pts[0] - pts[3])),
        ]

    def side_lengths_mm_on_estimated_plane(self, pts, rvec, tvec):
        """
        Convert detected image corners into camera 3D points by intersecting
        each pixel ray with the marker plane estimated from solvePnP.

        This is useful for observing calibration/detection consistency.
        """
        pts = pts.reshape(4, 2).astype(np.float64)

        undistorted = cv2.undistortPoints(
            pts.reshape(-1, 1, 2),
            self.camera_matrix,
            self.dist_coeffs,
        ).reshape(-1, 2)

        rays = np.column_stack(
            [
                undistorted[:, 0],
                undistorted[:, 1],
                np.ones((4,), dtype=np.float64),
            ]
        )

        R, _ = cv2.Rodrigues(rvec)
        n = R[:, 2].reshape(3).astype(np.float64)
        p0 = tvec.reshape(3).astype(np.float64)

        points_3d = []
        denom_eps = 1e-9

        for ray in rays:
            denom = float(np.dot(n, ray))
            if abs(denom) < denom_eps:
                return None, None
            scale = float(np.dot(n, p0) / denom)
            points_3d.append(scale * ray)

        points_3d = np.array(points_3d, dtype=np.float64)

        sides_m = [
            float(np.linalg.norm(points_3d[1] - points_3d[0])),
            float(np.linalg.norm(points_3d[2] - points_3d[1])),
            float(np.linalg.norm(points_3d[3] - points_3d[2])),
            float(np.linalg.norm(points_3d[0] - points_3d[3])),
        ]

        sides_mm = [x * 1000.0 for x in sides_m]
        return sides_mm, points_3d

    def reprojection_error_px(self, object_points, rvec, tvec, detected_pts):
        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )

        projected = projected.reshape(4, 2).astype(np.float64)
        detected_pts = detected_pts.reshape(4, 2).astype(np.float64)

        errors = np.linalg.norm(projected - detected_pts, axis=1)
        return [float(x) for x in errors], float(np.mean(errors)), projected

    def front_parallel_size_mm_estimate(self, pts, z_m):
        """
        Simple pinhole approximation. Works best when marker is nearly parallel
        to image plane. This is helpful as a quick visual check.
        """
        pts = pts.reshape(4, 2).astype(np.float64)
        fx = float(self.camera_matrix[0, 0])
        fy = float(self.camera_matrix[1, 1])
        cx = float(self.camera_matrix[0, 2])
        cy = float(self.camera_matrix[1, 2])

        points = []
        for u, v in pts:
            x = (u - cx) * z_m / fx
            y = (v - cy) * z_m / fy
            points.append([x, y, z_m])

        points = np.array(points, dtype=np.float64)

        sides_mm = [
            float(np.linalg.norm(points[1] - points[0]) * 1000.0),
            float(np.linalg.norm(points[2] - points[1]) * 1000.0),
            float(np.linalg.norm(points[3] - points[2]) * 1000.0),
            float(np.linalg.norm(points[0] - points[3]) * 1000.0),
        ]

        return sides_mm

    def image_callback(self, msg):
        self.frame_count += 1

        if self.camera_matrix is None or self.dist_coeffs is None:
            self.get_logger().warn("Waiting for camera_info...")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        annotated = frame.copy()

        output = {
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "image_frame_id": msg.header.frame_id,
            "camera_info_frame_id": self.camera_frame,
            "camera_info_valid": bool(self.camera_info_valid),
            "real_marker_size_mm": float(self.marker_size_mm),
            "status": "unknown",
            "detected_count": 0,
            "markers": [],
        }

        if not self.camera_info_valid:
            output["status"] = "invalid_camera_info"
            self.publish_json(output)
            self.publish_debug(msg, annotated)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params,
            )
        except Exception as e:
            self.get_logger().error(f"aruco detect error: {e}")
            return

        if ids is None or len(ids) == 0:
            output["status"] = "no_markers_detected"
            self.publish_json(output)
            self.publish_debug(msg, annotated)
            return

        ids_flat = ids.flatten().astype(int)
        output["status"] = "ok"
        output["detected_count"] = int(len(ids_flat))

        if self.draw_debug:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

        for i, marker_id in enumerate(ids_flat):
            marker_id = int(marker_id)
            pts = corners[i].reshape(4, 2)
            center_px = np.mean(pts, axis=0)

            side_px = self.side_lengths_px(pts)
            avg_side_px = float(np.mean(side_px))

            rvec, tvec, object_points = self.estimate_pose_solvepnp(corners[i])
            if rvec is None or tvec is None:
                continue

            z_mm = float(tvec.reshape(3)[2] * 1000.0)

            side_mm_plane, _points_3d = self.side_lengths_mm_on_estimated_plane(
                pts,
                rvec,
                tvec,
            )

            if side_mm_plane is None:
                continue

            avg_size_mm = float(np.mean(side_mm_plane))
            min_size_mm = float(np.min(side_mm_plane))
            max_size_mm = float(np.max(side_mm_plane))
            error_mm = avg_size_mm - self.marker_size_mm
            error_percent = 100.0 * error_mm / self.marker_size_mm if self.marker_size_mm > 0 else 0.0

            side_mm_front_parallel = self.front_parallel_size_mm_estimate(
                pts,
                z_mm / 1000.0,
            )
            avg_front_parallel_mm = float(np.mean(side_mm_front_parallel))

            reproj_errors, reproj_mean, projected_pts = self.reprojection_error_px(
                object_points,
                rvec,
                tvec,
                pts,
            )

            marker_out = {
                "id": marker_id,
                "center_px": {
                    "x": float(center_px[0]),
                    "y": float(center_px[1]),
                },
                "distance_z_mm": z_mm,
                "side_px": {
                    "top": side_px[0],
                    "right": side_px[1],
                    "bottom": side_px[2],
                    "left": side_px[3],
                    "avg": avg_side_px,
                },
                "measured_size_mm_plane": {
                    "top": float(side_mm_plane[0]),
                    "right": float(side_mm_plane[1]),
                    "bottom": float(side_mm_plane[2]),
                    "left": float(side_mm_plane[3]),
                    "avg": avg_size_mm,
                    "min": min_size_mm,
                    "max": max_size_mm,
                },
                "front_parallel_estimate_mm": {
                    "top": float(side_mm_front_parallel[0]),
                    "right": float(side_mm_front_parallel[1]),
                    "bottom": float(side_mm_front_parallel[2]),
                    "left": float(side_mm_front_parallel[3]),
                    "avg": avg_front_parallel_mm,
                    "note": "Approximate. Best when marker is nearly parallel to camera.",
                },
                "real_marker_size_mm": float(self.marker_size_mm),
                "size_error_mm": float(error_mm),
                "size_error_percent": float(error_percent),
                "reprojection_error_px": {
                    "corner_errors": reproj_errors,
                    "mean": reproj_mean,
                },
            }

            output["markers"].append(marker_out)

            if self.frame_count % max(1, self.log_every_n_frames) == 0:
                self.get_logger().info(
                    f"ID {marker_id}: measured={avg_size_mm:.2f} mm, "
                    f"real={self.marker_size_mm:.2f} mm, "
                    f"err={error_mm:+.2f} mm ({error_percent:+.2f}%), "
                    f"z={z_mm:.1f} mm, reproj={reproj_mean:.2f} px"
                )

            if self.draw_debug:
                cv2.drawFrameAxes(
                    annotated,
                    self.camera_matrix,
                    self.dist_coeffs,
                    rvec,
                    tvec,
                    self.marker_size_m * 0.5,
                )

                x = int(center_px[0])
                y = int(center_px[1])

                label1 = f"ID {marker_id} size={avg_size_mm:.1f}mm"
                label2 = f"err={error_mm:+.1f}mm reproj={reproj_mean:.1f}px"
                label3 = f"Z={z_mm:.0f}mm px={avg_side_px:.1f}"

                cv2.putText(
                    annotated,
                    label1,
                    (x + 8, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    label2,
                    (x + 8, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    label3,
                    (x + 8, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 255, 255),
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
    node = ArucoSizeCheckNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
