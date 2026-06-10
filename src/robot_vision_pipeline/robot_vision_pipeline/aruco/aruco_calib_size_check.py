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


class ArucoCalibSizeCheckNode(Node):
    """
    Check camera calibration using an ArUco marker.

    Main idea used in this version:
    1. Detect 4 marker corners in image pixel coordinates.
    2. Use the real marker size + camera intrinsic matrix K + distortion D to solvePnP.
    3. Get marker pose relative to camera: rvec, tvec.
       - tvec[0] = X marker center, in mm
       - tvec[1] = Y marker center, in mm
       - tvec[2] = Z marker center, in mm
    4. Use Z_center = tvec[2] and undistorted normalized corner rays to back-project
       the 4 image corners to camera coordinates:
           X = x_norm * Z_center
           Y = y_norm * Z_center
           Z = Z_center
       This is useful for a quick calibration/scale check when marker is nearly
       parallel to the image plane.
    5. Compute 4 side lengths and one diagonal length from the back-projected corners.
    6. Also compute reprojection error by projecting the ideal marker corners back
       onto the image using rvec/tvec.

    Important note:
    - Because solvePnP uses marker_size_mm as input, the pose-based side length is
      not an independent measurement of marker size.
    - The useful checks are:
        + reprojection_error_px should be small.
        + back_project_using_marker_z side/diagonal should be close to real size
          only when the marker is nearly front-parallel.
        + if known_distance_mm is provided from a ruler, it gives an independent
          distance/size sanity check.
    """

    def __init__(self):
        super().__init__("aruco_calib_size_check_node") # tên node

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_size_mm", 28.0)
        self.declare_parameter("known_distance_mm", 0.0)
        self.declare_parameter("draw_debug", True)
        self.declare_parameter("log_every_n_frames", 15)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.dictionary_name = self.get_parameter("dictionary").value
        self.marker_size_mm = float(self.get_parameter("marker_size_mm").value)
        self.known_distance_mm = float(self.get_parameter("known_distance_mm").value)
        self.draw_debug = bool(self.get_parameter("draw_debug").value)
        self.log_every_n_frames = int(self.get_parameter("log_every_n_frames").value)

        # biến lưu trạng thái 
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None # hệ số méo 
        self.camera_info_valid = False
        self.camera_info_frame = ""
        self.frame_count = 0
        self.printed_camera_info = False

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

        self.pub_json = self.create_publisher(String, "/aruco/calib_size_check_json", 10)
        self.pub_debug = self.create_publisher(Image, "/aruco/image_annotated", 10)

        self.get_logger().info("ArucoCalibSizeCheckNode started")
        self.get_logger().info(f"Image topic        : {self.image_topic}")
        self.get_logger().info(f"CameraInfo topic   : {self.camera_info_topic}")
        self.get_logger().info(f"Dictionary         : {self.dictionary_name}")
        self.get_logger().info(f"Real marker size   : {self.marker_size_mm:.3f} mm")
        self.get_logger().info(
            f"Expected diagonal  : {self.expected_diagonal_mm():.3f} mm"
        )
        if self.known_distance_mm > 0.0:
            self.get_logger().info(f"Known distance     : {self.known_distance_mm:.3f} mm")
        else:
            self.get_logger().warn(
                "known_distance_mm is 0. Independent ruler-distance check is disabled. "
                "The node will still estimate marker pose and use Z=tvec[2] for back-projection."
            )

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

        self.camera_info_frame = msg.header.frame_id
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
                f"CameraInfo | frame={self.camera_info_frame}, "
                f"width={msg.width}, height={msg.height}, "
                f"fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}, "
                f"D={self.dist_coeffs.tolist()}"
            )
            if not self.camera_info_valid:
                self.get_logger().error(
                    "Invalid CameraInfo. K may be all zero or incomplete. "
                    "Calibration check cannot run."
                )

    def expected_diagonal_mm(self):
        return float(self.marker_size_mm * math.sqrt(2.0))

    def marker_object_points_mm(self):
        """
        Marker coordinate convention must match OpenCV ArUco corner order:
        top-left, top-right, bottom-right, bottom-left.
        Unit: millimeter.
        """
        half = self.marker_size_mm / 2.0
        return np.array(
            [
                [-half, half, 0.0],
                [half, half, 0.0],
                [half, -half, 0.0],
                [-half, -half, 0.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def side_lengths_2d(points_2d):
        p = points_2d.reshape(4, 2).astype(np.float64)
        return [
            float(np.linalg.norm(p[1] - p[0])),
            float(np.linalg.norm(p[2] - p[1])),
            float(np.linalg.norm(p[3] - p[2])),
            float(np.linalg.norm(p[0] - p[3])),
        ]

    @staticmethod
    def side_lengths_3d(points_3d):
        p = points_3d.reshape(4, 3).astype(np.float64)
        return [
            float(np.linalg.norm(p[1] - p[0])),
            float(np.linalg.norm(p[2] - p[1])),
            float(np.linalg.norm(p[3] - p[2])),
            float(np.linalg.norm(p[0] - p[3])),
        ]

    @staticmethod
    def diagonal_02(points):
        p = np.asarray(points, dtype=np.float64)
        return float(np.linalg.norm(p[2] - p[0]))

    @staticmethod
    def stats(values):
        arr = np.array(values, dtype=np.float64)
        return {
            "top": float(arr[0]),
            "right": float(arr[1]),
            "bottom": float(arr[2]),
            "left": float(arr[3]),
            "avg": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "spread": float(np.max(arr) - np.min(arr)),
        }

    @staticmethod
    def point_list(points):
        arr = np.asarray(points, dtype=np.float64)
        return [
            {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
            for p in arr.reshape(-1, 3)
        ]

    @staticmethod
    def point_list_2d(points):
        arr = np.asarray(points, dtype=np.float64)
        return [
            {"u": float(p[0]), "v": float(p[1])}
            for p in arr.reshape(-1, 2)
        ]

    def undistorted_normalized_corners(self, pts):
        pts = pts.reshape(4, 1, 2).astype(np.float64)
        norm = cv2.undistortPoints(
            pts,
            self.camera_matrix,
            self.dist_coeffs,
        )
        return norm.reshape(4, 2)

    def solve_marker_pose(self, image_points):
        object_points = self.marker_object_points_mm()
        image_points = image_points.reshape(4, 2).astype(np.float64)

        # IPPE_SQUARE is designed for square planar markers.
        # Fallback to ITERATIVE if IPPE_SQUARE is unavailable or fails.
        flags = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
        success, rvec, tvec = cv2.solvePnP( # hàm tính pose 3d so với camera
            object_points,      # tọa độ thật 4 góc maker
            image_points,       # tọa độ pixel 4 góc maker trên ảnh
            self.camera_matrix, # K
            self.dist_coeffs,   # D
            flags=flags,        # giải thuật 
        )
        if not success and flags != cv2.SOLVEPNP_ITERATIVE:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        return success, rvec, tvec

    def camera_corners_from_pose(self, rvec, tvec): # chuyển 4 góc marker từ hệ marker sang hệ camera.
        object_points = self.marker_object_points_mm()
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        camera_points = (rotation_matrix @ object_points.T).T + tvec.reshape(1, 3)
        return camera_points

    def project_marker_corners(self, rvec, tvec):
        object_points = self.marker_object_points_mm()
        projected_points, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        return projected_points.reshape(4, 2)

    def back_project_corners_using_z(self, norm_pts, z_mm):
        """
        Convert undistorted normalized corners into camera coordinates with one common Z.
        This assumes the marker plane is approximately front-parallel.
        """
        norm_pts = norm_pts.reshape(4, 2).astype(np.float64)
        return np.column_stack(
            [
                norm_pts[:, 0] * z_mm, # X = (u - cx) * Z / fx
                norm_pts[:, 1] * z_mm, # Y = (v - cy) * Z / fy
                np.full(4, z_mm, dtype=np.float64),
            ]
        )

    def image_callback(self, msg):
        #
        self.frame_count += 1
        # Chuẩn bị dữ liệu đầu ra chung cho JSON, sẽ được cập nhật thêm thông tin sau khi xử lý ảnh
        output = {
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "image_frame_id": msg.header.frame_id,
            "camera_info_frame_id": self.camera_info_frame,
            "camera_info_valid": bool(self.camera_info_valid),
            "real_marker_size_mm": float(self.marker_size_mm),
            "expected_diagonal_mm": self.expected_diagonal_mm(),
            "known_distance_mm": float(self.known_distance_mm),
            "status": "unknown",
            "note": (
                "Pose is estimated from marker_size_mm. Z_center=tvec[2] is then used "
                "to back-project 4 image corners and compute 4 sides plus diagonal_02. "
                "Back-projected size is most meaningful when marker is nearly front-parallel."
            ),
            "detected_count": 0,
            "markers": [],
        }

        if self.camera_matrix is None or self.dist_coeffs is None:
            output["status"] = "waiting_for_camera_info"
            self.publish_json(output)
            self.get_logger().warn("Waiting for camera_info...")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        annotated = frame.copy()

        if not self.camera_info_valid:
            output["status"] = "invalid_camera_info"
            self.publish_json(output)
            self.publish_debug(msg, annotated)
            return
        # Chuyển ảnh sang grayscale để phát hiện marker dễ dàng hơn
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Phát hiện marker và lấy về các góc (corners) và ID của marker
        try:
            corners, ids, _rejected = cv2.aruco.detectMarkers(
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

        output["status"] = "ok"
        ids_flat = ids.flatten().astype(int)
        output["detected_count"] = int(len(ids_flat))

        if self.draw_debug:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        # Xử lý từng marker được phát hiện
        for i, marker_id in enumerate(ids_flat):

            marker_id = int(marker_id)
            # Chuẩn bị dữ liệu đầu ra cho marker này
            pts = corners[i].reshape(4, 2).astype(np.float64)
            center = np.mean(pts, axis=0)
            # Tính toán độ dài các cạnh và đường chéo của marker trong ảnh 
            raw_side_px = self.side_lengths_2d(pts)
            raw_diag_px = self.diagonal_02(pts)
            # Chuẩn hóa các góc marker bằng cách loại bỏ méo ảnh và chuyển sang hệ tọa độ chuẩn (normalized coordinates)
            norm_pts = self.undistorted_normalized_corners(pts)
            norm_sides = self.side_lengths_2d(norm_pts)
            norm_diag = self.diagonal_02(norm_pts)

            marker_out = {
                "id": marker_id,
                "center_px": {"x": float(center[0]), "y": float(center[1])},
                "corner_px": self.point_list_2d(pts),
                "raw_side_px": self.stats(raw_side_px),
                "raw_diagonal_02_px": float(raw_diag_px),
                "normalized_side": self.stats(norm_sides),
                "normalized_diagonal_02": float(norm_diag),
                "pose_from_marker_size": None,
                "back_project_using_marker_z": None,
                "measured_size_from_known_distance_mm": None,
                "estimated_distance_from_marker_size_mm": None,
                "warnings": [],
            }

            if self.marker_size_mm <= 0.0:
                marker_out["warnings"].append("marker_size_mm must be > 0 to solve marker pose.")
                output["markers"].append(marker_out)
                continue
            # Giải bài toán PnP để tìm pose của marker so với camera
            success, rvec, tvec = self.solve_marker_pose(pts)
            if not success:
                marker_out["warnings"].append("solvePnP failed for this marker.")
                output["markers"].append(marker_out)
                continue
            # Lấy Z_center từ tvec để back-project các góc marker vào hệ tọa độ camera
            z_center_mm = float(tvec.reshape(3)[2])
            # Tính toán vị trí 4 góc marker trong hệ tọa độ camera dựa trên pose đã giải được
            camera_corners_pose = self.camera_corners_from_pose(rvec, tvec)
            pose_sides_mm = self.side_lengths_3d(camera_corners_pose)
            pose_diag_mm = self.diagonal_02(camera_corners_pose)
            projected_pts = self.project_marker_corners(rvec, tvec)
            reproj_errors = np.linalg.norm(pts - projected_pts, axis=1)

            back_projected_points = self.back_project_corners_using_z(norm_pts, z_center_mm)
            # Tính toán độ dài các cạnh và đường chéo của marker dựa trên phép back-projection sử dụng Z_center từ tvec. So sánh với kích thước thực tế để đánh giá độ chính xác của phép back-projection.
            back_sides_mm = self.side_lengths_3d(back_projected_points)
            back_diag_mm = self.diagonal_02(back_projected_points)
            back_avg_size_mm = float(np.mean(back_sides_mm))
            # Tính toán sai số giữa kích thước back-projected và kích thước thực tế của marker, cả về mm và phần trăm
            back_size_err_mm = back_avg_size_mm - self.marker_size_mm
            back_size_err_percent = 100.0 * back_size_err_mm / self.marker_size_mm
            back_diag_err_mm = back_diag_mm - self.expected_diagonal_mm()
            back_diag_err_percent = 100.0 * back_diag_err_mm / self.expected_diagonal_mm()

            marker_out["pose_from_marker_size"] = {
                "rvec": [float(x) for x in rvec.reshape(3)],
                "tvec_mm": {
                    "x": float(tvec.reshape(3)[0]),
                    "y": float(tvec.reshape(3)[1]),
                    "z": z_center_mm,
                },
                "camera_corners_from_pose_mm": self.point_list(camera_corners_pose),
                "corner_z_mm": [float(p[2]) for p in camera_corners_pose],
                "pose_side_mm": self.stats(pose_sides_mm),
                "pose_diagonal_02_mm": float(pose_diag_mm),
                "projected_corner_px": self.point_list_2d(projected_pts),
                "reprojection_error_px": {
                    "corner_0": float(reproj_errors[0]),
                    "corner_1": float(reproj_errors[1]),
                    "corner_2": float(reproj_errors[2]),
                    "corner_3": float(reproj_errors[3]),
                    "avg": float(np.mean(reproj_errors)),
                    "max": float(np.max(reproj_errors)),
                },
            }

            marker_out["back_project_using_marker_z"] = {
                "z_center_from_tvec_mm": z_center_mm,
                "assumption": "All 4 corners use the same Z=tvec[2]. Best when marker is front-parallel.",
                "camera_corners_mm": self.point_list(back_projected_points),
                "side_mm": self.stats(back_sides_mm),
                "diagonal_02_mm": float(back_diag_mm),
                "expected_side_mm": float(self.marker_size_mm),
                "expected_diagonal_mm": self.expected_diagonal_mm(),
                "side_error_mm": float(back_size_err_mm),
                "side_error_percent": float(back_size_err_percent),
                "diagonal_error_mm": float(back_diag_err_mm),
                "diagonal_error_percent": float(back_diag_err_percent),
            }

            if self.known_distance_mm > 0.0:
                measured_sizes = [s * self.known_distance_mm for s in norm_sides]
                avg_size = float(np.mean(measured_sizes))
                err_mm = avg_size - self.marker_size_mm
                err_percent = 100.0 * err_mm / self.marker_size_mm if self.marker_size_mm > 0 else 0.0
                marker_out["measured_size_from_known_distance_mm"] = {
                    **self.stats(measured_sizes),
                    "diagonal_02_mm": float(norm_diag * self.known_distance_mm),
                    "expected_diagonal_mm": self.expected_diagonal_mm(),
                    "error_mm": float(err_mm),
                    "error_percent": float(err_percent),
                }

            z_estimates = [self.marker_size_mm / max(s, 1e-12) for s in norm_sides]
            avg_z = float(np.mean(z_estimates))
            z_out = {
                **self.stats(z_estimates),
                "from_diagonal_02_mm": float(self.expected_diagonal_mm() / max(norm_diag, 1e-12)),
                "pose_z_center_mm": z_center_mm,
                "error_vs_pose_z_mm": float(avg_z - z_center_mm),
                "error_vs_pose_z_percent": float(100.0 * (avg_z - z_center_mm) / max(abs(z_center_mm), 1e-12)),
            }
            if self.known_distance_mm > 0.0:
                z_err = avg_z - self.known_distance_mm
                z_err_percent = 100.0 * z_err / self.known_distance_mm
                z_out["error_vs_known_distance_mm"] = float(z_err)
                z_out["error_vs_known_distance_percent"] = float(z_err_percent)
            marker_out["estimated_distance_from_marker_size_mm"] = z_out

            # Basic quality warnings.
            spread_ratio = float(
                (max(norm_sides) - min(norm_sides)) / max(np.mean(norm_sides), 1e-12)
            )
            if spread_ratio > 0.10:
                marker_out["warnings"].append(
                    "Large side-length spread. Marker may be tilted, blurred, or corners are inaccurate."
                )
            if center[0] < 80 or center[0] > frame.shape[1] - 80 or center[1] < 80 or center[1] > frame.shape[0] - 80:
                marker_out["warnings"].append(
                    "Marker is near image border. Move it closer to image center for calibration checking."
                )
            if np.mean(reproj_errors) > 2.0:
                marker_out["warnings"].append(
                    "Average reprojection error is > 2 px. Check camera calibration, marker print quality, focus, or corner detection."
                )
            if abs(back_diag_err_percent) > 5.0:
                marker_out["warnings"].append(
                    "Back-projected diagonal error is > 5%. If marker is tilted, this is expected when using one common Z."
                )

            output["markers"].append(marker_out)

            if self.frame_count % max(1, self.log_every_n_frames) == 0:
                reproj_avg = marker_out["pose_from_marker_size"]["reprojection_error_px"]["avg"]
                bp = marker_out["back_project_using_marker_z"]
                self.get_logger().info(
                    f"ID {marker_id}: Zpose={z_center_mm:.1f} mm, "
                    f"back_side={bp['side_mm']['avg']:.2f} mm "
                    f"err={bp['side_error_mm']:+.2f} mm ({bp['side_error_percent']:+.2f}%), "
                    f"diag02={bp['diagonal_02_mm']:.2f} mm "
                    f"diag_err={bp['diagonal_error_mm']:+.2f} mm ({bp['diagonal_error_percent']:+.2f}%), "
                    f"reproj={reproj_avg:.2f} px"
                )

            if self.draw_debug:
                x = int(center[0])
                y = int(center[1])
                bp = marker_out["back_project_using_marker_z"]
                reproj_avg = marker_out["pose_from_marker_size"]["reprojection_error_px"]["avg"]

                # Draw the requested diagonal: corner 0 -> corner 2.
                p0 = tuple(np.round(pts[0]).astype(int))
                p2 = tuple(np.round(pts[2]).astype(int))
                cv2.line(annotated, p0, p2, (255, 0, 255), 2, cv2.LINE_AA)

                # Draw projected corners from solvePnP for visual reprojection check.
                for pp in projected_pts:
                    cv2.circle(
                        annotated,
                        tuple(np.round(pp).astype(int)),
                        4,
                        (0, 255, 0),
                        -1,
                        cv2.LINE_AA,
                    )

                # Draw pose axes if available.
                if hasattr(cv2, "drawFrameAxes"):
                    try:
                        cv2.drawFrameAxes(
                            annotated,
                            self.camera_matrix,
                            self.dist_coeffs,
                            rvec,
                            tvec,
                            self.marker_size_mm * 0.5,
                        )
                    except Exception:
                        pass

                label1 = f"ID {marker_id} Z={z_center_mm:.0f}mm reproj={reproj_avg:.2f}px"
                label2 = f"side={bp['side_mm']['avg']:.1f}mm diag={bp['diagonal_02_mm']:.1f}mm"
                label3 = f"err_side={bp['side_error_percent']:+.1f}% err_diag={bp['diagonal_error_percent']:+.1f}%"

                cv2.putText(
                    annotated,
                    label1,
                    (x + 8, y - 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    label2,
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    label3,
                    (x + 8, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 0),
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
    node = ArucoCalibSizeCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()