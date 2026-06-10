#!/usr/bin/env python3

import json
import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from robot_vision_pipeline.msg import ArucoPose, ArucoPoseArray  # type: ignore msg custom
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from cv_bridge import CvBridge


class ArucoDetectNode(Node):
    """
    Node detect ArUco + lấy tọa độ 3D bằng RGB pixel + aligned depth.

    Điểm quan trọng:
    - ArUco chỉ dùng để tìm tâm pixel của vật và hướng yaw trên ảnh RGB.
    - X/Y/Z không lấy từ solvePnP/tvec nữa.
    - Depth đã align với RGB, nên ảnh depth chỉ dùng để lấy Z tại đúng pixel RGB (u, v).
    - X/Y tính bằng phép chiếu ngược từ tâm RGB pixel + Z + RGB camera_info:
        X = (u_rgb - cx_rgb) * Z / fx_rgb
        Y = (v_rgb - cy_rgb) * Z / fy_rgb
        Z = depth tại pixel RGB đã align
    Vì vậy khai báo sai marker_size sẽ KHÔNG làm sai tọa độ X/Y/Z.
    """

    def __init__(self):
        super().__init__("aruco_detect_node")

        # Ảnh RGB chỉ dùng để detect ArUco.
        self.declare_parameter("image_topic", "/astra_sim/rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/astra_sim/rgb/camera_info")

        # Ảnh depth đã align với RGB. Depth chỉ dùng để lấy Z tại pixel RGB.
        # depth_camera_info vẫn subscribe để debug, nhưng X/Y dùng RGB camera_info.
        self.declare_parameter("depth_image_topic", "/astra_sim/depth/image_raw")
        self.declare_parameter("depth_camera_info_topic", "/astra_sim/depth/camera_info")

        self.declare_parameter("dictionary", "DICT_4X4_50")

        # Giữ lại để tương thích launch cũ, nhưng KHÔNG dùng để tính X/Y/Z.
        self.declare_parameter("marker_size", 0.028)

        self.declare_parameter("enable_pose", True)
        self.declare_parameter("draw_debug", True)

        # TF theo cây hiện tại của bạn:
        # world -> base_link -> astra_base -> astra_optical
        # Bước 1: astra_optical  -> astra_base
        # Bước 2: astra_base     -> base_link
        self.declare_parameter("camera_base_frame", "astra_base")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("log_every_n_frames", 10)

        # Để rỗng thì dùng frame_id từ RGB camera_info / RGB image.
        # Vì X/Y được tính bằng RGB camera_info nên pose_cam thuộc RGB optical frame.
        # Chỉ set nếu bạn chắc chắn frame TF nguồn phải bị ép sang frame khác.
        self.declare_parameter("camera_frame", "")

        # Nếu depth là 16UC1 thì thường đơn vị là mm, nhân 0.001 để ra mét.
        # Nếu depth là 32FC1 thì thường đã là mét, không dùng hệ số này.
        self.declare_parameter("depth_unit_scaling", 0.001)
        self.declare_parameter("depth_window", 5)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 5.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.depth_image_topic = self.get_parameter("depth_image_topic").value
        self.depth_camera_info_topic = self.get_parameter("depth_camera_info_topic").value
        self.dictionary_name = self.get_parameter("dictionary").value
        self.marker_size = float(self.get_parameter("marker_size").value)
        self.enable_pose = bool(self.get_parameter("enable_pose").value)
        self.draw_debug = bool(self.get_parameter("draw_debug").value)
        self.camera_base_frame = self.get_parameter("camera_base_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.camera_frame_override = self.get_parameter("camera_frame").value
        self.log_every_n_frames = int(self.get_parameter("log_every_n_frames").value)
        self.depth_unit_scaling = float(self.get_parameter("depth_unit_scaling").value)
        self.depth_window = int(self.get_parameter("depth_window").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        dict_id = getattr(cv2.aruco, self.dictionary_name, cv2.aruco.DICT_4X4_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

        try:
            self.aruco_params = cv2.aruco.DetectorParameters_create()
        except AttributeError:
            self.aruco_params = cv2.aruco.DetectorParameters()

        self.bridge = CvBridge()

        # RGB camera info dùng để tính X/Y từ pixel RGB + Z.
        self.rgb_camera_matrix = None
        self.rgb_dist_coeffs = None
        self.rgb_frame = ""
        self._logged_rgb_info = False

        # Depth state dùng cho tính tọa độ 3D.
        self.depth_camera_matrix = None
        self.depth_frame = ""
        self.depth_image = None
        self.depth_encoding = ""
        self.depth_header = None
        self._logged_depth_info = False
        self._warned_aligned_size = False
        self.frame_count = 0

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.sub_info = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.sub_depth_info = self.create_subscription(
            CameraInfo,
            self.depth_camera_info_topic,
            self.depth_camera_info_callback,
            10,
        )

        self.sub_depth = self.create_subscription(
            Image,
            self.depth_image_topic,
            self.depth_callback,
            10,
        )

        self.sub_img = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.pub_debug = self.create_publisher(
            Image,
            "/aruco/image_annotated",
            10,
        )

        self.pub_json = self.create_publisher(
            String,
            "/aruco/detections_json",
            10,
        )

        self.pub_pose = self.create_publisher(
            ArucoPoseArray,
            "/aruco_pose",
            10,
        )

        self.get_logger().info("ArucoDetectNode RGB-XY + ALIGNED-DEPTH-Z started")
        self.get_logger().info(f"RGB image topic       : {self.image_topic}")
        self.get_logger().info(f"RGB CameraInfo topic  : {self.camera_info_topic}")
        self.get_logger().info(f"Depth image topic     : {self.depth_image_topic}")
        self.get_logger().info(f"Depth CameraInfo topic: {self.depth_camera_info_topic}")
        self.get_logger().info(f"Dictionary            : {self.dictionary_name}")
        self.get_logger().info(f"Marker size           : {self.marker_size} m (ignored for XYZ)")
        self.get_logger().info(f"Enable pose           : {self.enable_pose}")
        self.get_logger().info(f"Camera source frame   : {self.camera_frame_override or '<from RGB CameraInfo>'}")
        self.get_logger().info(f"Camera base frame     : {self.camera_base_frame}")
        self.get_logger().info(f"Robot base frame      : {self.base_frame}")

    @staticmethod
    def stamp_is_zero(stamp):
        return stamp.sec == 0 and stamp.nanosec == 0

    def camera_info_callback(self, msg):
        self.rgb_camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        if len(msg.d) > 0:
            self.rgb_dist_coeffs = np.array(msg.d, dtype=np.float64)
        else:
            self.rgb_dist_coeffs = np.zeros((5,), dtype=np.float64)

        self.rgb_frame = msg.header.frame_id or ""

        if not self._logged_rgb_info:
            fx = self.rgb_camera_matrix[0, 0]
            fy = self.rgb_camera_matrix[1, 1]
            cx = self.rgb_camera_matrix[0, 2]
            cy = self.rgb_camera_matrix[1, 2]
            self.get_logger().info(
                f"RGB camera frame: {self.rgb_frame} | fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}"
            )
            self._logged_rgb_info = True

    def depth_camera_info_callback(self, msg):
        self.depth_camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.depth_frame = msg.header.frame_id or ""

        if not self._logged_depth_info:
            fx = self.depth_camera_matrix[0, 0]
            fy = self.depth_camera_matrix[1, 1]
            cx = self.depth_camera_matrix[0, 2]
            cy = self.depth_camera_matrix[1, 2]
            self.get_logger().info(
                f"Depth camera frame: {self.depth_frame} | fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}"
            )
            self._logged_depth_info = True

    def depth_callback(self, msg):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"depth cv_bridge error: {e}")
            return

        if depth is None:
            return

        if depth.ndim == 3:
            depth = depth[:, :, 0]

        self.depth_image = depth
        self.depth_encoding = msg.encoding
        self.depth_header = msg.header

        if not self.depth_frame:
            self.depth_frame = msg.header.frame_id or ""

    @staticmethod
    def marker_yaw_image_deg(pts):
        """
        Tính hướng marker trên mặt phẳng ảnh từ 4 góc ArUco.
        Không dùng marker_size, không dùng solvePnP.
        """
        pts = pts.reshape(4, 2).astype(np.float64)
        # Vector trục X của marker: trung bình cạnh trên và cạnh dưới.
        v_top = pts[1] - pts[0]
        v_bottom = pts[2] - pts[3]
        v = 0.5 * (v_top + v_bottom)
        yaw = math.atan2(v[1], v[0])
        return math.degrees(yaw)

    @staticmethod
    def yaw_deg_to_quaternion(yaw_deg):
        """Quaternion cho góc yaw quanh trục Z của camera frame."""
        yaw = math.radians(yaw_deg)
        half = yaw * 0.5
        return 0.0, 0.0, math.sin(half), math.cos(half)

    def pose_stamped_from_xyz_yaw(self, x, y, z, yaw_deg, frame_id, stamp=None):
        qx, qy, qz, qw = self.yaw_deg_to_quaternion(yaw_deg)

        pose = PoseStamped()
        pose.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        pose.header.frame_id = frame_id
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = float(qx)
        pose.pose.orientation.y = float(qy)
        pose.pose.orientation.z = float(qz)
        pose.pose.orientation.w = float(qw)
        return pose

    def get_pose_frame(self):
        """
        Frame của pose_cam.

        Vì X/Y được tính bằng RGB camera_info và pixel RGB, tọa độ camera
        phải được hiểu trong RGB/color optical frame, không phải depth frame.
        """
        if self.camera_frame_override:
            return self.camera_frame_override
        if self.rgb_frame:
            return self.rgb_frame
        return ""

    def get_depth_m_at_rgb_pixel(self, rgb_u, rgb_v, rgb_width, rgb_height):
        """
        Lấy Z tại tâm ArUco từ ảnh depth đã align với RGB.

        Vì depth đã align/register sang RGB, pixel depth dùng để lấy Z chính là
        pixel RGB (u, v). Không scale sang hệ pixel depth riêng nữa.
        """
        if self.depth_image is None:
            return None, None, "missing_depth_image"

        depth_h, depth_w = self.depth_image.shape[:2]
        if depth_w <= 0 or depth_h <= 0 or rgb_width <= 0 or rgb_height <= 0:
            return None, None, "invalid_image_size"

        if (depth_w != rgb_width or depth_h != rgb_height) and not self._warned_aligned_size:
            self._warned_aligned_size = True
            self.get_logger().warn(
                f"Aligned depth size differs from RGB size: depth={depth_w}x{depth_h}, "
                f"rgb={rgb_width}x{rgb_height}. Code still uses RGB pixel directly. "
                "Check that your depth image is really aligned to RGB."
            )

        depth_u = float(rgb_u)
        depth_v = float(rgb_v)

        u = int(round(depth_u))
        v = int(round(depth_v))

        if u < 0 or u >= depth_w or v < 0 or v >= depth_h:
            return None, {"x": depth_u, "y": depth_v}, "depth_pixel_out_of_range"

        win = max(1, self.depth_window)
        if win % 2 == 0:
            win += 1
        half = win // 2

        x0 = max(0, u - half)
        x1 = min(depth_w, u + half + 1)
        y0 = max(0, v - half)
        y1 = min(depth_h, v + half + 1)

        patch = self.depth_image[y0:y1, x0:x1]
        values = patch.astype(np.float32).reshape(-1)

        # 16UC1 thường là mm. 32FC1 thường là m.
        if self.depth_encoding in ("16UC1", "mono16") or np.issubdtype(self.depth_image.dtype, np.integer):
            values = values * self.depth_unit_scaling

        valid = np.isfinite(values)
        valid &= values > self.min_depth_m
        valid &= values < self.max_depth_m

        if not np.any(valid):
            return None, {"x": float(depth_u), "y": float(depth_v)}, "invalid_depth_value"

        z_m = float(np.median(values[valid]))
        z_m = -z_m  # Đổi dấu Z nếu cần thiết tùy convention của camera frame
        return z_m, {"x": float(depth_u), "y": float(depth_v)}, "ok"

    def back_project_rgb_pixel(self, rgb_u, rgb_v, z_m):
        """
        Chiếu ngược pixel RGB sang tọa độ 3D trong RGB camera frame.

        Depth chỉ cung cấp Z. X/Y được tính bằng RGB camera_info:
            X = (u_rgb - cx_rgb) * Z / fx_rgb
            Y = (v_rgb - cy_rgb) * Z / fy_rgb
            Z = depth_aligned
        """
        z_m = -z_m  # Đổi dấu Z nếu cần thiết tùy convention của camera frame
        if self.rgb_camera_matrix is None:
            raise RuntimeError("Missing RGB camera intrinsics. Cannot compute X/Y from RGB pixel.")

        fx = float(self.rgb_camera_matrix[0, 0])
        fy = float(self.rgb_camera_matrix[1, 1])
        cx = float(self.rgb_camera_matrix[0, 2])
        cy = float(self.rgb_camera_matrix[1, 2])

        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            raise RuntimeError("Invalid RGB camera intrinsics: fx/fy is zero")

        x_m = (float(rgb_u) - cx) * z_m / fx
        y_m = (float(rgb_v) - cy) * z_m / fy
        return x_m, y_m, z_m

    def lookup_transform_between(self, target_frame, source_frame, stamp):
        """
        Lấy transform source_frame -> target_frame.

        tf2 dùng thứ tự lookup_transform(target, source, time).
        Nghĩa là transform nhận được dùng để đổi pose từ source sang target.
        """
        if not target_frame or not source_frame:
            self.get_logger().warn(
                f"TF lookup skipped because target/source frame is empty: {source_frame} -> {target_frame}"
            )
            return None

        if target_frame == source_frame:
            return "identity"

        exact_time = rclpy.time.Time.from_msg(stamp) if not self.stamp_is_zero(stamp) else rclpy.time.Time()

        try:
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                exact_time,
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except Exception as e_exact:
            self.get_logger().warn(
                f"TF exact lookup failed: {source_frame} -> {target_frame}: {e_exact}. Trying latest transform."
            )

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            self.get_logger().warn(
                f"Using latest TF transform for {source_frame} -> {target_frame}."
            )
            return transform
        except Exception as e:
            self.get_logger().warn(
                f"TF transform failed: {source_frame} -> {target_frame}: {e}"
            )
            return None

    def transform_pose_to_frame(self, pose_stamped, target_frame):
        """
        Transform PoseStamped sang target_frame.
        Trả về PoseStamped hoặc None nếu TF lỗi.

        Fix:
        - Một số bản tf2_geometry_msgs.do_transform_pose() chỉ nhận geometry_msgs/Pose,
        không nhận PoseStamped.
        - Vì vậy truyền pose_stamped.pose vào do_transform_pose(),
        sau đó tự bọc kết quả lại thành PoseStamped.
        """
        source_frame = pose_stamped.header.frame_id

        if not target_frame or not source_frame:
            self.get_logger().warn(
                f"TF transform skipped because target/source frame is empty: "
                f"{source_frame} -> {target_frame}"
            )
            return None

        if target_frame == source_frame:
            out = PoseStamped()
            out.header.stamp = pose_stamped.header.stamp
            out.header.frame_id = source_frame
            out.pose = pose_stamped.pose
            return out

        transform = self.lookup_transform_between(
            target_frame,
            source_frame,
            pose_stamped.header.stamp,
        )

        if transform is None:
            return None

        if transform == "identity":
            out = PoseStamped()
            out.header.stamp = pose_stamped.header.stamp
            out.header.frame_id = target_frame
            out.pose = pose_stamped.pose
            return out

        try:
            # QUAN TRỌNG:
            # truyền Pose, không truyền PoseStamped
            transformed_pose = do_transform_pose(pose_stamped.pose, transform)
        except Exception as e:
            self.get_logger().warn(
                f"do_transform_pose failed: {source_frame} -> {target_frame}: {e}"
            )
            return None

        out = PoseStamped()
        out.header.stamp = pose_stamped.header.stamp
        out.header.frame_id = target_frame

        # Nếu hàm trả về PoseStamped thì lấy .pose, nếu trả về Pose thì dùng trực tiếp.
        if hasattr(transformed_pose, "pose"):
            out.pose = transformed_pose.pose
        else:
            out.pose = transformed_pose

        return out

    def image_callback(self, msg):
        self.frame_count += 1

        # 1) Chuyển ROS Image message thành OpenCV image.
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        if frame is None:
            return

        rgb_h, rgb_w = frame.shape[:2]

        # 2) Chuyển ảnh sang grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        annotated = frame.copy()

        # 3) Phát hiện marker ArUco trong ảnh RGB.
        try:
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dict,
                parameters=self.aruco_params,
            )
        except Exception as e:
            self.get_logger().error(f"aruco detect error: {e}")
            return

        detections = []
        pose_array = ArucoPoseArray()
        # Array này ưu tiên biểu diễn pose đã transform về base_frame.
        # Từng marker vẫn có frame_cam/frame_base riêng trong ArucoPose.
        pose_array.header.stamp = msg.header.stamp
        pose_array.header.frame_id = self.base_frame

        # 4) Nếu phát hiện marker, lấy tâm pixel + yaw từ ArUco, lấy XYZ từ depth.
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten()

            if self.draw_debug:
                cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

            for i, marker_id in enumerate(ids_flat):
                pts = corners[i].reshape(4, 2)

                # Tâm vật = tâm ArUco trên ảnh RGB.
                center_u = float(np.mean(pts[:, 0]))
                center_v = float(np.mean(pts[:, 1]))

                # Hướng marker trên ảnh, không phụ thuộc marker_size.
                yaw_deg = float(self.marker_yaw_image_deg(pts))

                det = {
                    "id": int(marker_id),
                    "center_px": {
                        "x": center_u,
                        "y": center_v,
                    },
                    "corners_px": pts.astype(float).tolist(),
                    "yaw_deg": yaw_deg,
                    "position_source": "rgb_intrinsic_xy_aligned_depth_z",
                    "marker_size_used_for_xyz": False,
                }

                pose_msg = ArucoPose()
                pose_msg.header = msg.header
                pose_msg.id = int(marker_id)
                pose_msg.frame_cam = self.get_pose_frame() or ""
                pose_msg.frame_base = self.base_frame
                pose_msg.has_pose_base = False
                pose_msg.yaw_deg = yaw_deg

                if self.enable_pose:
                    try:
                        z_m, depth_px, depth_status = self.get_depth_m_at_rgb_pixel(
                            center_u,
                            center_v,
                            rgb_w,
                            rgb_h,
                        )
                        det["depth_status"] = depth_status
                        if depth_px is not None:
                            det["depth_px"] = depth_px
                            det["rgb_px_used_for_xy"] = {"x": float(center_u), "y": float(center_v)}

                        if z_m is not None and depth_px is not None:
                            # Depth đã align với RGB:
                            # - depth_px chỉ dùng để báo/debug pixel lấy Z
                            # - X/Y tính trực tiếp từ center_u/center_v bằng RGB camera_info
                            x_m, y_m, z_m = self.back_project_rgb_pixel(
                                center_u,
                                center_v,
                                z_m,
                            )

                            det["position_m"] = {
                                "x": float(x_m),
                                "y": float(y_m),
                                "z": float(z_m),
                            }

                            source_frame = self.get_pose_frame() or (msg.header.frame_id or "")
                            pose_msg.frame_cam = source_frame or ""

                            # X/Y thuộc RGB camera frame, nhưng Z lấy từ depth đã align.
                            # Ưu tiên stamp RGB vì pixel u/v đến từ ảnh RGB.
                            pose_stamp = msg.header.stamp

                            pose_cam = self.pose_stamped_from_xyz_yaw(
                                x_m,
                                y_m,
                                z_m,
                                yaw_deg,
                                source_frame,
                                stamp=pose_stamp if not self.stamp_is_zero(pose_stamp) else self.get_clock().now().to_msg(),
                            )

                            det["pose_cam_m"] = {
                                "frame_id": source_frame or "",
                                "x": float(pose_cam.pose.position.x),
                                "y": float(pose_cam.pose.position.y),
                                "z": float(pose_cam.pose.position.z), 
                            }

                            # Nếu custom msg có field pose_cam thì giữ luôn pose trong camera/depth frame.
                            # Dùng hasattr để không làm lỗi nếu msg cũ không có field này.
                            if hasattr(pose_msg, "pose_cam"):
                                pose_msg.pose_cam = pose_cam.pose

                            # TF theo 2 bước đúng với cây của bạn:
                            #   astra_optical -> astra_base -> base_link
                            det["tf_source_frame"] = source_frame
                            det["tf_camera_base_frame"] = self.camera_base_frame
                            det["tf_robot_base_frame"] = self.base_frame

                            pose_camera_base_stamped = None
                            pose_robot_base_stamped = None

                            if not source_frame:
                                det["tf_status"] = "missing_source_frame"
                            else:
                                # Bước 1: camera optical frame -> astra_base
                                pose_camera_base_stamped = self.transform_pose_to_frame(
                                    pose_cam,
                                    self.camera_base_frame,
                                )

                                if pose_camera_base_stamped is not None:
                                    p_astra = pose_camera_base_stamped.pose
                                    det["pose_astra_base_m"] = {
                                        "frame_id": self.camera_base_frame,
                                        "x": float(p_astra.position.x),
                                        "y": float(p_astra.position.y),
                                        "z": float(p_astra.position.z),
                                    }
                                    det["tf_camera_base_status"] = "ok"

                                    # Bước 2: astra_base -> base_link
                                    pose_robot_base_stamped = self.transform_pose_to_frame(
                                        pose_camera_base_stamped,
                                        self.base_frame,
                                    )
                                else:
                                    det["tf_camera_base_status"] = "failed"

                            if pose_robot_base_stamped is not None:
                                pose_base = pose_robot_base_stamped.pose

                                pose_msg.has_pose_base = True
                                pose_msg.frame_base = self.base_frame
                                pose_msg.pose_base = pose_base
                                det["tf_robot_base_status"] = "ok"
                                det["tf_status"] = "ok"
                                det["pose_base_m"] = {
                                    "frame_id": self.base_frame,
                                    "x": float(pose_base.position.x),
                                    "y": float(pose_base.position.y),
                                    "z": float(pose_base.position.z),
                                }
                            else:
                                pose_msg.has_pose_base = False
                                det.setdefault("tf_robot_base_status", "failed")
                                det.setdefault("tf_status", "failed")

                            if self.frame_count % max(1, self.log_every_n_frames) == 0:
                                cam_txt = (
                                    f"cam[{source_frame}]=({x_m:.3f},{y_m:.3f},{z_m:.3f})m"
                                )
                                astra_txt = "astra_base=NA"
                                base_txt = "base_link=NA"
                                if "pose_astra_base_m" in det:
                                    pa = det["pose_astra_base_m"]
                                    astra_txt = (
                                        f"{pa['frame_id']}=({pa['x']:.3f},{pa['y']:.3f},{pa['z']:.3f})m"
                                    )
                                if "pose_base_m" in det:
                                    pb = det["pose_base_m"]
                                    base_txt = (
                                        f"{pb['frame_id']}=({pb['x']:.3f},{pb['y']:.3f},{pb['z']:.3f})m"
                                    )
                                self.get_logger().info(
                                    f"ID {int(marker_id)} | {cam_txt} | {astra_txt} | {base_txt} | "
                                    f"depth={det.get('depth_status')} tf={det.get('tf_status')}"
                                )

                    except Exception as e:
                        det["pose_status"] = "error"
                        det["pose_error"] = str(e)
                        self.get_logger().warn(f"depth pose error marker {marker_id}: {e}")
                else:
                    det["pose_status"] = "disabled"

                if self.draw_debug:
                    c = (int(round(center_u)), int(round(center_v)))
                    cv2.circle(annotated, c, 4, (0, 255, 255), -1)

                    # Vẽ hướng marker trên ảnh.
                    yaw_rad = math.radians(yaw_deg)
                    length = 45
                    p2 = (
                        int(round(center_u + length * math.cos(yaw_rad))),
                        int(round(center_v + length * math.sin(yaw_rad))),
                    )
                    cv2.arrowedLine(annotated, c, p2, (0, 255, 0), 2, tipLength=0.2)

                    label = f"ID {int(marker_id)} u={center_u:.1f} v={center_v:.1f} yaw={yaw_deg:.1f}"
                    if "position_m" in det:
                        p = det["position_m"]
                        label += f" cam=({p['x']:.3f},{p['y']:.3f},{p['z']:.3f})m"
                    elif "depth_status" in det:
                        label += f" {det['depth_status']}"
                    cv2.putText(
                        annotated,
                        label,
                        (c[0] + 8, c[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    if "pose_astra_base_m" in det:
                        pa = det["pose_astra_base_m"]
                        label_astra = f"{pa['frame_id']}=({pa['x']:.3f},{pa['y']:.3f},{pa['z']:.3f})m"
                    else:
                        label_astra = f"{self.camera_base_frame} TF {det.get('tf_camera_base_status', 'no_pose')}"
                    cv2.putText(
                        annotated,
                        label_astra,
                        (c[0] + 8, c[1] + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                    if "pose_base_m" in det:
                        pb = det["pose_base_m"]
                        label_base = f"{pb['frame_id']}=({pb['x']:.3f},{pb['y']:.3f},{pb['z']:.3f})m"
                    else:
                        label_base = f"{self.base_frame} TF {det.get('tf_robot_base_status', 'no_pose')}"
                    cv2.putText(
                        annotated,
                        label_base,
                        (c[0] + 8, c[1] + 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                detections.append(det)
                pose_array.poses.append(pose_msg)

        out = {
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "frame_id": msg.header.frame_id,
            "rgb_frame_id": self.rgb_frame,
            "depth_frame_id": self.depth_frame,
            "camera_base_frame": self.camera_base_frame,
            "base_frame": self.base_frame,
            "xyz_rule": "X/Y from RGB pixel + RGB camera_info; Z from aligned depth at same RGB pixel",
            "count": len(detections),
            "detections": detections,
        }

        json_msg = String()
        json_msg.data = json.dumps(out, ensure_ascii=False)
        self.pub_json.publish(json_msg)
        self.pub_pose.publish(pose_array)

        if self.draw_debug:
            try:
                # Vẽ trục pixel ảnh ở góc trái trên để dễ kiểm tra u/v.
                origin_size = 50
                cv2.arrowedLine(annotated, (0, 0), (origin_size, 0), (0, 0, 255), 2, tipLength=0.1)
                cv2.arrowedLine(annotated, (0, 0), (0, origin_size), (255, 0, 0), 2, tipLength=0.1)
                cv2.putText(
                    annotated,
                    "(0,0)",
                    (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    "u",
                    (origin_size + 5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    annotated,
                    "v",
                    (5, origin_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                debug_msg.header = msg.header
                self.pub_debug.publish(debug_msg)
            except Exception as e:
                self.get_logger().error(f"publish debug image error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
