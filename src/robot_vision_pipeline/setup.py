from setuptools import find_packages, setup
from glob import glob
import os

package_name = "robot_vision_pipeline"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=[
        "setuptools",
        "PyQt6",
        "opencv-python",
        "numpy",
    ],
    zip_safe=True,
    maintainer="minhquang",
    maintainer_email="minhquang@example.com",
    description="Robot vision pipeline using YOLO for object detection",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_detect_node = robot_vision_pipeline.yolo.yolo_detect_node:main",
            "yolo_json_to_box_detection_node = robot_vision_pipeline.yolo_json_to_box_detection_node:main",
            "pixel_to_base_mapper_node = robot_vision_pipeline.pose_estimation.pixel_to_base_mapper_node:main",
            "vision_detection_marker_node = robot_vision_pipeline.pose_estimation.vision_detection_marker_node:main",
            "static_image_camera_node = robot_vision_pipeline.static_image_camera_node:main",
            "aruco_detect_node = robot_vision_pipeline.aruco.aruco_detect_node:main",
            "vision_gui = robot_vision_pipeline.vision_gui.vision_gui_main:main",
            "vision_gui_astra = robot_vision_pipeline.vision_gui.vision_gui_astra:main",
        ],
    },
)
