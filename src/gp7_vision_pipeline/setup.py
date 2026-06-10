from setuptools import find_packages, setup

package_name = "gp7_vision_pipeline"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="norman",
    maintainer_email="norman@todo.todo",
    description="GP7 vision pipeline (perception hooks, no motion).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_box_detector_node = gp7_vision_pipeline.yolo_box_detector_node:main",
            "pixel_to_base_mapper_node = gp7_vision_pipeline.pixel_to_base_mapper_node:main",
            "vision_detection_marker_node = gp7_vision_pipeline.vision_detection_marker_node:main",
            "realsense_depth_debug = gp7_vision_pipeline.realsense_depth_debug_node:main",
        ],
    },
)
