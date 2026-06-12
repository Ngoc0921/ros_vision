"""Compatibility wrapper for robot_vision_pipeline Layer 3 marker node."""

from robot_vision_pipeline.vision_detection_marker_node import (  # noqa: F401
    VisionDetectionMarkerNode,
    main,
)


if __name__ == "__main__":
    main()
