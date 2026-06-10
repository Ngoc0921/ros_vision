#!/usr/bin/env python3
"""
Vision GUI Launcher Script
Provides a simple way to run the Vision GUI with ROS2
"""

import sys
import os

# Ensure ROS2 is properly initialized
def main():
    try:
        from robot_vision_pipeline.vision_gui.vision_gui_main import VisionGUI
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        gui = VisionGUI()
        gui.show()
        sys.exit(app.exec())
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("Please install: pip install PyQt6 opencv-python numpy")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
