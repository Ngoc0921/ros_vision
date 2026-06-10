#!/usr/bin/env python3
"""Test the camera extrinsic transform on a single observation.

Usage::

    ros2 run gp7_vision_pipeline test_camera_extrinsics \\
        --u 364 --v 259 --z 566 \\
        --yaml config/camera_extrinsics.yaml

The script:
1. Loads the saved R, t, and intrinsics from the YAML.
2. Back-projects (u, v, z_mm) to a camera-frame 3-D point.
3. Applies base = R @ camera + t.
4. Prints the predicted base-frame (x_b, y_b, z_b) in both metres and millimetres.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


def load_extrinsics_yaml(yaml_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load R, t, intrinsics from a camera_extrinsics.yaml file.

    Returns:
        R (3×3), t (3,), intrinsics dict {fx, fy, cx, cy}.
    """
    with open(yaml_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    rot = doc["rotation"]
    t_dict = doc["translation"]
    T_full = doc.get("transform", {})
    intr = doc.get("intrinsics", {})

    if len(rot["data"]) != 9:
        raise ValueError("rotation.data must have 9 entries")
    if len(t_dict["data"]) != 3:
        raise ValueError("translation.data must have 3 entries")

    R = np.array(rot["data"], dtype=np.float64).reshape(3, 3)
    t = np.array(t_dict["data"], dtype=np.float64).flatten()

    return R, t, intr


def backproject_to_camera(
    u: float, v: float, z_mm: float,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Back-project a pixel + raw depth to a camera-frame 3-D point in metres.

    Args:
        u, v:    pixel coordinates
        z_mm:    RealSense raw depth in millimetres (16UC1)
        fx,fy,cx,cy: pinhole intrinsics
    """
    z_m = z_mm / 1000.0
    X = (u - cx) * z_m / fx
    Y = (v - cy) * z_m / fy
    return np.array([X, Y, z_m], dtype=np.float64)


def transform_to_base(cam_pt: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply base = R @ camera + t."""
    return R @ cam_pt + t


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test the camera extrinsic transform on a single pixel+depth observation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--u", type=float, required=True,
        help="Pixel x coordinate (column index)",
    )
    p.add_argument(
        "--v", type=float, required=True,
        help="Pixel y coordinate (row index)",
    )
    p.add_argument(
        "--z", type=float, required=True,
        help="RealSense raw depth in millimetres (16UC1)",
    )
    p.add_argument(
        "--yaml", type=str, default="config/camera_extrinsics.yaml",
        help="Path to camera_extrinsics.yaml (default: config/camera_extrinsics.yaml)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    yaml_path = args.yaml
    if not os.path.isfile(yaml_path):
        print(f"[ERROR] YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    # Load extrinsic + intrinsics
    print(f"[test_camera_extrinsics] Loading {yaml_path}")
    try:
        R, t, intr = load_extrinsics_yaml(yaml_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load YAML: {exc}", file=sys.stderr)
        sys.exit(1)

    fx = intr.get("fx")
    fy = intr.get("fy")
    cx = intr.get("cx")
    cy = intr.get("cy")

    if None in (fx, fy, cx, cy):
        print(f"[ERROR] Intrinsics not fully populated in YAML: {intr}", file=sys.stderr)
        sys.exit(1)

    # Back-project to camera frame
    cam_pt = backproject_to_camera(args.u, args.v, args.z, fx, fy, cx, cy)

    # Transform to base frame
    base_pt = transform_to_base(cam_pt, R, t)

    # Print result
    print()
    print("=" * 60)
    print("  EXTRINSIC TRANSFORM TEST")
    print("=" * 60)
    print()
    print("  INPUT OBSERVATION")
    print(f"  u={args.u}  v={args.v}  z={args.z} mm")
    print()
    print("  CAMERA INTRINSICS USED")
    print(f"  fx={fx:.4f}  fy={fy:.4f}  cx={cx:.4f}  cy={cy:.4f}")
    print()
    print("  CAMERA-FRAME POINT  [m]")
    print(f"  X_c = {cam_pt[0]:.6f}")
    print(f"  Y_c = {cam_pt[1]:.6f}")
    print(f"  Z_c = {cam_pt[2]:.6f}  (= {args.z} / 1000)")
    print()
    print("  EXTRINSIC TRANSFORM  base = R @ camera + t")
    print("  R matrix (camera → base_link):")
    for row in R:
        print(f"    [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.6f}]")
    print("  t vector (camera → base_link) [m]:")
    print(f"    [{t[0]:10.6f}  {t[1]:10.6f}  {t[2]:10.6f}]")
    print(f"  t vector [mm]:")
    print(f"    [{t[0]*1000:10.3f}  {t[1]*1000:10.3f}  {t[2]*1000:10.3f}]")
    print()
    print("  PREDICTED BASE-FRAME POINT")
    print(f"  x_b = {base_pt[0]:.6f} m  ({base_pt[0]*1000:.3f} mm)")
    print(f"  y_b = {base_pt[1]:.6f} m  ({base_pt[1]*1000:.3f} mm)")
    print(f"  z_b = {base_pt[2]:.6f} m  ({base_pt[2]*1000:.3f} mm)")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
