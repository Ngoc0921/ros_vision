#!/usr/bin/env python3
"""Test the pixel → base XY homography on a single pixel observation.

Usage::

    ros2 run gp7_vision_pipeline test_pixel_to_base_homography \\
        --u 364 --v 259 \\
        --yaml config/pixel_to_base_homography.yaml

The script:
1. Loads the homography H and fixed_z_b_mm from the YAML.
2. Applies H to map (u, v) → (x_b, y_b) in millimetres.
3. Prints the predicted base-frame XY, with z_b = fixed_z_b_mm.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import yaml

from gp7_vision_pipeline.compute_pixel_to_base_homography import (
    load_homography_yaml,
    pixel_to_base_xy,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Test the pixel → base XY homography on a single pixel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--u", type=float, required=True,
        help="Pixel x coordinate (column index, u)",
    )
    p.add_argument(
        "--v", type=float, required=True,
        help="Pixel y coordinate (row index, v)",
    )
    p.add_argument(
        "--yaml",
        type=str,
        default="config/pixel_to_base_homography.yaml",
        help="Path to pixel_to_base_homography.yaml (default: config/pixel_to_base_homography.yaml)",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    yaml_path = args.yaml
    if not os.path.isfile(yaml_path):
        print(f"[ERROR] YAML not found: {yaml_path}", file=sys.stderr)
        sys.exit(1)

    # Load H and fixed_z_b_mm
    print(f"[test_pixel_to_base_homography] Loading {yaml_path}")
    try:
        H, fixed_z_mm = load_homography_yaml(yaml_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load YAML: {exc}", file=sys.stderr)
        sys.exit(1)

    # Apply homography
    x_mm, y_mm = pixel_to_base_xy(args.u, args.v, H)

    # Print result
    print()
    print("=" * 60)
    print("  PIXEL → BASE HOMOGRAPHY TEST")
    print("=" * 60)
    print()
    print("  INPUT PIXEL")
    print(f"  u={args.u}  v={args.v}")
    print()
    print("  HOMOGRAPHY MATRIX H")
    print("  H @ [u, v, 1]^T = [X, Y, Z]^T  →  x = X/Z, y = Y/Z")
    for row in H:
        print(f"    [{row[0]:16.8f}  {row[1]:16.8f}  {row[2]:16.8f}]")
    print()
    print("  PREDICTED BASE-FRAME POINT")
    print(f"  x_b = {x_mm:.3f} mm")
    print(f"  y_b = {y_mm:.3f} mm")
    print(f"  z_b = {fixed_z_mm:.1f} mm  (fixed picking plane)")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
