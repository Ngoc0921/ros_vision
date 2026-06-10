#!/usr/bin/env python3
"""Compute a 2-D homography from camera pixels to robot base XY for top-down planar picking.

Usage::

    ros2 run gp7_vision_pipeline compute_pixel_to_base_homography \\
        --csv config/calibration_points.csv \\
        --output config/pixel_to_base_homography.yaml \\
        --fixed-z 55

Assumptions:
- Camera is mounted top-down; optical axis is perpendicular to the base/work plane.
- Object picking plane has fixed z_b (default 55 mm).
- Rotation is assumed identity (not estimated).
- A 2-D perspective homography maps pixel (u, v) → base (x_b, y_b) in millimetres.

Steps:
1. Load CSV columns: point, x_b, y_b, z_b, u_c, v_c, z_c
2. Override z_b → fixed_z_mm (NOT taken from CSV).
3. Build source array: [u_c, v_c, 1]^T for each point (N×3, homogeneous).
4. Build target array: [x_b, y_b] in millimetres for each point (N×2).
5. Solve homography H via DLT (direct linear transform) using OpenCV or NumPy SVD.
6. Report per-point XY re-projection error in mm, plus mean/max/RMS.
7. Save H and diagnostics to YAML.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Homography computation — DLT (Direct Linear Transform)
# ─────────────────────────────────────────────────────────────────────────────

def compute_homography_dlt(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """Compute 3×3 homography H minimizing ||dst - H @ src||.

    Uses the Direct Linear Transform (DLT) algorithm.
    src_pts and dst_pts are N×2 arrays of (u, v) → (x, y).
    Returns H as a 3×3 NumPy array (row-major).
    Requires N ≥ 4.
    """
    if src_pts.shape != dst_pts.shape:
        raise ValueError(f"src {src_pts.shape} and dst {dst_pts.shape} must match")
    n = src_pts.shape[0]
    if n < 4:
        raise ValueError("At least 4 point correspondences are required for homography")

    # Build the 2n × 9 design matrix A
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        u, v = src_pts[i]
        x, y = dst_pts[i]
        # Row 2i: [-u, -v, -1, 0, 0, 0, x*u, x*v, x]
        A[2 * i] = [-u, -v, -1, 0, 0, 0, x * u, x * v, x]
        # Row 2i+1: [0, 0, 0, -u, -v, -1, y*u, y*v, y]
        A[2 * i + 1] = [0, 0, 0, -u, -v, -1, y * u, y * v, y]

    # Solve Ah = 0 in the least-squares sense (SVD of A)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]          # last row of V = last singular vector = nullspace

    H = h.reshape(3, 3)

    # Normalise so H[2,2] = 1
    H = H / H[2, 2]
    return H


def pixel_to_base_xy(u: float, v: float, H: np.ndarray) -> Tuple[float, float]:
    """Apply homography H to map a pixel (u, v) to base XY millimetres.

    Args:
        u, v:    pixel coordinates
        H:       3×3 homography matrix (NumPy array, row-major)

    Returns:
        (x_mm, y_mm) in robot base frame millimetres.

    This is the reusable utility referenced in task 7.
    """
    src = np.array([u, v, 1.0], dtype=np.float64)
    dst_h = H @ src          # homogeneous result
    x_mm = dst_h[0] / dst_h[2]
    y_mm = dst_h[1] / dst_h[2]
    return float(x_mm), float(y_mm)


def apply_homography_batch(
    pixels: np.ndarray, H: np.ndarray
) -> np.ndarray:
    """Apply homography to N×2 pixel array. Returns N×2 base XY (mm)."""
    ones = np.ones((pixels.shape[0], 1), dtype=pixels.dtype)
    homogeneous = np.hstack([pixels, ones])        # N×3
    dst_h = homogeneous @ H.T                       # N×3
    xy = dst_h[:, :2] / dst_h[:, 2:3]              # N×2
    return xy


def xy_errors(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-point Euclidean XY error in mm."""
    predicted = apply_homography_batch(src, H)
    return np.linalg.norm(predicted - dst, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────────────────────────────────────

def load_calibration_csv(
    path: str,
    fixed_z_mm: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Load CSV and return arrays for homography fitting.

    Args:
        path:       CSV file path
        fixed_z_mm: override z_b with this value (mm); NOT taken from CSV

    Returns:
        (uv_coords N×2, xy_base N×2, z_base N×1, raw_rows list-of-dict)
        All XY values are in millimetres.
    """
    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k.strip(): v.strip() for k, v in r.items()})

    if not rows:
        raise ValueError(f"No data rows in CSV: {path}")

    uv_list: List[np.ndarray] = []
    xy_list: List[np.ndarray] = []
    z_list: List[np.ndarray] = []

    for r in rows:
        u_c = float(r["u_c"])
        v_c = float(r["v_c"])
        x_b = float(r["x_b"])
        y_b = float(r["y_b"])
        z_b_raw = float(r["z_b"])
        uv_list.append(np.array([u_c, v_c]))
        xy_list.append(np.array([x_b, y_b]))
        z_list.append(np.array([fixed_z_mm]))

    return (
        np.array(uv_list, dtype=np.float64),
        np.array(xy_list, dtype=np.float64),
        np.array(z_list, dtype=np.float64),
        rows,
    )


# ─────────────────────────────────────────────────────────────────────────────
# YAML I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_homography_yaml(
    output_path: str,
    H: np.ndarray,
    fixed_z_mm: float,
    diag: Dict,
) -> None:
    """Save homography and diagnostics to a YAML file."""
    doc = {
        "calibration_type": "top_down_planar_homography",
        "source_frame": "camera_pixel",
        "target_frame": "base_link",
        "fixed_z_b_mm": float(fixed_z_mm),
        "homography": {
            "rows": 3,
            "cols": 3,
            "data": H.flatten().tolist(),
            "_note": "H @ [u, v, 1]^T = [x_mm/z, y_mm/z, 1]^T  →  x_mm = X/z, y_mm = Y/z",
        },
        "diagnostics": {
            "num_points": diag["num_points"],
            "mean_error_mm": round(float(diag["mean_error_mm"]), 4),
            "max_error_mm": round(float(diag["max_error_mm"]), 4),
            "rms_error_mm": round(float(diag["rms_error_mm"]), 4),
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, default_flow_style=None, sort_keys=False)

    print(f"[compute_pixel_to_base_homography] Saved {output_path}")


def load_homography_yaml(yaml_path: str) -> Tuple[np.ndarray, float]:
    """Load H and fixed_z_b_mm from a pixel_to_base_homography.yaml file."""
    with open(yaml_path, encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    H_data = doc["homography"]["data"]
    if len(H_data) != 9:
        raise ValueError(f"homography.data must have 9 entries; got {len(H_data)}")
    H = np.array(H_data, dtype=np.float64).reshape(3, 3)
    fixed_z = float(doc.get("fixed_z_b_mm", 55.0))
    return H, fixed_z


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics printing
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    csv_path: str,
    uv: np.ndarray,
    xy: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    fixed_z_mm: float,
    diag: Dict,
) -> None:
    n = diag["num_points"]

    print("=" * 70)
    print("  TOP-DOWN PLANAR HOMOGRAPHY REPORT")
    print("  pixel (u, v)  →  base (x_b, y_b, z_b)")
    print("=" * 70)
    print(f"  CSV:             {csv_path}")
    print(f"  Points used:     {n}")
    print(f"  Fixed z_b:      {fixed_z_mm} mm  (z_b from CSV overridden)")
    print()
    print("  HOMOGRAPHY H  [pixel → base_mm]")
    print("  H @ [u, v, 1]^T = [X, Y, Z]^T  →  x = X/Z, y = Y/Z")
    for row in H:
        print(f"    [{row[0]:15.8f}  {row[1]:15.8f}  {row[2]:15.8f}]")
    print()
    print("  DIAGNOSTICS — per-point XY error (mm):")
    print(f"  {'pt':>4}  {'u':>6}  {'v':>6}  {'x_b_mm':>10}  {'y_b_mm':>10}  "
          f"{'pred_x':>10}  {'pred_y':>10}  {'err_mm':>8}")
    per_err = diag["per_point_errors_mm"]
    for i in range(n):
        px, py = pixel_to_base_xy(uv[i, 0], uv[i, 1], H)
        print(
            f"  {i+1:3d}  "
            f"{uv[i,0]:6.1f}  {uv[i,1]:6.1f}  "
            f"{xy[i,0]:10.3f}  {xy[i,1]:10.3f}  "
            f"{px:10.3f}  {py:10.3f}  "
            f"{per_err[i]:8.3f}"
        )
    print()
    print(f"  Mean XY error:  {diag['mean_error_mm']:.4f} mm")
    print(f"  Max  XY error:  {diag['max_error_mm']:.4f} mm")
    print(f"  RMS  XY error:  {diag['rms_error_mm']:.4f} mm")
    print()
    print("  ERROR METRIC EXPLANATION")
    print("  ─────────────────────────────────────────────────────────────")
    print("  Per-point XY error: Euclidean distance in mm between known")
    print("  (x_b, y_b) from the CSV and the (x, y) predicted by H.")
    print()
    print("  Mean XY error: average of all per-point errors — overall fit.")
    print("  Max  XY error: worst single point — may be a measurement outlier.")
    print("  RMS  XY error: sqrt(mean(error^2)) — penalises large residuals.")
    print("=" * 70)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute top-down planar homography: pixel → base XY (mm).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--csv",
        default="calibration_points.csv",
        help="Path to calibration CSV (default: calibration_points.csv in package root)",
    )
    p.add_argument(
        "--output",
        default="config/pixel_to_base_homography.yaml",
        help="Output YAML path (default: config/pixel_to_base_homography.yaml)",
    )
    p.add_argument(
        "--fixed-z",
        type=float,
        default=55.0,
        dest="fixed_z",
        help="Fixed z_b in mm for the picking plane (default: 55.0)",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    fixed_z_mm = args.fixed_z

    # Load CSV (z_b from CSV is ignored; overridden to fixed_z_mm)
    print(f"[compute_pixel_to_base_homography] Loading {csv_path}")
    print(f"[compute_pixel_to_base_homography] Overriding all z_b values to {fixed_z_mm} mm")
    try:
        uv, xy, z, _ = load_calibration_csv(csv_path, fixed_z_mm)
    except Exception as exc:
        print(f"[ERROR] Failed to load CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    n = uv.shape[0]
    print(f"[compute_pixel_to_base_homography] {n} points loaded")

    if n < 4:
        print(f"[ERROR] Need at least 4 points, got {n}", file=sys.stderr)
        sys.exit(1)

    # Compute homography via DLT
    print("[compute_pixel_to_base_homography] Solving homography via DLT...")
    try:
        H = compute_homography_dlt(uv, xy)
    except Exception as exc:
        print(f"[ERROR] Homography computation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Compute per-point XY errors in mm
    per_err_mm = xy_errors(H, uv, xy)      # mm
    diag = {
        "num_points": n,
        "mean_error_mm": float(np.mean(per_err_mm)),
        "max_error_mm": float(np.max(per_err_mm)),
        "rms_error_mm": float(np.sqrt(np.mean(per_err_mm ** 2))),
        "per_point_errors_mm": per_err_mm.tolist(),
    }

    # Report
    print_report(csv_path, uv, xy, z, H, fixed_z_mm, diag)

    # Save YAML
    try:
        save_homography_yaml(args.output, H, fixed_z_mm, diag)
    except Exception as exc:
        print(f"[ERROR] Failed to write output YAML: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[compute_pixel_to_base_homography] Done.")


if __name__ == "__main__":
    main()
