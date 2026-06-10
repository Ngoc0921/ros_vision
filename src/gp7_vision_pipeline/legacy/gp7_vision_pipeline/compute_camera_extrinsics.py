#!/usr/bin/env python3
"""Compute rigid camera-to-base extrinsic transform from calibration point correspondences.

Usage::

    ros2 run gp7_vision_pipeline compute_camera_extrinsics \\
        --csv config/calibration_points.csv \\
        --output config/camera_extrinsics.yaml \\
        --fx 609.0 --fy 609.0 --cx 424.0 --cy 236.0

The script:
1. Loads (point, x_b, y_b, z_b, u_c, v_c, z_c) from the CSV (all in mm / px).
2. Converts mm → m internally.
3. Reconstructs 3-D camera points via the pinhole model using (u_c, v_c, z_c) and
   intrinsics (fx, fy, cx, cy).
4. Fits a rigid transform camera → base using SVD / Kabsch (rotation + translation,
   NO scale) on the N point correspondences.
5. Reports diagnostics (R, t, 4x4 homogeneous transform, per-point reprojection
   error in mm, mean, max, RMS).
6. Saves the result to a YAML file.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# Kabsch / Umeyama rigid alignment (no scale)
# ─────────────────────────────────────────────────────────────────────────────

def _svd_rigid_align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Find R, t minimizing ||target - (R @ source.T + t[:, None]).T||^2.

    Uses SVD-based Kabsch algorithm. Enforces det(R) = +1 (proper rotation).
    Returns (R 3×3, t 3×1).  source and target are N×3.
    """
    if source.shape != target.shape:
        raise ValueError(f"source {source.shape} and target {target.shape} must have same shape")
    if source.shape[0] < 3:
        raise ValueError("At least 3 point correspondences are required")

    # Centroids
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    # Centered clouds
    Ss = source - mu_s          # N×3
    St = target - mu_t          # N×3

    # Cross-covariance  H = Ss^T @ St
    H = Ss.T @ St              # 3×3

    # SVD
    U, _S, Vt = np.linalg.svd(H)

    # Enforce proper rotation: if det(V @ U^T) == -1, flip the sign of the
    # singular value corresponding to the smallest singular vector.
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    # Optimal translation
    t = mu_t - R @ mu_s         # 3×1 as 1-D array

    return R, t


def _reproject_error(source: np.ndarray, target: np.ndarray,
                     R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-point Euclidean error in metres.  source and target are N×3."""
    transformed = (R @ source.T).T + t      # N×3
    return np.linalg.norm(transformed - target, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# CSV loading
# ─────────────────────────────────────────────────────────────────────────────

def load_calibration_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV and return (camera_points_m, base_points_m) as N×3 arrays (metres).

    CSV columns: point,x_b,y_b,z_b,u_c,v_c,z_c
    - x_b,y_b,z_b : robot TCP position in base frame, millimetres
    - u_c,v_c     : image pixel coordinates
    - z_c          : RealSense raw depth, 16UC1, millimetres

    Pinhole back-projection (z is along the optical axis):
        X_c = (u_c - cx) / fx * z_c
        Y_c = (v_c - cy) / fy * z_c
        Z_c = z_c
    """
    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            stripped = {k.strip(): v.strip() for k, v in r.items()}
            rows.append(stripped)

    if not rows:
        raise ValueError(f"No data rows in CSV: {path}")

    base_pts: List[np.ndarray] = []
    cam_pts: List[np.ndarray] = []

    for r in rows:
        # Base-frame TCP position in millimetres
        x_b = float(r["x_b"]) / 1000.0   # mm → m
        y_b = float(r["y_b"]) / 1000.0
        z_b = float(r["z_b"]) / 1000.0
        base_pts.append(np.array([x_b, y_b, z_b]))

        # Pixel
        u_c = float(r["u_c"])
        v_c = float(r["v_c"])
        # Raw depth in millimetres
        z_c_mm = float(r["z_c"])
        z_c = z_c_mm / 1000.0              # mm → m

        cam_pts.append(np.array([u_c, v_c, z_c]))

    base_pts_arr = np.array(base_pts)      # N×3
    cam_pts_arr = np.array(cam_pts)        # N×3

    return cam_pts_arr, base_pts_arr


def backproject_to_camera(pixel_u: float, pixel_v: float, depth_m: float,
                          fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Back-project a single pixel + depth to a camera-frame 3-D point (metres)."""
    X = (pixel_u - cx) * depth_m / fx
    Y = (pixel_v - cy) * depth_m / fy
    return np.array([X, Y, depth_m])


def compute_extrinsics(cam_pts_px: np.ndarray, base_pts_m: np.ndarray,
                       fx: float, fy: float, cx: float, cy: float,
                       z_c_all_mm: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Full pipeline: back-project → align → diagnose.

    Args:
        cam_pts_px: N×3 array where cols are (u, v, z_c_mm)
        base_pts_m: N×3 array of (x_b, y_b, z_b) in metres
        fx,fy,cx,cy: pinhole intrinsics
        z_c_all_mm: N-element array of raw depth in millimetres

    Returns:
        R, t, cam_pts_m (back-projected camera points), diagnostics dict
    """
    n = cam_pts_px.shape[0]

    # Back-project to camera frame (metres)
    cam_pts_m = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        cam_pts_m[i] = backproject_to_camera(
            cam_pts_px[i, 0], cam_pts_px[i, 1],
            z_c_all_mm[i] / 1000.0,
            fx, fy, cx, cy,
        )

    # SVD rigid alignment: camera → base
    R, t = _svd_rigid_align(cam_pts_m, base_pts_m)

    # Reprojection error
    errors_m = _reproject_error(cam_pts_m, base_pts_m, R, t)
    errors_mm = errors_m * 1000.0

    diag = {
        "num_points": int(n),
        "mean_error_mm": float(np.mean(errors_mm)),
        "max_error_mm": float(np.max(errors_mm)),
        "rms_error_mm": float(np.sqrt(np.mean(errors_mm ** 2))),
        "per_point_errors_mm": errors_mm.tolist(),
    }

    return R, t, cam_pts_m, diag


# ─────────────────────────────────────────────────────────────────────────────
# YAML I/O
# ─────────────────────────────────────────────────────────────────────────────

def build_transform_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4×4 homogeneous transform: base_point = R @ camera_point + t."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def save_extrinsics_yaml(
    output_path: str,
    R: np.ndarray,
    t: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    diag: Dict,
) -> None:
    """Save extrinsic calibration to YAML."""
    T = build_transform_matrix(R, t)

    doc = {
        "camera_frame": "camera_color_optical_frame",
        "base_frame": "base_link",
        "units": "meters",
        "intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "_note": "Obtained from ros2 topic echo /camera/camera/color/camera_info",
        },
        "rotation": {
            "rows": 3,
            "cols": 3,
            "data": R.flatten().tolist(),
            "_note": "Camera → base_link rotation (column-major for use with numpy or tf)",
        },
        "translation": {
            "rows": 3,
            "cols": 1,
            "data": t.flatten().tolist(),   # metres
            "_note": "Camera → base_link translation in metres",
        },
        "transform": {
            "rows": 4,
            "cols": 4,
            "data": T.flatten().tolist(),
            "_note": "4×4 homogeneous transform: base = R @ camera + t",
        },
        "diagnostics": {
            "num_points": diag["num_points"],
            "mean_error_mm": round(diag["mean_error_mm"], 4),
            "max_error_mm": round(diag["max_error_mm"], 4),
            "rms_error_mm": round(diag["rms_error_mm"], 4),
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, default_flow_style=None, sort_keys=False)

    print(f"[compute_camera_extrinsics] Saved {output_path}")


def load_intrinsics_from_yaml(yaml_path: str) -> Tuple[float, float, float, float]:
    """Load fx, fy, cx, cy from a camera_info YAML (K matrix flattened row-major)."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    K = data.get("camera_matrix", {}) or data.get("K", {})
    data_list = K.get("data", [])
    if len(data_list) != 9:
        raise ValueError(
            f"camera_matrix.data in {yaml_path} must have 9 entries; "
            f"got {len(data_list)}"
        )
    fx, _, cx, _, fy, cy, _, _, _ = data_list
    return float(fx), float(fy), float(cx), float(cy)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics printing
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    csv_path: str,
    cam_pts_px: np.ndarray,
    base_pts_m: np.ndarray,
    cam_pts_m: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    diag: Dict,
    z_c_all_mm: np.ndarray,
) -> None:
    n = diag["num_points"]

    print("=" * 70)
    print("  CAMERA EXTRINSIC CALIBRATION REPORT")
    print("=" * 70)
    print(f"  CSV:            {csv_path}")
    print(f"  Points used:    {n}")
    print()
    print("  CAMERA INTRINSICS")
    print(f"  fx={fx:.4f}  fy={fy:.4f}  cx={cx:.4f}  cy={cy:.4f}")
    print()
    print("  BACK-PROJECTED CAMERA POINTS (m) — first 5:")
    header = "  {:>4} {:>12} {:>12} {:>12}  {:>12} {:>12} {:>12}  {:>8}".format(
        "pt", "X_c", "Y_c", "Z_c", "x_b", "y_b", "z_b", "z_c(mm)"
    )
    print(header)
    for i in range(min(5, n)):
        print(
            f"  {i+1:3d}  "
            f"{cam_pts_m[i,0]:12.5f} {cam_pts_m[i,1]:12.5f} {cam_pts_m[i,2]:12.5f}  "
            f"{base_pts_m[i,0]:12.5f} {base_pts_m[i,1]:12.5f} {base_pts_m[i,2]:12.5f}  "
            f"{z_c_all_mm[i]:8.0f}"
        )
    if n > 5:
        print(f"  ... ({n - 5} more rows)")
    print()
    print("  RIGID TRANSFORM  camera → base_link")
    print("  R (rotation, det(R)={:.6f}):".format(np.linalg.det(R)))
    for row in R:
        print(f"    [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.6f}]")
    print()
    print("  t (translation):")
    print(f"    [{t[0]:10.6f}  {t[1]:10.6f}  {t[2]:10.6f}]  (metres)")
    print(f"    [{t[0]*1000:10.3f}  {t[1]*1000:10.3f}  {t[2]*1000:10.3f}]  (millimetres)")
    print()
    T = build_transform_matrix(R, t)
    print("  4×4 homogeneous transform  [base = R @ camera + t]:")
    for row in T:
        print(f"    [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.6f}  {row[3]:10.6f}]")
    print()
    print("  DIAGNOSTICS — per-point transform error (Euclidean, mm):")
    print(f"  {'pt':>4}  {'error_mm':>10}")
    for i in range(n):
        print(f"  {i+1:3d}  {diag['per_point_errors_mm'][i]:10.3f}")
    print()
    print(f"  Mean error:  {diag['mean_error_mm']:.4f} mm")
    print(f"  Max error:   {diag['max_error_mm']:.4f} mm")
    print(f"  RMS error:   {diag['rms_error_mm']:.4f} mm")
    print()

    # ── Planarity warning ───────────────────────────────────────────────────
    z_b_values = base_pts_m[:, 2]
    z_unique = np.unique(np.round(z_b_values * 1000).astype(int))
    if len(z_unique) == 1:
        print("  *** PLANAR WARNING ***")
        print("  All base-frame z_b values are identical (planar calibration).")
        print("  Full 3-D extrinsic calibration is UNDERCONSTRAINED —")
        print("  the rotation about the optical axis cannot be resolved from planar data.")
        print("  For this project, the top-down planar homography is RECOMMENDED:")
        print("    ros2 run gp7_vision_pipeline compute_pixel_to_base_homography")
        print("  because the camera is assumed perpendicular to the robot base plane")
        print("  and z_b is fixed at the picking-plane height.")
        print("  For robust 3-D extrinsics, collect points at 3+ different heights.")
        print("=" * 70)
    else:
        print(f"  z_b values: {sorted(z_unique)} mm  (non-planar — good)")
        print("=" * 70)

    print()
    print("  ERROR METRIC EXPLANATION")
    print("  ─────────────────────────────────────────────────────────────")
    print("  Per-point error: Euclidean distance between the known base-frame")
    print("  TCP position (x_b, y_b, z_b) and the predicted base-frame position")
    print("  obtained by applying R, t to the back-projected camera point.")
    print()
    print("  Mean error: average of all per-point errors — indicates overall fit.")
    print("  Max error:  worst single point — may indicate a measurement outlier.")
    print("  RMS error:  sqrt(mean(error^2)) — penalises large residuals more.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_intrinsics(args: argparse.Namespace) -> Tuple[float, float, float, float]:
    """Resolve fx, fy, cx, cy from CLI args or camera_info YAML."""
    if args.camera_info_yaml:
        return load_intrinsics_from_yaml(args.camera_info_yaml)

    missing = [n for n, v in [
        ("fx", args.fx), ("fy", args.fy),
        ("cx", args.cx), ("cy", args.cy),
    ] if v is None]

    if missing:
        raise argparse.ArgumentError(
            None,
            f"intrinsics not provided. Supply either --camera-info-yaml "
            f"or all of --fx --fy --cx --cy. Missing: {missing}"
        )

    return args.fx, args.fy, args.cx, args.cy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute camera → base_link rigid extrinsic transform "
                    "from calibration correspondences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--csv", default="config/calibration_points.csv",
        help="Path to calibration CSV (default: config/calibration_points.csv)",
    )
    p.add_argument(
        "--output", default="config/camera_extrinsics.yaml",
        help="Output YAML path (default: config/camera_extrinsics.yaml)",
    )
    p.add_argument(
        "--fx", type=float, default=None,
        help="Focal length x (pixels)",
    )
    p.add_argument(
        "--fy", type=float, default=None,
        help="Focal length y (pixels)",
    )
    p.add_argument(
        "--cx", type=float, default=None,
        help="Principal point x (pixels)",
    )
    p.add_argument(
        "--cy", type=float, default=None,
        help="Principal point y (pixels)",
    )
    p.add_argument(
        "--camera-info-yaml", type=str, default=None,
        help="Path to YAML containing camera_matrix/K (fx,fy,cx,cy from camera_info topic)",
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

    # Resolve intrinsics
    try:
        fx, fy, cx, cy = parse_intrinsics(args)
    except argparse.ArgumentError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    # Load and validate CSV
    print(f"[compute_camera_extrinsics] Loading {csv_path}")
    try:
        cam_pts_px, base_pts_m = load_calibration_csv(csv_path)
    except Exception as exc:
        print(f"[ERROR] Failed to load CSV: {exc}", file=sys.stderr)
        sys.exit(1)

    if cam_pts_px.shape[0] < 3:
        print(f"[ERROR] Need at least 3 points, got {cam_pts_px.shape[0]}", file=sys.stderr)
        sys.exit(1)

    # Raw depth column (mm) for reporting
    z_c_all_mm = cam_pts_px[:, 2]

    print(f"[compute_camera_extrinsics] {cam_pts_px.shape[0]} points loaded")

    # Compute
    print("[compute_camera_extrinsics] Computing rigid transform (camera → base_link)...")
    R, t, cam_pts_m, diag = compute_extrinsics(
        cam_pts_px, base_pts_m, fx, fy, cx, cy, z_c_all_mm
    )

    # Report
    print_report(csv_path, cam_pts_px, base_pts_m, cam_pts_m, R, t,
                 fx, fy, cx, cy, diag, z_c_all_mm)

    # Save YAML
    try:
        save_extrinsics_yaml(args.output, R, t, fx, fy, cx, cy, diag)
    except Exception as exc:
        print(f"[ERROR] Failed to write output YAML: {exc}", file=sys.stderr)
        sys.exit(1)

    print("[compute_camera_extrinsics] Done.")


if __name__ == "__main__":
    main()
