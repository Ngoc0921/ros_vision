"""Depth image helpers: ROI median, unit conversion, invalid filtering, single-pixel raw read."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def depth_at_pixel(
    depth_image: np.ndarray,
    u: int,
    v: int,
    encoding: str,
) -> Tuple[Optional[int], Optional[float], str]:
    """Read the raw depth value and converted distance at a single pixel.

    Args:
        depth_image: 2D depth array (H, W).
        u: Column index (x / width direction).
        v: Row index (y / height direction).
        encoding: sensor_msgs/Image encoding, e.g. ``16UC1`` or ``32FC1``.

    Returns:
        Tuple of (raw_value, dist_meters, encoding).
        - raw_value: int for 16UC1 (millimetres), float for 32FC1 (metres), or None if OOB.
        - dist_meters: depth in metres, or None if invalid/zero.
        - encoding: echoed back so callers don't need to track it separately.
    """
    if depth_image is None or depth_image.size == 0:
        return None, None, encoding

    h, w = depth_image.shape[:2]
    if u < 0 or u >= w or v < 0 or v >= h:
        return None, None, encoding

    raw = depth_image[v, u]

    if encoding in ("16UC1", "mono16"):
        raw_int = int(raw)
        if raw_int == 0:
            return raw_int, None, encoding
        return raw_int, raw_int * 0.001, encoding

    if encoding in ("32FC1",):
        if not math.isfinite(raw) or raw <= 0.0:
            return int(raw), None, encoding
        return int(raw), float(raw), encoding

    # Fallback: best-effort interpretation
    if np.issubdtype(depth_image.dtype, np.floating):
        if not math.isfinite(raw) or raw <= 0.0:
            return None, None, encoding
        return None, float(raw), encoding
    else:
        raw_int = int(raw)
        if raw_int == 0:
            return raw_int, None, encoding
        return raw_int, raw_int * 0.001, encoding


def median_depth_meters(
    depth_image: np.ndarray,
    center_x: int,
    center_y: int,
    half_size: int,
    encoding: str,
) -> Tuple[float, Optional[int]]:
    """Return median depth in meters for a square ROI around (center_x, center_y).

    Invalid values (0, NaN, Inf) are ignored. If the depth image is uint16,
    values are treated as millimeters and converted to meters (RealSense
    aligned depth convention).

    Args:
        depth_image: 2D depth array (H, W).
        center_x: Column index of ROI center.
        center_y: Row index of ROI center.
        half_size: Half side length of the square ROI in pixels (inclusive).
        encoding: sensor_msgs/Image encoding, e.g. ``16UC1`` or ``32FC1``.

    Returns:
        Tuple of (median_depth_meters, roi_median_raw).
        - median_depth_meters: median depth in metres, or ``-1.0`` if no valid samples exist.
        - roi_median_raw: raw uint16 median in mm (for 16UC1), raw float median in metres
          (for 32FC1), or None if no valid samples.
        The caller can display ``roi_median_raw`` to show the raw depth value alongside
        the converted metres value.
    """
    if depth_image is None or depth_image.size == 0:
        return -1.0, None

    h, w = depth_image.shape[:2]
    if h < 1 or w < 1:
        return -1.0, None

    x0 = max(0, center_x - half_size)
    y0 = max(0, center_y - half_size)
    x1 = min(w, center_x + half_size + 1)
    y1 = min(h, center_y + half_size + 1)
    if x0 >= x1 or y0 >= y1:
        return -1.0, None

    roi = depth_image[y0:y1, x0:x1]

    if encoding in ("16UC1", "mono16"):
        roi_f = roi.astype(np.float64)
        valid = roi_f > 0.0
        if not np.any(valid):
            return -1.0, None
        mm = roi_f[valid]
        meters = mm * 0.001
        raw_median = float(np.median(mm))
        roi_median_raw = int(round(raw_median)) if math.isfinite(raw_median) else None
    elif encoding in ("32FC1",):
        roi_f = roi.astype(np.float64)
        valid = np.isfinite(roi_f) & (roi_f > 0.0)
        if not np.any(valid):
            return -1.0, None
        meters = roi_f[valid]
        raw_median = float(np.median(meters))
        roi_median_raw = raw_median if math.isfinite(raw_median) else None
    else:
        # Best effort: treat as float meters if float-like, else mm uint
        if np.issubdtype(roi.dtype, np.floating):
            roi_f = roi.astype(np.float64)
            valid = np.isfinite(roi_f) & (roi_f > 0.0)
            if not np.any(valid):
                return -1.0, None
            meters = roi_f[valid]
            raw_median = float(np.median(meters))
            roi_median_raw = raw_median if math.isfinite(raw_median) else None
        else:
            roi_f = roi.astype(np.float64)
            valid = roi_f > 0.0
            if not np.any(valid):
                return -1.0, None
            mm = roi_f[valid]
            meters = mm * 0.001
            raw_median = float(np.median(mm))
            roi_median_raw = int(round(raw_median)) if math.isfinite(raw_median) else None

    med = float(np.median(meters))
    if not math.isfinite(med) or med <= 0.0:
        return -1.0, None
    return med, roi_median_raw


def bbox_clip_to_image(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Clip bbox integer coordinates to image bounds."""
    x_min = int(max(0, min(x_min, width - 1)))
    y_min = int(max(0, min(y_min, height - 1)))
    x_max = int(max(0, min(x_max, width - 1)))
    y_max = int(max(0, min(y_max, height - 1)))
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    return x_min, y_min, x_max, y_max


def robust_center_depth(
    depth_image: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    encoding: str,
    min_depth_m: float = 0.1,
    max_depth_m: float = 2.0,
    outlier_threshold_m: float = 0.02,
    min_valid_samples: int = 5,
) -> Tuple[Optional[float], int, Optional[float], Optional[float], Optional[float]]:
    """Estimate robust center depth from a small window around the bbox center.

    Algorithm:
      1. Collect up to (2*radius+1)^2 raw samples from the window.
      2. Convert each sample to metres, reject invalid (zero/NaN/Inf/out-of-range).
      3. Take the median of valid samples as the anchor.
      4. Reject outliers more than outlier_threshold_m from the median.
      5. Return the mean of remaining samples (or median if too few survive).

    Args:
        depth_image: 2D depth array (H, W).
        center_x: Column index of window center.
        center_y: Row index of window center.
        radius: Half-side of the square window in pixels (window = 2*radius+1 square).
        encoding: sensor_msgs/Image encoding, e.g. ``16UC1`` or ``32FC1``.
        min_depth_m: Minimum plausible depth in metres.
        max_depth_m: Maximum plausible depth in metres.
        outlier_threshold_m: Samples farther than this from the median are rejected.
        min_valid_samples: Minimum valid samples after outlier rejection; if fewer
            remain, return None.

    Returns:
        Tuple of (final_depth_m, valid_count, raw_center_depth_m, median_depth_m, filtered_mean_m):
        - final_depth_m: robust filtered depth in metres, or None if not enough valid samples.
        - valid_count: number of raw valid samples collected (before outlier rejection).
        - raw_center_depth_m: depth at the exact center pixel (single-point reference), in metres.
        - median_depth_m: median of all valid raw samples (before outlier rejection).
        - filtered_mean_m: mean of samples within outlier_threshold_m of the median.
    """
    if depth_image is None or depth_image.size == 0:
        return None, 0, None, None, None

    h, w = depth_image.shape[:2]
    if h < 1 or w < 1:
        return None, 0, None, None, None

    # ── Step 1: collect raw samples from window ────────────────────────────────
    x0 = max(0, center_x - radius)
    y0 = max(0, center_y - radius)
    x1 = min(w, center_x + radius + 1)
    y1 = min(h, center_y + radius + 1)

    if x0 >= x1 or y0 >= y1:
        return None, 0, None, None, None

    window = depth_image[y0:y1, x0:x1]

    # ── Convert to metres and filter invalid ───────────────────────────────────
    raw_center_m: Optional[float] = None

    def _to_meters(val) -> Optional[float]:
        if encoding in ("16UC1", "mono16"):
            if int(val) == 0:
                return None
            return float(val) * 0.001
        if encoding in ("32FC1",):
            v = float(val)
            if not math.isfinite(v) or v <= 0.0:
                return None
            return v
        # Fallback: treat as float metres if float-like, else mm uint
        if np.issubdtype(window.dtype, np.floating):
            v = float(val)
            if not math.isfinite(v) or v <= 0.0:
                return None
            return v
        else:
            if int(val) == 0:
                return None
            return float(val) * 0.001

    valid_meters: list[float] = []
    center_row = center_y - y0
    center_col = center_x - x0

    for row in range(window.shape[0]):
        for col in range(window.shape[1]):
            m = _to_meters(window[row, col])
            if m is None:
                continue
            if m < min_depth_m or m > max_depth_m:
                continue
            if row == center_row and col == center_col:
                raw_center_m = m
            valid_meters.append(m)

    valid_count = len(valid_meters)

    if valid_count < min_valid_samples:
        return None, valid_count, raw_center_m, None, None

    valid_arr = np.array(valid_meters, dtype=np.float64)

    # ── Step 3: median anchor ─────────────────────────────────────────────────
    median_m = float(np.median(valid_arr))

    # ── Step 4: reject outliers ──────────────────────────────────────────────
    deviations = np.abs(valid_arr - median_m)
    mask = deviations <= outlier_threshold_m
    filtered = valid_arr[mask]

    if len(filtered) < min_valid_samples:
        # Not enough after outlier rejection → fall back to median
        return float(median_m), valid_count, raw_center_m, median_m, float(median_m)

    filtered_mean_m = float(np.mean(filtered))

    return filtered_mean_m, valid_count, raw_center_m, median_m, filtered_mean_m
