"""Draw YOLO box and labels on a BGR image for debug publishing."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from robot_vision_pipeline_msgs.msg import BoxDetection


def _measure_text_block(
    lines: List[str],
    font: int,
    font_scale: float,
    thickness: int,
    line_gap: int,
    padding: int,
) -> Tuple[int, int, List[Tuple[str, int, int, int]]]:
    """Return (block_w, block_h, rows) where each row is (text, tw, th, baseline)."""
    rows: List[Tuple[str, int, int, int]] = []
    max_tw = 0
    cursor = padding
    for line in lines:
        (tw, th), bl = cv2.getTextSize(line, font, font_scale, thickness)
        rows.append((line, tw, th, bl))
        max_tw = max(max_tw, tw)
        cursor += th + bl + line_gap
    block_h = cursor - line_gap + padding
    block_w = max_tw + 2 * padding
    return block_w, block_h, rows


def _draw_multiline_text_block(
    img: np.ndarray,
    lines: List[str],
    top_left_x: int,
    top_left_y: int,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.55,
    thickness: int = 1,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    line_gap: int = 2,
    padding: int = 6,
) -> None:
    """Draw several left-aligned lines with one solid background rectangle."""
    h, w = img.shape[:2]
    block_w, block_h, rows = _measure_text_block(
        lines, font, font_scale, thickness, line_gap, padding
    )

    x0 = max(0, min(top_left_x, w - block_w))
    y0 = max(0, min(top_left_y, h - block_h))
    x1 = min(w - 1, x0 + block_w)
    y1 = min(h - 1, y0 + block_h)

    cv2.rectangle(img, (x0, y0), (x1, y1), bg_color, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (80, 80, 80), 1, cv2.LINE_AA)

    cursor = y0 + padding
    for line, _tw, th, bl in rows:
        baseline_y = min(cursor + th, h - 1)
        cv2.putText(
            img,
            line,
            (x0 + padding, baseline_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )
        cursor = baseline_y + bl + line_gap


def _clamp_text_block_origin(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    block_w: int,
    block_h: int,
    img_w: int,
    img_h: int,
    margin: int,
) -> Tuple[int, int]:
    """Place block above bbox if possible, else below; keep inside image."""
    tx = x_min
    ty_above = y_min - margin - block_h
    if ty_above >= 0:
        ty = ty_above
    else:
        ty = min(y_max + margin, img_h - block_h)
    tx = max(0, min(tx, img_w - block_w))
    ty = max(0, min(ty, img_h - block_h))
    return tx, ty


# Per-class colours for the debug image overlay text (BGR convention).
OVERLAY_TEXT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "box":    (0, 255, 255),      # yellow
    "target": (255, 128, 0),      # blue-orange (read as BGR: bluish)
}


def draw_all_detections(
    bgr: np.ndarray,
    detections: List[BoxDetection],
    class_colors: Dict[str, Tuple[int, int, int]],
    *,
    physical_sizes_mm: Optional[List[Optional[Tuple[float, float, float]]]] = None,
) -> np.ndarray:
    """Draw all detections on the image with pixel + distance overlay.

    Overlay format:
      target: "target conf=X.XX" / "center=(u=XXX,v=XXX)" / "dist=X.XXXm"
      box:    "box conf=X.XX"     / "center=(u=XXX,v=XXX)" / "dist=X.XXXm"
              / "size_px=(w=XXX,h=XXX)"

    Args:
        bgr: Input BGR image.
        detections: List of BoxDetection messages.
        class_colors: Mapping from class_name (lower) → (B, G, R) colour tuple.
        physical_sizes_mm: Optional parallel list of (x_mm, y_mm, z_mm) tuples for
            physical size in millimetres, one per detection. If None or an entry is
            None, physical size is omitted for that detection.
    """
    out = bgr.copy()
    ih, iw = out.shape[:2]

    if not detections:
        return draw_no_detection(out, "No detection")

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 1
    gap = 2
    pad = 6

    for i, det in enumerate(detections):
        color = class_colors.get(det.class_name.strip().lower(), (0, 255, 0))
        x_min, y_min = det.x_min, det.y_min
        x_max, y_max = det.x_max, det.y_max

        # Clamp
        x_min = max(0, min(x_min, iw - 1))
        y_min = max(0, min(y_min, ih - 1))
        x_max = max(0, min(x_max, iw - 1))
        y_max = max(0, min(y_max, ih - 1))

        cx = det.center_x if det.center_x >= 0 else (x_min + x_max) // 2
        cy = det.center_y if det.center_y >= 0 else (y_min + y_max) // 2
        wp = det.width_px if det.width_px > 0 else int(x_max - x_min)
        hp = det.height_px if det.height_px > 0 else int(y_max - y_min)

        # Draw bbox
        cv2.rectangle(out, (x_min, y_min), (x_max, y_max), color, 2, cv2.LINE_AA)

        # Draw crosshairs
        cv2.line(out, (0, cy), (iw - 1, cy), color, 1, cv2.LINE_AA)
        cv2.line(out, (cx, 0), (cx, ih - 1), color, 1, cv2.LINE_AA)

        # Draw centre marker
        r = max(4, min(iw, ih) // 120)
        r = min(r, 14)
        cv2.circle(out, (cx, cy), r, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), r, (255, 255, 255), 1, cv2.LINE_AA)

        # Build overlay lines
        label = det.class_name.strip().lower()
        text_color = OVERLAY_TEXT_COLORS.get(label, (255, 255, 255))

        dist_m = det.distance_m
        if dist_m >= 0.0:
            dist_txt = f"dist={dist_m:.3f}m"
        else:
            dist_txt = "dist=n/a"

        lines = [
            f"{label} conf={det.confidence:.2f}",
            f"center=(u={cx},v={cy})",
            dist_txt,
        ]

        if label == "box":
            lines.append(f"size_px=(w={wp},h={hp})")
            if physical_sizes_mm is not None and i < len(physical_sizes_mm):
                phys = physical_sizes_mm[i]
                if phys is not None:
                    px, py, pz = phys
                    lines.append(f"size_mm=(x={px:.1f},y={py:.1f},z={pz:.1f})")

        block_w, block_h, _ = _measure_text_block(lines, font, scale, thick, gap, pad)
        tx, ty = _clamp_text_block_origin(
            x_min, y_min, x_max, y_max, block_w, block_h, iw, ih, margin=8
        )
        _draw_multiline_text_block(
            out,
            lines,
            tx,
            ty,
            font=font,
            font_scale=scale,
            thickness=thick,
            text_color=text_color,
            bg_color=(0, 0, 0),
            line_gap=gap,
            padding=pad,
        )

    return out


def draw_no_detection(bgr: np.ndarray, message: str = "No detection") -> np.ndarray:
    """Draw status text with solid background when no target detection is present."""
    out = bgr.copy()
    h, w = out.shape[:2]
    lines = [message]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thick = 2
    gap = 2
    pad = 8
    block_w, block_h, _ = _measure_text_block(lines, font, scale, thick, gap, pad)
    tx, ty = 8, 8
    tx = max(0, min(tx, w - block_w))
    ty = max(0, min(ty, h - block_h))
    _draw_multiline_text_block(
        out,
        lines,
        tx,
        ty,
        font=font,
        font_scale=scale,
        thickness=thick,
        text_color=(0, 200, 255),
        bg_color=(0, 0, 0),
        line_gap=gap,
        padding=pad,
    )
    return out
