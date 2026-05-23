from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    """Output of segmenting court surface from a single frame."""
    mask: np.ndarray              # uint8 binary, 0 / 255
    largest_area_ratio: float     # area(largest_component) / total_pixels
    sampled_hue: float            # the hue we used as anchor
    sampled_sat_lo: int           # saturation threshold used


def segment_court_hsv(
    image: np.ndarray,
    hue_tolerance: int = 25,
    center_patch_ratio: float = 0.25,
    min_area_ratio: float = 0.05,
) -> Optional[SegmentationResult]:
    
    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ── Sample hue from central patch ─────────────────────────────────────
    cx, cy = w // 2, h // 2
    half_w = max(8, int(w * center_patch_ratio / 2))
    half_h = max(8, int(h * center_patch_ratio / 2))
    x0, x1 = max(0, cx - half_w), min(w, cx + half_w)
    y0, y1 = max(0, cy - half_h), min(h, cy + half_h)
    patch = hsv[y0:y1, x0:x1]

    if patch.size == 0:
        return None

    not_white = patch[:, :, 1] > 30
    hues = patch[:, :, 0][not_white] if not_white.any() else patch[:, :, 0].flatten()
    sampled_hue = float(np.median(hues))
    sampled_sat = float(np.median(patch[:, :, 1]))
    sat_lo = max(20, int(sampled_sat * 0.40))

    # ── Build hue mask ────────────────────────────────────────────────────
    lo_h = max(0, int(sampled_hue) - hue_tolerance)
    hi_h = min(180, int(sampled_hue) + hue_tolerance)
    mask = cv2.inRange(hsv, (lo_h, sat_lo, 15), (hi_h, 255, 255))

    # Drop the saturated whites (advertising boards, score overlays)
    bright_white = cv2.inRange(hsv, (0, 0, 220), (180, 25, 255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(bright_white))

    # ── Morphological cleanup ────────────────────────────────────────────
    # Close: bridge across the painted court lines (these are inside the court
    # surface but appear as gaps in the HSV mask). Kernel size scales with
    # frame width to track resolution.
    # Court lines are ~3-8 px wide so a small kernel bridges them.
    # w//30 was too large and bridged the run-off zone into the advertising boards.
    close_px = max(10, w // 80)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # Open: kill stray small specks (crowd, sponsor patches that match hue).
    open_px = max(8, w // 160)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # ── Keep only the largest connected component ───────────────────────
    largest = _largest_component(mask)
    if largest is None:
        return None

    area_ratio = float(largest.sum()) / 255.0 / (h * w)
    if area_ratio < min_area_ratio:
        return None

    return SegmentationResult(
        mask=largest,
        largest_area_ratio=area_ratio,
        sampled_hue=sampled_hue,
        sampled_sat_lo=sat_lo,
    )


def _largest_component(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return a uint8 mask containing only the largest connected component."""
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return None
    # Component 0 is background; pick largest among the rest by area.
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return None
    biggest_idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(mask)
    out[labels == biggest_idx] = 255
    return out
