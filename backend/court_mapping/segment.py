"""
Court-surface segmentation strategies.

Operates on a clean median frame (no players) so the segmentation step gets a
huge head-start over single-frame approaches:
  - colour clusters are stable (no flicker from passing players)
  - the largest connected region of court colour is uncontested
  - white-line pixels stay white (median of "white most of the time" is white)

The primary strategy here is HSV-based with adaptive sampling. We pick the hue
from the central image patch (where the court almost always is on a static
broadcast or amateur cam), build an inRange mask, mask out brightest white
(advertising boards), and clean up morphologically.

A SAM-based strategy can be slotted in later as a drop-in replacement.
"""

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
    """
    Segment the court surface using HSV colour sampling.

    Sampling strategy
    -----------------
    Median of the central image patch (default 25% × 25%) gives the dominant
    court hue. Using the median (not mean) keeps the result stable when a
    player or scoreboard happens to be at the centre. Saturated white pixels
    are excluded from sampling so painted lines don't pull the hue.

    Parameters
    ----------
    image : np.ndarray
        BGR uint8 frame (typically the temporal median frame).
    hue_tolerance : int, default 25
        Half-width of the hue window in OpenCV's 0..180 hue space.
    center_patch_ratio : float, default 0.25
        Fraction of frame edge length used for the sampling patch.
    min_area_ratio : float, default 0.05
        Reject the segmentation if the largest connected component is smaller
        than this fraction of the frame. We expect the court to fill ≥5% of
        the frame on any sane camera setup.

    Returns
    -------
    SegmentationResult | None
        The cleaned binary mask (largest component only) or None on failure.
    """
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
    close_px = max(20, w // 30)
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
