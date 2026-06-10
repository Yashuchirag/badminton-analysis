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

    # ── Sample near court (lower half, centre strip) ──────────────────────
    y0 = max(0, int(h * 0.62))
    y1 = min(h, int(h * 0.88))
    x0 = max(0, int(w * 0.25))
    x1 = min(w, int(w * 0.75))
    patch_near = hsv[y0:y1, x0:x1]
    if patch_near.size == 0:
        return None

    not_white = patch_near[:, :, 1] > 50
    hues_n = patch_near[:, :, 0][not_white] if not_white.any() else patch_near[:, :, 0].flatten()
    near_hue = float(np.median(hues_n))
    near_sat = float(np.median(patch_near[:, :, 1]))

    # ── Near-court mask — restricted to y ≥ 48 % (at or below the net) ───
    sat_lo = max(20, int(near_sat * 0.40))
    lo_h = max(0, int(near_hue) - hue_tolerance)
    hi_h = min(180, int(near_hue) + hue_tolerance)
    mask = cv2.inRange(hsv, (lo_h, sat_lo, 20), (hi_h, 255, 255))
    mask[:int(h * 0.48), :] = 0

    # ── Far-court zone: y = 38–49 % of frame height ──────────────────────
    # The far court lies just above the net and below the advertising boards.
    # At y=38-49 % the frame reliably contains only the far court surface;
    # sampling here avoids the ad banners above (y < 38 %) entirely.
    # NOTE: on two-colour courts the far half can be a completely different
    # hue from the near half (e.g. teal H≈114 vs yellow-green H≈65), so we
    # sample and mask them independently.
    fy0 = max(0, int(h * 0.38))
    fy1 = min(h, int(h * 0.49))
    fx0 = max(0, int(w * 0.30))
    fx1 = min(w, int(w * 0.70))
    patch_far = hsv[fy0:fy1, fx0:fx1]

    if patch_far.size > 0:
        # Exclude white-painted court lines (low saturation) from the hue sample.
        not_line = patch_far[:, :, 1] > 15
        hues_f = patch_far[:, :, 0][not_line]
        if len(hues_f) >= 100:
            far_hue = float(np.median(hues_f))
            far_sat = float(np.median(patch_far[:, :, 1][not_line]))
            sat_lo_f = max(8, int(far_sat * 0.30))
            lo_hf = max(0, int(far_hue) - (hue_tolerance + 10))
            hi_hf = min(180, int(far_hue) + (hue_tolerance + 10))
            mask_far = cv2.inRange(hsv, (lo_hf, sat_lo_f, 20), (hi_hf, 255, 255))
            # Restrict far mask to y = 35–52 % — keeps ad banners above 35 % clean
            mask_far[:int(h * 0.35), :] = 0
            mask_far[int(h * 0.52):, :] = 0
            # Restrict far mask x-range: the far court in perspective occupies
            # roughly the centre 50% of the frame. Audience seating at the far end
            # shares the same teal hue, so x-position is the only discriminator.
            mask_far[:, :int(w * 0.25)] = 0
            mask_far[:, int(w * 0.75):] = 0
            mask = cv2.bitwise_or(mask, mask_far)

    # ── Drop very bright whites (advertising boards, score overlays) ──────
    bright_white = cv2.inRange(hsv, (0, 0, 220), (180, 25, 255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(bright_white))

    # ── Morphological cleanup ─────────────────────────────────────────────
    # 1. Narrow vertical close: bridges only the actual net gap (≈ 15–30 px).
    #    Kept small so it does not merge the court with ad banners above it.
    net_gap_px = max(50, h // 12)
    bridge_w   = max(12, w // 120)
    k_net = cv2.getStructuringElement(cv2.MORPH_RECT, (bridge_w, net_gap_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_net)

    # 2. Isotropic close: fill small gaps along court lines.
    close_px = max(8, w // 100)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # 3. Open: remove isolated specks (crowd pixels, sponsor patches).
    open_px = max(6, w // 180)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # ── Keep only the largest connected component ─────────────────────────
    largest = _largest_component(mask)
    if largest is None:
        return None

    area_ratio = float(largest.sum()) / 255.0 / (h * w)
    if area_ratio < min_area_ratio:
        return None

    return SegmentationResult(
        mask=largest,
        largest_area_ratio=area_ratio,
        sampled_hue=near_hue,
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
