from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RefinementDebug:
    """Optional intermediate artefacts for debug visualisation."""
    white_mask: Optional[np.ndarray]                          # white pixels inside search region
    component_mask: Optional[np.ndarray]                      # largest connected white component
    boundary_points: List[np.ndarray]                         # extreme points per side (top, right, bottom, left)
    fitted_lines: List[Optional[Tuple[float, float, float]]]  # (a,b,c) per side
    side_was_refined: List[bool]                              # per side, True if the robust fit succeeded


@dataclass
class RefinementResult:
    refined_corners: np.ndarray                  # (4, 2) float32
    # Per-CORNER mask (TL, TR, BR, BL): True iff both adjacent sides were refined
    # by robust line fitting (not just the rough-quad fallback).
    used_refinement_mask: List[bool]
    debug: Optional[RefinementDebug] = None


# Side ordering: TL→TR (top), TR→BR (right), BR→BL (bottom), BL→TL (left)
_SIDES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def refine_corners(
    image: np.ndarray,
    rough_corners: np.ndarray,
    surface_mask: Optional[np.ndarray] = None,
    v_min: int = 160,
    s_max: int = 70,
    debug: bool = False,
) -> RefinementResult:
    """
    Refine the rough court corners by fitting lines to the white line grid.

    The painted court lines form a single connected white component on the
    court surface, by far the largest white blob in the playing area. Its
    extreme pixels (topmost per column, bottommost per column, leftmost and
    rightmost per row) trace the OUTER edge of the doubles boundary lines,
    which is the official court boundary. Each side is fitted with a robust
    RANSAC line through those extreme points, and adjacent lines are
    intersected to obtain the corners.

    This avoids the failure mode of segment/corridor search around the rough
    quad: the HSV surface mask bleeds onto the green apron and sponsor-banner
    strip, so the rough sides can sit far from the true lines and corridor
    search then locks onto white banner text instead of the baseline.

    `surface_mask` should be the HSV court-surface mask. If omitted, the
    filled rough quadrilateral is used as the search region instead.
    """
    h, w = image.shape[:2]
    fallback = RefinementResult(
        refined_corners=rough_corners.copy(),
        used_refinement_mask=[False] * 4,
        debug=RefinementDebug(None, None, [], [None] * 4, [False] * 4) if debug else None,
    )

    # ── 1. white-pixel mask restricted to the court region ──────────────
    if image.ndim == 2:
        white = cv2.inRange(image, v_min, 255)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))

    if surface_mask is not None:
        region = surface_mask
    else:
        region = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(region, [rough_corners.astype(np.int32)], 255)
    # Dilate so the white lines on the region border are fully included.
    region = cv2.dilate(region, np.ones((31, 31), np.uint8))
    white = cv2.bitwise_and(white, region)

    # ── 2. largest connected component = the court line grid ────────────
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    if n_labels < 2:
        return fallback
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[big, cv2.CC_STAT_AREA])
    bbox_w = int(stats[big, cv2.CC_STAT_WIDTH])
    bbox_h = int(stats[big, cv2.CC_STAT_HEIGHT])

    rough_w = float(rough_corners[:, 0].max() - rough_corners[:, 0].min())
    rough_h = float(rough_corners[:, 1].max() - rough_corners[:, 1].min())
    # Sanity: the grid must be substantial and span most of the rough quad.
    if area < 3000 or bbox_w < 0.5 * rough_w or bbox_h < 0.5 * rough_h:
        return fallback

    component = (labels == big)
    ys, xs = np.nonzero(component)

    # ── 3. extreme boundary points per side ──────────────────────────────
    top_pts, bottom_pts = _column_extremes(xs, ys)      # far / near baseline
    left_pts, right_pts = _row_extremes(xs, ys)         # left / right sideline

    # ── 4. robust line fit per side ──────────────────────────────────────
    # Baselines are near-horizontal; sidelines lean strongly with perspective,
    # so they get a wide tolerance around vertical.
    fitted: List[Optional[Tuple[float, float, float]]] = [
        _fit_line_to_points(top_pts, expected_angle_deg=0.0, angle_tol_deg=20.0),
        _fit_line_to_points(right_pts, expected_angle_deg=90.0, angle_tol_deg=45.0),
        _fit_line_to_points(bottom_pts, expected_angle_deg=0.0, angle_tol_deg=20.0),
        _fit_line_to_points(left_pts, expected_angle_deg=90.0, angle_tol_deg=45.0),
    ]
    side_was_refined = [ln is not None for ln in fitted]
    for side_idx, (i, j) in enumerate(_SIDES):
        if fitted[side_idx] is None:
            fitted[side_idx] = _line_through_two_points(rough_corners[i], rough_corners[j])

    # ── 5. intersect adjacent lines → corners ────────────────────────────
    # Side order: top (0), right (1), bottom (2), left (3).
    # Corner k is the intersection of side[(k-1)%4] and side[k]:
    #   k=0 (TL): left ∩ top, k=1 (TR): top ∩ right,
    #   k=2 (BR): right ∩ bottom, k=3 (BL): bottom ∩ left
    refined_pts: List[Optional[np.ndarray]] = []
    used: List[bool] = []
    for k in range(4):
        prev_idx = (k - 1) % 4
        pt = _intersect_lines(fitted[prev_idx], fitted[k])
        refined_pts.append(pt)
        used.append(
            pt is not None
            and side_was_refined[prev_idx]
            and side_was_refined[k]
        )

    final = np.array([
        refined_pts[i] if refined_pts[i] is not None else rough_corners[i]
        for i in range(4)
    ], dtype=np.float32)

    debug_info = None
    if debug:
        debug_info = RefinementDebug(
            white_mask=white,
            component_mask=component.astype(np.uint8) * 255,
            boundary_points=[top_pts, right_pts, bottom_pts, left_pts],
            fitted_lines=list(fitted),
            side_was_refined=side_was_refined,
        )

    return RefinementResult(
        refined_corners=final,
        used_refinement_mask=used,
        debug=debug_info,
    )


# ─── Boundary point extraction ──────────────────────────────────────────────

def _column_extremes(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Topmost and bottommost component pixel per column, as (x, y) arrays."""
    order = np.lexsort((ys, xs))
    x_s, y_s = xs[order], ys[order]
    ux, start = np.unique(x_s, return_index=True)
    end = np.r_[start[1:], len(x_s)] - 1
    top = np.column_stack([ux, y_s[start]]).astype(np.float64)
    bottom = np.column_stack([ux, y_s[end]]).astype(np.float64)
    return top, bottom


def _row_extremes(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Leftmost and rightmost component pixel per row, as (x, y) arrays."""
    order = np.lexsort((xs, ys))
    x_s, y_s = xs[order], ys[order]
    uy, start = np.unique(y_s, return_index=True)
    end = np.r_[start[1:], len(y_s)] - 1
    left = np.column_stack([x_s[start], uy]).astype(np.float64)
    right = np.column_stack([x_s[end], uy]).astype(np.float64)
    return left, right


# ─── Line fitting & intersection ────────────────────────────────────────────

def _fit_line_to_points(
    points: np.ndarray,
    expected_angle_deg: float,
    angle_tol_deg: float,
    iterations: int = 400,
    inlier_threshold_px: float = 2.0,
    min_inliers: int = 30,
) -> Optional[Tuple[float, float, float]]:
    """
    RANSAC line fit through boundary points with an angle prior, refined by
    total least squares on the inliers. Returns implicit (a, b, c) with
    a*x + b*y + c = 0, or None when no acceptable line is found.
    """
    P = np.asarray(points, dtype=np.float64)
    if len(P) < min_inliers:
        return None

    rng = np.random.default_rng(seed=42)
    best: Optional[Tuple[float, float, float]] = None
    best_n = 0
    expected = expected_angle_deg % 180.0

    for _ in range(iterations):
        i, j = rng.integers(0, len(P), size=2)
        d = P[j] - P[i]
        span = float(np.hypot(d[0], d[1]))
        if span < 20.0:
            continue
        angle = float(np.degrees(np.arctan2(d[1], d[0]))) % 180.0
        diff = abs(angle - expected)
        if min(diff, 180.0 - diff) > angle_tol_deg:
            continue
        a, b = d[1] / span, -d[0] / span
        c = -(a * P[i, 0] + b * P[i, 1])
        dist = np.abs(P[:, 0] * a + P[:, 1] * b + c)
        n = int((dist <= inlier_threshold_px).sum())
        if n > best_n:
            best_n, best = n, (a, b, c)

    if best is None or best_n < min_inliers:
        return None

    # Total-least-squares refit on the inliers.
    a, b, c = best
    dist = np.abs(P[:, 0] * a + P[:, 1] * b + c)
    inliers = P[dist <= inlier_threshold_px]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    direction = vt[0]
    a, b = float(direction[1]), float(-direction[0])
    c = float(-(a * centroid[0] + b * centroid[1]))
    return (a, b, c)


def _line_through_two_points(
    p1: np.ndarray, p2: np.ndarray
) -> Tuple[float, float, float]:
    """Return implicit line (a, b, c) such that a*x + b*y + c = 0."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    a = y2 - y1
    b = x1 - x2
    c = -(a * x1 + b * y1)
    return (a, b, c)


def _intersect_lines(
    line1: Optional[Tuple[float, float, float]],
    line2: Optional[Tuple[float, float, float]],
) -> Optional[np.ndarray]:
    if line1 is None or line2 is None:
        return None
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y], dtype=np.float32)
