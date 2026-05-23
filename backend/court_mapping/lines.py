from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class RefinementDebug:
    """Optional intermediate artefacts for debug visualisation."""
    corridor_masks: List[np.ndarray]                                   # 4 masks, one per side
    detected_segments: List[List[Tuple[float, float, float, float]]]   # per side
    fitted_lines: List[Optional[Tuple[float, float, float]]]           # (a,b,c) per side
    side_was_refined: List[bool]                                       # per side, True if RANSAC succeeded


@dataclass
class RefinementResult:
    refined_corners: np.ndarray                  # (4, 2) float32
    # Per-CORNER mask (TL, TR, BR, BL): True iff both adjacent sides were refined
    # by RANSAC line fitting (not just the rough-quad fallback).
    used_refinement_mask: List[bool]
    debug: Optional[RefinementDebug] = None


# Side ordering: TL→TR (top), TR→BR (right), BR→BL (bottom), BL→TL (left)
_SIDES = [(0, 1), (1, 2), (2, 3), (3, 0)]


def refine_corners(
    image: np.ndarray,
    rough_corners: np.ndarray,
    corridor_outward_sideline: int = 50,
    corridor_inward_sideline: int = 90,
    corridor_outward_baseline: int = 50,
    corridor_inward_baseline: int = 30,
    angle_tol_deg: float = 10.0,
    debug: bool = False,
) -> RefinementResult:
    
    h, w = image.shape[:2]

    # Detect line segments on white-painted regions only.
    segs = _detect_line_segments_white(image)
    if segs is None or len(segs) == 0:
        return RefinementResult(
            refined_corners=rough_corners.copy(),
            used_refinement_mask=[False] * 4,
            debug=RefinementDebug([], [[]] * 4, [None] * 4, [False] * 4) if debug else None,
        )

    # Centroid of the rough quad — used to determine which way is "inward"
    # for each side.
    centroid = np.mean(rough_corners, axis=0).astype(np.float32)

    fitted: List[Tuple[float, float, float]] = []
    side_was_refined: List[bool] = []
    per_side_segments: List[List[Tuple[float, float, float, float]]] = []
    corridor_masks: List[np.ndarray] = []

    for side_idx, (i, j) in enumerate(_SIDES):
        p1, p2 = rough_corners[i], rough_corners[j]
        side_mid = ((p1 + p2) / 2.0).astype(np.float32)
        # Vector from side midpoint toward the centroid = "inward".
        inward = centroid - side_mid
        n_inward = float(np.linalg.norm(inward))
        inward_dir = inward / n_inward if n_inward > 1e-6 else None

        # Sides 0 and 2 are baselines (top/bottom); 1 and 3 are sidelines
        # (right/left). Each gets a different inward search depth — see the
        # docstring for the rationale.
        is_baseline = side_idx % 2 == 0
        if is_baseline:
            outward_extent = corridor_outward_baseline
            inward_extent = corridor_inward_baseline
        else:
            outward_extent = corridor_outward_sideline
            inward_extent = corridor_inward_sideline

        corridor = _make_corridor_mask(
            (h, w), p1, p2,
            outward_extent=outward_extent,
            inward_extent=inward_extent,
            inward_dir=inward_dir,
        )
        if debug:
            corridor_masks.append(corridor)
        candidates = _filter_segments_to_corridor(segs, corridor)
        candidates = _filter_by_angle(candidates, p1, p2, angle_tol_deg)
        # Prefer segments on the outward side of the rough boundary so RANSAC
        # snaps to the outer edge of the white line, not the inner edge.
        candidates = _prefer_outer_segments(candidates, p1, p2, inward_dir)
        if debug:
            per_side_segments.append(candidates)

        line_eq = _ransac_fit_line(candidates) if candidates else None
        if line_eq is None:
            # No usable segments — fall back to the rough side itself.
            line_eq = _line_through_two_points(p1, p2)
            side_was_refined.append(False)
        else:
            side_was_refined.append(True)
        fitted.append(line_eq)

    # Intersect adjacent fitted lines → refined corners.
    # Side order: top (0), right (1), bottom (2), left (3).
    # Corner k is the intersection of side[(k-1)%4] (CCW-prev) and side[k]:
    #   k=0 (TL): left  ∩ top
    #   k=1 (TR): top   ∩ right
    #   k=2 (BR): right ∩ bottom
    #   k=3 (BL): bottom ∩ left
    refined_pts: List[Optional[np.ndarray]] = []
    used: List[bool] = []
    for k in range(4):
        prev_idx = (k - 1) % 4
        next_idx = k % 4
        pt = _intersect_lines(fitted[prev_idx], fitted[next_idx])
        refined_pts.append(pt)
        # Corner is "refined" only if BOTH adjacent sides got a real RANSAC fit
        # AND the intersection succeeded.
        used.append(
            pt is not None
            and side_was_refined[prev_idx]
            and side_was_refined[next_idx]
        )

    final = np.array([
        refined_pts[i] if refined_pts[i] is not None else rough_corners[i]
        for i in range(4)
    ], dtype=np.float32)

    debug_info = None
    if debug:
        debug_info = RefinementDebug(
            corridor_masks=corridor_masks,
            detected_segments=per_side_segments,
            fitted_lines=list(fitted),
            side_was_refined=side_was_refined,
        )

    return RefinementResult(
        refined_corners=final,
        used_refinement_mask=used,
        debug=debug_info,
    )


# ─── Line detection ─────────────────────────────────────────────────────────

def _detect_line_segments_white(
    image_bgr: np.ndarray,
    v_min: int = 160,
    s_max: int = 70,
) -> List[Tuple[float, float, float, float]]:
    """
    Detect line segments restricted to WHITE-PAINTED regions.

    Court lines are painted white (high V, low S in HSV). By masking the
    grayscale image to white-only pixels before running LSD we suppress
    false-positive lines from sponsor banners, scoreboard borders, and the
    floor's run-off-zone outer contrast edge.
    """
    if image_bgr is None or image_bgr.size == 0:
        return []
    if image_bgr.ndim == 2:
        gray = image_bgr
        # Without colour information we can't isolate white; threshold on intensity.
        white_mask = cv2.inRange(gray, v_min, 255)
    else:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Light morphological close to bridge tiny gaps in the painted line mask.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, k)

    # Apply the mask to grayscale: lines stand out as bright pixels on a
    # uniformly-dark background, which gives LSD very clean input.
    masked = cv2.bitwise_and(gray, gray, mask=white_mask)

    return _run_line_detector(masked)


def _run_line_detector(gray: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """Try LSD (cv2 / ximgproc) then fall back to Hough."""
    # Try LSD via the main cv2 module.
    try:
        lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        result = lsd.detect(gray)
        # detect() returns either lines or (lines, ...). Newer OpenCV returns a tuple.
        if isinstance(result, tuple):
            lines = result[0]
        else:
            lines = result
        if lines is None:
            return []
        return [tuple(map(float, ln[0])) for ln in lines]
    except (AttributeError, cv2.error):
        pass

    # Try LSD via ximgproc (separate OpenCV-contrib package).
    try:
        lsd = cv2.ximgproc.createFastLineDetector()
        lines = lsd.detect(gray)
        if lines is None:
            return []
        return [tuple(map(float, ln[0])) for ln in lines]
    except (AttributeError, cv2.error):
        pass

    # Final fallback: Hough segments on Canny edges.
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    h, w = gray.shape[:2]
    min_len = max(int(min(h, w) * 0.05), 30)
    segs = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360, threshold=50,
        minLineLength=min_len, maxLineGap=8,
    )
    if segs is None:
        return []
    return [tuple(map(float, s[0])) for s in segs]


# ─── Corridor masking ───────────────────────────────────────────────────────

def _make_corridor_mask(
    shape: Tuple[int, int],
    p1: np.ndarray,
    p2: np.ndarray,
    outward_extent: int,
    inward_extent: int,
    inward_dir: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Render an oriented rectangle mask along the segment p1→p2.

    The corridor is asymmetric in the perpendicular direction:
    `outward_extent` px on the side AWAY from the court interior, and
    `inward_extent` px on the side TOWARD the court interior. If
    `inward_dir` is None the corridor is symmetric (outward = inward
    perpendicular extent on each side).
    """
    h, w = shape
    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)
    diff = p2 - p1
    length = float(np.linalg.norm(diff))
    if length < 1e-3:
        return np.zeros((h, w), dtype=np.uint8)
    direction = diff / length
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    # Orient `normal` to point INWARD if an inward direction is provided.
    if inward_dir is not None and float(np.dot(normal, inward_dir)) < 0:
        normal = -normal

    # Extend along axis a bit past the corners so candidate segments that
    # overshoot still register.
    extend = max(int(length * 0.05), 10)
    a = p1 - direction * extend
    b = p2 + direction * extend

    # Build rectangle: +normal direction is INWARD; -normal direction is OUTWARD.
    rect = np.array([
        a - normal * outward_extent,   # back-outward
        b - normal * outward_extent,   # forward-outward
        b + normal * inward_extent,    # forward-inward
        a + normal * inward_extent,    # back-inward
    ], dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [rect.astype(np.int32)], 255)
    return mask


def _filter_segments_to_corridor(
    segments: List[Tuple[float, float, float, float]],
    corridor: np.ndarray,
) -> List[Tuple[float, float, float, float]]:
    """Keep only segments whose midpoint is inside the corridor mask."""
    h, w = corridor.shape[:2]
    out = []
    for x1, y1, x2, y2 in segments:
        mx = int(round((x1 + x2) / 2))
        my = int(round((y1 + y2) / 2))
        if 0 <= mx < w and 0 <= my < h and corridor[my, mx]:
            out.append((x1, y1, x2, y2))
    return out


def _filter_by_angle(
    segments: List[Tuple[float, float, float, float]],
    p1: np.ndarray,
    p2: np.ndarray,
    tol_deg: float,
) -> List[Tuple[float, float, float, float]]:
    """Keep segments whose direction is within `tol_deg` of the rough side direction."""
    rough_angle = _normalise_angle(_segment_angle_deg(p1[0], p1[1], p2[0], p2[1]))
    out = []
    for x1, y1, x2, y2 in segments:
        a = _normalise_angle(_segment_angle_deg(x1, y1, x2, y2))
        diff = abs(a - rough_angle)
        diff = min(diff, 180.0 - diff)   # angles are mod 180 (line, not vector)
        if diff <= tol_deg:
            out.append((x1, y1, x2, y2))
    return out


def _segment_angle_deg(x1, y1, x2, y2) -> float:
    return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))


def _normalise_angle(a: float) -> float:
    """Return angle in [0, 180)."""
    a = a % 180.0
    return a + 180.0 if a < 0 else a


def _prefer_outer_segments(
    segments: List[Tuple[float, float, float, float]],
    p1: np.ndarray,
    p2: np.ndarray,
    inward_dir: Optional[np.ndarray],
) -> List[Tuple[float, float, float, float]]:
    """Return segments on the outward side of the rough boundary (p1→p2).

    The rough corners from the HSV mask land on the inner edge of the white
    court lines. This filter keeps only segments whose midpoint is on the
    outward side (away from the court interior), which forces RANSAC to fit
    the outer edge of the white line rather than the inner edge.
    Falls back to all segments when fewer than 2 qualify.
    """
    if inward_dir is None or len(segments) == 0:
        return segments
    side_mid = (np.asarray(p1, dtype=np.float64) + np.asarray(p2, dtype=np.float64)) / 2.0
    outer = []
    for seg in segments:
        mx = (seg[0] + seg[2]) / 2.0
        my = (seg[1] + seg[3]) / 2.0
        vec = np.array([mx - side_mid[0], my - side_mid[1]], dtype=np.float64)
        # dot > 0 → segment is inward; dot <= 0 → segment is outward or on the line
        if float(np.dot(vec, inward_dir)) <= 0.0:
            outer.append(seg)
    return outer if len(outer) >= 2 else segments


# ─── Line fitting & intersection ────────────────────────────────────────────

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


def _line_from_segment(seg: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    return _line_through_two_points(np.array([seg[0], seg[1]]),
                                     np.array([seg[2], seg[3]]))


def _ransac_fit_line(
    segments: List[Tuple[float, float, float, float]],
    iterations: int = 200,
    inlier_threshold_px: float = 3.0,
) -> Optional[Tuple[float, float, float]]:
    
    if not segments:
        return None
    if len(segments) == 1:
        return _line_from_segment(segments[0])

    # Build weighted endpoint set: each segment contributes its endpoints
    # weighted by its length (longer segments are more reliable).
    pts_x: List[float] = []
    pts_y: List[float] = []
    weights: List[float] = []
    for x1, y1, x2, y2 in segments:
        L = float(np.hypot(x2 - x1, y2 - y1))
        if L <= 1.0:
            continue
        pts_x += [x1, x2]
        pts_y += [y1, y2]
        weights += [L, L]
    if not pts_x:
        return None

    P = np.column_stack([pts_x, pts_y]).astype(np.float64)
    W = np.array(weights, dtype=np.float64)

    rng = np.random.default_rng(seed=42)
    best_score = -1.0
    best_line: Optional[Tuple[float, float, float]] = None

    for _ in range(min(iterations, len(segments) * 4)):
        idx = int(rng.integers(0, len(segments)))
        cand = _line_from_segment(segments[idx])
        a, b, c = cand
        norm = float(np.hypot(a, b))
        if norm < 1e-6:
            continue
        # Distance of every point to the candidate line.
        dist = np.abs(P[:, 0] * a + P[:, 1] * b + c) / norm
        inliers = dist <= inlier_threshold_px
        score = float(W[inliers].sum())
        if score > best_score:
            best_score = score
            # Refit using inliers via SVD-based total-least-squares.
            best_line = _refit_total_least_squares(P[inliers], W[inliers]) or cand

    return best_line


def _refit_total_least_squares(
    points: np.ndarray, weights: np.ndarray
) -> Optional[Tuple[float, float, float]]:
    """Weighted total least squares: fit a line minimising perpendicular distance."""
    if len(points) < 2:
        return None
    w = weights.reshape(-1, 1)
    centroid = (w * points).sum(axis=0) / w.sum()
    centred = points - centroid
    # Weighted covariance.
    cov = (centred * w).T @ centred / w.sum()
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Smallest eigenvalue → normal to the line.
    normal = eigvecs[:, 0]
    a, b = float(normal[0]), float(normal[1])
    c = -(a * centroid[0] + b * centroid[1])
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
