from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np


# BWF court dimensions (metres)
COURT = {
    "singles": {"length": 13.40, "width": 5.18},
    "doubles": {"length": 13.40, "width": 6.1},
}

# Zone polygons in metres from court centre (0, 0)
ZONES = {
    "singles_back":   [(-6.7, -2.59), (6.7, -2.59), (6.7, 2.59), (-6.7, 2.59)],
    "doubles_back":   [(-6.7, -3.05), (6.7, -3.05), (6.7, 3.05), (-6.7, 3.05)],
    "service_near":   [(-6.7, -2.59), (0,   -2.59), (0,  2.59),  (-6.7, 2.59)],
    "service_far":    [(0,    -2.59), (6.7, -2.59), (6.7, 2.59),  (0,   2.59)],
}

# Margin within which we escalate to Gemma4 (metres)
LINE_MARGIN = 0.15

class CourtHomography:
    def __init__(self):
        self.H = None
        self.H_inv = None

    def calibrate(self, pixel_pts: np.ndarray, court_pts: np.ndarray):
        self.H, _ = cv2.findHomography(pixel_pts, court_pts)
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)

    def to_court(self, px: float, py: float) -> tuple:
        if self.H is None:
            raise RuntimeError("Homography not calibrated — call calibrate() first")
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0][0][0]), float(out[0][0][1])


def point_in_polygon(x: float, y: float, polygon: list) -> bool:
    pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, (float(x), float(y)), measureDist=False) >= 0


def distance_to_boundary(x: float, y: float, polygon: list) -> float:
    """Returns signed distance (metres); positive = inside, negative = outside."""
    pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, (float(x), float(y)), measureDist=True)


# ─── Corner ordering ────────────────────────────────────────────────────────

def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]   # y - x
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ], dtype=np.float32)


# ─── Mask → quadrilateral ───────────────────────────────────────────────────

def mask_to_quadrilateral(mask: np.ndarray) -> Optional[np.ndarray]:
    
    if mask is None or mask.size == 0:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    main = max(contours, key=cv2.contourArea)
    if cv2.contourArea(main) <= 0:
        return None

    hull = cv2.convexHull(main)
    peri = cv2.arcLength(hull, True)

    # Try increasing epsilon until we land on exactly 4 vertices.
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
            return order_corners(quad)

    # approxPolyDP failed → fall back to extremal-hull-points selection.
    hull_pts = hull.reshape(-1, 2).astype(np.float32)
    if len(hull_pts) < 4:
        return None
    s = hull_pts.sum(axis=1)
    d = hull_pts[:, 1] - hull_pts[:, 0]
    quad = np.array([
        hull_pts[np.argmin(s)],   # TL
        hull_pts[np.argmin(d)],   # TR
        hull_pts[np.argmax(s)],   # BR
        hull_pts[np.argmax(d)],   # BL
    ], dtype=np.float32)
    return order_corners(quad)


# ─── Validation ─────────────────────────────────────────────────────────────

@dataclass
class ValidationReport:
    ok: bool
    reasons: List[str]
    reprojection_error_px: Optional[float] = None  # mean px error for canonical lines


def validate_corners(
    corners: np.ndarray,
    image_shape: Tuple[int, int],
    margin_ratio: float = 0.35,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.97,
) -> ValidationReport:
    
    reasons: List[str] = []
    h, w = image_shape

    if corners is None or corners.shape != (4, 2):
        return ValidationReport(False, ["corners shape is not (4, 2)"])

    margin_x = w * margin_ratio
    margin_y = h * margin_ratio
    for i, (x, y) in enumerate(corners):
        if not (-margin_x <= x <= w + margin_x and -margin_y <= y <= h + margin_y):
            reasons.append(f"corner {i} out of frame bounds: ({x:.0f}, {y:.0f})")

    # Convex check (court is always convex when seen through a normal lens).
    if not cv2.isContourConvex(corners.reshape(4, 1, 2).astype(np.int32)):
        reasons.append("quadrilateral is non-convex")

    area = cv2.contourArea(corners.reshape(4, 1, 2))
    image_area = float(w * h)
    if area / image_area < min_area_ratio:
        reasons.append(f"quad area too small: {area/image_area:.2%} of frame")
    elif area / image_area > max_area_ratio:
        reasons.append(f"quad area too large: {area/image_area:.2%} of frame")

    # Aspect-ratio sanity. We don't know perspective, but a degenerate quad will
    # show as very unequal opposite-side lengths or near-zero diagonal.
    tl, tr, br, bl = corners
    top = np.linalg.norm(tr - tl)
    bot = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    if min(top, bot, left, right) < min(w, h) * 0.05:
        reasons.append("quadrilateral has a degenerately short side")

    return ValidationReport(ok=(len(reasons) == 0), reasons=reasons)


# ─── Homography fitting ─────────────────────────────────────────────────────

def fit_homography(
    pixel_corners: np.ndarray,
    mode: str = "doubles",
) -> Optional[CourtHomography]:
    
    spec = COURT.get(mode) or COURT["doubles"]
    half_len = spec["length"] / 2.0   # x extent (court length runs along x in metres)
    half_wid = spec["width"] / 2.0    # y extent

    # Image convention: TL is "far left" (small x in image, on far baseline) →
    # world: far baseline at +x, near baseline at -x; left sideline at +y.
    # The exact orientation is arbitrary as long as it's a valid homography.
    court_pts = np.array([
        [+half_len, +half_wid],   # TL → far  left
        [+half_len, -half_wid],   # TR → far  right
        [-half_len, -half_wid],   # BR → near right
        [-half_len, +half_wid],   # BL → near left
    ], dtype=np.float32)

    pixel_pts = np.asarray(pixel_corners, dtype=np.float32).reshape(4, 2)
    homo = CourtHomography()
    homo.calibrate(pixel_pts, court_pts)
    if homo.H is None:
        return None
    return homo


# ─── Canonical line set + projection ────────────────────────────────────────

@dataclass
class CourtLine:
    """A single canonical court line segment in court (metres) coordinates."""
    name: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    kind: str   # "boundary" | "service" | "center" | "singles"


def canonical_court_lines(mode: str = "doubles") -> List[CourtLine]:
    
    half_len = 6.70
    doubles_half = 3.05
    singles_half = 2.59
    short_service = 1.98
    long_service_doubles = 6.70 - 0.76   # = 5.94

    lines: List[CourtLine] = []

    # Outer boundary width depends on mode — must match what fit_homography used.
    outer_half = singles_half if mode == "singles" else doubles_half

    lines += [
        CourtLine("baseline_far",    (+half_len, +outer_half), (+half_len, -outer_half), "boundary"),
        CourtLine("baseline_near",   (-half_len, +outer_half), (-half_len, -outer_half), "boundary"),
        CourtLine("sideline_left",   (+half_len, +outer_half), (-half_len, +outer_half), "boundary"),
        CourtLine("sideline_right",  (+half_len, -outer_half), (-half_len, -outer_half), "boundary"),
    ]

    if mode == "doubles":
        # Singles sidelines drawn as thinner internal lines
        lines += [
            CourtLine("singles_left",  (+half_len, +singles_half), (-half_len, +singles_half), "singles"),
            CourtLine("singles_right", (+half_len, -singles_half), (-half_len, -singles_half), "singles"),
        ]
        # Doubles long service lines
        lines += [
            CourtLine("doubles_long_service_far",
                      (+long_service_doubles, +doubles_half),
                      (+long_service_doubles, -doubles_half),
                      "service"),
            CourtLine("doubles_long_service_near",
                      (-long_service_doubles, +doubles_half),
                      (-long_service_doubles, -doubles_half),
                      "service"),
        ]

    # Net (centre) — spans the outer boundary width for the current mode.
    lines.append(CourtLine("net", (0.0, +outer_half), (0.0, -outer_half), "center"))

    # Short service lines (both sides of net)
    half = doubles_half if mode == "doubles" else singles_half
    lines += [
        CourtLine("short_service_far",  (+short_service, +half), (+short_service, -half), "service"),
        CourtLine("short_service_near", (-short_service, +half), (-short_service, -half), "service"),
    ]

    # Centre service line: runs y=0 between the two short service lines.
    lines.append(CourtLine("center_service",
                            (+short_service, 0.0), (-short_service, 0.0), "center"))

    return lines


def project_court_lines(
    homography: CourtHomography,
    mode: str = "doubles",
) -> List[Tuple[CourtLine, Tuple[Tuple[int, int], Tuple[int, int]]]]:
    
    if homography.H_inv is None:
        raise RuntimeError("Homography not calibrated (H_inv is None)")

    lines = canonical_court_lines(mode)
    out = []
    pts_court = np.array(
        [[ln.start, ln.end] for ln in lines], dtype=np.float32
    ).reshape(-1, 1, 2)
    pts_pixel = cv2.perspectiveTransform(pts_court, homography.H_inv).reshape(-1, 2)

    for i, ln in enumerate(lines):
        p1 = tuple(int(round(v)) for v in pts_pixel[2 * i])
        p2 = tuple(int(round(v)) for v in pts_pixel[2 * i + 1])
        out.append((ln, (p1, p2)))
    return out
