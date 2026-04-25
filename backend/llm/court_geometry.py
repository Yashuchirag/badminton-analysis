import numpy as np
import cv2

# BWF singles court: 13.4m long x 5.18m wide
# Doubles: 13.4m x 6.1m
COURT = {
    "singles": {"length": 13.40, "width": 5.18},
    "doubles": {"length": 13.40, "width": 6.1},
}

# All in metres from court centre (0,0)
ZONES = {
    "singles_back":   [(-6.7, -2.59), (6.7, -2.59), (6.7, 2.59), (-6.7, 2.59)],
    "doubles_back":   [(-6.7, -3.05), (6.7, -3.05), (6.7, 3.05), (-6.7, 3.05)],
    "service_near":   [(-6.7, -2.59), (0,   -2.59), (0,  2.59),  (-6.7, 2.59)],
    "service_far":    [(0,    -2.59), (6.7, -2.59), (6.7, 2.59),  (0,   2.59)],
}

# Margin within which we escalate to Gemma4 (metres).
# Tracking has ~5-10px error which maps to ~10cm on court; 15cm gives safe headroom.
LINE_MARGIN = 0.15   # 15cm

# Court real-world corners in TL→TR→BR→BL order (matches _order_corners output).
# For IN/OUT checking the specific near/far or left/right orientation doesn't
# matter because the court boundary is a rectangle that is symmetric on all axes.
_COURT_PTS = {
    "singles": [[6.7, 2.59], [6.7, -2.59], [-6.7, -2.59], [-6.7, 2.59]],
    "doubles": [[6.7, 3.05], [6.7, -3.05], [-6.7, -3.05], [-6.7, 3.05]],
}


class CourtHomography:
    def __init__(self):
        self.H = None
        self.H_inv = None   # pixel ← court (for overlay drawing)

    def calibrate(self, pixel_pts: np.ndarray, court_pts: np.ndarray):
        self.H, _ = cv2.findHomography(pixel_pts, court_pts)
        if self.H is not None:
            self.H_inv = np.linalg.inv(self.H)

    def auto_calibrate(self, video_path: str, mode: str = "singles",
                       n_frames: int = 90, debug: bool = False) -> list:
        """
        Auto-detect court corners from the video and calibrate.
        Returns the detected pixel_pts list (useful for debugging).
        """
        pixel_pts, court_pts = auto_calibrate_court(
            video_path, n_frames=n_frames, mode=mode, debug=debug
        )
        self.calibrate(np.array(pixel_pts, dtype=np.float32),
                       np.array(court_pts, dtype=np.float32))
        return pixel_pts

    def to_court(self, px: float, py: float) -> tuple[float, float]:
        if self.H is None:
            raise RuntimeError("Homography not calibrated — call calibrate() or auto_calibrate() first")
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0][0][0]), float(out[0][0][1])


def point_in_polygon(x: float, y: float, polygon: list) -> bool:
    pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(pts, (float(x), float(y)), measureDist=False)
    return result >= 0


def distance_to_boundary(x: float, y: float, polygon: list) -> float:
    """Returns signed distance (metres); positive = inside, negative = outside."""
    # OpenCV pointPolygonTest requires contour shape (N, 1, 2)
    pts = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
    return cv2.pointPolygonTest(pts, (float(x), float(y)), measureDist=True)


# ─────────────────────────────────────────────────────────────────────────────
# Auto-calibration: detect court corners from video without manual interaction
# ─────────────────────────────────────────────────────────────────────────────

def auto_calibrate_court(
        video_path: str,
        n_frames: int = 30,
        frame_skip: int = 3,
        mode: str = "singles",
        debug: bool = False,
) -> tuple[list, list]:
    """
    Detect court corners automatically from any video — no manual interaction.

    Primary strategy: detect the coloured court surface (green/blue) as a large
    solid region, then approximate its boundary as a quadrilateral.
    This is robust to advertising boards, crowd, scoreboards and player motion
    because those are all outside or on top of the court surface.

    Falls back to a Hough-line approach if the surface mask fails.

    Returns:
        (pixel_pts, court_pts) ready for CourtHomography.calibrate().
    Raises:
        RuntimeError with a diagnostic hint if all attempts fail.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if vid_h == 0 or vid_w == 0:
        cap.release()
        raise RuntimeError(f"Video has zero dimensions: {video_path}")

    all_corners: list[np.ndarray] = []
    sampled = 0

    for i in range(n_frames * frame_skip):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_skip != 0:
            continue

        corners = _detect_court_surface(frame, vid_w, vid_h, debug=(debug and sampled == 0))
        if corners is not None:
            all_corners.append(corners)
        sampled += 1

    cap.release()

    if sampled < 1:
        raise RuntimeError(f"No frames readable from '{video_path}'.")

    if not all_corners:
        raise RuntimeError(
            "Could not auto-detect the court surface in any sampled frame.\n"
            "Run with debug=True to save a diagnostic image (debug_court_mask.png)."
        )

    # Median of all per-frame corner estimates — robust to outlier frames
    stacked = np.array(all_corners)          # (N, 4, 2)
    corners = np.median(stacked, axis=0).astype(np.float32)

    court_pts = _COURT_PTS.get(mode, _COURT_PTS["singles"])
    pixel_pts = corners.tolist()
    print(f"✓ Court auto-calibrated from {len(all_corners)}/{sampled} frames")
    print(f"  Corners (TL,TR,BR,BL): {[[round(v) for v in p] for p in pixel_pts]}")
    return pixel_pts, court_pts


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_court_lines(frame: np.ndarray, w: int, h: int,
                        debug: bool = False) -> np.ndarray | None:
    """
    PRIMARY detector: find court corners from the WHITE court lines.

    Uses adaptive thresholding + Hough line transform so it is immune to the
    brightness difference between the near and far court halves (stadium
    lighting makes the far half much brighter / more washed-out in HSV).

    Strategy:
      1. Adaptive threshold finds bright-relative-to-local-neighbourhood pixels
         (= white lines) in BOTH the dark near half and bright far half.
      2. Hough P-Lines finds long straight segments.
      3. Classify as horizontal (baselines, service lines) or vertical (sidelines).
      4. Filter to plausible court-line lengths.
      5. Take the extremal lines → 4 corners via line intersection.

    Returns ordered (TL, TR, BR, BL) or None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── Step 1: isolate white lines ────────────────────────────────────────
    # Adaptive threshold is key: it normalises for brightness variation across
    # the frame so white lines pop equally in both near and far halves.
    block = max(11, (w // 55) | 1)   # must be odd; ~23px for 1280-wide frame
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=block, C=-12,
    )
    # Also include globally bright lines (near half where contrast is high)
    _, glob = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
    bright = cv2.bitwise_or(adapt, glob)

    # Small open: remove isolated noise, keep line structures
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k3)

    if debug:
        cv2.imwrite("debug_lines_bright.png", bright)
        print("  Saved debug_lines_bright.png")

    # ── Step 2: Hough line transform ───────────────────────────────────────
    edges = cv2.Canny(bright, 50, 150, apertureSize=3)
    min_len = max(int(w * 0.28), 60)
    max_gap = max(int(w * 0.03), 12)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 360, threshold=50,
        minLineLength=min_len, maxLineGap=max_gap,
    )
    if lines is None or len(lines) < 4:
        return None

    # ── Step 3: classify into horizontal / vertical ────────────────────────
    h_lines: list[dict] = []
    v_lines: list[dict] = []
    for seg in lines:
        x1, y1, x2, y2 = [float(v) for v in seg[0]]
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 20:
            continue
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))  # 0=horizontal, 90=vertical
        entry = dict(x1=x1, y1=y1, x2=x2, y2=y2, length=length)
        if angle < 25:
            entry['y'] = (y1 + y2) / 2.0
            h_lines.append(entry)
        elif angle > 65:
            entry['x'] = (x1 + x2) / 2.0
            v_lines.append(entry)

    # Filter to plausible court-line lengths
    # Baseline spans ~70-90% of frame width; sidelines ~45-75% of frame height
    h_lines = [l for l in h_lines if l['length'] >= w * 0.28]
    v_lines = [l for l in v_lines if l['length'] >= h * 0.18]

    if len(h_lines) < 2 or len(v_lines) < 2:
        return None

    # ── Step 4: pick extremal lines ────────────────────────────────────────
    top_h = min(h_lines, key=lambda l: l['y'])   # far baseline (smallest y)
    bot_h = max(h_lines, key=lambda l: l['y'])   # near baseline (largest y)
    lft_v = min(v_lines, key=lambda l: l['x'])
    rgt_v = max(v_lines, key=lambda l: l['x'])

    # The far baseline must be in the top 40% of the frame.
    # (If only the net and near baseline are found, top_h['y'] ≈ h/2 > h*0.4 → reject)
    if top_h['y'] > h * 0.40:
        return None
    # Near baseline must be in the bottom 30% of the frame.
    if bot_h['y'] < h * 0.65:
        return None
    # The two baselines must be far enough apart vertically.
    if bot_h['y'] - top_h['y'] < h * 0.35:
        return None
    # Sidelines must straddle the horizontal centre of the frame.
    if lft_v['x'] > w * 0.45 or rgt_v['x'] < w * 0.55:
        return None

    # ── Step 5: compute corners as line intersections ──────────────────────
    def _leq(l: dict):
        """Return (a, b, c) s.t. a*x + b*y + c = 0 for the given segment."""
        dx = l['x2'] - l['x1']
        dy = l['y2'] - l['y1']
        return dy, -dx, -dy * l['x1'] + dx * l['y1']

    def _intersect(la: dict, lb: dict):
        a1, b1, c1 = _leq(la)
        a2, b2, c2 = _leq(lb)
        det = a1 * b2 - b1 * a2
        if abs(det) < 1e-6:
            return None
        x = (-c1 * b2 + b1 * c2) / det
        y = (-a1 * c2 + c1 * a2) / det
        return np.array([x, y], dtype=np.float32)

    tl = _intersect(top_h, lft_v)
    tr = _intersect(top_h, rgt_v)
    br = _intersect(bot_h, rgt_v)
    bl = _intersect(bot_h, lft_v)

    if any(p is None for p in (tl, tr, br, bl)):
        return None

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    if debug:
        viz = frame.copy()
        for lbl, pt in zip(["TL", "TR", "BR", "BL"], corners):
            p = tuple(pt.astype(int))
            cv2.circle(viz, p, 10, (0, 255, 0), -1)
            cv2.putText(viz, f"L:{lbl}", (p[0]+12, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite("debug_lines_corners.png", viz)
        print("  Saved debug_lines_corners.png")

    return corners if _quad_valid(corners, w, h) else None


def _project_far_corners(near_corners: np.ndarray) -> np.ndarray | None:
    """
    Given the near-half trapezoid (TL=net-left, TR=net-right, BR=near-right, BL=near-left),
    find the far-baseline corners by projective extension of the sidelines.

    Both court halves are equal in real-world length (6.7 m each side of the net).
    We parameterise each sideline using the projective 1-D mapping:
        pixel_y(world_y) = (a·world_y + b) / (c·world_y + 1)
    with known anchors:
        world_y = -6.7 → pixel at near baseline  (BL or BR)
        world_y =  0   → pixel at net line        (TL or TR)
        world_y → +∞   → vanishing point          (VP)
    Then query at world_y = +6.7 to get the far baseline pixel position.
    """
    TL, TR, BR, BL = near_corners

    # ── Vanishing point: intersection of the two sidelines extended ───────
    def _isect(p1, p2, p3, p4):
        x1, y1, x2, y2 = map(float, (*p1, *p2))
        x3, y3, x4, y4 = map(float, (*p3, *p4))
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(d) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float64)

    VP = _isect(BL, TL, BR, TR)    # sidelines converge above the net
    if VP is None:
        return None

    # ── Projective 1-D mapping along one sideline ─────────────────────────
    def _proj_y(y_near, y_net, y_vp, wy=6.7, wy_near=-6.7):
        """Pixel-y at world_y=wy given anchor points."""
        b = float(y_net)
        yn, yv = float(y_near), float(y_vp)
        if abs(yn - yv) < 1e-6:
            return b
        c = (b - yn) / (wy_near * (yn - yv))
        a = yv * c
        denom = c * wy + 1.0
        return None if abs(denom) < 1e-6 else (a * wy + b) / denom

    def _x_at_y(p1, p2, ty):
        dy = float(p2[1]) - float(p1[1])
        if abs(dy) < 1e-6:
            return float(p1[0])
        return float(p1[0]) + (ty - float(p1[1])) / dy * (float(p2[0]) - float(p1[0]))

    fl_y = _proj_y(BL[1], TL[1], VP[1])
    fr_y = _proj_y(BR[1], TR[1], VP[1])
    if fl_y is None or fr_y is None:
        return None

    fl_x = _x_at_y(BL, TL, fl_y)
    fr_x = _x_at_y(BR, TR, fr_y)

    far_left  = np.array([fl_x, fl_y], dtype=np.float32)
    far_right = np.array([fr_x, fr_y], dtype=np.float32)

    # Sanity: far corners must be above the net and inside a loose frame margin
    margin = 100
    if fl_y >= float(TL[1]) or fr_y >= float(TR[1]):   # must be above net
        return None
    if fl_y < -margin or fr_y < -margin:                # not too far above frame
        return None
    if fl_x >= fr_x:                                    # left must be left of right
        return None

    return np.array([far_left, far_right,
                     np.array(BR, dtype=np.float32),
                     np.array(BL, dtype=np.float32)])


def _detect_court_surface(frame: np.ndarray, w: int, h: int,
                           debug: bool = False) -> np.ndarray | None:
    """
    Detect court corners. Strategy:
      1. Detect the near-half court surface (single dominant hue from centre patch).
         The near half forms a trapezoid: top = net line, bottom = near baseline.
      2. Project the sidelines forward to find the far-baseline corners using
         the fact that both court halves are equal length (6.7 m) in real space.
      3. Fallback: Hough line detection if the near-half mask fails.

    This avoids having to detect the far half by colour — which fails when the
    two court halves use different colours (e.g. green near + teal far).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cx  = w // 2

    # ── Sample near-half hue & saturation from the frame centre ──────────
    cy = h // 2
    y0, y1 = max(0, cy - h // 8), min(h, cy + h // 8)
    x0, x1 = max(0, cx - w // 8), min(w, cx + w // 8)
    patch = hsv[y0:y1, x0:x1]
    not_white = patch[:, :, 1] > 30
    hues = patch[:, :, 0][not_white] if not_white.any() else patch[:, :, 0].flatten()
    near_hue = float(np.median(hues))
    near_s   = float(np.median(patch[:, :, 1]))
    sat_lo   = max(20, int(near_s * 0.40))   # 40% of patch median keeps court, drops crowd

    lo_h = max(0,   int(near_hue) - 25)
    hi_h = min(180, int(near_hue) + 25)
    near_mask = cv2.inRange(hsv, (lo_h, sat_lo, 15), (hi_h, 255, 255))

    bright_white = cv2.inRange(hsv, (0, 0, 220), (180, 25, 255))
    near_mask = cv2.bitwise_and(near_mask, cv2.bitwise_not(bright_white))

    close_px = max(20, w // 30)
    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_CLOSE, k_close)
    open_px  = max(8, w // 160)
    k_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    near_mask = cv2.morphologyEx(near_mask, cv2.MORPH_OPEN, k_open)

    if debug:
        cv2.imwrite("debug_near_mask.png", near_mask)
        print(f"  Near hue={near_hue:.1f}  sat_lo={sat_lo}")
        print("  Saved debug_near_mask.png")

    contours, _ = cv2.findContours(near_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    large = [c for c in contours if cv2.contourArea(c) >= w * h * 0.05]

    near_corners = None
    if large:
        main_c   = max(large, key=cv2.contourArea)
        hull_pts = cv2.convexHull(main_c)
        peri     = cv2.arcLength(hull_pts, True)
        for eps in (0.01, 0.02, 0.03, 0.05, 0.08, 0.12):
            approx = cv2.approxPolyDP(hull_pts, eps * peri, True)
            if len(approx) == 4:
                c4 = approx.reshape(4, 2).astype(np.float32)
                if _quad_valid(c4, w, h):
                    near_corners = _order_corners(c4)
                    break
        if near_corners is None:
            hull = hull_pts.reshape(-1, 2).astype(np.float32)
            s = hull.sum(axis=1); d = hull[:, 1] - hull[:, 0]
            c4 = np.array([hull[np.argmin(s)], hull[np.argmin(d)],
                           hull[np.argmax(s)], hull[np.argmax(d)]], dtype=np.float32)
            if _quad_valid(c4, w, h):
                near_corners = c4

    if near_corners is not None:
        if debug:
            print(f"  Near-half corners: {[[int(v) for v in p] for p in near_corners]}")
        full_corners = _project_far_corners(near_corners)
        if full_corners is not None and _quad_valid(full_corners, w, h):
            return full_corners

    # ── Fallback: Hough line detection ────────────────────────────────────
    return _detect_court_lines(frame, w, h, debug=debug)


def _quad_valid(corners: np.ndarray, w: int, h: int) -> bool:
    """Return True if the quadrilateral looks like a plausible court region."""
    margin_x, margin_y = w * 0.35, h * 0.35
    for cx, cy in corners:
        if not (-margin_x <= cx <= w + margin_x and
                -margin_y <= cy <= h + margin_y):
            return False
    area = cv2.contourArea(corners.reshape(4, 1, 2))
    if not (w * h * 0.05 < area < w * h * 0.97):
        return False
    if not cv2.isContourConvex(corners.reshape(4, 1, 2).astype(np.int32)):
        return False
    return True


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points as: Top-Left, Top-Right, Bottom-Right, Bottom-Left.

    Uses the classic sum/diff trick:
      TL  → smallest (x + y)
      BR  → largest  (x + y)
      TR  → smallest (y - x)
      BL  → largest  (y - x)
    """
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]   # y - x for each point
    return np.array([
        pts[np.argmin(s)],   # TL
        pts[np.argmin(d)],   # TR
        pts[np.argmax(s)],   # BR
        pts[np.argmax(d)],   # BL
    ], dtype=np.float32)
