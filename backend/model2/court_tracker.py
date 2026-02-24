import cv2
import numpy as np
from collections import deque
from typing import Optional, Tuple, List
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════
# Real-world badminton court dimensions (meters) — BWF standard
#   Origin = bottom-left corner (near side)
#   Court is 13.4m long × 6.1m wide (doubles)
# ══════════════════════════════════════════════════════════════════════════
COURT_KEYPOINTS_REAL = np.array([
    [0.0,  0.0],   # 0: bottom-left  (near-left)
    [6.1,  0.0],   # 1: bottom-right (near-right)
    [6.1, 13.4],   # 2: top-right    (far-right)
    [0.0, 13.4],   # 3: top-left     (far-left)
    [0.0,  4.715], # 4: left service-short line (near)
    [6.1,  4.715], # 5: right service-short line (near)
    [0.0,  8.685], # 6: left service-short line (far)
    [6.1,  8.685], # 7: right service-short line (far)
    [3.05, 0.0],   # 8: center-near
    [3.05, 13.4],  # 9: center-far
    [3.05, 4.715], # 10: center-T near
    [3.05, 8.685], # 11: center-T far
], dtype=np.float32)


@dataclass
class CourtDetection:
    """Result of court detection for a single frame."""
    corners: np.ndarray             # shape (4,2) pixel coords [TL,TR,BR,BL]
    homography: np.ndarray          # 3×3 H mapping pixel → real-world (meters)
    homography_inv: np.ndarray      # 3×3 H mapping real-world → pixel
    confidence: float               # 0–1
    lines: List = field(default_factory=list)  # raw detected line segments


# ══════════════════════════════════════════════════════════════════════════
# Core detector
# ══════════════════════════════════════════════════════════════════════════

class CourtTracker:
    """
    Detect and track badminton court edges across video frames.

    Pipeline
    --------
    1. Colour / brightness preprocessing  → isolate bright court lines
    2. Canny edge detection
    3. Probabilistic Hough transform      → line segments
    4. Cluster into horizontal / vertical groups
    5. Find the 4 strongest boundary lines → intersect for corners
    6. Compute homography (pixel ↔ real-world)
    7. Temporal EMA smoothing of corners  → stable overlay

    Usage
    -----
        tracker = CourtTracker()
        detection = tracker.detect(frame)
        if detection:
            tracker.draw(frame, detection)
            real_xy = tracker.pixel_to_court(detection, shuttle_pixel_xy)
    """

    def __init__(
        self,
        smoothing_alpha: float = 0.3,   # EMA weight for new frame (0=frozen, 1=raw)
        min_line_length: int = 80,      # Hough min line length (px)
        max_line_gap: int = 20,         # Hough max gap within a line (px)
        hough_threshold: int = 60,      # Hough accumulator threshold
        canny_low: int = 30,
        canny_high: int = 100,
        blur_ksize: int = 5,
        angle_tolerance_deg: float = 15.0,  # Tolerance for H/V classification
    ):
        self.alpha = smoothing_alpha
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.hough_threshold = hough_threshold
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_ksize = blur_ksize
        self.angle_tol = np.deg2rad(angle_tolerance_deg)

        # EMA state
        self._smoothed_corners: Optional[np.ndarray] = None  # (4,2) float

    def detect(self, frame: np.ndarray) -> Optional[CourtDetection]:
        """
        Run full court detection on a BGR frame.

        Returns CourtDetection or None if the court cannot be found.
        """
        h, w = frame.shape[:2]

        # 1. Pre-process
        gray = self._preprocess(frame)

        # 2. Edges
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        # 3. Hough lines
        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        if raw_lines is None:
            return self._fallback()

        lines = raw_lines[:, 0, :]  # (N,4)

        # 4. Classify into horizontal / vertical
        h_lines, v_lines = self._classify_lines(lines)

        if len(h_lines) < 2 or len(v_lines) < 2:
            return self._fallback()

        # 5. Pick boundary lines
        top, bottom = self._boundary_pair(h_lines, axis="y", frame_size=(w, h))
        left, right  = self._boundary_pair(v_lines, axis="x", frame_size=(w, h))

        if any(l is None for l in [top, bottom, left, right]):
            return self._fallback()

        # 6. Compute 4 corners from line intersections
        corners = self._quad_corners(top, bottom, left, right)
        if corners is None:
            return self._fallback()

        # Validate: corners should form a reasonable quadrilateral
        if not self._is_valid_quad(corners, w, h):
            return self._fallback()

        # 7. EMA smoothing
        corners = self._smooth(corners)

        # 8. Homography  (corners order: TL, TR, BR, BL)
        H, H_inv = self._compute_homography(corners)
        if H is None:
            return self._fallback()

        conf = self._compute_confidence(corners, w, h)

        return CourtDetection(
            corners=corners,
            homography=H,
            homography_inv=H_inv,
            confidence=conf,
            lines=lines.tolist(),
        )

    def pixel_to_court(
        self,
        detection: CourtDetection,
        pixel_xy: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Map a pixel coordinate to real-world court coordinates (meters).

        Origin = bottom-left corner of the court.
        """
        pt = np.array([[pixel_xy]], dtype=np.float32)
        real = cv2.perspectiveTransform(pt, detection.homography)[0, 0]
        return float(real[0]), float(real[1])

    def court_to_pixel(
        self,
        detection: CourtDetection,
        court_xy: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Map real-world court coordinates (meters) back to pixel."""
        pt = np.array([[court_xy]], dtype=np.float32)
        pix = cv2.perspectiveTransform(pt, detection.homography_inv)[0, 0]
        return float(pix[0]), float(pix[1])

    def draw(
        self,
        frame: np.ndarray,
        detection: CourtDetection,
        color: Tuple[int, int, int] = (0, 220, 255),
        thickness: int = 2,
        draw_keypoints: bool = True,
        draw_mini_map: bool = True,
        shuttle_pixel: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Draw court overlay on frame (in-place).

        Args:
            frame:          BGR frame to annotate
            detection:      CourtDetection from detect()
            color:          Line color (BGR)
            thickness:      Line thickness
            draw_keypoints: Draw corner & T-junction keypoints
            draw_mini_map:  Draw bird's-eye mini-map in corner
            shuttle_pixel:  If provided, also plot shuttle on mini-map
        """
        corners = detection.corners.astype(np.int32)  # TL, TR, BR, BL

        # Outer boundary
        cv2.polylines(frame, [corners], isClosed=True, color=color, thickness=thickness)

        # Interpolate inner lines (service lines, center line)
        self._draw_inner_lines(frame, detection, color, thickness)

        # Corner dots
        if draw_keypoints:
            for pt in corners:
                cv2.circle(frame, tuple(pt), 6, (0, 0, 255), -1)

        # Confidence badge
        conf_pct = int(detection.confidence * 100)
        cv2.putText(frame, f"Court: {conf_pct}%",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Mini bird's-eye map
        if draw_mini_map:
            self._draw_minimap(frame, detection, shuttle_pixel, color)

        return frame

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Isolate bright court lines.

        Strategy:
          • Convert to LAB and use L channel (luminance) — court lines are
            typically brighter than the floor regardless of floor colour.
          • Alternatively apply adaptive threshold to handle uneven lighting.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]  # 0–255 luminance

        # Gentle Gaussian blur to reduce noise before edge detection
        blurred = cv2.GaussianBlur(l_channel, (self.blur_ksize, self.blur_ksize), 0)

        # CLAHE to enhance contrast in dimly lit arenas
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        return enhanced

    def _classify_lines(self, lines: np.ndarray):
        """Split lines into near-horizontal and near-vertical groups."""
        h_lines, v_lines = [], []
        for x1, y1, x2, y2 in lines:
            angle = np.arctan2(y2 - y1, x2 - x1)
            # Near-horizontal: angle ≈ 0 or ±π
            if abs(angle) < self.angle_tol or abs(abs(angle) - np.pi) < self.angle_tol:
                h_lines.append((x1, y1, x2, y2))
            # Near-vertical: angle ≈ ±π/2
            elif abs(abs(angle) - np.pi / 2) < self.angle_tol:
                v_lines.append((x1, y1, x2, y2))
        return h_lines, v_lines

    def _line_to_infinite(self, x1, y1, x2, y2):
        """Return (a, b, c) for ax + by + c = 0 (infinite line through two points)."""
        a = y2 - y1
        b = x1 - x2
        c = a * x1 + b * y1
        return a, b, -c  # ax + by = c form

    def _intersect(self, l1, l2):
        """Intersection point of two infinite lines (a,b,c). Returns None if parallel."""
        a1, b1, c1 = l1
        a2, b2, c2 = l2
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        return x, y

    def _boundary_pair(self, lines, axis: str, frame_size):
        """
        From a list of near-H or near-V line segments, pick the two lines
        that form the outer boundary (minimum and maximum position on `axis`).

        Uses median position of each line segment to vote for two clusters.
        """
        w, h = frame_size
        positions = []
        for x1, y1, x2, y2 in lines:
            pos = (y1 + y2) / 2 if axis == "y" else (x1 + x2) / 2
            positions.append(pos)

        positions = np.array(positions)

        # Simple: just pick the two extreme positions
        sorted_idx = np.argsort(positions)
        min_line = lines[sorted_idx[0]]
        max_line = lines[sorted_idx[-1]]

        return (
            self._line_to_infinite(*min_line),   # top / left
            self._line_to_infinite(*max_line),   # bottom / right
        )

    def _quad_corners(self, top, bottom, left, right):
        """Intersect 4 boundary lines to get TL, TR, BR, BL corners."""
        tl = self._intersect(top, left)
        tr = self._intersect(top, right)
        br = self._intersect(bottom, right)
        bl = self._intersect(bottom, left)

        if any(p is None for p in [tl, tr, br, bl]):
            return None

        return np.array([tl, tr, br, bl], dtype=np.float32)

    def _is_valid_quad(self, corners, w, h, margin=0.05):
        """Sanity-check: corners should be inside (or near) the frame."""
        for x, y in corners:
            if x < -margin * w or x > (1 + margin) * w:
                return False
            if y < -margin * h or y > (1 + margin) * h:
                return False

        # Area should be at least 5% of frame
        area = cv2.contourArea(corners.reshape(4, 1, 2).astype(np.int32))
        if area < 0.05 * w * h:
            return False
        return True

    def _smooth(self, corners: np.ndarray) -> np.ndarray:
        """Exponential moving average over corner positions."""
        if self._smoothed_corners is None:
            self._smoothed_corners = corners.copy()
        else:
            self._smoothed_corners = (
                self.alpha * corners + (1 - self.alpha) * self._smoothed_corners
            )
        return self._smoothed_corners.copy()

    def _compute_homography(self, corners: np.ndarray):
        """
        Compute homography from pixel corners → real-world court corners.

        Pixel corners order: TL, TR, BR, BL
        Real-world (meters): TL=(0,13.4), TR=(6.1,13.4), BR=(6.1,0), BL=(0,0)
        """
        real_corners = np.array([
            [0.0,  13.4],  # TL
            [6.1,  13.4],  # TR
            [6.1,   0.0],  # BR
            [0.0,   0.0],  # BL
        ], dtype=np.float32)

        H, status = cv2.findHomography(corners, real_corners, cv2.RANSAC, 5.0)
        if H is None:
            return None, None

        H_inv, _ = cv2.findHomography(real_corners, corners, cv2.RANSAC, 5.0)
        return H, H_inv

    def _compute_confidence(self, corners, w, h) -> float:
        """
        Heuristic confidence based on:
          • How rectangular the quad is (90° corners)
          • Aspect ratio closeness to 6.1/13.4 ≈ 0.455
        """
        # Compute angles at each corner
        pts = corners
        angles = []
        for i in range(4):
            p0 = pts[(i - 1) % 4]
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            v1 = p0 - p1
            v2 = p2 - p1
            cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

        angle_score = 1 - np.mean(np.abs(np.array(angles) - 90)) / 90

        # Aspect ratio score
        width  = np.linalg.norm(pts[1] - pts[0])
        height = np.linalg.norm(pts[3] - pts[0])
        ar = width / (height + 1e-9)
        expected_ar = 6.1 / 13.4
        ar_score = max(0, 1 - abs(ar - expected_ar) / expected_ar)

        return float(np.clip(0.5 * angle_score + 0.5 * ar_score, 0, 1))

    def _fallback(self) -> Optional[CourtDetection]:
        """Return smoothed corners from previous frame if available."""
        if self._smoothed_corners is not None:
            H, H_inv = self._compute_homography(self._smoothed_corners)
            if H is not None:
                return CourtDetection(
                    corners=self._smoothed_corners,
                    homography=H,
                    homography_inv=H_inv,
                    confidence=0.1,
                )
        return None

    def _draw_inner_lines(self, frame, detection: CourtDetection, color, thickness):
        """
        Project standard badminton court lines onto the frame using the
        inverse homography (real-world → pixel).
        """
        real_lines = [
            # Service short lines (doubles)
            ([0.0, 4.715], [6.1, 4.715]),
            ([0.0, 8.685], [6.1, 8.685]),
            # Center line
            ([3.05, 0.0], [3.05, 13.4]),
            # Back boundary for singles (0.76m from back)
            ([0.76, 0.0], [0.76, 4.715]),
            ([5.34, 0.0], [5.34, 4.715]),
            ([0.76, 8.685], [0.76, 13.4]),
            ([5.34, 8.685], [5.34, 13.4]),
        ]

        for (rx1, ry1), (rx2, ry2) in real_lines:
            p1 = self.court_to_pixel(detection, (rx1, ry1))
            p2 = self.court_to_pixel(detection, (rx2, ry2))
            if p1 and p2:
                cv2.line(frame,
                         (int(p1[0]), int(p1[1])),
                         (int(p2[0]), int(p2[1])),
                         color, max(1, thickness - 1))

    def _draw_minimap(
        self,
        frame: np.ndarray,
        detection: CourtDetection,
        shuttle_pixel: Optional[Tuple[float, float]],
        color: Tuple[int, int, int],
        map_w: int = 120,
        map_h: int = 220,
        margin: int = 10,
    ):
        """Draw a bird's-eye mini-map of the court in the top-right corner."""
        fh, fw = frame.shape[:2]
        x0 = fw - map_w - margin
        y0 = margin

        # Dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + map_w, y0 + map_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Court boundary
        court_rect = np.array([
            [x0 + 10,          y0 + 10],
            [x0 + map_w - 10,  y0 + 10],
            [x0 + map_w - 10,  y0 + map_h - 10],
            [x0 + 10,          y0 + map_h - 10],
        ], dtype=np.int32)
        cv2.polylines(frame, [court_rect], True, color, 1)

        # Helper: real-world → mini-map pixel
        def real_to_map(rx, ry):
            mx = int(x0 + 10 + rx / 6.1 * (map_w - 20))
            my = int(y0 + map_h - 10 - ry / 13.4 * (map_h - 20))
            return mx, my

        # Draw standard inner lines on mini-map
        inner = [
            ((0, 4.715), (6.1, 4.715)),
            ((0, 8.685), (6.1, 8.685)),
            ((3.05, 0), (3.05, 13.4)),
        ]
        for (r1, r2) in inner:
            cv2.line(frame, real_to_map(*r1), real_to_map(*r2), color, 1)

        # Shuttle position on mini-map
        if shuttle_pixel is not None:
            try:
                sx, sy = self.pixel_to_court(detection, shuttle_pixel)
                if 0 <= sx <= 6.1 and 0 <= sy <= 13.4:
                    cv2.circle(frame, real_to_map(sx, sy), 5, (0, 255, 0), -1)
            except Exception:
                pass

        cv2.putText(frame, "Court", (x0 + 35, y0 + map_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


# ══════════════════════════════════════════════════════════════════════════
# Integration with ShuttleTracker from the existing codebase
# ══════════════════════════════════════════════════════════════════════════

def track_video_with_court(
    shuttle_tracker,          # ShuttleTracker instance from your existing code
    video_path: str,
    output_path: str,
    mode: str = "hybrid",
    conf_threshold: float = 0.25,
    court_smoothing: float = 0.25,
):
    """
    Wrapper that runs both shuttle tracking AND court detection on a video.

    Drop-in replacement for ShuttleTracker.track_video() with court overlay.

    Args:
        shuttle_tracker:   Your existing ShuttleTracker instance
        video_path:        Input video
        output_path:       Output annotated video
        mode:              Shuttle tracking mode ('yolo','obb','tracknet','hybrid')
        conf_threshold:    YOLO confidence threshold
        court_smoothing:   EMA alpha for court corner smoothing (lower = more stable)
    """
    import cv2
    import numpy as np
    from collections import deque

    cap = cv2.VideoCapture(video_path)
    fps     = int(cap.get(cv2.CAP_PROP_FPS))
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    court_tracker = CourtTracker(smoothing_alpha=court_smoothing)
    trail         = deque(maxlen=30)
    frame_idx     = 0

    print(f"\nTracking (shuttle + court): {video_path}")
    print(f"  Mode: {mode} | FPS: {fps} | {width}×{height}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Court detection ---
        court = court_tracker.detect(frame)

        # --- Shuttle detection (reuse your existing logic) ---
        shuttle_pos = None
        if mode == "yolo" and shuttle_tracker.yolo_model:
            shuttle_pos = shuttle_tracker._detect_yolo(
                frame, shuttle_tracker.yolo_model, conf_threshold)
        elif mode == "obb" and shuttle_tracker.obb_model:
            shuttle_pos = shuttle_tracker._detect_yolo(
                frame, shuttle_tracker.obb_model, conf_threshold, is_obb=True)
        elif mode == "tracknet" and shuttle_tracker.tracknet_model:
            shuttle_pos = shuttle_tracker._detect_tracknet(frame)
        elif mode == "hybrid":
            yolo_pos = None
            if shuttle_tracker.obb_model:
                yolo_pos = shuttle_tracker._detect_yolo(
                    frame, shuttle_tracker.obb_model, conf_threshold, is_obb=True)
            elif shuttle_tracker.yolo_model:
                yolo_pos = shuttle_tracker._detect_yolo(
                    frame, shuttle_tracker.yolo_model, conf_threshold)
            tracknet_pos = (shuttle_tracker._detect_tracknet(frame)
                            if shuttle_tracker.tracknet_model else None)
            if yolo_pos and tracknet_pos:
                import numpy as np
                dist = np.hypot(yolo_pos[0] - tracknet_pos[0],
                                yolo_pos[1] - tracknet_pos[1])
                shuttle_pos = tracknet_pos if dist < 50 else yolo_pos
            else:
                shuttle_pos = yolo_pos or tracknet_pos

        # --- Draw court overlay ---
        if court:
            court_tracker.draw(frame, court, shuttle_pixel=shuttle_pos)

        # --- Draw shuttle ---
        if shuttle_pos:
            x, y = int(shuttle_pos[0]), int(shuttle_pos[1])
            trail.append((x, y))
            cv2.circle(frame, (x, y), 12, (0, 255, 0), 2)
            cv2.circle(frame, (x, y),  3, (0, 255, 0), -1)

            # Trail
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                cv2.line(frame, trail[i - 1], trail[i],
                         (0, int(255 * alpha), 0), 2)

            # Real-world position
            if court:
                cx, cy = court_tracker.pixel_to_court(court, shuttle_pos)
                cv2.putText(frame,
                            f"Court: ({cx:.2f}m, {cy:.2f}m)",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # HUD
        cv2.putText(frame, f"Frame {frame_idx}  Mode: {mode.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        if shuttle_pos:
            cv2.putText(frame,
                        f"Shuttle px: ({int(shuttle_pos[0])}, {int(shuttle_pos[1])})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames…")

    cap.release()
    out.release()
    print(f"\n✓ Done → {output_path}  ({frame_idx} frames)")


# ══════════════════════════════════════════════════════════════════════════
# Standalone CLI
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Badminton court + shuttle tracker")
    parser.add_argument("--video",        required=True, help="Input video path")
    parser.add_argument("--output",       required=True, help="Output video path")
    parser.add_argument("--mode",         default="hybrid",
                        choices=["yolo", "obb", "tracknet", "hybrid"])
    parser.add_argument("--yolo-weights",     default=None)
    parser.add_argument("--obb-weights",      default=None)
    parser.add_argument("--tracknet-weights", default=None)
    parser.add_argument("--device",       default="0")
    parser.add_argument("--conf",         type=float, default=0.25)
    parser.add_argument("--court-only",   action="store_true",
                        help="Only show court overlay, skip shuttle tracking")
    parser.add_argument("--smoothing",    type=float, default=0.25,
                        help="EMA smoothing alpha for court corners (0–1)")
    args = parser.parse_args()

    if args.court_only:
        # Standalone court-only mode
        cap = cv2.VideoCapture(args.video)
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

        tracker   = CourtTracker(smoothing_alpha=args.smoothing)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            court = tracker.detect(frame)
            if court:
                tracker.draw(frame, court)
            out.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  {frame_idx} frames…")
        cap.release()
        out.release()
        print(f"✓ Court-only output → {args.output}")

    else:
        # Full shuttle + court tracking
        try:
            sys.path.insert(0, ".")
            from train_and_track import ShuttleTracker  # adjust import as needed
        except ImportError:
            print("⚠  Could not import ShuttleTracker – running court-only mode.")
            args.court_only = True

        if not args.court_only:
            shuttle_tracker = ShuttleTracker(
                yolo_weights=args.yolo_weights,
                obb_weights=args.obb_weights,
                tracknet_weights=args.tracknet_weights,
                device=args.device,
            )
            track_video_with_court(
                shuttle_tracker=shuttle_tracker,
                video_path=args.video,
                output_path=args.output,
                mode=args.mode,
                conf_threshold=args.conf,
                court_smoothing=args.smoothing,
            )