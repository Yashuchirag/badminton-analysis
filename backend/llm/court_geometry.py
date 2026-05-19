import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import cv2
from court_mapping.detector import detect_court_in_video

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
        Uses the court_mapping module (temporal median + projective extension)
        which is more reliable than the simple single-frame HSV approach.
        Returns the detected pixel corner list (TL, TR, BR, BL).
        """
        result = detect_court_in_video(
            video_path=video_path,
            mode=mode,
            n_frames=n_frames,
            refine=True,
            keep_artifacts=debug,
        )
        if not result.success or result.homography is None:
            raise RuntimeError(
                f"Court detection failed (confidence={result.confidence:.2f}). "
                f"Notes: {result.notes}"
            )
        self.H = result.homography.H
        self.H_inv = result.homography.H_inv
        return result.refined_corners.tolist() if result.refined_corners is not None else []

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
