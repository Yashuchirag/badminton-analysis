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


class CourtHomography:
    def __init__(self):
        self.H = None

    def calibrate(self, pixel_pts: np.ndarray, court_pts: np.ndarray):
        self.H, _ = cv2.findHomography(pixel_pts, court_pts)

    def to_court(self, px: float, py: float) -> tuple[float, float]:
        if self.H is None:
            raise RuntimeError("Homography not calibrated")
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