import numpy as np
from dataclasses import dataclass
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class _Track:
    track_id:    int
    box:         tuple  # (x1, y1, x2, y2) pixel coords
    frames_lost: int = 0

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def area(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class PlayerTracker:
    """
    Detects and tracks both players across frames using a pretrained YOLO model,
    maps them to LEFT/RIGHT court halves, and maintains player identities
    (player index 0 or 1) even when sides are switched between games.

    Usage in ScoringEngine:
        # initialise once:
        self.player_tracker = PlayerTracker()

        # call every detection_interval frames:
        side_map = self.player_tracker.update(frame, self.court)
        if side_map:
            self.state.side_to_player = side_map
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.4,
        iou_threshold: float = 0.3,
        max_frames_lost: int = 20,
        device: str = "cpu",
    ):
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._device = device
        self._conf = conf
        self._iou_threshold = iou_threshold
        self._max_frames_lost = max_frames_lost

        self._tracks: list[_Track] = []
        self._next_track_id = 0
        # Maps track_id → player index (0 or 1). Established on first 2-player detection.
        self._track_to_player: dict[int, int] = {}
        # Most recent {"LEFT": (cx, cy), "RIGHT": (cx, cy)} — for serve detection
        self._latest_side_positions: dict[str, tuple] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray, court) -> dict:
        """
        Run one detection + tracking cycle.

        Args:
            frame:  current video frame (BGR numpy array)
            court:  CourtHomography instance (must already be calibrated)

        Returns:
            dict with keys "LEFT" and/or "RIGHT" → player index (0 or 1).
            Returns empty dict if court not calibrated or fewer than 2 players visible.
        """
        if court.H is None:
            return {}

        detections = self._detect(frame)
        self._match_and_update(detections)
        self._assign_player_indices()
        return self._build_side_map(court)

    # ── Internal methods ──────────────────────────────────────────────────────

    def _detect(self, frame: np.ndarray) -> list[tuple]:
        results = self._model(frame, classes=[0], conf=self._conf, verbose=False, device=self._device)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append((float(x1), float(y1), float(x2), float(y2)))
        return boxes

    def _match_and_update(self, detections: list[tuple]):
        for t in self._tracks:
            t.frames_lost += 1

        matched_ids: set[int] = set()
        unmatched: list[tuple] = []

        for box in detections:
            best_iou = self._iou_threshold
            best_track: Optional[_Track] = None
            for t in self._tracks:
                score = _iou(box, t.box)
                if score > best_iou and t.track_id not in matched_ids:
                    best_iou = score
                    best_track = t

            if best_track is not None:
                best_track.box = box
                best_track.frames_lost = 0
                matched_ids.add(best_track.track_id)
            else:
                unmatched.append(box)

        for box in unmatched:
            self._tracks.append(_Track(track_id=self._next_track_id, box=box))
            self._next_track_id += 1

        self._tracks = [t for t in self._tracks
                        if t.frames_lost <= self._max_frames_lost]

    def _assign_player_indices(self):
        """Give each new track a player index (0 or 1). First-seen = 0, second = 1."""
        for t in self._tracks:
            if t.frames_lost > 0:
                continue
            if t.track_id not in self._track_to_player:
                used = set(self._track_to_player.values())
                for idx in (0, 1):
                    if idx not in used:
                        self._track_to_player[t.track_id] = idx
                        break

    def get_side_positions(self) -> dict:
        """Return {"LEFT": (cx, cy), "RIGHT": (cx, cy)} from the last update call."""
        return dict(self._latest_side_positions)

    def _build_side_map(self, court) -> dict:
        """Return {"LEFT": player_idx, "RIGHT": player_idx} for currently visible players."""
        left_candidates: list[_Track] = []
        right_candidates: list[_Track] = []

        for t in self._tracks:
            if t.frames_lost > 0:
                continue
            if t.track_id not in self._track_to_player:
                continue
            try:
                court_x, _ = court.to_court(t.cx, t.cy)
                if court_x < 0:
                    left_candidates.append(t)
                else:
                    right_candidates.append(t)
            except Exception:
                continue

        side_map: dict[str, int] = {}
        self._latest_side_positions = {}
        if left_candidates:
            best = max(left_candidates, key=lambda t: t.area)
            side_map["LEFT"] = self._track_to_player[best.track_id]
            self._latest_side_positions["LEFT"] = (best.cx, best.cy)
        if right_candidates:
            best = max(right_candidates, key=lambda t: t.area)
            side_map["RIGHT"] = self._track_to_player[best.track_id]
            self._latest_side_positions["RIGHT"] = (best.cx, best.cy)

        return side_map
