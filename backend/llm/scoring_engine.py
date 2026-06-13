# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import cv2
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from court_mapping.geometry import CourtHomography, ZONES, LINE_MARGIN, distance_to_boundary
from court_mapping.detector import detect_court_in_video
from court_mapping.visualize import draw_court_overlay
from model2.train_and_track import ShuttleTracker
from player_tracker import PlayerTracker


WINNING_SCORE  = 21
WINNING_MARGIN = 2
MAX_SCORE      = 30


class Verdict(str, Enum):
    IN = "IN"; OUT = "OUT"; LET = "LET"; PENDING = "PENDING"


@dataclass
class LandingEvent:
    frame_idx:  int
    pixel_x:    float
    pixel_y:    float
    court_x:    float
    court_y:    float
    confidence: float
    source:     str              # "yolo" | "obb" | "tracknet" | "fused"
    near_line:  bool  = False
    verdict:    Verdict = Verdict.PENDING
    reason:     str   = ""


@dataclass
class MatchState:
    score:            list[int] = field(default_factory=lambda: [0, 0])
    serving_side:     int  = 0
    last_hitter_side: Optional[str] = None   # "LEFT" or "RIGHT"
    side_to_player:   dict = field(default_factory=lambda: {"LEFT": 0, "RIGHT": 1})
    rally_state:      str  = "SERVE_PENDING"  # "SERVE_PENDING" | "RALLY_ACTIVE"
    game:             int  = 1
    history:          list[LandingEvent] = field(default_factory=list)


class ScoringEngine:
    """
    Geometric, event-driven rally scorer. The shuttle trajectory itself drives
    a small state machine:

      SERVE_PENDING ──sustained motion──▶ RALLY_ACTIVE ──rally end──▶ point
                                                ▲                        │
                                                └────── cooldown ────────┘

    Hits are sharp direction changes in the trajectory (the court half where
    they happen identifies the hitter). A rally ends when the shuttle comes to
    rest low in the frame, or vanishes while descending near the floor.
    Verdicts are pure homography geometry — no LLM is used or imported.
    """

    # ── Rally start: sustained launch motion ─────────────────────────────────
    _LAUNCH_SPEED  = 8.0   # px/frame
    _LAUNCH_FRAMES = 3     # consecutive fast frames to call the serve

    # ── Hit detection: sharp trajectory direction change ─────────────────────
    _HIT_MIN_SPEED = 6.0                       # px/frame on both legs
    _HIT_MAX_COS   = float(np.cos(np.radians(55)))  # direction change > 55°
    _HIT_GAP_MAX   = 8     # max frame gap between trajectory points
    _HIT_COOLDOWN  = 4     # min frames between two detected hits

    # ── Rally end ─────────────────────────────────────────────────────────────
    _FLOOR_RATIO   = 0.55  # fallback "low in frame" gate when uncalibrated
    _STILL_SPEED   = 3.0   # px/frame
    _STILL_FRAMES  = 5     # consecutive slow frames = shuttle at rest
    _GONE_FRAMES   = 10    # frames with no detection after a descending low ball
    _ABORT_FRAMES  = 150   # no detections at all → abort rally, no point

    # ── After a point: ignore the shuttle being picked up / walked back ──────
    _POINT_COOLDOWN = 75   # frames

    _MAX_PLAUSIBLE_OUT = 2.0  # metres past the line; further = false detection

    def __init__(
            self,
            yolo_weights: Optional[str] = None,
            obb_weights: Optional[str] = None,
            tracknet_weights: Optional[str] = None,
            mode: str = "singles",
            tracking_mode: str = "hybrid",    # passed to ShuttleTracker
            conf_threshold: float = 0.25,
            device: str = "cpu",
            player_model: str = "yolo11n.pt",   # pretrained model for player detection
            player_detection_interval: int = 30, # run player detector every N frames
            detect_stride: int = 1,              # run shuttle inference every Nth frame
            yolo_conf_skip_tracknet: float = 0.65,  # YOLO conf above which TrackNet is skipped
        ):
        self.game_mode = mode
        self.tracking_mode = tracking_mode
        self.conf_threshold = conf_threshold
        self.detect_stride = max(1, detect_stride)
        self.yolo_conf_skip_tracknet = yolo_conf_skip_tracknet
        self.state = MatchState()
        self._zone_poly = ZONES["doubles_back"] if mode == "doubles" else ZONES["singles_back"]

        self.tracker = ShuttleTracker(
            yolo_weights=yolo_weights,
            obb_weights=obb_weights,
            tracknet_weights=tracknet_weights,
            device=device,
        )

        self.court = CourtHomography()
        self._projected_lines = []   # canonical court lines in pixel space, set by auto_calibrate
        self.players = PlayerTracker(model_path=player_model, device=device)
        self._player_detection_interval = player_detection_interval

        self._frame_height: Optional[int] = None
        self.last_shuttle: Optional[tuple] = None  # (x, y, conf, source) for this frame, for visualization
        self._floor_min_y: Optional[float] = None  # pixel y of the far baseline, set by auto_calibrate
        self._traj: deque = deque(maxlen=12)   # (frame_idx, x, y)
        self._prev_court_x: Optional[float] = None
        self._last_det: Optional[tuple] = None  # (frame_idx, x, y, conf, source, vy)
        self._launch_count = 0
        self._slow_count = 0
        self._last_hit_frame = -10**9
        self._rally_start_frame = 0
        self._cooldown = 0

    def auto_calibrate(self, video_path: str, n_frames: int = 90,
                       debug: bool = False) -> list:
        """
        Auto-detect court corners from the video and calibrate the homography.
        Call this once before starting process_frame().
        """
        # The corner detector always finds the OUTERMOST painted lines, i.e. the
        # doubles boundary, so the homography must be fitted in the doubles
        # frame regardless of game mode. The game mode only selects which zone
        # polygon (self._zone_poly) is used for IN/OUT judgement.
        result = detect_court_in_video(
            video_path=video_path,
            mode="doubles",
            n_frames=n_frames,
            refine=True,
            keep_artifacts=debug,
        )
        if not result.success or result.homography is None:
            raise RuntimeError(
                f"Court detection failed (confidence={result.confidence}). "
                f"Notes: {result.notes}"
            )
        self.court.H = result.homography.H
        self.court.H_inv = result.homography.H_inv
        self._projected_lines = result.projected_lines

        # Pixel y of the far baseline: any floor landing appears at or below
        # this row, so it replaces the crude frame-height floor gate.
        corners = np.array(self._zone_poly, dtype=np.float32).reshape((-1, 1, 2))
        pix = cv2.perspectiveTransform(corners, self.court.H_inv).reshape(-1, 2)
        self._floor_min_y = float(pix[:, 1].min())

        return result.refined_corners.tolist() if result.refined_corners is not None else []

    def process_frame(self, frame, frame_idx: int) -> Optional[LandingEvent]:
        self._frame_height = frame.shape[0]
        self.last_shuttle = None

        # ── Player side mapping (runs every N frames, low overhead) ──────────
        if frame_idx % self._player_detection_interval == 0:
            side_map = self.players.update(frame, self.court)
            if len(side_map) == 2:
                self.state.side_to_player = side_map

        # ── Stride skip: no shuttle inference on non-stride frames ───────────
        # Interpolated positions are visualization-only and never enter _traj,
        # so hit detection still runs between real detections with correct
        # gap normalization.
        if self.detect_stride > 1 and frame_idx % self.detect_stride != 0:
            if self._cooldown > 0:
                self._cooldown -= 1
            self.tracker.frame_buffer.append(frame.copy())
            if len(self._traj) >= 2:
                (t0, x0, y0), (t1, x1, y1) = list(self._traj)[-2:]
                gap = t1 - t0
                if gap > 0 and 0 < frame_idx - t1 <= self._HIT_GAP_MAX:
                    k = frame_idx - t1
                    self.last_shuttle = (x1 + (x1 - x0) / gap * k,
                                         y1 + (y1 - y0) / gap * k,
                                         0.0, "interp")
            return None

        # ── Post-point cooldown: shuttle is being picked up, ignore it ───────
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        pos, conf, source = self.tracker.predict_frame(
            frame,
            mode=self.tracking_mode,
            conf_threshold=self.conf_threshold,
            yolo_conf_skip_tracknet=self.yolo_conf_skip_tracknet,
        )

        # ── Trajectory bookkeeping ────────────────────────────────────────────
        speed = None
        if pos is not None:
            self.last_shuttle = (pos[0], pos[1], conf, source)
            vy = 0.0
            if self._traj:
                lt, lx, ly = self._traj[-1]
                gap = frame_idx - lt
                if 0 < gap <= self._HIT_GAP_MAX:
                    speed = float(np.hypot(pos[0] - lx, pos[1] - ly)) / gap
                    vy = (pos[1] - ly) / gap
            self._traj.append((frame_idx, pos[0], pos[1]))
            self._last_det = (frame_idx, pos[0], pos[1], conf, source, vy)

        # ── Net crossing → last hitter is the side the shuttle came FROM ─────
        if pos is not None and self.court.H is not None:
            try:
                court_x, _ = self.court.to_court(pos[0], pos[1])
                if self._prev_court_x is not None:
                    l2r = self._prev_court_x < 0 and court_x > 0
                    r2l = self._prev_court_x > 0 and court_x < 0
                    if l2r or r2l:
                        self.state.last_hitter_side = "LEFT" if l2r else "RIGHT"
                        if self.state.rally_state == "SERVE_PENDING":
                            # Shuttle already over the net — rally is on
                            self._start_rally(frame_idx, announce="net crossing")
                self._prev_court_x = court_x
            except Exception:
                pass

        # ── SERVE_PENDING: rally starts on sustained shuttle motion ──────────
        if self.state.rally_state == "SERVE_PENDING":
            if speed is not None and speed >= self._LAUNCH_SPEED:
                self._launch_count += 1
                if self._launch_count >= self._LAUNCH_FRAMES:
                    self.state.last_hitter_side = self._court_side(pos)
                    self._start_rally(frame_idx, announce=f"launch {speed:.1f}px/f")
            elif pos is not None:
                self._launch_count = 0
            return None

        # ── RALLY_ACTIVE: hit detection ───────────────────────────────────────
        if pos is not None:
            self._detect_hit(frame_idx)

        # ── Rally end A: shuttle at rest low in the frame ─────────────────────
        if pos is not None:
            if speed is not None and speed < self._STILL_SPEED:
                self._slow_count += 1
            elif speed is not None:
                self._slow_count = 0
            if self._slow_count >= self._STILL_FRAMES and self._is_low(pos[1]):
                return self._finish_rally(frame_idx, pos[0], pos[1], conf, source,
                                          how="came to rest")

        # ── Rally end B: vanished while descending near the floor ────────────
        if pos is None and self._last_det is not None:
            lf, lx, ly, lconf, lsrc, lvy = self._last_det
            if (frame_idx - lf >= self._GONE_FRAMES
                    and self._is_low(ly)
                    and lvy >= 0):
                return self._finish_rally(frame_idx, lx, ly, lconf, lsrc,
                                          how="vanished while descending")

        # ── Rally end C: lost the shuttle entirely → abort, no point ─────────
        last_f = self._last_det[0] if self._last_det else self._rally_start_frame
        if frame_idx - last_f > self._ABORT_FRAMES:
            print(f"  ⚠ [{frame_idx}] Rally aborted — shuttle lost for "
                  f"{frame_idx - last_f} frames, no point awarded")
            self._reset_rally()

        return None

    # ------------------------------------------------------------------
    # Rally state machine helpers
    # ------------------------------------------------------------------

    def _is_low(self, py: float) -> bool:
        """True if the pixel row is within the court's floor band."""
        if self._floor_min_y is not None:
            return py >= self._floor_min_y
        return (self._frame_height is not None
                and py > self._FLOOR_RATIO * self._frame_height)

    def _court_side(self, pos) -> Optional[str]:
        if self.court.H is None:
            return None
        try:
            cx, _ = self.court.to_court(pos[0], pos[1])
            return "LEFT" if cx < 0 else "RIGHT"
        except Exception:
            return None

    def _start_rally(self, frame_idx: int, announce: str):
        self.state.rally_state = "RALLY_ACTIVE"
        self._rally_start_frame = frame_idx
        self._launch_count = 0
        self._slow_count = 0
        print(f"  🏸 [{frame_idx}] Rally active ({announce}) "
              f"hitter={self.state.last_hitter_side or '?'}")

    def _detect_hit(self, frame_idx: int):
        """
        A hit is a sharp direction change between two consecutive trajectory
        legs, both moving at racket speed. The court half where it happens
        identifies the hitter.
        """
        if frame_idx - self._last_hit_frame <= self._HIT_COOLDOWN:
            return
        if len(self._traj) < 3:
            return
        (t0, x0, y0), (t1, x1, y1), (t2, x2, y2) = list(self._traj)[-3:]
        if t1 - t0 > self._HIT_GAP_MAX or t2 - t1 > self._HIT_GAP_MAX:
            return

        v1 = np.array([(x1 - x0) / (t1 - t0), (y1 - y0) / (t1 - t0)])
        v2 = np.array([(x2 - x1) / (t2 - t1), (y2 - y1) / (t2 - t1)])
        s1, s2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if s1 < self._HIT_MIN_SPEED or s2 < self._HIT_MIN_SPEED:
            return
        if float(np.dot(v1, v2)) / (s1 * s2) >= self._HIT_MAX_COS:
            return

        side = self._court_side((x1, y1))
        if side is not None:
            self.state.last_hitter_side = side
            self._last_hit_frame = frame_idx
            print(f"  🏓 [{frame_idx}] Hit by {side} "
                  f"(turn {s1:.0f}→{s2:.0f}px/f)")

    def _finish_rally(self, frame_idx: int, px: float, py: float,
                      conf: float, source: str, how: str) -> Optional[LandingEvent]:
        if self.court.H is None:
            self._reset_rally()
            return None
        try:
            cx, cy = self.court.to_court(px, py)
        except Exception:
            self._reset_rally()
            return None

        event = LandingEvent(
            frame_idx=frame_idx,
            pixel_x=px, pixel_y=py,
            court_x=cx, court_y=cy,
            confidence=conf,
            source=source,
        )

        dist = distance_to_boundary(cx, cy, self._zone_poly)
        event.near_line = abs(dist) < LINE_MARGIN

        # Sanity guard: a real shot lands at most a short distance past the
        # line. Far beyond that = false detection in the crowd/net/background.
        if dist < -(LINE_MARGIN + self._MAX_PLAUSIBLE_OUT):
            event.reason = (f"Discarded — {abs(dist)*100:.0f}cm outside boundary "
                            f"(likely false detection), no point [{how}]")
        elif dist >= 0:
            event.verdict = Verdict.IN
            event.reason = (f"IN: {dist*100:.1f}cm inside "
                            f"[{how}, {source} conf={conf:.2f}]")
        else:
            event.verdict = Verdict.OUT
            event.reason = (f"OUT: {abs(dist)*100:.1f}cm outside "
                            f"[{how}, {source} conf={conf:.2f}]")

        if event.verdict in (Verdict.IN, Verdict.OUT):
            self._update_score(event)
        self.state.history.append(event)
        self._reset_rally()
        self._cooldown = self._POINT_COOLDOWN
        return event

    def _reset_rally(self):
        self.state.rally_state = "SERVE_PENDING"
        self._traj.clear()
        self._prev_court_x = None
        self._last_det = None
        self._launch_count = 0
        self._slow_count = 0

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _update_score(self, event: LandingEvent):
        _opp = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
        landing_side = "LEFT" if event.court_x < 0 else "RIGHT"

        if event.verdict == Verdict.IN:
            # Shuttle landed in-bounds on landing_side's half → other player scored
            scorer_side = _opp[landing_side]
        else:
            # OUT → the player who hit it out loses the point. If we never saw
            # the hitter, the shuttle still fell on landing_side's end of the
            # court, so the hit came from the other side.
            hitter = self.state.last_hitter_side or _opp[landing_side]
            scorer_side = _opp[hitter]

        scorer = self.state.side_to_player[scorer_side]
        self.state.score[scorer] += 1
        print(f"  ✓ Point → {scorer_side} side (Player {scorer + 1}) | "
              f"{event.verdict.value} on {landing_side} | score={self.state.score}")
        self._check_serve_change(scorer)

    def _check_serve_change(self, scorer: int):
        s = self.state.score
        if scorer != self.state.serving_side:
            self.state.serving_side = scorer
        if (s[scorer] >= WINNING_SCORE and s[scorer] - s[1-scorer] >= WINNING_MARGIN) \
                or s[scorer] == MAX_SCORE:
            self._game_over(scorer)

    def _game_over(self, winner: int):
        print(f"Game {self.state.game} won by player {winner + 1}: {self.state.score}")
        self.state.score = [0, 0]
        self.state.game += 1
        self.state.last_hitter_side = None
        self._prev_court_x = None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_court_boundaries(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the detected court grid onto the frame and return the annotated
        copy. Uses the canonical lines projected by court_mapping during
        auto_calibrate(); returns the frame unchanged if not calibrated.
        """
        if not self._projected_lines:
            return frame
        return draw_court_overlay(frame, self._projected_lines, corners=None)
