import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from court_geometry import CourtHomography, ZONES, LINE_MARGIN, distance_to_boundary
from model2.train_and_track import ShuttleTracker
from landing_detector import LandingDetector, RawDetection
from gemma_client import LineJudge
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
    gemma_used: bool  = False
    reason:     str   = ""


@dataclass
class MatchState:
    score:            list[int] = field(default_factory=lambda: [0, 0])
    serving_side:     int  = 0
    last_hitter_side: Optional[str] = None   # "LEFT" or "RIGHT", set by net crossing
    side_to_player:   dict = field(default_factory=lambda: {"LEFT": 0, "RIGHT": 1})
    rally_state:      str  = "SERVE_PENDING"  # "SERVE_PENDING" | "RALLY_ACTIVE"
    game:             int  = 1
    history:          list[LandingEvent] = field(default_factory=list)


class ScoringEngine:

    def __init__(
            self,
            yolo_weights: Optional[str] = None,
            obb_weights: Optional[str] = None,
            tracknet_weights: Optional[str] = None,
            gemma: Optional[LineJudge] = None,
            mode: str = "singles",
            tracking_mode: str = "hybrid",    # passed to ShuttleTracker
            conf_threshold: float = 0.25,
            device: str = "cpu",
            player_model: str = "yolov8n.pt",   # pretrained model for player detection
            player_detection_interval: int = 30, # run player detector every N frames
        ):
        self.game_mode = mode
        self.tracking_mode = tracking_mode
        self.conf_threshold = conf_threshold
        self.state = MatchState()
        self.gemma = gemma
        self._zone_poly = ZONES["doubles_back"] if mode == "doubles" else ZONES["singles_back"]

        # Your real ShuttleTracker — loads models exactly as your CLI does
        self.tracker = ShuttleTracker(
            yolo_weights=yolo_weights,
            obb_weights=obb_weights,
            tracknet_weights=tracknet_weights,
            device=device,
        )

        self.court = CourtHomography()
        self.landing = LandingDetector()
        self.players = PlayerTracker(model_path=player_model)
        self._player_detection_interval = player_detection_interval

        # ── Post-candidate confirmation window ───────────────────────────────
        # LandingDetector emits *candidates* (includes racket hits).
        # We hold each candidate for CONFIRM_WINDOW frames and watch whether
        # the shuttle comes back moving fast.  If it does → racket hit → cancel.
        # If it stays gone / slow → floor landing → score.
        self._CONFIRM_WINDOW  = 10    # frames to observe after a candidate
        self._CANCEL_SPEED    = 22.0  # px/frame: above this = still in flight
        self._FLOOR_HEIGHT_RATIO = 0.55  # shuttle must be below this % of frame height
        self._pending_event: Optional[LandingEvent] = None
        self._pending_countdown: int = 0
        self._pending_prev_pos: Optional[tuple] = None

        # ── Net crossing + pixel speed tracking ──────────────────────────────
        self._prev_court_x: Optional[float] = None
        self._prev_pixel_pos: Optional[tuple] = None
        self._pixel_speed: float = 0.0
        self._frame_height: Optional[int] = None

        # ── Serve detection ───────────────────────────────────────────────────
        self._SERVE_LOW_SPEED    = 5.0   # px/frame — shuttle "at rest" near server
        self._SERVE_LAUNCH_SPEED = 8.0   # px/frame — shuttle launched
        self._SERVE_STILL_FRAMES = 3     # consecutive still frames before candidate set
        self._serve_candidate_side: Optional[str] = None
        self._serve_still_count: int = 0

    def auto_calibrate(self, video_path: str, n_frames: int = 90,
                       debug: bool = False) -> list:
        """
        Auto-detect court corners from the video and calibrate the homography.
        Call this once before starting process_frame().

        Args:
            video_path: path to the video file (same one you'll process).
            n_frames:   how many frames to sample for line accumulation.
            debug:      if True, saves debug_court_mask_*.png images showing
                        what the detector sees — useful when detection fails.
        Returns the detected pixel corner list.
        """
        return self.court.auto_calibrate(video_path, mode=self.game_mode,
                                         n_frames=n_frames, debug=debug)

    def calibrate(self, pixel_pts: list, court_pts: list):
        """Manual calibration fallback — prefer auto_calibrate() instead."""
        self.court.calibrate(np.array(pixel_pts, dtype=np.float32),
                             np.array(court_pts, dtype=np.float32))

    def process_frame(
            self,
            frame,
            frame_idx: int,
        ) -> Optional[LandingEvent]:
        self._frame_height = frame.shape[0]

        # ── Player side mapping (runs every N frames, low overhead) ──────────
        if frame_idx % self._player_detection_interval == 0:
            side_map = self.players.update(frame, self.court)
            if len(side_map) == 2:
                self.state.side_to_player = side_map

        pos, conf, source = self.tracker.predict_frame(
            frame,
            mode=self.tracking_mode,
            conf_threshold=self.conf_threshold,
        )

        # ── Pixel speed (used by serve detection) ────────────────────────────
        if pos is not None:
            self._pixel_speed = (
                np.hypot(pos[0] - self._prev_pixel_pos[0],
                         pos[1] - self._prev_pixel_pos[1])
                if self._prev_pixel_pos else 0.0
            )
            self._prev_pixel_pos = pos
        else:
            self._pixel_speed = 0.0

        # ── Net crossing → last_hitter_side (active in both states) ──────────
        if pos is not None and self.court.H is not None:
            try:
                court_x, _ = self.court.to_court(pos[0], pos[1])
                if self._prev_court_x is not None:
                    l2r = self._prev_court_x < 0 and court_x > 0
                    r2l = self._prev_court_x > 0 and court_x < 0
                    if l2r or r2l:
                        new_side = "LEFT" if l2r else "RIGHT"
                        self.state.last_hitter_side = new_side
                        if self.state.rally_state == "SERVE_PENDING":
                            # Shuttle already over the net — serve happened, rally active
                            self.state.rally_state = "RALLY_ACTIVE"
                            self._serve_candidate_side = None
                            self._serve_still_count = 0
                            print(f"  🏸 Rally active (net crossing) hitter={new_side}")
                self._prev_court_x = court_x
            except Exception:
                pass

        # ── SERVE_PENDING: watch for serve launch ─────────────────────────────
        if self.state.rally_state == "SERVE_PENDING":
            self._handle_serve_detection(pos)
            return None

        # ── RALLY_ACTIVE: confirmation window ────────────────────────────────
        if self._pending_event is not None:
            confirmed = self._tick_confirmation(pos, conf)
            return confirmed

        # ── RALLY_ACTIVE: new landing candidate ──────────────────────────────
        det: Optional[RawDetection] = self.landing.update(pos, conf, source)
        if det is None:
            return None

        # Height gate: floor landings happen in the lower portion of the frame.
        if self._frame_height is not None and det.y < self._FLOOR_HEIGHT_RATIO * self._frame_height:
            return None

        event = self._build_landing_event(frame, det, frame_idx)
        self._pending_event    = event
        self._pending_countdown = self._CONFIRM_WINDOW
        self._pending_prev_pos  = (det.x, det.y)
        return None

    # ------------------------------------------------------------------
    # Serve detection
    # ------------------------------------------------------------------

    def _handle_serve_detection(self, pos: Optional[tuple]):
        """
        Called each frame while in SERVE_PENDING.
        Watches for the shuttle to be still near a player then launch.
        Transitions to RALLY_ACTIVE and sets last_hitter_side on launch.
        """
        if pos is None:
            return

        if self._pixel_speed < self._SERVE_LOW_SPEED:
            self._serve_still_count += 1
            if self._serve_still_count >= self._SERVE_STILL_FRAMES:
                nearest = self._nearest_player_side(pos)
                if nearest:
                    self._serve_candidate_side = nearest
        else:
            # Shuttle is moving — if we had a still candidate, reset only if
            # speed hasn't reached launch threshold yet
            if self._serve_candidate_side is None:
                self._serve_still_count = 0

        if (self._serve_candidate_side is not None
                and self._pixel_speed >= self._SERVE_LAUNCH_SPEED):
            self.state.last_hitter_side = self._serve_candidate_side
            self.state.rally_state = "RALLY_ACTIVE"
            self._serve_candidate_side = None
            self._serve_still_count = 0
            self._prev_court_x = None
            print(f"  🏸 Serve detected from {self.state.last_hitter_side} "
                  f"(speed={self._pixel_speed:.1f}px/f)")

    def _nearest_player_side(self, shuttle_pos: tuple) -> Optional[str]:
        """Return which court side's player is closest to the shuttle pixel position."""
        side_positions = self.players.get_side_positions()
        if not side_positions:
            return None
        sx, sy = shuttle_pos
        best_side, best_dist = None, float("inf")
        for side, (px, py) in side_positions.items():
            d = np.hypot(sx - px, sy - py)
            if d < best_dist:
                best_dist = d
                best_side = side
        return best_side

    # ------------------------------------------------------------------
    # Confirmation window
    # ------------------------------------------------------------------

    def _tick_confirmation(self, pos, conf) -> Optional[LandingEvent]:
        """
        Called every frame while a candidate is pending.
        Returns the confirmed LandingEvent when the window expires, or None
        if still waiting / cancelled.
        """
        self._pending_countdown -= 1

        if pos is not None and conf > 0.25 and self._pending_prev_pos is not None:
            px, py = self._pending_prev_pos
            speed = np.hypot(pos[0] - px, pos[1] - py)
            self._pending_prev_pos = pos

            if speed > self._CANCEL_SPEED:
                # Shuttle is still moving fast → this was a racket hit, not a landing
                print(f"  ⚬ Candidate cancelled — shuttle still at {speed:.1f} px/frame")
                self._pending_event    = None
                self._pending_prev_pos = None
                return None

        elif pos is not None:
            self._pending_prev_pos = pos

        if self._pending_countdown <= 0:
            # Window expired without cancellation → confirm as floor landing
            event = self._pending_event
            self._pending_event    = None
            self._pending_prev_pos = None
            self._confirm_landing(event)
            return event

        return None   # still watching

    # ------------------------------------------------------------------
    # Event building + scoring
    # ------------------------------------------------------------------

    def _build_landing_event(self, frame, det: RawDetection, frame_idx: int) -> LandingEvent:
        """Compute court coords and IN/OUT verdict. Does NOT touch the score."""
        cx, cy = self.court.to_court(det.x, det.y)

        event = LandingEvent(
            frame_idx=frame_idx,
            pixel_x=det.x, pixel_y=det.y,
            court_x=cx, court_y=cy,
            confidence=det.conf,
            source=det.source,
        )

        dist = distance_to_boundary(cx, cy, self._zone_poly)
        event.near_line = abs(dist) < LINE_MARGIN

        # Sanity guard: a real badminton shot can be at most ~50cm past the line.
        # Anything further is a false detection in the crowd, ceiling, or net area —
        # discard it rather than counting a bogus OUT.
        MAX_PLAUSIBLE_OUT = 2.0   # metres
        if dist < -(LINE_MARGIN + MAX_PLAUSIBLE_OUT):
            event.verdict = Verdict.PENDING
            event.reason  = (f"Discarded — {abs(dist)*100:.0f}cm outside boundary "
                             f"(likely false detection in background)")
            return event

        if dist > LINE_MARGIN:
            event.verdict = Verdict.IN
            event.reason = f"Geometric IN: {dist*100:.1f}cm inside [{det.source} conf={det.conf:.2f}]"

        elif dist < -LINE_MARGIN:
            event.verdict = Verdict.OUT
            event.reason = f"Geometric OUT: {abs(dist)*100:.1f}cm outside [{det.source} conf={det.conf:.2f}]"

        else:
            if self.gemma:
                annotated = self._draw_court_overlay(frame.copy(), cx, cy, det.x, det.y)
                result = self.gemma.judge(
                    annotated, cx, cy,
                    zone_name=f"{self.game_mode} court",
                    distance_to_line=dist,
                )
                event.verdict  = Verdict(result.get("verdict", "OUT"))
                event.gemma_used = True
                event.reason = (
                    f"Gemma4 [{det.source} conf={det.conf:.2f}]: "
                    f"{result.get('reason', '')} "
                    f"(gemma_conf={result.get('confidence', 0):.2f})"
                )
            else:
                event.verdict = Verdict.IN if dist >= 0 else Verdict.OUT
                event.reason  = f"Geometric fallback (no Gemma): dist={dist*100:.1f}cm"

        return event

    def _confirm_landing(self, event: LandingEvent):
        """Score the confirmed floor landing and append to history."""
        if event.verdict in (Verdict.IN, Verdict.OUT):
            self._update_score(event)
        self.state.history.append(event)
        # Reset rally — wait for next serve
        self.state.rally_state = "SERVE_PENDING"
        self._serve_candidate_side = None
        self._serve_still_count = 0
        self._prev_court_x = None
        self._prev_pixel_pos = None

    def draw_court_boundaries(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw all standard court lines onto *frame* in-place and return it.

        Lines drawn (singles):
          • Outer boundary          — solid yellow
          • Net                     — solid cyan
          • Short service lines     — white, both sides
          • Center line             — white, full length

        For doubles, the outer boundary switches to the wider sidelines and
        the doubles long-service line is added.
        """
        if self.court.H_inv is None:
            return frame

        def _proj(*pts):
            """Project one or more (x, y) court-coord tuples → int pixel pairs."""
            arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
            out = cv2.perspectiveTransform(arr, self.court.H_inv)
            return [tuple(p.astype(int)) for p in out.reshape(-1, 2)]

        half_l = 6.70          # court half-length (metres)
        sw     = 2.59          # singles half-width
        dw     = 3.05          # doubles half-width
        ssl    = 1.98          # short service line distance from net

        half_w = dw if self.game_mode == "doubles" else sw

        # ── Outer boundary ───────────────────────────────────────────────────
        corners = _proj(
            (-half_l, -half_w), ( half_l, -half_w),
            ( half_l,  half_w), (-half_l,  half_w),
        )
        cv2.polylines(frame,
                      [np.array(corners, dtype=np.int32).reshape(-1, 1, 2)],
                      isClosed=True, color=(0, 220, 220), thickness=2)

        # ── Net ──────────────────────────────────────────────────────────────
        n1, n2 = _proj((0, -half_w), (0, half_w))
        cv2.line(frame, n1, n2, color=(255, 220, 0), thickness=2)

        # ── Short service lines (both sides of net) ───────────────────────────
        for sx in (ssl, -ssl):
            p1, p2 = _proj((sx, -half_w), (sx, half_w))
            cv2.line(frame, p1, p2, color=(200, 200, 200), thickness=1)

        # ── Center line (full length) ─────────────────────────────────────────
        c1, c2 = _proj((-half_l, 0), (half_l, 0))
        cv2.line(frame, c1, c2, color=(200, 200, 200), thickness=1)

        # ── Singles sidelines (inner) when in doubles mode ────────────────────
        if self.game_mode == "doubles":
            for sy in (sw, -sw):
                p1, p2 = _proj((-half_l, sy), (half_l, sy))
                cv2.line(frame, p1, p2, color=(160, 160, 160), thickness=1)
            # Doubles long service line (0.76 m inside back boundary)
            for sx in (half_l - 0.76, -(half_l - 0.76)):
                p1, p2 = _proj((sx, -dw), (sx, dw))
                cv2.line(frame, p1, p2, color=(160, 160, 160), thickness=1)

        return frame

    def _draw_court_overlay(
            self, frame: np.ndarray,
            court_x: float, court_y: float,
            px: float, py: float,
        ) -> np.ndarray:
        """
        Project the court boundary polygon back into pixel space and draw it
        on a copy of the frame, plus mark the shuttle landing position.
        This gives Gemma a visual reference for where the court lines are.
        """
        if self.court.H_inv is None:
            return frame

        # Project zone polygon corners court → pixel
        corners = np.array(self._zone_poly, dtype=np.float32).reshape((-1, 1, 2))
        pix_corners = cv2.perspectiveTransform(corners, self.court.H_inv)
        pix_corners = pix_corners.reshape((-1, 2)).astype(np.int32)

        # Draw court boundary in cyan
        cv2.polylines(frame, [pix_corners], isClosed=True,
                      color=(255, 255, 0), thickness=2)

        # Mark shuttle landing position with a bright circle + crosshair
        ipx, ipy = int(round(px)), int(round(py))
        cv2.circle(frame, (ipx, ipy), 8,  (0, 0, 255), 2)
        cv2.line(frame, (ipx - 12, ipy), (ipx + 12, ipy), (0, 0, 255), 1)
        cv2.line(frame, (ipx, ipy - 12), (ipx, ipy + 12), (0, 0, 255), 1)

        # Label with court coordinates
        label = f"({court_x:.2f}m, {court_y:.2f}m)"
        cv2.putText(frame, label, (ipx + 10, ipy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return frame

    def _update_score(self, event: LandingEvent):
        _opp = {"LEFT": "RIGHT", "RIGHT": "LEFT"}
        landing_side = "LEFT" if event.court_x < 0 else "RIGHT"

        if event.verdict == Verdict.IN:
            # Shuttle landed in-bounds on landing_side's half → other player scored
            scorer_side = _opp[landing_side]
        else:
            # OUT → the player who hit it out loses the point
            if self.state.last_hitter_side is None:
                print("  ⚠ OUT but last_hitter_side unknown — point skipped")
                return
            scorer_side = _opp[self.state.last_hitter_side]

        scorer = self.state.side_to_player[scorer_side]
        self.state.score[scorer] += 1
        print(f"  ✓ Point → {scorer_side} side opponent (Player {scorer + 1}) | {event.verdict.value} on {landing_side} | score={self.state.score}")
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