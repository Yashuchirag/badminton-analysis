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
    score:        list[int] = field(default_factory=lambda: [0, 0])
    serving_side: int  = 0
    last_hitter:  int  = 0    # 0 or 1 — update externally from shot tracker
    rally_active: bool = False
    game:         int  = 1
    history:      list[LandingEvent] = field(default_factory=list)


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
        self._H_inv: Optional[np.ndarray] = None

    def calibrate(self, pixel_pts: list, court_pts: list):
        self.court.calibrate(np.array(pixel_pts), np.array(court_pts))
        # Also store the inverse homography so we can project court → pixel
        if self.court.H is not None:
            self._H_inv = np.linalg.inv(self.court.H)
        else:
            self._H_inv = None

    def process_frame(
            self,
            frame,
            frame_idx: int,
        ) -> Optional[LandingEvent]:
        pos, conf, source = self.tracker.predict_frame(
            frame,
            mode=self.tracking_mode,
            conf_threshold=self.conf_threshold,
        )

        det: Optional[RawDetection] = self.landing.update(pos, conf, source)
        if det is None:
            return None

        return self._score_landing(frame, det, frame_idx)

    # ------------------------------------------------------------------
    # Scoring — blocking, sequential, no threads
    # ------------------------------------------------------------------

    def _score_landing(self, frame, det: RawDetection, frame_idx: int) -> LandingEvent:
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

        if dist > LINE_MARGIN:
            event.verdict = Verdict.IN
            event.reason = f"Geometric IN: {dist*100:.1f}cm inside [{det.source} conf={det.conf:.2f}]"

        elif dist < -LINE_MARGIN:
            event.verdict = Verdict.OUT
            event.reason = f"Geometric OUT: {abs(dist)*100:.1f}cm outside [{det.source} conf={det.conf:.2f}]"

        else:
            # Near line — block and wait for Gemma4 (intentional, keeps score in sync)
            if self.gemma:
                annotated = self._draw_court_overlay(frame.copy(), cx, cy, det.x, det.y)
                result = self.gemma.judge(
                    annotated, cx, cy,
                    zone_name=f"{self.game_mode} court",
                    distance_to_line=dist,
                )
                event.verdict = Verdict(result.get("verdict", "OUT"))
                event.gemma_used = True
                event.reason = (
                    f"Gemma4 [{det.source} conf={det.conf:.2f}]: "
                    f"{result.get('reason', '')} "
                    f"(gemma_conf={result.get('confidence', 0):.2f})"
                )
            else:
                # No Gemma available — fall back to geometric best-guess
                event.verdict = Verdict.IN if dist >= 0 else Verdict.OUT
                event.reason = f"Geometric fallback (no Gemma): dist={dist*100:.1f}cm"

        if event.verdict in (Verdict.IN, Verdict.OUT):
            self._update_score(event)

        self.state.history.append(event)
        return event

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
        if self._H_inv is None:
            return frame

        # Project zone polygon corners court → pixel
        corners = np.array(self._zone_poly, dtype=np.float32).reshape((-1, 1, 2))
        pix_corners = cv2.perspectiveTransform(corners, self._H_inv)
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
        last_hitter = self.state.last_hitter
        scorer = last_hitter if event.verdict == Verdict.IN else 1 - last_hitter
        self.state.score[scorer] += 1
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