import numpy as np
from collections import deque
from typing import Optional
from dataclasses import dataclass


@dataclass
class RawDetection:
    x: float
    y: float
    conf: float
    source: str
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0


class LandingDetector:
    """
    Emits *candidate* landing events based on shuttle kinematics.

    Two trigger conditions (speed-crash removed — too noisy on racket hits):

      Condition A — arc reversal:
        The shuttle was moving downward (vy > vy_down_thresh) and its vertical
        velocity has now gone negative (upward) or stopped.  This catches the
        bottom of a descending arc — both floor bounces and racket lifts.

      Condition B — disappearance after downward fast flight:
        The shuttle was tracked at speed > disappear_min_speed, was moving
        downward, and has not been detected this frame.  Catches smashes that
        hit the floor before the arc reversal is observable.

    ⚠ These candidates include BOTH floor landings AND racket hits.
    The ScoringEngine's confirmation window is responsible for distinguishing
    them (racket hits cause the shuttle to reappear at high speed within a
    few frames; floor landings do not).
    """

    def __init__(
            self,
            vy_down_thresh:      float = 3.0,   # px/frame — min downward speed to consider
            disappear_min_speed: float = 7.0,   # px/frame — min speed on last frame before disappearance
            cooldown_frames:     int   = 35,    # frames blocked after any candidate fires
            history_len:         int   = 10,
        ):
        self.vy_down_thresh    = vy_down_thresh
        self.disappear_speed   = disappear_min_speed
        self.cooldown          = cooldown_frames

        self._history: deque[Optional[RawDetection]] = deque(maxlen=history_len)
        self._frames_since: int = cooldown_frames   # start ready

    def update(
            self,
            pos:    Optional[tuple[float, float]],
            conf:   float,
            source: str,
        ) -> Optional[RawDetection]:
        self._frames_since += 1

        det: Optional[RawDetection] = None
        if pos is not None:
            det = RawDetection(x=pos[0], y=pos[1], conf=conf, source=source)
            prev = next((d for d in reversed(self._history) if d is not None), None)
            if prev:
                det.vx    = det.x - prev.x
                det.vy    = det.y - prev.y
                det.speed = np.hypot(det.vx, det.vy)

        self._history.append(det)

        if self._frames_since < self.cooldown:
            return None

        valid = [d for d in self._history if d is not None]

        # ── Condition A: arc reversal ────────────────────────────────────────
        # Shuttle was moving down (positive vy), now moving up (negative vy)
        if det is not None and len(valid) >= 2:
            prev_det = valid[-2]
            if prev_det.vy > self.vy_down_thresh and det.vy < 0:
                return self._register(det)

        # ── Condition B: disappearance after downward fast flight ────────────
        if det is None and valid:
            last = valid[-1]
            if last.vy > self.vy_down_thresh and last.speed > self.disappear_speed:
                return self._register(last)

        return None

    def _register(self, det: RawDetection) -> RawDetection:
        self._frames_since = 0
        return det
