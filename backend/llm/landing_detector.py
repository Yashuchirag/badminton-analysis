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

    def __init__(
            self,
            min_speed_px: float = 3.0,        
            speed_drop_ratio: float = 0.55,    
            disappear_min_speed: float = 6.0,  
            cooldown_frames: int = 20,         
            history_len: int = 8,
        ):
        self.min_speed = min_speed_px
        self.speed_drop_ratio = speed_drop_ratio
        self.disappear_speed = disappear_min_speed
        self.cooldown = cooldown_frames

        self._history: deque[Optional[RawDetection]] = deque(maxlen=history_len)
        self._frames_since: int = cooldown_frames   # start ready

    def update(
            self,
            pos: Optional[tuple[float, float]],
            conf: float,
            source: str,
        ) -> Optional[RawDetection]:
        self._frames_since += 1

        det: Optional[RawDetection] = None
        if pos is not None:
            det = RawDetection(x=pos[0], y=pos[1], conf=conf, source=source)

            # Compute velocity from previous valid detection
            prev = next((d for d in reversed(self._history) if d is not None), None)
            if prev:
                det.vx = det.x - prev.x
                det.vy = det.y - prev.y
                det.speed = np.hypot(det.vx, det.vy)

        self._history.append(det)

        if self._frames_since < self.cooldown:
            return None

        valid = [d for d in self._history if d is not None]

        # ── Condition 1: arc bottom (vy reversal) ────────────────────────
        if det and len(valid) >= 3:
            vy_window = [d.vy for d in valid[-3:]]
            # Was moving down (positive vy in image coords) and stopped/reversed
            if vy_window[-2] > 2.0 and vy_window[-1] < vy_window[-2] * 0.25:
                return self._register(det)

        # ── Condition 2: speed crash ──────────────────────────────────────
        if det and len(valid) >= 2 and det.speed > self.min_speed:
            prev_speed = valid[-2].speed
            if prev_speed > self.min_speed and det.speed < prev_speed * (1 - self.speed_drop_ratio):
                return self._register(det)

        # ── Condition 3: disappearance after fast downward motion ─────────
        if det is None and len(valid) >= 2:
            last = valid[-1]
            if last.vy > 0 and last.speed > self.disappear_speed:
                # Return the last known position as the landing estimate
                return self._register(last)

        return None

    def _register(self, det: RawDetection) -> RawDetection:
        self._frames_since = 0
        return det