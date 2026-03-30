from scipy.signal import savgol_filter

class RallyDetector:
    """
    Detects rallies and counts shots from shuttle position data.
    Plug directly into ShuttleTracker.track_video().
    """

    def __init__(self, fps: int, gap_seconds: float = 0.5, min_rally_frames: int = 15):
        """
        Args:
            fps: Video FPS (already available in track_video)
            gap_seconds: Seconds of no detection = rally ended (tune this)
            min_rally_frames: Ignore rallies shorter than this (removes false positives)
        """
        self.fps = fps
        self.gap_threshold = int(gap_seconds * fps)  # e.g. 0.5s * 30fps = 15 frames
        self.min_rally_frames = min_rally_frames

        self.state = "IDLE"
        self.rallies = []                  # completed rallies
        self.current_positions = []        # (frame_idx, x, y) for current rally
        self.current_start_frame = None
        self.no_detect_count = 0

    def update(self, frame_idx: int, position):
        """Call once per frame with current position (or None)."""
        if position is not None:
            self.no_detect_count = 0

            if self.state == "IDLE":
                # New rally started
                self.state = "IN_PLAY"
                self.current_start_frame = frame_idx
                self.current_positions = []

            self.current_positions.append((
                frame_idx, 
                float(position[0]), 
                float(position[1])
            ))

        else:
            self.no_detect_count += 1

            if self.state == "IN_PLAY" and self.no_detect_count >= self.gap_threshold:
                # Rally ended — finalize it
                self.state = "IDLE"
                self._finalize_rally(frame_idx)

    def _finalize_rally(self, end_frame: int):
        """Compute shot count and save the rally."""
        duration_frames = end_frame - self.current_start_frame

        # Skip very short detections (noise)
        if duration_frames < self.min_rally_frames:
            self.current_positions = []
            return

        shot_count = self._count_shots(self.current_positions)

        rally = {
            "rally_id": len(self.rallies) + 1,
            "start_frame": self.current_start_frame,
            "end_frame": end_frame,
            "duration_seconds": round(duration_frames / self.fps, 2),
            "shot_count": shot_count,
            "positions": self.current_positions,  # full trajectory
        }
        self.rallies.append(rally)

        print(f"  ✓ Rally {rally['rally_id']} ended | "
              f"Duration: {rally['duration_seconds']}s | "
              f"Shots: {shot_count}")

        self.current_positions = []

    def _count_shots(self, positions: list) -> int:
        """
        Count shots by detecting direction reversals in the Y-axis trajectory.
        Uses Savitzky-Golay smoothing to suppress noise.
        """
        if len(positions) < 7:  # Not enough points to smooth
            return 0

        ys = [p[2] for p in positions]  # extract Y coords

        # Smooth the trajectory — this is critical to avoid false peaks from jitter
        window = min(11, len(ys) if len(ys) % 2 == 1 else len(ys) - 1)
        try:
            smoothed = savgol_filter(ys, window_length=window, polyorder=2)
        except Exception:
            smoothed = ys

        # Count direction changes (local minima + local maxima = shots)
        shot_count = 0
        for i in range(1, len(smoothed) - 1):
            is_peak = smoothed[i - 1] < smoothed[i] > smoothed[i + 1]
            is_trough = smoothed[i - 1] > smoothed[i] < smoothed[i + 1]
            if is_peak or is_trough:
                shot_count += 1

        return shot_count

    def get_summary(self) -> dict:
        """Return match-level stats."""
        if not self.rallies:
            return {}
        return {
            "total_rallies": len(self.rallies),
            "total_shots": sum(r["shot_count"] for r in self.rallies),
            "avg_shots_per_rally": round(
                sum(r["shot_count"] for r in self.rallies) / len(self.rallies), 1
            ),
            "longest_rally_seconds": max(r["duration_seconds"] for r in self.rallies),
            "rallies": self.rallies,
        }