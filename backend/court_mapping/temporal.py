from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np


@dataclass
class VideoInfo:
    """Static metadata about the input video."""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_sec: float


@dataclass
class MedianResult:
    """Output of compute_median_frame()."""
    median_frame: np.ndarray  # BGR uint8, (H, W, 3)
    n_sampled: int            # number of frames actually used
    info: VideoInfo


def probe_video(video_path: Union[str, Path]) -> VideoInfo:
    """Open the video, read header info, close. Cheap — no decoding."""
    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if w <= 0 or h <= 0:
            raise RuntimeError(f"Video has zero or unreadable dimensions: {path}")
        duration = (n / fps) if fps > 0 else 0.0
        return VideoInfo(path=path, width=w, height=h, fps=fps,
                         total_frames=n, duration_sec=duration)
    finally:
        cap.release()


def compute_median_frame(
    video_path: Union[str, Path],
    n_frames: int = 60,
    max_dim: int | None = None,
) -> MedianResult:
    
    info = probe_video(video_path)

    if info.total_frames <= 0:
        # Container reported no frame count — fall back to sequential read.
        frames = _read_sequential_sample(info, n_frames, max_dim)
    else:
        frames = _read_indexed_sample(info, n_frames, max_dim)

    if not frames:
        raise RuntimeError(f"Could not read any frames from video: {info.path}")

    # Stack along axis=0 → (N, H, W, 3). float32 keeps median stable for uint8 inputs.
    stack = np.stack(frames, axis=0)
    median = np.median(stack, axis=0).astype(np.uint8)

    if max_dim is not None and (median.shape[1] != info.width or median.shape[0] != info.height):
        median = cv2.resize(median, (info.width, info.height), interpolation=cv2.INTER_LINEAR)

    return MedianResult(median_frame=median, n_sampled=len(frames), info=info)


# ─── internal helpers ───────────────────────────────────────────────────────

def _maybe_downscale(frame: np.ndarray, max_dim: int | None) -> np.ndarray:
    if max_dim is None:
        return frame
    h, w = frame.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_dim:
        return frame
    scale = max_dim / float(long_edge)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _read_indexed_sample(info: VideoInfo, n_frames: int,
                          max_dim: int | None) -> list[np.ndarray]:
    """Seek to evenly-spaced indices (preferred path)."""
    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video for sampling: {info.path}")
    try:
        target = min(n_frames, max(1, info.total_frames))
        # Skip the first/last few frames in case of fade-in/black borders.
        margin = max(1, info.total_frames // 50)
        lo = margin
        hi = max(margin + 1, info.total_frames - margin)
        idxs = np.linspace(lo, hi - 1, num=target, dtype=int)

        frames: list[np.ndarray] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(_maybe_downscale(frame, max_dim))
        return frames
    finally:
        cap.release()


def _read_sequential_sample(info: VideoInfo, n_frames: int,
                             max_dim: int | None) -> list[np.ndarray]:
    """Fallback: total-frames unknown. Read sequentially with a stride guess."""
    cap = cv2.VideoCapture(info.path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot reopen video for sampling: {info.path}")
    try:
        # Assume ~10s of useful content × 30 fps ≈ 300 frames if duration unknown.
        guess_total = int(info.duration_sec * info.fps) if info.fps > 0 else 300
        guess_total = max(n_frames, guess_total)
        stride = max(1, guess_total // n_frames)

        frames: list[np.ndarray] = []
        i = 0
        while len(frames) < n_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if i % stride == 0:
                frames.append(_maybe_downscale(frame, max_dim))
            i += 1
        return frames
    finally:
        cap.release()
