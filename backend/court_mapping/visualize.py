"""
Render the detected court grid onto the input video and write a marked MP4.

Since the camera is static, the projected court lines are constant for the
entire video — we compute them once and overlay the same shape on every frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np

from .geometry import CourtLine


# Per-line-kind drawing styles (BGR).
_LINE_STYLES = {
    "boundary": dict(color=(0, 255, 0),   thickness=3, alpha=0.85),
    "service":  dict(color=(0, 200, 255), thickness=2, alpha=0.70),
    "center":   dict(color=(255, 200, 0), thickness=2, alpha=0.70),
    "singles":  dict(color=(255, 0, 255), thickness=2, alpha=0.55),
}

_CORNER_STYLE = dict(color=(0, 0, 255), radius=8, thickness=-1)
_CORNER_LABELS = ("TL", "TR", "BR", "BL")


def draw_court_overlay(
    frame: np.ndarray,
    projected_lines: List[Tuple[CourtLine, Tuple[Tuple[int, int], Tuple[int, int]]]],
    corners: np.ndarray | None = None,
    label_corners: bool = True,
) -> np.ndarray:
    """
    Draw the court grid onto a single frame and return the annotated frame.

    The drawing is done on a transparent overlay copy and alpha-blended
    according to per-line-kind style; this keeps the broadcast image visible
    underneath the lines.
    """
    annotated = frame.copy()

    # Group lines by their alpha so we can blend each group in one step.
    by_alpha: dict[float, List[Tuple[CourtLine, Tuple[Tuple[int, int], Tuple[int, int]]]]] = {}
    for ln, (p1, p2) in projected_lines:
        style = _LINE_STYLES.get(ln.kind, _LINE_STYLES["service"])
        by_alpha.setdefault(style["alpha"], []).append((ln, (p1, p2)))

    h, w = frame.shape[:2]
    for alpha, group in by_alpha.items():
        layer = annotated.copy()
        for ln, (p1, p2) in group:
            style = _LINE_STYLES.get(ln.kind, _LINE_STYLES["service"])
            cv2.line(layer, _clip_pt(p1, w, h), _clip_pt(p2, w, h),
                     style["color"], style["thickness"], cv2.LINE_AA)
        cv2.addWeighted(layer, alpha, annotated, 1.0 - alpha, 0.0, dst=annotated)

    if corners is not None:
        for i, pt in enumerate(corners):
            x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
            cv2.circle(annotated, (x, y), _CORNER_STYLE["radius"],
                       _CORNER_STYLE["color"], _CORNER_STYLE["thickness"], cv2.LINE_AA)
            if label_corners:
                cv2.putText(annotated, _CORNER_LABELS[i], (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            _CORNER_STYLE["color"], 2, cv2.LINE_AA)

    return annotated


def annotate_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    projected_lines: List[Tuple[CourtLine, Tuple[Tuple[int, int], Tuple[int, int]]]],
    corners: np.ndarray | None = None,
    progress_every: int = 60,
) -> int:
    """
    Read input_path frame-by-frame, overlay the court grid, write to output_path
    as an MP4. Returns the number of frames written.
    """
    input_path = str(input_path)
    output_path = str(output_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video for writing: {output_path}")

    n_written = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            annotated = draw_court_overlay(frame, projected_lines, corners=corners)
            writer.write(annotated)
            n_written += 1
            if progress_every and n_written % progress_every == 0:
                pct = (n_written / n_total * 100.0) if n_total else 0.0
                print(f"  ...{n_written} frames written ({pct:.1f}%)")
    finally:
        cap.release()
        writer.release()

    return n_written


def save_debug_artifacts(
    out_dir: Union[str, Path],
    median_frame: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    rough_corners: np.ndarray | None = None,
    refined_corners: np.ndarray | None = None,
    projected_lines: List[Tuple[CourtLine, Tuple[Tuple[int, int], Tuple[int, int]]]] | None = None,
) -> List[str]:
    """Save median frame, mask, and an annotated still showing the detected court."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    if median_frame is not None:
        p = out_dir / "median_frame.png"
        cv2.imwrite(str(p), median_frame)
        saved.append(str(p))

    if mask is not None:
        p = out_dir / "court_mask.png"
        cv2.imwrite(str(p), mask)
        saved.append(str(p))

    if median_frame is not None and refined_corners is not None and projected_lines is not None:
        annotated = draw_court_overlay(median_frame, projected_lines, corners=refined_corners)
        if rough_corners is not None:
            for i, pt in enumerate(rough_corners):
                x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
                cv2.drawMarker(annotated, (x, y), (255, 255, 255),
                               cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        p = out_dir / "detected_court_still.png"
        cv2.imwrite(str(p), annotated)
        saved.append(str(p))

    return saved


# ─── helpers ────────────────────────────────────────────────────────────────

def _clip_pt(pt: Tuple[int, int], w: int, h: int) -> Tuple[int, int]:
    """Clip a point to a slightly extended frame so cv2.line handles huge offsets."""
    margin = max(w, h)
    x = max(-margin, min(w + margin, int(pt[0])))
    y = max(-margin, min(h + margin, int(pt[1])))
    return (x, y)
