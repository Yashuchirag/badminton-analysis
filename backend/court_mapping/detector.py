"""
End-to-end orchestrator: video file → detected court corners + homography.

Pipeline:
    1. Compute temporal-median frame from the video (removes players/shuttle).
    2. Segment the court surface from the median frame (HSV mask).
    3. Approximate the largest mask blob as a 4-point quadrilateral.
    4. Refine each corner sub-pixel using LSD line fitting along corridors.
    5. Validate the resulting quad (convexity, area, in-bounds).
    6. Fit a homography from the refined corners to the canonical court.
    7. Project all canonical court lines back into image space.

The output is a CourtDetectionResult bundling the detected geometry, the
projected line set ready for drawing, and (optionally) intermediate artefacts
for debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from llm.court_geometry import CourtHomography

from .geometry import (
    CourtLine,
    fit_homography,
    mask_to_quadrilateral,
    project_court_lines,
    validate_corners,
)
from .lines import refine_corners
from .segment import segment_court_hsv
from .temporal import MedianResult, VideoInfo, compute_median_frame


@dataclass
class CourtDetectionResult:
    """Full output of the detector."""
    success: bool
    confidence: str                        # "high" | "medium" | "low" | "failed"
    rough_corners: Optional[np.ndarray]     # (4, 2) before line refinement
    refined_corners: Optional[np.ndarray]   # (4, 2) final
    homography: Optional[CourtHomography]
    projected_lines: List[Tuple[CourtLine, Tuple[Tuple[int, int], Tuple[int, int]]]] = field(default_factory=list)
    mode: str = "doubles"
    median_frame: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    info: Optional[VideoInfo] = None
    refinement_used: List[bool] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def detect_court_in_video(
    video_path: Union[str, Path],
    mode: str = "doubles",
    n_frames: int = 60,
    refine: bool = True,
    keep_artifacts: bool = False,
) -> CourtDetectionResult:
    """
    Run the full pipeline on a video file.

    Parameters
    ----------
    video_path : str | Path
    mode : "doubles" | "singles"
        Which canonical court width to fit. Default doubles is safest because
        the doubles outer line is the visible boundary in nearly every video.
    n_frames : int
        Frames to sample for the temporal median.
    refine : bool, default True
        Whether to run sub-pixel line-refinement after mask quad approximation.
    keep_artifacts : bool, default False
        If True, populate `median_frame` and `mask` on the result for caller
        inspection / debug saving.

    Returns
    -------
    CourtDetectionResult
        With success/confidence flags. Even on partial failure the orchestrator
        attempts to return whatever it has (e.g. rough corners with no
        refinement) so downstream visualization can still produce something.
    """
    notes: List[str] = []

    # ── 1. temporal median ───────────────────────────────────────────────
    median: MedianResult = compute_median_frame(video_path, n_frames=n_frames)
    notes.append(f"Sampled {median.n_sampled} frames for median (target {n_frames}).")

    # ── 2. court surface segmentation ────────────────────────────────────
    seg = segment_court_hsv(median.median_frame)
    if seg is None:
        notes.append("Court-surface HSV segmentation failed.")
        return CourtDetectionResult(
            success=False, confidence="failed",
            rough_corners=None, refined_corners=None, homography=None,
            mode=mode,
            median_frame=median.median_frame if keep_artifacts else None,
            mask=None,
            info=median.info,
            notes=notes,
        )
    notes.append(
        f"Surface mask covers {seg.largest_area_ratio:.1%} of frame, "
        f"hue={seg.sampled_hue:.0f}, sat_lo={seg.sampled_sat_lo}."
    )

    # ── 3. mask → rough quadrilateral ────────────────────────────────────
    rough = mask_to_quadrilateral(seg.mask)
    if rough is None:
        notes.append("Could not approximate mask as a 4-point quadrilateral.")
        return CourtDetectionResult(
            success=False, confidence="failed",
            rough_corners=None, refined_corners=None, homography=None,
            mode=mode,
            median_frame=median.median_frame if keep_artifacts else None,
            mask=seg.mask if keep_artifacts else None,
            info=median.info,
            notes=notes,
        )

    rough_report = validate_corners(rough, median.median_frame.shape[:2])
    if not rough_report.ok:
        notes.append(f"Rough quad failed validation: {rough_report.reasons}")

    # ── 4. line-based corner refinement ──────────────────────────────────
    if refine:
        refinement = refine_corners(median.median_frame, rough)
        refined = refinement.refined_corners
        used_refinement = refinement.used_refinement_mask
        n_refined = sum(used_refinement)
        notes.append(f"Sub-pixel line refinement used on {n_refined}/4 corners.")
    else:
        refined = rough.copy()
        used_refinement = [False, False, False, False]
        notes.append("Sub-pixel refinement skipped (refine=False).")

    # ── 5. validate refined corners ──────────────────────────────────────
    refined_report = validate_corners(refined, median.median_frame.shape[:2])
    if not refined_report.ok:
        notes.append(f"Refined quad failed validation: {refined_report.reasons}")
        # Fall back to rough if rough was OK and refined isn't.
        if rough_report.ok:
            notes.append("Falling back to rough corners (pre-refinement).")
            refined = rough.copy()
            used_refinement = [False, False, False, False]

    final_report = validate_corners(refined, median.median_frame.shape[:2])

    # ── 6. fit homography ────────────────────────────────────────────────
    homo = fit_homography(refined, mode=mode)
    if homo is None or homo.H is None:
        notes.append("Failed to fit homography from corners.")
        return CourtDetectionResult(
            success=False, confidence="low",
            rough_corners=rough, refined_corners=refined, homography=None,
            mode=mode,
            median_frame=median.median_frame if keep_artifacts else None,
            mask=seg.mask if keep_artifacts else None,
            info=median.info,
            refinement_used=used_refinement,
            notes=notes,
        )

    # ── 7. project canonical lines ──────────────────────────────────────
    projected = project_court_lines(homo, mode=mode)

    # ── confidence score ────────────────────────────────────────────────
    confidence = _classify_confidence(
        n_refined=sum(used_refinement),
        validation_ok=final_report.ok,
        area_ratio=seg.largest_area_ratio,
    )

    return CourtDetectionResult(
        success=final_report.ok,
        confidence=confidence,
        rough_corners=rough,
        refined_corners=refined,
        homography=homo,
        projected_lines=projected,
        mode=mode,
        median_frame=median.median_frame if keep_artifacts else None,
        mask=seg.mask if keep_artifacts else None,
        info=median.info,
        refinement_used=used_refinement,
        notes=notes,
    )


def _classify_confidence(n_refined: int, validation_ok: bool, area_ratio: float) -> str:
    if not validation_ok:
        return "low"
    if n_refined >= 3 and area_ratio >= 0.10:
        return "high"
    if n_refined >= 2 and area_ratio >= 0.07:
        return "medium"
    return "low"
