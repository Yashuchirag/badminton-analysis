"""
court_mapping — automatic badminton court corner detection from video.

Public API
----------
    from court_mapping import detect_court_in_video, annotate_video
    from court_mapping import CourtDetectionResult

The recommended entry points are:
    detect_court_in_video(video_path, mode="doubles") -> CourtDetectionResult
    annotate_video(input_path, output_path, projected_lines, corners=...) -> int

CLI:
    python -m court_mapping.run --input <video.mp4> [--debug]
"""

from .detector import CourtDetectionResult, detect_court_in_video
from .geometry import (
    CourtLine,
    canonical_court_lines,
    fit_homography,
    mask_to_quadrilateral,
    order_corners,
    project_court_lines,
    validate_corners,
)
from .lines import refine_corners
from .segment import SegmentationResult, segment_court_hsv
from .temporal import MedianResult, VideoInfo, compute_median_frame, probe_video
from .visualize import annotate_video, draw_court_overlay, save_debug_artifacts

__all__ = [
    # detector
    "CourtDetectionResult",
    "detect_court_in_video",
    # geometry
    "CourtLine",
    "canonical_court_lines",
    "fit_homography",
    "mask_to_quadrilateral",
    "order_corners",
    "project_court_lines",
    "validate_corners",
    # lines
    "refine_corners",
    # segment
    "SegmentationResult",
    "segment_court_hsv",
    # temporal
    "MedianResult",
    "VideoInfo",
    "compute_median_frame",
    "probe_video",
    # visualize
    "annotate_video",
    "draw_court_overlay",
    "save_debug_artifacts",
]
