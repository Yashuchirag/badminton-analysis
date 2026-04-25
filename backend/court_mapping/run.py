"""
CLI entry point for the court-mapping pipeline.

Usage
-----
    cd backend
    python -m court_mapping.run --input path/to/video.mp4
    python -m court_mapping.run --input video.mp4 --output marked.mp4 --debug

The default output goes to backend/court_mapping/output/<input_stem>_marked.mp4.
With --debug, additional intermediate artefacts are written next to the video
(median frame, court mask, single-still detection viz, JSON of corners).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from .detector import CourtDetectionResult, detect_court_in_video
from .visualize import annotate_video, save_debug_artifacts


# Default output location: backend/court_mapping/output/
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="court_mapping",
        description="Detect badminton court corners and produce a video with the court grid drawn on every frame.",
    )
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="Path to input video (mp4 / mov / avi).")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Path to write the annotated MP4. "
                             "Default: backend/court_mapping/output/<stem>_marked.mp4.")
    parser.add_argument("--mode", choices=("doubles", "singles"), default="doubles",
                        help="Which court width to fit (default: doubles — outermost line).")
    parser.add_argument("--n-frames", type=int, default=60,
                        help="Number of frames to sample for the temporal median.")
    parser.add_argument("--no-refine", action="store_true",
                        help="Skip sub-pixel line refinement (faster, slightly less accurate).")
    parser.add_argument("--debug", action="store_true",
                        help="Save median frame, mask, detection still, and corners.json next to the output.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    input_path: Path = args.input
    if not input_path.exists():
        print(f"ERROR: input video not found: {input_path}", file=sys.stderr)
        return 2

    output_path: Path = args.output or (_DEFAULT_OUTPUT_DIR / f"{input_path.stem}_marked.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"▶  Detecting court corners in: {input_path}")
    print(f"   mode={args.mode}  n_frames={args.n_frames}  refine={not args.no_refine}")

    result: CourtDetectionResult = detect_court_in_video(
        video_path=input_path,
        mode=args.mode,
        n_frames=args.n_frames,
        refine=not args.no_refine,
        keep_artifacts=args.debug,
    )

    print()
    print("── Detection notes ──")
    for note in result.notes:
        print(f"   • {note}")
    print(f"── confidence: {result.confidence}  success: {result.success}")

    if result.refined_corners is not None:
        print("── corners (TL, TR, BR, BL) in pixels:")
        for label, pt in zip(("TL", "TR", "BR", "BL"), result.refined_corners):
            print(f"     {label}: ({pt[0]:8.2f}, {pt[1]:8.2f})")

    if not result.success or result.homography is None:
        print()
        print("⚠  Detection did not produce a usable homography.")
        if args.debug and result.median_frame is not None:
            saved = save_debug_artifacts(
                output_path.parent,
                median_frame=result.median_frame,
                mask=result.mask,
                rough_corners=result.rough_corners,
                refined_corners=result.refined_corners,
                projected_lines=result.projected_lines,
            )
            for p in saved:
                print(f"   debug artefact saved: {p}")
        return 1

    # ── Write annotated video ────────────────────────────────────────────
    print()
    print(f"▶  Writing annotated video to: {output_path}")
    n_frames_written = annotate_video(
        input_path=input_path,
        output_path=output_path,
        projected_lines=result.projected_lines,
        corners=result.refined_corners,
    )
    print(f"   wrote {n_frames_written} frames")

    # ── Optional debug artefacts ─────────────────────────────────────────
    if args.debug:
        saved = save_debug_artifacts(
            output_path.parent,
            median_frame=result.median_frame,
            mask=result.mask,
            rough_corners=result.rough_corners,
            refined_corners=result.refined_corners,
            projected_lines=result.projected_lines,
        )
        # Dump corners + homography to JSON alongside the output.
        meta_path = output_path.with_suffix(".corners.json")
        meta = {
            "input": str(input_path),
            "output": str(output_path),
            "mode": result.mode,
            "confidence": result.confidence,
            "success": result.success,
            "rough_corners_TLTRBRBL": _np_to_list(result.rough_corners),
            "refined_corners_TLTRBRBL": _np_to_list(result.refined_corners),
            "homography_H": _np_to_list(result.homography.H),
            "homography_H_inv": _np_to_list(result.homography.H_inv),
            "refinement_used": list(result.refinement_used),
            "notes": result.notes,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        saved.append(str(meta_path))
        print()
        print("── Debug artefacts ──")
        for p in saved:
            print(f"   • {p}")

    return 0


def _np_to_list(arr) -> Optional[list]:
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return list(arr)


if __name__ == "__main__":
    raise SystemExit(main())
