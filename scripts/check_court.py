"""
Quick court-tracking sanity check.
Detects the court from a short median sample, then renders the first
MAX_FRAMES frames with the court overlay and saves a short clip.

Usage:
    python scripts/check_court.py <video_path> [--frames 1000] [--mode singles|doubles]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Allow imports from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from court_mapping.detector import detect_court_in_video
from court_mapping.visualize import draw_court_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Court overlay sanity check")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--frames", type=int, default=1000, help="Max frames to render (default 1000)")
    parser.add_argument("--mode", choices=("singles", "doubles"), default="singles")
    parser.add_argument("--output", type=Path, default=None, help="Output path (default: court_check.mp4 next to input)")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 2

    output = "court_check.mp4"

    print(f"Detecting court (mode={args.mode}) ...")
    result = detect_court_in_video(
        video_path=args.video,
        mode=args.mode,
        n_frames=40,
        refine=True,
        keep_artifacts=False,
    )

    print(f"  confidence : {result.confidence}")
    print(f"  success    : {result.success}")
    for note in result.notes:
        print(f"  • {note}")

    if not result.success or result.homography is None:
        print("\nCourt detection failed — cannot render overlay.", file=sys.stderr)
        return 1

    if result.refined_corners is not None:
        print("  corners (TL TR BR BL):")
        for label, pt in zip(("TL", "TR", "BR", "BL"), result.refined_corners):
            print(f"    {label}: ({pt[0]:.1f}, {pt[1]:.1f})")

    cap = cv2.VideoCapture(str(args.video))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output), fourcc, fps, (width, height))

    print(f"\nRendering first {args.frames} frames → {output}")
    count = 0
    while count < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        annotated = draw_court_overlay(
            frame,
            result.projected_lines,
            corners=result.refined_corners,
            label_corners=(count == 0),
        )
        out.write(annotated)
        count += 1
        if count % 200 == 0:
            print(f"  {count}/{args.frames} frames written ...")

    cap.release()
    out.release()
    print(f"Done. Wrote {count} frames to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
