"""
Quick court-tracking sanity check.
Detects the court from a short median sample, then renders the first
MAX_FRAMES frames with the court overlay and saves a short clip.

Usage:
    python scripts/check_court.py <video_path> [--frames 1000] [--mode singles|doubles] [--debug]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from court_mapping.detector import detect_court_in_video
from court_mapping.visualize import draw_court_overlay


def main() -> int:
    parser = argparse.ArgumentParser(description="Court overlay sanity check")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--frames", type=int, default=1000, help="Max frames to render (default 1000)")
    parser.add_argument("--mode", choices=("singles", "doubles"), default="doubles")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output video path (default: <video_stem>_court_check.mp4 next to input)")
    parser.add_argument("--debug", action="store_true",
                        help="Save median frame, HSV mask, and rough corners alongside the output")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 2

    video_dir = args.video.parent
    stem = args.video.stem
    output: Path = args.output if args.output is not None else video_dir / f"{stem}_court_check.mp4"
    snapshot_path: Path = output.with_name(output.stem + "_snapshot.png")
    debug_dir: Path = output.parent / f"{stem}_court_debug"

    print(f"Detecting court (mode={args.mode}) ...")
    result = detect_court_in_video(
        video_path=args.video,
        mode=args.mode,
        n_frames=40,
        refine=True,
        keep_artifacts=args.debug,
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

    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        if result.median_frame is not None:
            p = debug_dir / "median_frame.png"
            cv2.imwrite(str(p), result.median_frame)
            print(f"  debug: median_frame  → {p}")
        if result.mask is not None:
            p = debug_dir / "court_mask.png"
            cv2.imwrite(str(p), result.mask)
            print(f"  debug: court_mask    → {p}")
        if result.rough_corners is not None and result.median_frame is not None:
            rough_vis = result.median_frame.copy()
            for i, pt in enumerate(result.rough_corners):
                cv2.drawMarker(rough_vis, (int(pt[0]), int(pt[1])), (255, 255, 255),
                               cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
                cv2.putText(rough_vis, ("TL", "TR", "BR", "BL")[i],
                            (int(pt[0]) + 8, int(pt[1]) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            p = debug_dir / "rough_corners.png"
            cv2.imwrite(str(p), rough_vis)
            print(f"  debug: rough_corners → {p}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"ERROR: could not open video for rendering: {args.video}", file=sys.stderr)
        return 2

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"ERROR: could not open video writer at {output}", file=sys.stderr)
        return 2

    snapshot_target = max(0, args.frames // 2)
    snapshot_saved = False
    last_annotated = None

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
        writer.write(annotated)
        last_annotated = annotated
        if not snapshot_saved and count >= snapshot_target:
            cv2.imwrite(str(snapshot_path), annotated)
            snapshot_saved = True
        count += 1
        if count % 200 == 0:
            print(f"  {count}/{args.frames} frames written ...")

    if not snapshot_saved and last_annotated is not None:
        cv2.imwrite(str(snapshot_path), last_annotated)
        snapshot_saved = True

    cap.release()
    writer.release()

    print(f"Done. Wrote {count} frames to {output}")
    if snapshot_saved:
        print(f"Snapshot (frame ~{snapshot_target}) → {snapshot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
