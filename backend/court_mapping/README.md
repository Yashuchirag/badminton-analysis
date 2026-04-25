# court_mapping

Automatic badminton court corner detection for static-camera videos.

## What it does

Given an input video of a badminton match (broadcast or amateur, fixed camera),
this module:

1. Computes a temporal-median reference frame, which removes players, the
   shuttle, and line judges by averaging across many frames.
2. Segments the court surface from that clean frame using adaptive HSV.
3. Approximates the court boundary as a 4-point quadrilateral.
4. Refines each corner sub-pixel by fitting lines to the painted court boundary
   along corridors aligned with the rough sides (LSD line detector + RANSAC).
5. Validates the quadrilateral and fits a homography to the canonical
   13.40 m × 6.10 m doubles court (or 5.18 m singles court).
6. Projects all canonical court lines back into image space and overlays them
   on every frame of the input video, writing an annotated MP4.

Because the camera is static, the court grid is computed once and re-used for
every frame.

## Why this approach

Single-frame HSV / Hough detection is brittle on mixed-source videos because
players occlude lines, lighting varies across the frame, and broadcast overlays
add false signals. Temporal median sidesteps every one of those issues — once
the players are gone, even a simple HSV mask becomes reliable.

## Usage

### CLI

```bash
cd backend
python -m court_mapping.run --input path/to/match.mp4
```

Output goes to `backend/court_mapping/output/match_marked.mp4` by default.

Useful flags:

| flag                 | purpose                                                           |
|----------------------|-------------------------------------------------------------------|
| `--output PATH`      | Write the annotated video to a specific path.                     |
| `--mode singles`     | Fit the singles court (5.18 m wide) instead of doubles.           |
| `--n-frames 90`      | Sample more frames for the median (slower, more robust).          |
| `--no-refine`        | Skip the LSD line-refinement step.                                |
| `--debug`            | Save median frame, court mask, detection still, and corners JSON. |

### Library

```python
from court_mapping import detect_court_in_video, annotate_video

result = detect_court_in_video("match.mp4", mode="doubles")

if result.success:
    annotate_video(
        input_path="match.mp4",
        output_path="match_marked.mp4",
        projected_lines=result.projected_lines,
        corners=result.refined_corners,
    )
```

`result` is a `CourtDetectionResult` exposing:

- `refined_corners` — `(4, 2)` ndarray, ordered TL/TR/BR/BL
- `homography` — `CourtHomography` (reused from `llm.court_geometry`)
- `projected_lines` — list of `(CourtLine, ((px1, py1), (px2, py2)))` ready for drawing
- `confidence` — `"high" | "medium" | "low" | "failed"`
- `notes` — human-readable diagnostic strings

## File layout

```
backend/court_mapping/
├── __init__.py        public API exports
├── temporal.py        median frame extraction
├── segment.py         court-surface HSV segmentation
├── geometry.py        corner ordering / mask→quad / canonical lines / homography
├── lines.py           LSD + RANSAC sub-pixel corner refinement
├── visualize.py       drawing + MP4 writer + debug artefacts
├── detector.py        end-to-end orchestrator
├── run.py             CLI (`python -m court_mapping.run ...`)
└── output/            default output directory for annotated videos
```

## Reuses from existing code (read-only)

- `llm.court_geometry.CourtHomography` — homography class with `calibrate()` /
  `to_court()` is used directly; we don't redefine it.
- `llm.court_geometry.COURT` — canonical court dimensions.

`backend/model2/` is **not** modified or imported.

## Future extensions

- Pluggable segmentation strategies: SAM2 / SAM as a drop-in replacement for
  `segment_court_hsv` (zero training, much more robust on unusual courts at
  the cost of a ~400 MB model dependency).
- Save calibration JSON keyed by video ID and reuse it on subsequent runs.
- Frontend page to manually adjust auto-detected corners when needed.
