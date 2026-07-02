# Backend

FastAPI service that takes a badminton match video (uploaded or streamed live
from a phone camera) and returns an automated score, rally history, and an
annotated output video. Detection and tracking run entirely on geometry and
computer vision, no LLM is involved in scoring.

## Pipeline

1. **Court detection** (`court_mapping/`) — auto-detects the four court
   corners from a static-camera video and fits a homography to the canonical
   court. See `court_mapping/README.md` for details.
2. **Shuttle tracking** (`model2/`) — `ShuttleTracker` (from
   `model2/train_and_track.py`) fuses YOLO, YOLO-OBB, and TrackNetV2 to
   locate the shuttle in each frame.
3. **Player tracking** (`llm/player_tracker.py`) — lightweight YOLO-based
   player detection, run every N frames.
4. **Scoring** (`llm/scoring_engine.py`) — `ScoringEngine` drives an
   event-based state machine (`SERVE_PENDING` → `RALLY_ACTIVE` → point) off
   the shuttle trajectory alone: launch speed marks serves, sharp direction
   changes mark hits, and a shuttle coming to rest or vanishing near the
   floor ends the rally. Landing position is judged IN/OUT via homography
   against the court lines.
5. **Session orchestration** (`session.py`) — wraps one `ScoringEngine` per
   job/session so the court is calibrated once and score/trajectory state
   carries across video chunks.
6. **API** (`main.py`) — FastAPI app exposing upload and live-streaming
   endpoints (see below).

## Setup

```bash
cd backend
python -m venv ubenv
./ubenv/bin/pip install -r requirements.txt

# PyTorch with CUDA is not pinned in requirements.txt — install separately:
./ubenv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Model weights (`model2/models/`, `yolo*.pt`, `*.pth`) and datasets
(`model2/dataset/`) are gitignored and expected to exist locally; they are
not part of this repo.

## Running the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

`DETECT_STRIDE` (env var, default `2`) controls how often shuttle inference
runs, every Nth frame, to trade accuracy for speed.

Only one video (upload or live) is processed at a time; a second request
while one is active gets `409 Conflict`. This is a VRAM constraint, not a
design choice, `main.py` holds a single `ScoringEngine` per active session.

### Endpoints

**Recorded upload:**

| Endpoint | Purpose |
|---|---|
| `POST /upload-video` | Upload a video file (`mode=singles\|doubles`); returns a `job_id`. |
| `POST /process-video/{job_id}` | Start processing the uploaded video. |
| `GET /job-status/{job_id}` | Poll status, frame count, score, progress %. |
| `GET /results/{job_id}` | Full results once `status == "complete"`. |
| `GET /download/{job_id}` | Download the annotated result video. |

**Live (chunked, phone camera):**

| Endpoint | Purpose |
|---|---|
| `POST /live/start` | Start a live session (`mode`, `annotate`, `detect_stride`); returns `session_id`. |
| `POST /live/chunk/{session_id}?seq=N` | Upload the next video chunk in sequence order. |
| `POST /live/finish/{session_id}` | Signal no more chunks are coming. |
| `GET /live/status/{session_id}` | Poll current state. |
| `WS /live/ws/{session_id}` | Subscribe to state updates as they happen. |

## Standalone scripts (no API)

```bash
# Run the scoring engine over a single video, writes an annotated MP4
./ubenv/bin/python llm/scoring_main.py

# Same, plus shuttle trail overlay (used for combined tracking/scoring demos)
./ubenv/bin/python llm/integrated_main.py
```

Both scripts hardcode `VIDEO_PATH`/`OUTPUT_PATH` at the top of the file,
edit those before running.

## Court corner detection (standalone)

```bash
python -m court_mapping.run --input path/to/match.mp4
```

See `court_mapping/README.md` for flags and library usage.

## Training / re-tracking (`model2/`)

```bash
# Annotate raw frames
python model2/data_preparation.py --action annotate \
  --images dataset/raw_frames/match1/ --output dataset/annotations/match1

# Split annotated data into train/val/test
python model2/data_preparation.py --action split \
  --images dataset/raw_frames/match1 --annotations dataset/annotations/match1/ \
  --output dataset/processed/match1/ --method rally

# Train a YOLO-OBB shuttle detector
python model2/train_and_track.py --action train-obb \
  --split-dir dataset/processed/match1/ --output-dir yolo-obb \
  --yolo-version 8 --epochs 20 --batch 8 --device 0

# Run hybrid (YOLO + OBB + TrackNet) tracking over a video
python model2/train_and_track.py --action track \
  --video dataset/videos/Sample_3.mp4 --output-video Sample_3_tracked.mp4 \
  --yolo-weights runs/detect/yolo-runs/yolo_standard/weights/best.pt \
  --obb-weights runs/obb/train-track/yolo_obb/weights/best.pt \
  --tracknet-weights runs/train-track/tracknet_best.pth \
  --mode hybrid --device cuda
```

`--action` also accepts `train-yolo`, `train-tracknet`, and `save-config`.
Re-encode a tracked output for broad player compatibility with:

```bash
ffmpeg -i Sample_3_tracked.mp4 -vcodec libx264 -pix_fmt yuv420p -crf 23 Tracked.mp4
```

## Tests

```bash
./ubenv/bin/python -m pytest llm/test_scoring_corrections.py
```

Covers rally-end detection (high lobs vs. genuinely descending shuttles) and
the verdict-correction API (`ScoringEngine.correct_last_verdict`).
