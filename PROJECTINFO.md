# PROJECTINFO — Badminton Video Scoring App

**Author**: Chirag Chandrashekar (AI/ML Engineer)
**Last Updated**: 2026-05-17

---

## Final Objective

Upload a badminton match video → fully automated scoring and analytics:
- Detect shuttle trajectory and landing position
- Detect and track players
- Map the court geometry
- Segment rallies automatically
- Determine IN/OUT for each shot
- Compute match score following standard badminton rules (21-point)
- Return results (score, rally stats, player heatmaps) to the mobile app

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Expo SDK 54, React Native, TypeScript, Expo Router |
| Backend | FastAPI (Python), OpenCV |
| Shuttle Tracking | TrackNetV2 (custom U-Net) + YOLO-OBB |
| Player Detection | YOLOv8 standard |
| Court Mapping | Custom line-detection pipeline (OpenCV) |
| Scoring Engine | Python rules engine + Gemma4 LLM (for near-line calls) |
| GPU | NVIDIA CUDA (WSL Ubuntu, `ubenv` virtualenv) |

---

## Project Structure

```
game-tracker/
├── frontend/myApp/          # Expo React Native app
│   └── app/index.tsx        # Main screen (473 lines)
├── backend/
│   ├── main.py              # FastAPI entry point (169 lines)
│   ├── model2/              # Core CV/ML — shuttle + player tracking
│   │   ├── train_and_track.py      # ShuttleTracker class (1022 lines) — MAIN FILE
│   │   ├── TrackNetV2.py           # Custom U-Net architecture
│   │   ├── court_tracker.py        # Court tracking (inside model2)
│   │   ├── train_TrackNetV2.py     # TrackNet training script
│   │   ├── train_tracknet.py       # Alternate training script
│   │   ├── data_preparation.py
│   │   ├── data_splitter.py
│   │   ├── overlay_and_verify.py   # Verification tooling
│   │   └── verify_obb.py
│   ├── court_mapping/       # Court boundary detection — WORKING
│   │   ├── run.py
│   │   ├── detector.py
│   │   ├── geometry.py
│   │   ├── lines.py
│   │   ├── segment.py
│   │   ├── temporal.py
│   │   └── visualize.py
│   └── llm/                 # Scoring engine + LLM line judge — FAULTY
│       ├── scoring_engine.py       # ScoringEngine class (363 lines)
│       ├── scoring_main.py         # Entry point for scoring pipeline
│       ├── court_geometry.py       # CourtHomography, ZONES, court dimensions
│       ├── landing_detector.py     # LandingDetector, RawDetection
│       ├── gemma_client.py         # Gemma4 LLM wrapper for line calls
│       ├── event_detector.py       # Rally/event detection
│       └── rally_buffer.py         # Rally buffering logic
```

---

## Trained Model Weights (already trained, stored locally)

| Model | Path | Purpose |
|---|---|---|
| YOLO standard | `model2/runs/detect/yolo-runs/yolo_standard/weights/best.pt` | Player detection |
| YOLO OBB | `model2/runs/obb/yolo-obb/yolo_obb/weights/best.pt` | Shuttle detection (oriented bbox) |
| TrackNetV2 | `model2/runs/train-track/tracknet_best.pth` | Shuttle trajectory tracking |

---

## Component Status

### Frontend (`frontend/myApp/app/index.tsx`)
- **Status**: Working
- Lets user pick video from gallery or record directly with camera
- Uploads video as `multipart/form-data` to `POST /track-human-video-async`
- Polls `GET /job-status/{jobId}` every 500ms for live progress
- Shows: frame count, % complete, live preview image, people count, shuttle detection status
- Backend URL comes from `EXPO_PUBLIC_API_URL` (default `http://localhost:8001`, see `lib/config.ts`); port 8000 is reserved for the local LLM server
- **Planned change**: Instead of uploading raw video directly to the endpoint, upload to a database/storage bucket first; backend then reads from storage. This is more efficient for long videos.

### Backend API (`backend/main.py`)
- **Status**: Rewritten (2026-05-17) — clean two-step pipeline
- **Flow**: Upload video → get job_id → trigger processing → poll for progress → fetch results/download
- **Endpoints**:
  - `POST /upload-video?mode=singles|doubles` — saves video file, returns `{ job_id }`
  - `POST /process-video/{job_id}` — starts background pipeline (calibrate → track → score)
  - `GET /job-status/{job_id}` — live progress: status, frame, progress_percent, score
  - `GET /results/{job_id}` — full results once complete (score, game, landing_events)
  - `GET /download/{job_id}` — download annotated output MP4
- **Job state machine**: `uploaded → calibrating → processing → complete | error`
- **Processing**: `ThreadPoolExecutor` (non-blocking), `ScoringEngine` drives the full pipeline per-frame
- **Output per job**: `outputs/<job_id>_input.mp4` (deleted after processing), `outputs/<job_id>_result.mp4`

### ShuttleTracker (`backend/model2/train_and_track.py`) — MAIN TRACKING FILE
- **Status**: Working as standalone
- `ShuttleTracker` class uses producer-consumer threading: `_reader()` thread decodes frames into a Queue, `_writer()` thread consumes processed frames and writes output
- **Hybrid detection mode** fuses three sources per frame:
  - YOLO standard (inference at 416px)
  - YOLO-OBB (oriented bounding box for shuttle)
  - TrackNetV2 heatmap (takes 3 consecutive frames × 3 RGB = 9-channel input, processed at 256px)
- `YOLOTrainer` class handles training for both standard and OBB models with auto-generated YAML configs
- **Not yet connected** to the full API pipeline

### TrackNetV2 (`backend/model2/TrackNetV2.py`)
- **Status**: Trained, working
- Custom U-Net-style architecture (not a library model)
- Encoder (3 stages) + bottleneck + decoder with skip connections
- Input: 9 channels (3 frames × 3 RGB concatenated), Output: single-channel heatmap of shuttle position

### Court Mapping (`backend/court_mapping/`)
- **Status**: Working
- Detects court boundary corners from video frames using line detection + temporal smoothing
- Outputs: corner coordinates as JSON + annotated MP4
- Has been tested and produces correct results

### LLM Scoring Engine (`backend/llm/`)
- **Status**: Faulty / incomplete
- `ScoringEngine` class wraps ShuttleTracker + CourtHomography + LandingDetector + optional Gemma4 LLM
- Flow per frame: shuttle position → LandingDetector → 15-frame confirmation window (to distinguish floor landing from racket hit) → IN/OUT verdict → score update
- Geometric IN/OUT using homography-projected court coordinates; Gemma4 invoked only for near-line calls
- `MatchState` tracks score `[0, 0]`, serving side, rally state, game number
- Supports standard 21-point badminton rules with serve change and deuce logic
- `CourtHomography` in `court_geometry.py` handles pixel ↔ court-coordinate transforms
- **Issues**: Overall pipeline is faulty; needs debugging before it can be integrated

---

## Integration Status (What Is and Isn't Connected)

| Connection | Status |
|---|---|
| Frontend → Backend upload | Works (but sends raw video; planned to change to DB-first) |
| Backend → ShuttleTracker | Partially wired (has bugs in main.py) |
| ShuttleTracker → CourtHomography | Exists in scoring_engine.py but not in main pipeline |
| CourtMapping → ScoringEngine | Connected (fixed 2026-05-17) |
| ScoringEngine → Frontend (results) | Not connected |
| Full end-to-end pipeline | Not working |

The deliberate decision was made to **not connect everything until each module works standalone**, because debugging a fully-connected pipeline processing long videos takes too long per iteration.

---

## Immediate Next Work

1. **Video upload flow change**: Implement DB/storage upload on the frontend side; backend reads from storage instead of receiving raw bytes. Better for long videos (no in-memory buffering, resumable, async-friendly).
2. **Fix `main.py` bugs**: Initialize `jobs` dict, fix async execution, fix output path, fix response format.
3. **Connect court_mapping output → scoring_engine**: Pass detected court corners into `CourtHomography.calibrate()`.
4. **Debug scoring engine**: Identify and fix faults in the LLM pipeline.
5. **Full pipeline integration**: Video → ShuttleTracker → CourtMapping → ScoringEngine → structured results → Frontend display.

---

## Environment Notes

- **OS**: WSL2 (Ubuntu) on Windows
- **Python env**: `ubenv` virtualenv with all project dependencies installed
- **GPU**: NVIDIA CUDA
  - ⚠️ Current status: Driver too old (v12090), falling back to CPU
  - To use GPU: Update NVIDIA driver to compatible version (recommended)
  - PyTorch expects newer CUDA driver for optimal performance
- **Backend run**: Must be started inside WSL from `backend/` directory using `uvicorn main:app`
- **Frontend run**: `cd frontend/myApp && npx expo start`
- **Network**: Set `EXPO_PUBLIC_API_URL=http://<LAN-IP>:8001` when testing on a physical phone; the default is `http://localhost:8001`
