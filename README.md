# 🏸 Badminton Video Scoring App

# Video Preview

<video src="https://github.com/user-attachments/assets/9782b5b1-93be-4ccd-8e7e-540d336c683b" width="100%" controls></video>

An **Expo + React Native** app paired with a **FastAPI** backend that takes a badminton match video, whether uploaded from a phone or streamed live from the camera, and returns an automated score, rally history, and an annotated output video. Detection and scoring run entirely on computer vision and geometry; no LLM is involved in the scoring path.

---

## 🚀 Vision

Badminton scoring is fast-paced and difficult to track manually. This app:

- Accepts **video input**, either a recorded upload or a live phone camera feed
- Auto-detects the **court** from a static-camera video and calibrates a homography to it
- Tracks the **shuttle** frame by frame and detects hits, rallies, and landings
- Judges **IN/OUT** verdicts geometrically and drives a real scoring state machine
- Returns the **match score, rally history, and an annotated video** back to the app

---

## 🧱 Tech Stack

### Frontend (`frontend/myApp`)

- **Expo (SDK 54)**, **React Native**, **Expo Router**, **TypeScript**
- Video capture and gallery picking via `expo-camera` / `expo-image-picker`
- Playback via `expo-av`
- Two flows: upload-and-analyze (`app/index.tsx`) and live scoring (`app/live.tsx`)

### Backend (`backend`)

- **FastAPI** serving upload and live-streaming endpoints
- **OpenCV** for frame processing
- **YOLOv8/YOLOv11 (standard + OBB)** and **TrackNetV2** fused for shuttle tracking
- **YOLO** for lightweight player tracking
- A hand-written **scoring engine** (state machine, no ML) that turns shuttle trajectory into hits, rallies, and points

---

## 📁 Project Structure

```text
frontend/myApp/
 ├── app/
 │   ├── index.tsx        # Upload/record a video, poll job status, view results
 │   ├── live.tsx          # Live camera capture, chunked upload, WebSocket score updates
 │   └── _layout.tsx
 ├── components/
 │   └── ShuttleBackground.tsx   # Animated interactive background
 ├── lib/
 │   ├── config.ts         # API base URL
 │   ├── theme.ts
 │   └── videoForm.ts      # Cross-platform multipart upload helper (web vs. native)
 └── assets/

backend/
 ├── main.py               # FastAPI app: upload, processing, live endpoints
 ├── session.py             # ChunkSession: per-session engine + live worker thread
 ├── court_mapping/         # Auto court-corner detection + homography (see its README)
 ├── model2/                # TrackNetV2 + YOLO training/tracking pipeline
 └── llm/
     ├── scoring_engine.py   # Core scoring state machine
     ├── player_tracker.py
     ├── scoring_main.py     # Standalone scoring script (no API)
     └── integrated_main.py  # Standalone scoring + shuttle trail overlay demo
```

---

## 🛠️ Installation & Setup

### Frontend

```bash
cd frontend/myApp
npm install
npm start          # then press a / i / w, or scan the QR code
```

### Backend

```bash
cd backend
python -m venv ubenv
./ubenv/bin/pip install -r requirements.txt

# PyTorch with CUDA isn't pinned in requirements.txt, install separately:
./ubenv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

Model weights (`model2/models/`, `yolo*.pt`, `*.pth`) and datasets are gitignored and expected to exist locally. See `backend/README.md` for the full setup, endpoint reference, and training/tracking CLI.

---

## 📱 How the App Works

**Upload flow:**

1. User records or picks a video from the gallery.
2. The app uploads it to `POST /upload-video`, then triggers `POST /process-video/{job_id}`.
3. The backend calibrates the court once, then tracks the shuttle and players frame by frame, scoring as it goes.
4. The app polls `GET /job-status/{job_id}` for progress, then fetches `GET /results/{job_id}` and the annotated video from `GET /download/{job_id}`.

**Live flow:**

1. `POST /live/start` opens a session and returns a `session_id`.
2. The phone records short chunks and uploads each via `POST /live/chunk/{session_id}`.
3. The backend processes chunks in sequence on a worker thread, carrying score and trajectory state across chunks.
4. The app subscribes to `WS /live/ws/{session_id}` (or polls `GET /live/status/{session_id}`) for live score updates.
5. `POST /live/finish/{session_id}` signals the end of the stream.

Only one video (upload or live) is processed at a time; a second request while one is active gets `409 Conflict`, since a single GPU backs the whole service.

---

## 🧪 Current Status

### ✅ Done

- Court auto-detection from a static-camera video, fitted homography to the canonical court (`court_mapping/`)
- Hybrid shuttle tracking (YOLO + YOLO-OBB + TrackNetV2), batched over 32-frame windows for throughput (12 → 35 fps)
- Player tracking (lightweight YOLO, sampled every N frames)
- Scoring engine: serve detection, hit detection, rally-end detection, IN/OUT verdicts, 21-point rules with deuce and game transitions
- Side switch between games, cross-game verdict corrections, and a race-safe active-session count (previously buggy, now fixed)
- Verdict-correction logic in the scoring engine (`correct_last_verdict`), covered by tests, though not yet wired to a REST endpoint
- FastAPI backend: upload/process/poll/download flow, plus a chunked live-streaming flow with WebSocket updates
- Frontend: recording, gallery picking, upload progress, results playback, live scoring screen
- Cross-platform multipart upload helper (`lib/videoForm.ts`) fixing a web/native `FormData` mismatch
- H.264 transcode step on session finalize, so result videos actually play back in the browser
- Test suites for the scoring engine and the FastAPI app (see `backend/README.md` → Tests, and `E2E_TEST_REPORT.md`)

### ⚠️ Known issues (see `E2E_TEST_REPORT.md` for detail)

- Hit detection has no floor/height gate, so a landing impact can be misread as a racket hit, occasionally misattributing OUT points
- Live throughput still trails real time on CPU-bound setups; a dropped chunk can stall live scoring until `finish` is called
- Service-court rules (left/right box by score parity, faults) aren't modeled
- `mode` (singles/doubles) isn't sent from the upload screen, so scoring always runs in singles mode

### 🚧 Pending

- Shot classification / shot-type breakdown
- Player performance analytics, heatmaps
- Doubles-aware scoring end to end (the engine supports it; the frontend doesn't send it yet)

---

## 🌱 Future Enhancements

- Real-time scoring at full camera frame rate
- Match replay with shot/rally overlays
- Player comparison dashboards
- Coach & training mode

---

## 🤝 Contributions

Contributions, ideas, and feedback are welcome!

- Fork the repo
- Create a feature branch
- Submit a pull request

---

## 📜 License

This project is currently under development and not licensed for commercial use.

---

## 👤 Author

**Chirag Chandrashekar**
AI / ML Engineer | Sports Analytics Enthusiast

---

If you love **sports + computer vision + AI**, this project is for you 🏸🤖
