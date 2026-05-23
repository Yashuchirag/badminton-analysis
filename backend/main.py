import sys
import os
from pathlib import Path

# Allow scoring_engine's internal imports (landing_detector, gemma_client, etc.)
sys.path.insert(0, str(Path(__file__).parent / "llm"))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import cv2
import uuid
import asyncio
import torch

from llm.scoring_engine import ScoringEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

YOLO_WEIGHTS     = "model2/runs/detect/yolo-runs/yolo_standard/weights/best.pt"
OBB_WEIGHTS      = "model2/runs/obb/yolo-obb/yolo_obb/weights/best.pt"
TRACKNET_WEIGHTS = "model2/runs/train-track/tracknet_best.pth"

jobs: Dict[str, Any] = {}
_executor = ThreadPoolExecutor(max_workers=2)


def _run_pipeline(job_id: str, video_path: str, mode: str) -> None:
    try:
        jobs[job_id]["status"] = "calibrating"

        engine = ScoringEngine(
            yolo_weights=YOLO_WEIGHTS,
            obb_weights=OBB_WEIGHTS,
            tracknet_weights=TRACKNET_WEIGHTS,
            mode=mode,
            tracking_mode="hybrid",
            conf_threshold=0.25,
            device=DEVICE,
        )

        engine.auto_calibrate(video_path, n_frames=60)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video file")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = str(OUTPUT_DIR / f"{job_id}_result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError("Cannot create output video writer")

        jobs[job_id].update({
            "status": "processing",
            "total_frames": total_frames,
            "frame": 0,
            "progress_percent": 0.0,
            "score": [0, 0],
        })

        landing_events = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            event = engine.process_frame(frame, frame_idx)
            if event is not None:
                landing_events.append({
                    "frame": event.frame_idx,
                    "verdict": event.verdict.value,
                    "court_x": round(event.court_x, 3),
                    "court_y": round(event.court_y, 3),
                    "confidence": round(event.confidence, 3),
                    "source": event.source,
                    "reason": event.reason,
                })

            frame = engine.draw_court_boundaries(frame)
            out.write(frame)

            frame_idx += 1
            jobs[job_id]["frame"] = frame_idx
            jobs[job_id]["progress_percent"] = round(frame_idx / total_frames * 100, 1) if total_frames else 0
            jobs[job_id]["score"] = list(engine.state.score)
            jobs[job_id]["rally_state"] = engine.state.rally_state
            jobs[job_id]["last_hitter_side"] = engine.state.last_hitter_side

        cap.release()
        out.release()
        os.remove(video_path)

        jobs[job_id].update({
            "status": "complete",
            "output_video": output_path,
            "score": list(engine.state.score),
            "game": engine.state.game,
            "landing_events": landing_events,
            "total_landings": len(landing_events),
        })

    except Exception as exc:
        print(f"[pipeline error] job={job_id}: {exc}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(exc)


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...), mode: str = "singles"):
    """Step 1: Upload the video file. Returns a job_id to track processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    job_id = str(uuid.uuid4())
    video_path = str(OUTPUT_DIR / f"{job_id}_input.mp4")

    contents = await file.read()
    with open(video_path, "wb") as f:
        f.write(contents)

    jobs[job_id] = {
        "status": "uploaded",
        "video_path": video_path,
        "mode": mode,
        "frame": 0,
        "total_frames": 0,
        "progress_percent": 0.0,
        "score": [0, 0],
    }

    return {"job_id": job_id, "status": "uploaded"}


@app.post("/process-video/{job_id}")
async def process_video(job_id: str):
    """Step 2: Start the analysis pipeline for an uploaded video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] not in ("uploaded", "error"):
        raise HTTPException(
            status_code=400,
            detail=f"Job is already '{jobs[job_id]['status']}' — cannot start processing",
        )

    video_path = jobs[job_id]["video_path"]
    mode       = jobs[job_id].get("mode", "singles")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline, job_id, video_path, mode)

    return {"job_id": job_id, "status": "processing"}


@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    """Poll for live progress: status, frame count, score, progress %."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Fetch the full results once processing is complete."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if jobs[job_id]["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete (current status: {jobs[job_id]['status']})",
        )
    return jobs[job_id]


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download the annotated output video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    path = jobs[job_id].get("output_video")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Output video not ready")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"badminton_{job_id[:8]}_result.mp4",
    )
