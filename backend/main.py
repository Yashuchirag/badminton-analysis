import sys
import os
from pathlib import Path

# Allow scoring_engine's internal imports (landing_detector, player_tracker, etc.)
sys.path.insert(0, str(Path(__file__).parent / "llm"))

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import cv2
import uuid
import asyncio
import torch

import session as sessions

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

YOLO_WEIGHTS     = None
OBB_WEIGHTS      = "model2/models/claudette_trained/yolo_obb_finetune/weights/best.pt"
TRACKNET_WEIGHTS = "model2/models/claudette_trained/tracknetv2/tracknetv2_best.pth"

DETECT_STRIDE = int(os.environ.get("DETECT_STRIDE", "2"))

# Videos uploaded but not yet processing: job_id -> {video_path, mode}
uploads: Dict[str, Any] = {}


def _engine_kwargs(mode: str, detect_stride: int) -> dict:
    return {
        "yolo_weights": YOLO_WEIGHTS,
        "obb_weights": OBB_WEIGHTS,
        "tracknet_weights": TRACKNET_WEIGHTS,
        "mode": mode,
        "tracking_mode": "hybrid",
        "conf_threshold": 0.25,
        "device": DEVICE,
        "detect_stride": detect_stride,
    }


# ── Recorded video upload (single-chunk session) ─────────────────────────────

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

    uploads[job_id] = {"video_path": video_path, "mode": mode}
    return {"job_id": job_id, "status": "uploaded"}


@app.post("/process-video/{job_id}")
async def process_video(job_id: str):
    """Step 2: Start the analysis pipeline for an uploaded video."""
    if job_id not in uploads:
        raise HTTPException(status_code=404, detail="Job not found")
    if sessions.get_session(job_id) is not None:
        raise HTTPException(status_code=400, detail="Job is already processing")
    if sessions.active_sessions() >= 1:
        raise HTTPException(
            status_code=409,
            detail="Another session is processing; try again when it finishes",
        )

    info = uploads.pop(job_id)
    video_path = info["video_path"]

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    s = sessions.create_session(
        job_id,
        engine_kwargs=_engine_kwargs(info["mode"], DETECT_STRIDE),
        output_dir=OUTPUT_DIR,
        annotate=True,
        live=False,
        loop=asyncio.get_running_loop(),
    )
    s.expected_total_frames = total_frames
    s.add_chunk(0, video_path)
    s.finish()

    return {"job_id": job_id, "status": "processing"}


@app.get("/job-status/{job_id}")
async def job_status(job_id: str):
    """Poll for live progress: status, frame count, score, progress %."""
    if job_id in uploads:
        return {"status": "uploaded", "frame": 0, "total_frames": 0,
                "progress_percent": 0.0, "score": [0, 0]}
    s = sessions.get_session(job_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return s.latest_state


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Fetch the full results once processing is complete."""
    s = sessions.get_session(job_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if s.latest_state["status"] != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete (current status: {s.latest_state['status']})",
        )
    return s.latest_state


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download the annotated output video."""
    s = sessions.get_session(job_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Job not found")
    path = s.output_path
    if not path or not os.path.exists(path) or s.latest_state["status"] != "complete":
        raise HTTPException(status_code=404, detail="Output video not ready")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"badminton_{job_id[:8]}_result.mp4",
    )


# ── Live mode (chunked session fed by the phone camera) ──────────────────────

class LiveStartRequest(BaseModel):
    mode: str = "singles"
    annotate: bool = False
    detect_stride: int = DETECT_STRIDE


@app.post("/live/start")
async def live_start(req: LiveStartRequest):
    if sessions.active_live_sessions() >= 1:
        raise HTTPException(status_code=409, detail="A live session is already active")
    if sessions.active_sessions() >= 1:
        raise HTTPException(
            status_code=409,
            detail="An upload is processing; try again when it finishes",
        )

    session_id = str(uuid.uuid4())
    (OUTPUT_DIR / session_id).mkdir(exist_ok=True)

    sessions.create_session(
        session_id,
        engine_kwargs=_engine_kwargs(req.mode, max(1, req.detect_stride)),
        output_dir=OUTPUT_DIR,
        annotate=req.annotate,
        live=True,
        loop=asyncio.get_running_loop(),
    )
    return {"session_id": session_id}


@app.post("/live/chunk/{session_id}")
async def live_chunk(session_id: str, seq: int, file: UploadFile = File(...)):
    s = sessions.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if s.finished:
        raise HTTPException(status_code=400, detail="Session already finished")

    chunk_path = str(OUTPUT_DIR / session_id / f"chunk_{seq:05d}.mp4")
    contents = await file.read()
    with open(chunk_path, "wb") as f:
        f.write(contents)

    backlog = s.add_chunk(seq, chunk_path)
    return {"queued": True, "backlog": backlog}


@app.post("/live/finish/{session_id}")
async def live_finish(session_id: str):
    s = sessions.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    s.finish()
    return s.latest_state


@app.get("/live/status/{session_id}")
async def live_status(session_id: str):
    s = sessions.get_session(session_id)
    if s is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return s.latest_state


@app.websocket("/live/ws/{session_id}")
async def live_ws(websocket: WebSocket, session_id: str):
    s = sessions.get_session(session_id)
    if s is None:
        await websocket.close(code=4004)
        return
    await websocket.accept()
    q = s.subscribe()
    try:
        await websocket.send_json(s.latest_state)
        while True:
            state = await q.get()
            await websocket.send_json(state)
            if state["status"] in ("complete", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        s.unsubscribe(q)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "4000")))
