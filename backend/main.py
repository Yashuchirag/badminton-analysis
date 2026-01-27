from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from datetime import datetime
from pathlib import Path
from typing import Dict

from train.model_tracknet import TrackNet
from infer import shuttle_tracking_combined
from save_annotated_video_sync import process_video_sync

import cv2
import tempfile
import os
import torch
import json
import asyncio
import uuid
import asyncio

import base64

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")

tracknet = TrackNet().to(DEVICE)
tracknet.load_state_dict(torch.load("tracknet.pth", map_location=DEVICE))
tracknet.eval()
print("TrackNet model loaded successfully")

jobs: Dict[str, dict] = {}

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def process_video_job(job_id: str, video_path: str):
    """Background task to process video"""
    try:
        jobs[job_id]["status"] = "processing"
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = "Could not open video"
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        jobs[job_id]["total_frames"] = total_frames
        
        # Create output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"badminton_{timestamp}_annotated.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            jobs[job_id]["status"] = "error"
            jobs[job_id]["message"] = "Could not create output video"
            cap.release()
            return
        
        # Process video
        
        result = process_video_sync(
            cap, out, model, pose_model, tracknet, DEVICE, 
            total_frames, width, height, job_id, jobs
        )
        
        cap.release()
        out.release()
        os.remove(video_path)
        
        # Update job status
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["output_video"] = str(output_path)
        jobs[job_id]["unique_people"] = result["unique_people"]
        jobs[job_id]["shuttle_detections"] = result["shuttle_detections"]
        jobs[job_id]["total_frames"] = result["total_frames"]
        jobs[job_id]["summary"] = {
            "detection_rate": round((result["shuttle_detections"] / result["total_frames"]) * 100, 1) if result["total_frames"] > 0 else 0
        }
        
    except Exception as e:
        print(f"Error processing job {job_id}: {str(e)}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["message"] = str(e)



@app.post("/track-human-video-async")
async def track_human_video_async(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Stream processing updates in real-time"""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    job_id = str(uuid.uuid4())

    contents = await file.read()
    print("Request received for streaming")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Initialize job
    jobs[job_id] = {
        "status": "queued",
        "frame": 0,
        "total_frames": total_frames,
        "progress_percent": 0,
    }

    background_tasks.add_task(process_video_job, job_id, video_path)

    return {
        "job_id": job_id,
        "status": "queued",
        "total_frames": total_frames
    }


@app.get("/job-status/{job_id}")
async def get_job_status(job_id: str):
    """Get current status of processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.get("/download-video")
async def download_video(path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=os.path.basename(path)
    )

@app.get("/list-videos")
async def list_videos():
    if not OUTPUT_DIR.exists():
        return {"videos": []}
    
    videos = [
        {
            "filename": f.name,
            "path": str(f),
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "created": f.stat().st_mtime
        }
        for f in OUTPUT_DIR.glob("*.mp4")
    ]
    
    return {"videos": sorted(videos, key=lambda x: x['created'], reverse=True)}

@app.delete("/clear-jobs")
async def clear_old_jobs():
    """Clear completed jobs (call periodically)"""
    global jobs
    jobs = {k: v for k, v in jobs.items() if v["status"] not in ["complete", "error"]}
    return {"message": "Cleared old jobs"}