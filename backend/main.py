from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from ultralytics import YOLO
from datetime import datetime

import cv2

import numpy as np
import tempfile
import os
import torch

from train.model_tracknet import TrackNet
from infer import shuttle_tracking_combined

app = FastAPI()

model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

tracknet = TrackNet().to(DEVICE)
tracknet.load_state_dict(torch.load("tracknet.pth", map_location=DEVICE))
tracknet.eval()
print("TrackNet model loaded successfully")


DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


@app.post("/track-human-video")
async def track_human_video(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Read image bytes
    contents = await file.read()
    print("Request received")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        os.remove(video_path)
        raise HTTPException(status_code=400, detail="Could not open video")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"badminton_{timestamp}"

    
    try:
        shuttle_positions, wrist_data, unique_ids, output_path = shuttle_tracking_combined(
            cap, model, pose_model, tracknet, DEVICE, video_filename
        )
    finally:
        cap.release()
        os.remove(video_path)

    return {
        "status": "success",
        "device_used": DEVICE,
        "total_frames": len(shuttle_positions) + 2,
        "unique_ids": list(unique_ids),
        "total_unique_people": len(unique_ids),
        "wrist_data_points": len(wrist_data),
        "wrist_samples": wrist_data[:5],
        "shuttle_detections": sum(1 for pos in shuttle_positions if pos is not None),
        "shuttle_positions": shuttle_positions,
        "output_video": output_path
    }

@app.get("/download-video")
async def download_video(path: str):
    """Download the annotated video"""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=os.path.basename(path)
    )


@app.get("/list-videos")
async def list_videos():
    from pathlib import Path
    output_dir = Path("outputs")
    if not output_dir.exists():
        return {"videos": []}
    
    videos = [
        {
            "filename": f.name,
            "path": str(f),
            "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            "created": f.stat().st_mtime
        }
        for f in output_dir.glob("*.mp4")
    ]
    
    return {"videos": sorted(videos, key=lambda x: x['created'], reverse=True)}