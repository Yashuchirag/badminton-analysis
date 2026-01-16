from fastapi import FastAPI, UploadFile, File
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os

app = FastAPI()

model = YOLO("yolov8n.pt")

@app.post("/track-human")
async def track_human(file: UploadFile = File(...)):
    # Read image bytes
    contents = await file.read()

    # Convert bytes → OpenCV image
    np_img = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # Run YOLO
    results = model(img)

    detections = []
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])

            # COCO class 0 = person
            if cls_id == 0:
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections.append({
                    "x": x1,
                    "y": y1,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "confidence": float(box.conf[0])
                })

    return {
        "status": "success",
        "humans_detected": person_count,
        "boxes": detections
    }

@app.post("/track-human-video")
async def track_human_video(file: UploadFile = File(...)):
    # Read image bytes
    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"error": "Could not open video"}

    frame_idx = 0
    max_people = 0
    frame_stats = []
    unique_ids = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = model(frame, conf=0.4, iou=0.5)
        results1 = model.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5,
            tracker="bytetrack.yaml"
        )
        people_count = 0

        annotated_frame = results1[0].plot()
        cv2.imshow("YOLO Detection", annotated_frame)
        cv2.waitKey(1)


        for r in results1:
            if r.boxes.id is not None:
                for track_id, cls_id in zip(r.boxes.id, r.boxes.cls):
                    if int(cls_id) == 0:
                        unique_ids.add(int(track_id))
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # person
                    people_count += 1

        max_people = max(max_people, people_count)

        frame_stats.append({
            "frame": frame_idx,
            "humans_detected": people_count,
            "unique_ids": unique_ids
        })

    cap.release()
    os.remove(video_path)

    return {
        "status": "success",
        "total_frames": frame_idx,
        "max_humans_in_video": max_people,
        "max_unique_ids": max(unique_ids),
        "unique_ids": unique_ids,
        #   "per_frame_stats": frame_stats
    }
