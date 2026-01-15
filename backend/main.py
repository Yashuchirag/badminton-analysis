from fastapi import FastAPI, UploadFile, File
import cv2
from ultralytics import YOLO
import numpy as np

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
