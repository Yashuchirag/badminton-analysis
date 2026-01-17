from fastapi import FastAPI, UploadFile, File
import cv2
from ultralytics import YOLO
import numpy as np
import tempfile
import os
from infer import shuttle_tracking

app = FastAPI()

model = YOLO("yolov8n.pt")
pose_model = YOLO("yolov8n-pose.pt")

DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 540


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
    wrist_data = []
    shuttle_positions = []

    shuttle_positions = shuttle_tracking(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = model.track(
            frame,
            persist=True,
            conf=0.4,
            iou=0.5,
            tracker="bytetrack.yaml"
        )

        pose_results = pose_model(frame)
        people_count = 0


        for r in results:
            if r.boxes.id is not None:
                for track_id, cls_id in zip(r.boxes.id, r.boxes.cls):
                    if int(cls_id) == 0:
                        unique_ids.add(int(track_id))
                        people_count += 1
        

        max_people = max(max_people, people_count)

        for r in pose_results:
            if r.keypoints is None:
                continue

            for kp in r.keypoints.xy.cpu().numpy():
                left_wrist = kp[9]
                right_wrist = kp[10]

                wrist_data.append({
                    "frame": frame_idx,
                    "left_wrist": {
                        "x": float(left_wrist[0]),
                        "y": float(left_wrist[1])
                    },
                    "right_wrist": {
                        "x": float(right_wrist[0]),
                        "y": float(right_wrist[1])
                    }
                })

                # Draw wrists
                cv2.circle(frame, (int(left_wrist[0]), int(left_wrist[1])), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(right_wrist[0]), int(right_wrist[1])), 5, (255, 0, 0), -1)

        # ---------- Visualization ----------
        annotated_frame = results[0].plot()
        
        annotated_frame = cv2.resize(
            annotated_frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )
        cv2.imshow("YOLO Detection + Wrist", annotated_frame)
        cv2.waitKey(1)


        frame_stats.append({
            "frame": frame_idx,
            "humans_detected": people_count,
            "unique_ids": unique_ids
        })

    cap.release()
    os.remove(video_path)
    cv2.destroyAllWindows()

    return {
        "status": "success",
        "total_frames": frame_idx,
        "max_humans_in_video": max_people,
        "max_unique_ids": max(unique_ids),
        "unique_ids": unique_ids,
        "wrist_samples": wrist_data[:5],
        "shuttle_positions": shuttle_positions
        #   "per_frame_stats": frame_stats
    }
