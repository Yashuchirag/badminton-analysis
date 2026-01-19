import cv2
import os
import json
import numpy as np

# -------------------------
# CONFIG
# -------------------------
VIDEO_PATH = "Badminton_Training_Video.mp4"
FRAME_DIR = "frames"
ANNOTATION_FILE = "annotations.json"
INPUT_W, INPUT_H = 960, 540

os.makedirs(FRAME_DIR, exist_ok=True)

annotations = {}
current_frame = None
current_frame_idx = 0

# -------------------------
# MOUSE CALLBACK
# -------------------------
def click_event(event, x, y, flags, param):
    global annotations, current_frame_idx, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        annotations[str(current_frame_idx)] = [x, y]
        vis = current_frame.copy()
        cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
        cv2.imshow("Frame", vis)
        print(f"Frame {current_frame_idx}: {x}, {y}")

# -------------------------
# MAIN
# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.namedWindow("Frame")

print("🖱️ Click shuttle | SPACE = next | ESC = stop")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (INPUT_W, INPUT_H))
    frame_path = f"{FRAME_DIR}/frame_{frame_idx:05d}.jpg"
    cv2.imwrite(frame_path, frame)

    current_frame = frame.copy()
    current_frame_idx = frame_idx

    cv2.imshow("Frame", current_frame)
    cv2.setMouseCallback("Frame", click_event)

    key = cv2.waitKey(0)
    if key == 27:
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# Save annotations
with open(ANNOTATION_FILE, "w") as f:
    json.dump(annotations, f, indent=2)

print(f"✅ Saved {len(annotations)} annotations")
print(f"✅ Frames saved in {FRAME_DIR}/")
