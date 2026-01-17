import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from model_tracknet import TrackNet
from dataset_tracknet import TrackNetDataset
from train_tracknet import train_tracknet

# -------------------------
# CONFIG
# -------------------------
VIDEO_PATH = "Badminton_Training_Video.mp4"
INPUT_W, INPUT_H = 960, 540
HEATMAP_SIGMA = 3
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
MODEL_PATH = "tracknet.pth"

# -------------------------
# GLOBAL VARIABLES
# -------------------------
annotations = {}
current_frame_idx = 0
current_frame = None

# -------------------------
# MOUSE CALLBACK
# -------------------------
def click_event(event, x, y, flags, param):
    """
    Records coordinates when left mouse button is clicked.
    """
    global annotations, current_frame_idx, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        annotations[current_frame_idx] = (x, y)
        print(f"Frame {current_frame_idx}: {x}, {y}")

        vis = current_frame.copy()
        cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
        cv2.imshow("Frame", vis)

# -------------------------
# GENERATE HEATMAP
# -------------------------
def generate_heatmap(x, y, H, W, sigma=3):
    heatmap = np.zeros((H, W), dtype=np.float32)
    cv2.circle(heatmap, (x, y), 1, 1, -1)
    heatmap = cv2.GaussianBlur(heatmap, (7, 7), sigma)
    heatmap /= heatmap.max() + 1e-6
    return heatmap

# -------------------------
# MAIN FUNCTION
# -------------------------
if __name__ == "__main__":
    # --- Step 1: Read Video Frames ---
    cap = cv2.VideoCapture(VIDEO_PATH)
    frames = []
    frame_idx = 0

    print("🖱️ Click shuttle position. Press SPACE to continue, ESC to stop.")

    cv2.namedWindow("Frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (INPUT_W, INPUT_H))
        frames.append(frame.copy())

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

    print(f"Total frames read: {len(frames)}")
    print(f"Labeled frames: {len(annotations)}")

    # -------------------------
    # BUILD TRAINING DATA
    # -------------------------
    valid_indices = []
    labels = []

    for idx in sorted(annotations.keys()):
        if idx >= 2:
            valid_indices.append(idx)
            x, y = annotations[idx]
            heatmap = generate_heatmap(x, y, INPUT_H, INPUT_W, HEATMAP_SIGMA)
            labels.append(heatmap)

    labels = np.array(labels, dtype=np.float32)
    print(f"✅ Heatmaps shape: {labels.shape}")

    # -------------------------
    # TRAIN
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    train_tracknet(
        frame_paths=frames,
        labels=(valid_indices, labels),   # 👈 IMPORTANT
        dataset_class=TrackNetDataset,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lr=LR,
        save_path=MODEL_PATH,
        device=device
    )

    print(f"✅ TrackNet trained & saved → {MODEL_PATH}")