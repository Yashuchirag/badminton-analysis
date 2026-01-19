import cv2
import torch
import numpy as np
from train.model_tracknet import TrackNet

# -------------------------
# CONFIG
# -------------------------
INPUT_W, INPUT_H = 960, 540
HEATMAP_THRESH = 0.4


def shuttle_tracking(cap, train=False, labels=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (INPUT_W, INPUT_H))
        frames.append(frame)
    print(f"Total frames: {len(frames)}")

    
    model = TrackNet().to(device)
    model.load_state_dict(torch.load("tracknet.pth", map_location=device))
    model.eval()


    shuttle_positions = []

    for i in range(2, len(frames)):
        f1, f2, f3 = frames[i - 2], frames[i - 1], frames[i]

        # Stack frames → (H, W, 9)
        input_img = np.concatenate([f1, f2, f3], axis=2)
        input_tensor = (
            torch.from_numpy(input_img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        with torch.no_grad():
            heatmap = model(input_tensor)

        heatmap = heatmap.squeeze().cpu().numpy()

        # Get shuttle position
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)

        if heatmap[y, x] > HEATMAP_THRESH:
            shuttle_positions.append((x, y))
        else:
            shuttle_positions.append(None)

    # -------------------------
    # VISUALIZE
    # -------------------------
    for frame, pos in zip(frames[2:], shuttle_positions):
        if pos:
            cv2.circle(frame, pos, 5, (0, 0, 255), -1)

        cv2.imshow("TrackNet Shuttle Tracking", frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

    return shuttle_positions
