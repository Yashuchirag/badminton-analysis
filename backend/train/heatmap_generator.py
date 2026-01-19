import numpy as np
import cv2

def generate_heatmap(x, y, H, W, sigma=3):
    heatmap = np.zeros((H, W), dtype=np.float32)

    if 0 <= x < W and 0 <= y < H:
        heatmap[y, x] = 1.0

    heatmap = cv2.GaussianBlur(heatmap, (7, 7), sigma)
    return heatmap / (heatmap.max() + 1e-6)
