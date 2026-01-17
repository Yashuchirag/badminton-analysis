import numpy as np
import cv2

def generate_heatmap(x, y, h=288, w=512, sigma=3):
    heatmap = np.zeros((h, w), dtype=np.float32)
    cv2.circle(heatmap, (x, y), sigma, 1, -1)
    heatmap = cv2.GaussianBlur(heatmap, (7,7), 0)
    return heatmap
