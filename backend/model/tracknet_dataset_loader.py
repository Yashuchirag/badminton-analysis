import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


class TrackNetDataset(Dataset):
    def __init__(self, root_dir, split):
        self.img_dir = Path(root_dir) / split / "images"
        self.hm_dir  = Path(root_dir) / split / "labels"

        # IMPORTANT: keep temporal order
        self.frames = sorted(self.img_dir.glob("*.jpg"))

    def __len__(self):
        # because we use t-1, t, t+1
        return len(self.frames) - 2

    def __getitem__(self, idx):
        f_prev = self.frames[idx]
        f_curr = self.frames[idx + 1]
        f_next = self.frames[idx + 2]
        H, W = 960, 544

        # ---------- Load images ----------
        def load_img(p):
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (W, H))  # resize
            img = img.astype(np.float32) / 255.0
            return img

        img_prev = load_img(f_prev)
        img_curr = load_img(f_curr)
        img_next = load_img(f_next)

        # Stack frames → [H, W, 9]
        x = np.concatenate([img_prev, img_curr, img_next], axis=2)

        # Convert to torch → [9, H, W]
        x = torch.from_numpy(x).permute(2, 0, 1)

        # ---------- Load heatmap ----------
        hm_path = self.hm_dir / f"{f_curr.stem}.png"

        if hm_path.exists():
            heatmap = cv2.imread(str(hm_path), cv2.IMREAD_GRAYSCALE)
            heatmap = cv2.resize(heatmap, (W, H))
            heatmap = heatmap.astype(np.float32) / 255.0
        else:
            # missing shuttle → zero heatmap
            h, w = img_curr.shape[:2]
            heatmap = np.zeros((h, w), dtype=np.float32)

        y = torch.from_numpy(heatmap).unsqueeze(0)  # [1, H, W]

        return x, y