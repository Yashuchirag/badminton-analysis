from torch.utils.data import Dataset
import torch
import numpy as np
import cv2

class TrackNetDataset(Dataset):
    def __init__(self, frame_paths, indices, heatmaps, img_size):
        self.frame_paths = frame_paths
        self.indices = indices
        self.heatmaps = heatmaps
        self.W, self.H = img_size

    def _load_frame(self, idx):
        frame = cv2.imread(self.frame_paths[idx])
        if frame is None:
            raise RuntimeError(f"Missing frame: {self.frame_paths[idx]}")

        frame = cv2.resize(frame, (self.W, self.H))
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - 0.5) / 0.5
        return frame

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        f1 = self._load_frame(idx - 2)
        f2 = self._load_frame(idx - 1)
        f3 = self._load_frame(idx)

        x = np.concatenate([f1, f2, f3], axis=2)
        x = torch.from_numpy(x).permute(2, 0, 1)

        y = torch.from_numpy(self.heatmaps[i]).unsqueeze(0)

        return x, y
