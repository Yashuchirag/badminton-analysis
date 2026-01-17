from torch.utils.data import Dataset
import torch
import numpy as np

class TrackNetDataset(Dataset):
    def __init__(self, frames, label_data):
        self.frames = frames
        self.indices, self.heatmaps = label_data

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        frame_idx = self.indices[idx]

        # 3 consecutive frames
        f1 = self.frames[frame_idx - 2]
        f2 = self.frames[frame_idx - 1]
        f3 = self.frames[frame_idx]

        # (H, W, 9)
        input_img = np.concatenate([f1, f2, f3], axis=2)
        input_img = input_img.astype(np.float32) / 255.0

        x = torch.from_numpy(input_img).permute(2, 0, 1)
        y = torch.from_numpy(self.heatmaps[idx]).unsqueeze(0)

        return x, y
