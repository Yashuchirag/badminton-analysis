import numpy as np
import torch
import os
import json

from torch import nn
from torch.utils.data import DataLoader, random_split
from model_tracknet import TrackNet
from dataset_tracknet import TrackNetDataset
from heatmap_generator import generate_heatmap


# -------------------------
# CONFIG
# -------------------------

def main_train():
    FRAME_DIR = "frames"
    ANNOTATION_FILE = "annotations.json"
    INPUT_W, INPUT_H = 960, 540
    HEATMAP_SIGMA = 3
    BATCH_SIZE = 2
    EPOCHS = 15
    LR = 1e-4
    VAL_SPLIT = 0.2
    MODEL_PATH = "tracknet.pth"

    frame_paths = sorted([
        os.path.join(FRAME_DIR, f)
        for f in os.listdir(FRAME_DIR)
        if f.endswith(".jpg")
    ])

    print("Frame_paths loaded")

    # Load annotations
    with open(ANNOTATION_FILE) as f:
        ann = json.load(f)

    print("Annotations loaded")
    indices, heatmaps = [], []
    for k in sorted(ann.keys(), key=int):
        idx = int(k)
        if idx >= 2:
            x, y = ann[k]
            indices.append(idx)
            heatmaps.append(
                generate_heatmap(x, y, INPUT_W, INPUT_H, HEATMAP_SIGMA)
            )

    heatmaps = np.array(heatmaps, dtype=np.float32)
    print("Heatmaps generated")

    dataset = TrackNetDataset(
        frame_paths, indices, heatmaps, (INPUT_W, INPUT_H)
    )

    print("Dataset created")

    # Train/val split
    val_len = int(len(dataset) * VAL_SPLIT)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    print("Data loaders created")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: ", device)

    # Model
    model = TrackNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    print("Model created")

    best_val = float("inf")
    print("Best val: ", best_val)

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)   

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()

        val_loss /= len(val_loader)

        print(f"[{epoch+1:02d}] Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print("✅ Saved best model")

    print("🎯 Training complete")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    print("Starting training")
    main_train()