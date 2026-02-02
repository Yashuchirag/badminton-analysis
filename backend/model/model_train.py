import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tracknet_dataset_loader import TrackNetDataset
from tracknet_code import TrackNet

import matplotlib.pyplot as plt
from tqdm import tqdm  # progress bar
import os
import time

def visualize_prediction(model, dataset, device, idx=0):
    model.eval()

    x, y_gt = dataset[idx]      # x: [9,H,W], y_gt: [1,H,W]

    with torch.no_grad():
        pred = model(x.unsqueeze(0).to(device))[0].cpu()

    gt = y_gt[0]
    pred = pred[0]

    plt.figure(figsize=(12,4))

    # ---- Input visualization ----
    plt.subplot(1, 3, 1)
    plt.title("Input (sum of RGB frames)")
    plt.imshow(x[:3].sum(0), cmap='gray')
    plt.axis("off")

    # ---- Ground truth ----
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Heatmap")
    plt.imshow(gt, cmap='hot')
    plt.colorbar()

    # ---- Prediction ----
    plt.subplot(1, 3, 3)
    plt.title("Predicted Heatmap")
    plt.imshow(pred, cmap='hot')
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def train():
    # ---------------- Setup Dataset & DataLoader ----------------
    train_ds = TrackNetDataset("tracknet_data", split="train")
    val_ds   = TrackNetDataset("tracknet_data", split="val")
    print("Dataset loaded....")

    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False
    )
    print("DataLoader created....")

    # ---------------- Initialize model ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    torch.backends.cudnn.benchmark = True
    model = TrackNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print("Model initialized....")

    # ---------------- Training loop ----------------
    num_epochs = 5  # adjust as needed
    best_val_loss = float("inf")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Training started....")
    
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()


            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(x)
                loss = criterion(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.6f}")

        # ---------------- Validation ----------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                pred_val = model(x_val)
                val_loss += criterion(pred_val, y_val).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"🔹 Epoch {epoch+1}/{num_epochs} | Avg Val Loss: {avg_val_loss:.6f}")
        end_time = time.time()
        print(f"Epoch {epoch+1}/{num_epochs} | Time: {end_time - start_time:.2f} seconds")
        # ---------------- Save best model ----------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, "best_tracknet.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved best model to {checkpoint_path}")

    # ---------------- Sanity check ----------------
    x, y = train_ds[100]
    print("Input shape:", x.shape)   # [9, H, W]
    print("Heatmap shape:", y.shape) # [1, H, W]
    print("Max heatmap value:", y.max())  # 0 if shuttle missing, >0 otherwise

    visualize_prediction(model, train_ds, device, idx=100)

if __name__ == "__main__":
    torch.multiprocessing.freeze_support()  # optional on Windows
    train()