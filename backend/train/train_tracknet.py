import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model_tracknet import TrackNet

def train_tracknet(
    frame_paths,
    labels,
    dataset_class,
    batch_size=4,
    epochs=50,
    lr=1e-4,
    save_path="tracknet.pth",
    device=None
):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------
    # MODEL
    # -------------------------
    model = TrackNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # -------------------------
    # DATA
    # -------------------------
    dataset = dataset_class(frame_paths, labels)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )

    # -------------------------
    # TRAINING LOOP
    # -------------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

    # -------------------------
    # SAVE MODEL
    # -------------------------
    torch.save(model.state_dict(), save_path)
    print(f"✅ TrackNet model saved to: {save_path}")

    return model
