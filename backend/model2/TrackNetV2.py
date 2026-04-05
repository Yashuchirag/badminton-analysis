import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from typing import Optional
from pathlib import Path


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not installed. TrackNetV2 training disabled.")

def get_device():
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            return "0"
        return "cpu"
    return "cpu"

DEVICE = get_device()


class TrackNetV2(nn.Module):

    def __init__(self, sequence_length: int = 3):
        super().__init__()
        in_ch = sequence_length * 3  # 9 channels for 3 RGB frames

        # ── Encoder ───────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),    nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        # ── Bottleneck ────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

        # ── Decoder with skip connections ─────────────────────────────
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        self.output_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.output_conv(d1)).squeeze(1)


class TrackNetV2Trainer:
    @staticmethod
    def train(
        split_dir: str,
        output_dir: str,
        sequence_length: int = 3,
        img_size: int = 512,
        epochs: int = 20,
        batch_size: int = 16,
        lr: float = 1e-4,
        device: str = DEVICE,
        patience: int = 15,
        lr_patience: int = 5,
        resume_from: Optional[str] = None,
        finetune_from: Optional[str] = None,
        freeze_encoder: bool = False,
        log_every: int = 20,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed.")

        if device.isdigit():
            device = f"cuda:{device}"
        elif device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, using CPU")
            device = "cpu"

        if resume_from and finetune_from:
            raise ValueError("Cannot use both resume_from and finetune_from.")

        train_ds = TrackNetDataset(split_dir, "train", sequence_length, img_size)
        val_ds = TrackNetDataset(split_dir, "val",   sequence_length, img_size)
        use_pin = device != "cpu"

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=8, pin_memory=use_pin)
        val_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=8, pin_memory=use_pin)

        model = TrackNetV2(sequence_length).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=lr_patience
        )

        start_epoch = 0
        best_val_loss = float("inf")
        epochs_no_improve = 0
        training_mode = "FROM SCRATCH"
        prev_lr = lr

        if resume_from:
            ckpt = torch.load(resume_from, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            prev_lr = optimizer.param_groups[0]["lr"]
            training_mode = "RESUME"
            print(f"Resuming from epoch {start_epoch}")

        elif finetune_from:
            TrackNetV2Trainer._load_weights_flexible(model, finetune_from, device)
            training_mode = "FINE-TUNE"
            if freeze_encoder:
                for name, param in model.named_parameters():
                    if any(name.startswith(l) for l in ["enc1", "enc2", "enc3", "bottleneck"]):
                        param.requires_grad = False
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=lr
                )
                print("Encoder frozen — decoder only")

        folder = "tracknetv2_finetune" if finetune_from else "tracknetv2"
        save_dir = os.path.join(output_dir, folder)
        os.makedirs(save_dir, exist_ok=True)

        total_epochs = start_epoch + epochs

        print(f"\n{'='*70}")
        print("Training TrackNetV2")
        print(f"  Mode            : {training_mode}")
        print(f"  Device          : {device}")
        print(f"  Train sequences : {len(train_ds)}")
        print(f"  Val sequences   : {len(val_ds)}")
        print(f"  Max epochs      : {epochs}  (early stop: {patience})")
        print(f"  LR              : {lr}  (scheduler patience: {lr_patience})")
        print(f"{'='*70}\n")

        epoch_bar = tqdm(
            range(start_epoch, total_epochs), 
            desc="Training Tracknet", 
            unit="epoch",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        for epoch in epoch_bar:
            model.train()
            train_loss = 0.0

            train_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{total_epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )

            for batch_idx, (images, heatmaps) in enumerate(train_bar):
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                optimizer.zero_grad()
                loss = criterion(model(images), heatmaps)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                train_bar.set_postfix(
                    loss=f"{train_loss / (batch_idx+1):.6f}",
                    refresh=True
                )
            
            train_bar.close()
            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            val_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{total_epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )

            with torch.no_grad():
                for images, heatmaps in val_loader:
                    images = images.to(device)
                    heatmaps = heatmaps.to(device)
                    outputs = model(images)
                    val_loss += criterion(outputs, heatmaps).item()

                    for pred, gt in zip(outputs.cpu().numpy(), heatmaps.cpu().numpy()):
                        if gt.max() > 0.5:
                            py, px = np.unravel_index(pred.argmax(), pred.shape)
                            gy, gx = np.unravel_index(gt.argmax(),   gt.shape)
                            dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                            val_correct += 1 if dist <= 5 else 0
                            val_total   += 1

                    val_bar.set_postfix(
                        loss=f"{val_loss / (val_bar.n or 1):.6f}",
                        refresh=True
                    )
            
            val_bar.close()
            val_loss /= len(val_loader)
            accuracy = (val_correct / val_total * 100) if val_total > 0 else 0.0

            # Scheduler
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr != prev_lr:
                print(f"LR changed from {prev_lr:.2e} to {current_lr:.2e}")
                prev_lr = current_lr

            # Epoch bar
            epoch_bar.set_postfix(
                train_loss=f"{train_loss:.6f}",
                val_loss=f"{val_loss:.6f}",
                accuracy=f"{accuracy:.1f}%",
                lr=f"{current_lr:.2e}",
                best_val_loss=f"{best_val_loss:.6f}",
                refresh=True
            )

            tqdm.write(
                f"Epoch {epoch+1}/{start_epoch+epochs} | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Acc: {accuracy:.1f}% | LR: {current_lr:.2e}"
            )

            # Checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict":model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss":val_loss,
                "accuracy":accuracy,
            }, os.path.join(save_dir, f"tracknetv2_epoch_{epoch+1}.pth"))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss":val_loss,
                    "accuracy":accuracy,
                }, os.path.join(save_dir, "tracknetv2_best.pth"))
                tqdm.write(
                    f"  → Best saved (val: {val_loss:.6f}, acc: {accuracy:.1f}%)"
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    tqdm.write(f"\n⚡ Early stopping at epoch {epoch+1}")
                    break
        epoch_bar.close()
        tqdm.write(f"\n✓ Training complete | Best val loss: {best_val_loss:.6f}")
        return model

    @staticmethod
    def _load_weights_flexible(model: nn.Module, weights_path: str, device: str):
        """
        Load weights with automatic remapping from old TrackNet → TrackNetV2.
        Falls back to partial loading if shapes don't match.
        """
        ckpt = torch.load(weights_path, map_location=device)
        old_state = ckpt.get("model_state_dict", ckpt)
        new_state = model.state_dict()

        # Remap old TrackNet layer names to new TrackNetV2 names
        remap = {
            "conv1.0.weight": "enc1.0.weight", "conv1.0.bias": "enc1.0.bias",
            "conv1.2.weight": "enc1.3.weight", "conv1.2.bias": "enc1.3.bias",
            "conv2.0.weight": "enc2.0.weight", "conv2.0.bias": "enc2.0.bias",
            "conv2.2.weight": "enc2.3.weight", "conv2.2.bias": "enc2.3.bias",
            "conv3.0.weight": "enc3.0.weight", "conv3.0.bias": "enc3.0.bias",
            "conv3.2.weight": "enc3.3.weight", "conv3.2.bias": "enc3.3.bias",
        }
        remapped = {remap.get(k, k): v for k, v in old_state.items()}
        transferred = []

        for name, param in new_state.items():
            if name in remapped and remapped[name].shape == param.shape:
                new_state[name] = remapped[name]
                transferred.append(name)

        model.load_state_dict(new_state)
        skipped = len(new_state) - len(transferred)
        print(f"  ✓ Loaded {len(transferred)} layers | "
              f"⚠ {skipped} layers initialised from scratch")

class TrackNetDataset(Dataset):
    """Dataset for TrackNetV2. Compatible with both your format and TrackNetV2 format."""

    def __init__(self, split_dir: str, split: str = "train",
                 sequence_length: int = 3, img_size: int = 512):
        self.img_dir = Path(split_dir) / split / "images"
        self.ann_path = Path(split_dir) / split / "annotations.json"
        self.sequence_length = sequence_length
        self.img_size = img_size

        if not self.ann_path.exists():
            raise FileNotFoundError(f"annotations.json not found: {self.ann_path}")

        with open(self.ann_path) as f:
            self.annotations = json.load(f)

        self.frames = sorted(self.annotations.keys())
        self.valid_indices = [
            i for i in range(len(self.frames) - sequence_length + 1)
            if self._are_consecutive(self.frames[i:i + sequence_length])
        ]

        print(f"TrackNet {split}: {len(self.valid_indices)} sequences "
              f"(from {len(self.frames)} frames)")

    def _get_prefix_and_num(self, fname: str):
        import re
        m = re.match(r"^(.+_frame)(\d+)\.jpg$", fname)
        if m:
            return m.group(1), int(m.group(2))
        m2 = re.search(r"(\d+)", fname)
        return "default_", int(m2.group(1)) if m2 else 0

    def _are_consecutive(self, frame_list: list) -> bool:
        prefixes, numbers = [], []
        for fname in frame_list:
            p, n = self._get_prefix_and_num(fname)
            prefixes.append(p)
            numbers.append(n)
        if len(set(prefixes)) > 1:
            return False
        return all(numbers[i + 1] == numbers[i] + 1 for i in range(len(numbers) - 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start      = self.valid_indices[idx]
        seq_frames = self.frames[start:start + self.sequence_length]

        images = []
        for fname in seq_frames:
            img = cv2.imread(str(self.img_dir / fname))
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        images = np.array(images).transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        last_ann = self.annotations[seq_frames[-1]]
        heatmap  = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        if last_ann.get("visibility") == "visible" and last_ann.get("x") is not None:
            orig = cv2.imread(str(self.img_dir / seq_frames[-1]))
            if orig is not None:
                h_orig, w_orig = orig.shape[:2]
                x = max(0, min(int(float(last_ann["x"]) * self.img_size / w_orig), self.img_size - 1))
                y = max(0, min(int(float(last_ann["y"]) * self.img_size / h_orig), self.img_size - 1))
                sigma  = 5
                yy, xx = np.mgrid[0:self.img_size, 0:self.img_size]
                heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2)).astype(np.float32)

        return torch.FloatTensor(images), torch.FloatTensor(heatmap)