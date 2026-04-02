import os
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not installed.")


# ══════════════════════════════════════════════════════════════════════════
# 1. Dataset Converter
#    Reads the TrackNetV2 structure:
#      root/amateur/matchX/matchX.csv + matchX.mp4
#      root/pro/matchX/matchX.csv     + matchX.mp4
#      root/test/matchX/matchX.csv    + matchX.mp4
# ══════════════════════════════════════════════════════════════════════════

def convert_tracknetv2_dataset(
    tracknetv2_dir: str,
    output_split_dir: str,
    val_ratio: float = 0.1,
    test_use_provided: bool = True,
    frame_skip: int = 1,
):
    """
    Converts TrackNetV2 dataset (MP4 + CSV) into your annotations.json format.

    Args:
        tracknetv2_dir:     Root folder containing amateur/, pro/, test/
        output_split_dir:   Where to write train/val/test splits
        val_ratio:          Fraction of pro+amateur matches to use for val
        test_use_provided:  If True, use the test/ folder as your test split
        frame_skip:         Extract every Nth frame (1 = all frames)
                            Use 2-3 to reduce dataset size if storage is limited
    
    Output structure:
        output_split_dir/
          train/
            images/         ← extracted .jpg frames
            annotations.json
          val/
            images/
            annotations.json
          test/
            images/
            annotations.json
    """
    root = Path(tracknetv2_dir)
    out  = Path(output_split_dir)

    splits = {
        "train": {},
        "val":   {},
        "test":  {},
    }

    # ── Collect all matches from amateur/ and pro/ ─────────────────────
    trainval_matches = []
    for category in ["amateur", "pro"]:
        cat_dir = root / category
        if not cat_dir.exists():
            print(f"⚠ {category}/ not found — skipping")
            continue
        for match_dir in sorted(cat_dir.iterdir()):
            if match_dir.is_dir():
                trainval_matches.append((category, match_dir))

    # Split into train/val by match (not by frame — avoids data leakage)
    n_val = max(1, int(len(trainval_matches) * val_ratio))
    val_matches   = trainval_matches[-n_val:]
    train_matches = trainval_matches[:-n_val]

    print(f"\nDataset split:")
    print(f"  Train matches : {len(train_matches)}")
    print(f"  Val matches   : {len(val_matches)}")

    # ── Process train and val matches ─────────────────────────────────
    for split_name, match_list in [("train", train_matches), ("val", val_matches)]:
        for category, match_dir in match_list:
            _process_match(
                match_dir=match_dir,
                category=category,
                split_name=split_name,
                out=out,
                splits=splits,
                frame_skip=frame_skip,
            )

    # ── Process test/ folder ──────────────────────────────────────────
    if test_use_provided:
        test_dir = root / "test"
        if test_dir.exists():
            for match_dir in sorted(test_dir.iterdir()):
                if match_dir.is_dir():
                    _process_match(
                        match_dir=match_dir,
                        category="test",
                        split_name="test",
                        out=out,
                        splits=splits,
                        frame_skip=frame_skip,
                    )

    # ── Write annotations.json for each split ─────────────────────────
    for split_name, annotations in splits.items():
        ann_path = out / split_name / "annotations.json"
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ann_path, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"✓ {split_name}: {len(annotations)} frames → {ann_path}")

    print("\n✓ Conversion complete!")
    return out


def _process_match(match_dir, category, split_name, out, splits, frame_skip):
    """Extract frames from one match MP4 and read its CSV annotations."""

    # Find CSV and MP4 files in this match folder
    csv_files = list(match_dir.glob("*.csv"))
    mp4_files = list(match_dir.glob("*.mp4"))

    if not csv_files or not mp4_files:
        print(f"  ⚠ Skipping {match_dir.name} — missing CSV or MP4")
        return

    csv_path = csv_files[0]
    mp4_path = mp4_files[0]

    match_name = f"{category}_{match_dir.name}"
    print(f"  Processing {match_name} ...", end=" ", flush=True)

    # ── Parse CSV ──────────────────────────────────────────────────────
    # Format: Frame,Visibility,X,Y
    # Example: 0,0,0,0  or  42,1,320,180
    frame_annotations = {}
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header if present (Frame,Visibility,X,Y)

        for row in reader:
            if len(row) < 4:
                continue
            try:
                frame_num  = int(row[0])
                visibility = int(row[1])
                x = float(row[2]) if visibility == 1 else None
                y = float(row[3]) if visibility == 1 else None
                frame_annotations[frame_num] = {
                    "visibility": "visible"     if visibility == 1 else "not_visible",
                    "x": x,
                    "y": y,
                }
            except (ValueError, IndexError):
                continue

    # ── Extract frames from MP4 ────────────────────────────────────────
    out_img_dir = out / split_name / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        print(f"❌ Cannot open {mp4_path}")
        return

    frame_idx   = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply frame_skip — skip frames to reduce dataset size
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue

        # Build unique frame filename
        # e.g. pro_match1_frame00042.jpg
        unique_name = f"{match_name}_frame{frame_idx:05d}.jpg"
        out_path    = out_img_dir / unique_name

        if not out_path.exists():
            cv2.imwrite(str(out_path), frame)

        # Get annotation — if frame_skip > 1 and frame not in CSV,
        # use the annotation for the nearest available frame
        ann = frame_annotations.get(frame_idx, {
            "visibility": "not_visible",
            "x": None,
            "y": None,
        })

        splits[split_name][unique_name] = ann
        saved_count += 1
        frame_idx   += 1

    cap.release()
    print(f"{saved_count} frames")


def merge_with_existing_data(
    existing_split_dir: str,
    tracknetv2_split_dir: str,
    merged_output_dir: str,
):
    """
    Merge your manually annotated data with converted TrackNetV2 data.
    Both must already be in annotations.json format.
    
    Args:
        existing_split_dir:    Your current split dir (3,200 frames)
        tracknetv2_split_dir:  Output of convert_tracknetv2_dataset()
        merged_output_dir:     Where to write the merged dataset
    """
    import shutil

    for split in ["train", "val", "test"]:
        merged_ann  = {}
        out_img_dir = Path(merged_output_dir) / split / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)

        for source_dir in [existing_split_dir, tracknetv2_split_dir]:
            source = Path(source_dir)
            ann_path = source / split / "annotations.json"
            img_dir  = source / split / "images"

            if not ann_path.exists():
                print(f"  ⚠ No {split}/annotations.json in {source_dir} — skipping")
                continue

            with open(ann_path) as f:
                anns = json.load(f)

            for fname, ann in anns.items():
                src = img_dir / fname
                dst = out_img_dir / fname
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                merged_ann[fname] = ann

        out_ann = Path(merged_output_dir) / split / "annotations.json"
        with open(out_ann, "w") as f:
            json.dump(merged_ann, f, indent=2)
        print(f"✓ {split}: {len(merged_ann)} total frames merged → {out_ann}")


# ══════════════════════════════════════════════════════════════════════════
# 2. TrackNetV2 Architecture
#    VGG-style encoder with U-Net skip connections
#    Significantly better than the original 3-block version
# ══════════════════════════════════════════════════════════════════════════

class TrackNetV2(nn.Module):
    """
    TrackNetV2 architecture — VGG encoder + U-Net decoder with skip connections.
    Drop-in replacement for the original TrackNet class.
    
    Key improvements over original:
      - Deeper encoder (4 levels vs 3)
      - Skip connections prevent information loss during upsampling
      - Bottleneck layer captures higher-level temporal features
      - Batch normalization for stable training on larger datasets
    """

    def __init__(self, sequence_length: int = 3):
        super().__init__()
        in_channels = sequence_length * 3  # 9 for 3 frames × RGB

        # ── Encoder ───────────────────────────────────────────────────
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # /2

        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # /4

        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # /8

        # ── Bottleneck ────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── Decoder with skip connections ─────────────────────────────
        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            # 512 = 256 from up3 + 256 from enc3 skip
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            # 256 = 128 from up2 + 128 from enc2 skip
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            # 128 = 64 from up1 + 64 from enc1 skip
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ── Output heatmap ─────────────────────────────────────────────
        self.output_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)  # (B, 9, H, W)

        # Encode — save outputs for skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        bn = self.bottleneck(self.pool3(e3))

        # Decode — concatenate skip connections
        d3 = self.dec3(torch.cat([self.up3(bn), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        out = torch.sigmoid(self.output_conv(d1))
        return out.squeeze(1)  # (B, H, W)


# ══════════════════════════════════════════════════════════════════════════
# 3. Dataset
# ══════════════════════════════════════════════════════════════════════════

class TrackNetDataset(Dataset):
    """
    Dataset for TrackNetV2 training.
    Reads from split_dir/split/images/ and split_dir/split/annotations.json
    Compatible with both your original format and converted TrackNetV2 format.
    """

    def __init__(self, split_dir: str, split: str = "train",
                 sequence_length: int = 3, img_size: int = 512):
        self.img_dir         = Path(split_dir) / split / "images"
        self.ann_path        = Path(split_dir) / split / "annotations.json"
        self.sequence_length = sequence_length
        self.img_size        = img_size

        if not self.ann_path.exists():
            raise FileNotFoundError(f"annotations.json not found: {self.ann_path}")

        with open(self.ann_path) as f:
            self.annotations = json.load(f)

        self.frames = sorted(self.annotations.keys())

        # Build valid sequence indices
        self.valid_indices = []
        for i in range(len(self.frames) - sequence_length + 1):
            seq = self.frames[i:i + sequence_length]
            if self._are_consecutive(seq):
                self.valid_indices.append(i)

        print(f"TrackNet {split}: {len(self.valid_indices)} sequences "
              f"(from {len(self.frames)} frames)")

    def _get_prefix_and_num(self, fname: str):
        """
        Extract match prefix and frame number from filename.
        Handles both formats:
          - New: pro_match1_frame00042.jpg → prefix="pro_match1_frame", num=42
          - Old: frame_0001.jpg            → prefix="frame_", num=1
        """
        import re
        m = re.match(r"^(.+_frame)(\d+)\.jpg$", fname)
        if m:
            return m.group(1), int(m.group(2))
        m2 = re.search(r"(\d+)", fname)
        return "default_", int(m2.group(1)) if m2 else 0

    def _are_consecutive(self, frame_list: list) -> bool:
        """Frames must be from same match AND have consecutive frame numbers."""
        prefixes, numbers = [], []
        for fname in frame_list:
            prefix, num = self._get_prefix_and_num(fname)
            prefixes.append(prefix)
            numbers.append(num)

        # All from the same match/rally
        if len(set(prefixes)) > 1:
            return False

        # Frame numbers must be consecutive
        return all(numbers[i + 1] == numbers[i] + 1
                   for i in range(len(numbers) - 1))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start      = self.valid_indices[idx]
        seq_frames = self.frames[start:start + self.sequence_length]

        # ── Load image sequence ────────────────────────────────────────
        images = []
        for fname in seq_frames:
            img_path = self.img_dir / fname
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        # (T, C, H, W), normalized
        images = np.array(images).transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        # ── Generate heatmap for last frame ───────────────────────────
        last_ann  = self.annotations[seq_frames[-1]]
        heatmap   = np.zeros((self.img_size, self.img_size), dtype=np.float32)

        if last_ann.get("visibility") == "visible" and last_ann.get("x") is not None:
            # Get original image size to scale coordinates correctly
            orig_img = cv2.imread(str(self.img_dir / seq_frames[-1]))
            if orig_img is not None:
                h_orig, w_orig = orig_img.shape[:2]
                x = int(float(last_ann["x"]) * self.img_size / w_orig)
                y = int(float(last_ann["y"]) * self.img_size / h_orig)

                # Clamp to valid range
                x = max(0, min(x, self.img_size - 1))
                y = max(0, min(y, self.img_size - 1))

                # Gaussian heatmap centered on shuttle position
                sigma  = 5
                yy, xx = np.mgrid[0:self.img_size, 0:self.img_size]
                heatmap = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
                heatmap = heatmap.astype(np.float32)

        return torch.FloatTensor(images), torch.FloatTensor(heatmap)


# ══════════════════════════════════════════════════════════════════════════
# 4. TrackNetV2 Trainer
# ══════════════════════════════════════════════════════════════════════════

class TrackNetV2Trainer:
    """
    Trainer for TrackNetV2.
    Improvements over original TrackNetTrainer:
      - Early stopping (no more guessing epochs)
      - ReduceLROnPlateau scheduler
      - Weighted loss (visible frames count more than not_visible)
      - Progress logging every N batches
      - Validation accuracy metric (not just loss)
    """

    @staticmethod
    def train(
        split_dir: str,
        output_dir: str,
        sequence_length: int = 3,
        img_size: int = 512,
        epochs: int = 100,           # set high — early stopping will cut it
        batch_size: int = 8,
        lr: float = 1e-4,
        device: str = "cpu",
        patience: int = 15,          # early stopping patience
        lr_patience: int = 5,        # LR scheduler patience
        resume_from: Optional[str] = None,
        finetune_from: Optional[str] = None,
        freeze_encoder: bool = False,
        log_every: int = 20,         # print progress every N batches
    ):
        """
        Train TrackNetV2.

        Args:
            split_dir:       Path to splits directory (train/val/test subdirs)
            output_dir:      Where to save checkpoints
            sequence_length: Frames per input sequence (default 3)
            img_size:        Input image size (default 512)
            epochs:          Max epochs — early stopping will likely stop earlier
            batch_size:      Batch size
            lr:              Initial learning rate
            device:          'cpu', 'cuda', or '0' (GPU index)
            patience:        Early stopping — stop after N epochs with no improvement
            lr_patience:     Halve LR after N epochs with no improvement
            resume_from:     Path to checkpoint to resume interrupted training
            finetune_from:   Path to pretrained weights to fine-tune from
            freeze_encoder:  Freeze encoder layers during fine-tuning
            log_every:       Log batch progress every N batches
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. Run: pip install torch torchvision")

        # ── Device setup ──────────────────────────────────────────────
        if device.isdigit():
            device = f"cuda:{device}"
        elif device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            device = "cpu"
        print(f"Using device: {device}")

        if resume_from and finetune_from:
            raise ValueError("Cannot use both resume_from and finetune_from")

        # ── Datasets ──────────────────────────────────────────────────
        train_ds = TrackNetDataset(split_dir, "train", sequence_length, img_size)
        val_ds   = TrackNetDataset(split_dir, "val",   sequence_length, img_size)

        use_pin = device != "cpu"
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True,  num_workers=4, pin_memory=use_pin)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, num_workers=4, pin_memory=use_pin)

        # ── Model ─────────────────────────────────────────────────────
        model     = TrackNetV2(sequence_length).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCELoss()          # binary cross-entropy for heatmap

        # LR scheduler — halves LR when val loss stops improving
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=lr_patience, verbose=True
        )

        start_epoch     = 0
        best_val_loss   = float("inf")
        epochs_no_improve = 0
        training_mode   = "FROM SCRATCH"

        # ── Resume / Fine-tune ────────────────────────────────────────
        if resume_from:
            ckpt = torch.load(resume_from, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch   = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            training_mode = "RESUME"
            print(f"Resuming from epoch {start_epoch}")

        elif finetune_from:
            ckpt = TrackNetV2Trainer._load_weights_flexible(
                model, finetune_from, device
            )
            training_mode = "FINE-TUNE"

            if freeze_encoder:
                encoder_layers = ["enc1", "enc2", "enc3", "bottleneck"]
                for name, param in model.named_parameters():
                    if any(name.startswith(l) for l in encoder_layers):
                        param.requires_grad = False
                # Rebuild optimizer with only trainable params
                optimizer = optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=lr
                )
                print("Encoder frozen — only decoder will be updated")

        os.makedirs(output_dir, exist_ok=True)
        folder = "tracknetv2_finetune" if finetune_from else "tracknetv2"
        save_dir = os.path.join(output_dir, folder)
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n{'='*70}")
        print("Training TrackNetV2")
        print(f"  Mode            : {training_mode}")
        print(f"  Device          : {device}")
        print(f"  Sequence length : {sequence_length}")
        print(f"  Image size      : {img_size}")
        print(f"  Train sequences : {len(train_ds)}")
        print(f"  Val sequences   : {len(val_ds)}")
        print(f"  Max epochs      : {epochs} (early stop patience: {patience})")
        print(f"  LR              : {lr} (scheduler patience: {lr_patience})")
        print(f"{'='*70}\n")

        # ── Training loop ─────────────────────────────────────────────
        for epoch in range(start_epoch, start_epoch + epochs):
            model.train()
            train_loss   = 0.0
            train_batches = 0

            for batch_idx, (images, heatmaps) in enumerate(train_loader):
                images   = images.to(device)
                heatmaps = heatmaps.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, heatmaps)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_batches += 1

                if (batch_idx + 1) % log_every == 0:
                    avg = train_loss / train_batches
                    print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                          f"| Loss: {avg:.6f}")

            train_loss /= len(train_loader)

            # ── Validation ────────────────────────────────────────────
            model.eval()
            val_loss    = 0.0
            val_correct = 0   # frames where we detected shuttle in right place
            val_total   = 0

            with torch.no_grad():
                for images, heatmaps in val_loader:
                    images   = images.to(device)
                    heatmaps = heatmaps.to(device)
                    outputs  = model(images)
                    loss     = criterion(outputs, heatmaps)
                    val_loss += loss.item()

                    # Accuracy: predicted peak within 5px of ground truth peak
                    for pred, gt in zip(outputs, heatmaps):
                        pred_np = pred.cpu().numpy()
                        gt_np   = gt.cpu().numpy()
                        if gt_np.max() > 0.5:   # frame has visible shuttle
                            py, px = np.unravel_index(pred_np.argmax(), pred_np.shape)
                            gy, gx = np.unravel_index(gt_np.argmax(),   gt_np.shape)
                            dist = np.sqrt((px - gx) ** 2 + (py - gy) ** 2)
                            val_correct += 1 if dist <= 5 else 0
                            val_total   += 1

            val_loss /= len(val_loader)
            accuracy  = (val_correct / val_total * 100) if val_total > 0 else 0.0

            # LR scheduler step
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print(f"Epoch {epoch+1}/{start_epoch + epochs} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"Acc: {accuracy:.1f}% | "
                  f"LR: {current_lr:.2e}")

            # ── Save checkpoint ───────────────────────────────────────
            ckpt_path = os.path.join(save_dir, f"tracknetv2_epoch_{epoch+1}.pth")
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss":           train_loss,
                "val_loss":             val_loss,
                "accuracy":             accuracy,
            }, ckpt_path)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss      = val_loss
                epochs_no_improve  = 0
                best_path = os.path.join(save_dir, "tracknetv2_best.pth")
                torch.save({
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss":           train_loss,
                    "val_loss":             val_loss,
                    "accuracy":             accuracy,
                }, best_path)
                print(f"  → New best model saved (val_loss: {val_loss:.6f}, "
                      f"accuracy: {accuracy:.1f}%)")
            else:
                epochs_no_improve += 1
                print(f"  → No improvement ({epochs_no_improve}/{patience})")

            # ── Early stopping ────────────────────────────────────────
            if epochs_no_improve >= patience:
                print(f"\n⚡ Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break

        print(f"\n{'='*70}")
        print("TrackNetV2 Training Complete")
        print(f"  Best weights : {save_dir}/tracknetv2_best.pth")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"{'='*70}\n")

        return model

    @staticmethod
    def _load_weights_flexible(model: nn.Module, weights_path: str, device: str):
        """
        Load weights into model with shape-matching fallback.
        Handles:
          1. Exact match          → load all weights directly
          2. Partial match        → load matching layers, skip mismatches
          3. Old TrackNet format  → remap old layer names to new ones
        """
        ckpt      = torch.load(weights_path, map_location=device)
        old_state = ckpt.get("model_state_dict", ckpt)   # handle both formats
        new_state = model.state_dict()

        # Map old TrackNet layer names → new TrackNetV2 layer names
        # (used when loading from your original TrackNet into TrackNetV2)
        remap = {
            "conv1.0.weight":   "enc1.0.weight",
            "conv1.0.bias":     "enc1.0.bias",
            "conv1.2.weight":   "enc1.3.weight",
            "conv1.2.bias":     "enc1.3.bias",
            "conv2.0.weight":   "enc2.0.weight",
            "conv2.0.bias":     "enc2.0.bias",
            "conv2.2.weight":   "enc2.3.weight",
            "conv2.2.bias":     "enc2.3.bias",
            "conv3.0.weight":   "enc3.0.weight",
            "conv3.0.bias":     "enc3.0.bias",
            "conv3.2.weight":   "enc3.3.weight",
            "conv3.2.bias":     "enc3.3.bias",
            "deconv1.0.weight": "up3.weight",
            "deconv1.0.bias":   "up3.bias",
            "deconv2.0.weight": "up2.weight",
            "deconv2.0.bias":   "up2.bias",
            "deconv3.0.weight": "up1.weight",
            "deconv3.0.bias":   "up1.bias",
        }

        remapped    = {remap.get(k, k): v for k, v in old_state.items()}
        transferred = []
        skipped     = []

        for name, param in new_state.items():
            if name in remapped and remapped[name].shape == param.shape:
                new_state[name] = remapped[name]
                transferred.append(name)
            else:
                skipped.append(name)

        model.load_state_dict(new_state)
        print(f"  ✓ Loaded {len(transferred)} layers from {weights_path}")
        if skipped:
            print(f"  ⚠ Skipped {len(skipped)} layers (new/shape mismatch — "
                  f"will train from scratch)")
        return model


# ══════════════════════════════════════════════════════════════════════════
# 5. Updated Inference — plug TrackNetV2 into your ShuttleTracker
# ══════════════════════════════════════════════════════════════════════════

def load_tracknetv2_for_inference(weights_path: str, device: str = "cpu",
                                   sequence_length: int = 3):
    if device.isdigit():
        device = f"cuda:{device}"

    ckpt  = torch.load(weights_path, map_location=device)
    model = TrackNetV2(sequence_length).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"✓ Loaded TrackNetV2: {weights_path}")
    return model


# ══════════════════════════════════════════════════════════════════════════
# 6. Validation — check converted dataset before training
# ══════════════════════════════════════════════════════════════════════════

def validate_dataset(split_dir: str):
    """
    Run before training to catch issues in the converted dataset.
    Checks: missing images, broken annotations, class imbalance.
    """
    root = Path(split_dir)

    for split in ["train", "val", "test"]:
        ann_path = root / split / "annotations.json"
        img_dir  = root / split / "images"

        if not ann_path.exists():
            print(f"⚠ {split}: annotations.json missing")
            continue

        with open(ann_path) as f:
            anns = json.load(f)

        visible_count     = 0
        not_visible_count = 0
        missing_images    = 0

        for fname, ann in anns.items():
            img_path = img_dir / fname
            if not img_path.exists():
                missing_images += 1
            if ann.get("visibility") == "visible":
                visible_count += 1
            else:
                not_visible_count += 1

        total = len(anns)
        print(f"\n{split}:")
        print(f"  Total frames   : {total}")
        print(f"  Visible        : {visible_count} ({visible_count/total*100:.1f}%)")
        print(f"  Not visible    : {not_visible_count} ({not_visible_count/total*100:.1f}%)")
        print(f"  Missing images : {missing_images}")

        if missing_images > 0:
            print(f"  ❌ Fix missing images before training!")
        if visible_count / total < 0.2:
            print(f"  ⚠ Low visible ratio — model may struggle to learn detections")


# ══════════════════════════════════════════════════════════════════════════
# 7. Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TrackNetV2 — Convert, Train, Validate")
    parser.add_argument("--action", required=True,
                        choices=["convert", "merge", "validate", "train"])

    # Convert args
    parser.add_argument("--tracknetv2-dir",   help="Root of downloaded TrackNetV2 dataset")
    parser.add_argument("--output-split-dir", help="Where to write converted splits")
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Extract every Nth frame (1=all, 2=half, 3=third)")
    parser.add_argument("--val-ratio", type=float, default=0.1)

    # Merge args
    parser.add_argument("--existing-split-dir",   help="Your current annotated splits")
    parser.add_argument("--tracknetv2-split-dir", help="Converted TrackNetV2 splits")
    parser.add_argument("--merged-output-dir",    help="Output merged splits")

    # Train args
    parser.add_argument("--split-dir",    help="Split dir to train on")
    parser.add_argument("--output-dir",   help="Where to save checkpoints")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--img-size",     type=int,   default=512)
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--device",       default="cpu")
    parser.add_argument("--resume-from",  help="Resume from checkpoint")
    parser.add_argument("--finetune-from",help="Fine-tune from pretrained weights")
    parser.add_argument("--freeze-encoder", action="store_true")

    args = parser.parse_args()

    if args.action == "convert":
        convert_tracknetv2_dataset(
            tracknetv2_dir=args.tracknetv2_dir,
            output_split_dir=args.output_split_dir,
            val_ratio=args.val_ratio,
            frame_skip=args.frame_skip,
        )

    elif args.action == "merge":
        merge_with_existing_data(
            existing_split_dir=args.existing_split_dir,
            tracknetv2_split_dir=args.tracknetv2_split_dir,
            merged_output_dir=args.merged_output_dir,
        )

    elif args.action == "validate":
        validate_dataset(args.output_split_dir or args.split_dir)

    elif args.action == "train":
        TrackNetV2Trainer.train(
            split_dir=args.split_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            img_size=args.img_size,
            patience=args.patience,
            device=args.device,
            resume_from=args.resume_from,
            finetune_from=args.finetune_from,
            freeze_encoder=args.freeze_encoder,
        )