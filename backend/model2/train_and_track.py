import os
import yaml
import json
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠ ultralytics not installed. YOLO training disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not installed. TrackNet training disabled.")

def get_device():
    """Auto-detect best available device."""
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():  # Works for both NVIDIA and AMD ROCm
            return "0"                 # GPU device 0
        else:
            return "cpu"
    return "cpu"

DEVICE = get_device()
# ══════════════════════════════════════════════════════════════════════════
# 1. YOLO Training (Standard + OBB)
# ══════════════════════════════════════════════════════════════════════════

class YOLOTrainer:
    """Train YOLOv8/v11 for shuttle detection (standard or OBB mode)."""
    
    @staticmethod
    def create_dataset_yaml(split_dir: str, use_obb: bool = False) -> str:
        """Generate dataset.yaml for YOLO training.
        
        Args:
            split_dir: Path to splits/ directory (contains train/val/test)
            use_obb: If True, use obb_labels; else use standard labels
        
        Returns:
            Path to generated dataset.yaml
        """
        yaml_path = os.path.join(split_dir, "dataset.yaml")
        
        if use_obb:
            # For OBB, YOLO expects labels in 'labels/' subdirectory
            # Our OBB labels are in 'obb_labels/', so we need to link/copy them
            import shutil
            
            for split in ['train', 'val', 'test']:
                labels_dir = os.path.join(split_dir, split, 'labels')
                obb_labels_dir = os.path.join(split_dir, split, 'obb_labels')
                
                if not os.path.exists(obb_labels_dir):
                    print(f"⚠ Warning: {split}/obb_labels not found")
                    continue
                
                # Remove existing labels if it's pointing to wrong place
                if os.path.exists(labels_dir):
                    if os.path.islink(labels_dir):
                        os.unlink(labels_dir)
                    elif os.path.isdir(labels_dir):
                        # Check if it's already our OBB labels
                        if os.path.samefile(labels_dir, obb_labels_dir):
                            continue  # Already correct
                        else:
                            shutil.rmtree(labels_dir)
                
                # Create symlink or copy
                try:
                    # Try symlink first (Windows requires admin or dev mode)
                    os.symlink(os.path.abspath(obb_labels_dir), 
                             labels_dir, 
                             target_is_directory=True)
                    print(f"  ✓ Symlinked {split}/labels → obb_labels")
                except (OSError, NotImplementedError):
                    # Fall back to copying
                    shutil.copytree(obb_labels_dir, labels_dir)
                    print(f"  ✓ Copied {split}/obb_labels → labels")
        
        # Build config
        config = {
            "path": os.path.abspath(split_dir),
            "train": "train/images",
            "val": "val/images", 
            "test": "test/images",
            "nc": 1,
            "names": ["shuttle"],
        }
        
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Created {yaml_path}")
        print(f"  Mode: {'OBB' if use_obb else 'Standard'}")
        if use_obb:
            print(f"  YOLO will use labels/ (pointing to obb_labels/)")
        return yaml_path
    
    @staticmethod
    def train_standard(
        split_dir: str,
        output_dir: str,
        model_size: str = "n",          # n, s, m, l, x
        epochs: int = 10,
        imgsz: int = 640,
        batch: int = 8,
        device: str = DEVICE,
        pretrained: bool = True,
        yolo_version: str = "8",
        resume_from: Optional[str] = None,      # Resume interrupted training
        finetune_from: Optional[str] = None,    # Fine-tune from pre-trained weights
        freeze_layers: int = 0,                 # Number of layers to freeze for fine-tuning
        **kwargs
    ):
        """Train standard YOLOv8/v11 with axis-aligned boxes.
        
        Best for:
          • Clear, slow shuttles
          • Real-time inference requirements
          • General-purpose detection
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
        
        output_dir = os.path.abspath(output_dir)
        yaml_path = YOLOTrainer.create_dataset_yaml(split_dir, use_obb=False)

        # Fine-tuning mode: load your own model
        if resume_from:
            if not os.path.exists(resume_from):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
            print(f"Resuming training from: {resume_from}")
            model = YOLO(resume_from)
            training_mode = "RESUME"
            use_pretrained = False

        # Mode 2: Fine-tune from existing weights
        elif finetune_from:
            if not os.path.exists(finetune_from):
                raise FileNotFoundError(f"Fine-tune weights not found: {finetune_from}")
            print(f"Fine-tuning from: {finetune_from}")
            model = YOLO(finetune_from)
            training_mode = "FINE-TUNE"
            use_pretrained = False
            
            # Freeze layers if requested
            if freeze_layers > 0:
                print(f"  Freezing first {freeze_layers} layers")
                for i, (name, param) in enumerate(model.model.named_parameters()):
                    if i < freeze_layers:
                        param.requires_grad = False

        # Standard mode: pretrained or from scratch
        elif pretrained:
            try:
                model_name = f"yolo{yolo_version}{model_size}.pt"
                model = YOLO(model_name)
                print(f"Using pretrained YOLOv{yolo_version}{model_size}")
                training_mode = "PRETRAINED"
            except Exception as e:
                print(f"⚠ YOLOv{yolo_version} not available, falling back to YOLOv8")
                model_name = f"yolov8{model_size}.pt"
                model = YOLO(model_name)
                yolo_version = "8"
                training_mode = "PRETRAINED"
            use_pretrained = True
        else:
            model_name = f"yolo{yolo_version}{model_size}.yaml"
            model = YOLO(model_name)
            training_mode = "FROM-SCRATCH"
            use_pretrained = False
        
        print(f"\n{'='*70}")
        print(f"Training YOLOv{yolo_version}{model_size.upper()} - Standard Detection")
        print(f"  Mode: {training_mode}")
        if freeze_layers > 0:
            print(f"  Frozen layers: {freeze_layers}")
        print(f"{'='*70}")
        
        # Train
        results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project=output_dir,
                name="yolo_standard",
                exist_ok=True,
                pretrained=use_pretrained,
                resume=bool(resume_from),  # Only resume if resume_from is set
                **kwargs
            )
        
        # Validate
        metrics = model.val()
        
        print(f"\n{'='*70}")
        print("Training Complete - Standard YOLO")
        print(f"{'='*70}")
        print(f"  Best weights: {output_dir}/yolo_standard/weights/best.pt")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"{'='*70}\n")
        
        return model, metrics
    
    @staticmethod
    def train_obb(
        split_dir: str,
        output_dir: str,
        model_size: str = "n",
        epochs: int = 10,
        imgsz: int = 640,
        batch: int = 8,
        device: str = DEVICE,
        pretrained: bool = True,
        yolo_version: str = "8",
        resume_from: Optional[str] = None,
        finetune_from: Optional[str] = None,
        freeze_layers: int = 0,
        **kwargs
    ):
        """Train YOLOv8/v11-OBB with oriented bounding boxes.
        
        Best for:
          • Motion-blurred shuttles
          • Fast rallies with elongated streaks
          • Precise orientation estimation
        
        Note: Requires YOLOv8/v11 OBB variant
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
        
        output_dir = os.path.abspath(output_dir)
        yaml_path = YOLOTrainer.create_dataset_yaml(split_dir, use_obb=True)

        # Determine training mode
        if resume_from and finetune_from:
            raise ValueError("Cannot use both resume_from and finetune_from simultaneously")
        

        # Mode 1: Resume interrupted training
        if resume_from:
            if not os.path.exists(resume_from):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
            print(f"Resuming training from: {resume_from}")
            model = YOLO(resume_from)
            training_mode = "RESUME"
            use_pretrained = False
        
        # Mode 2: Fine-tune from existing weights
        elif finetune_from:
            if not os.path.exists(finetune_from):
                raise FileNotFoundError(f"Fine-tune weights not found: {finetune_from}")
            print(f"Fine-tuning from: {finetune_from}")
            model = YOLO(finetune_from)
            training_mode = "FINE-TUNE"
            use_pretrained = False
            
            # Freeze layers if requested
            if freeze_layers > 0:
                print(f"  Freezing first {freeze_layers} layers")
                for i, (name, param) in enumerate(model.model.named_parameters()):
                    if i < freeze_layers:
                        param.requires_grad = False


        # Mode 3: Train from pretrained or scratch
        elif pretrained:
            try:
                model_name = f"yolo{yolo_version}{model_size}-obb.pt"
                model = YOLO(model_name)
                print(f"Using pretrained YOLOv{yolo_version}{model_size}-OBB")
                training_mode = "PRETRAINED"
            except Exception as e:
                print(f"⚠ YOLOv{yolo_version}-OBB not available, falling back to YOLOv8-OBB")
                model_name = f"yolov8{model_size}-obb.pt"
                model = YOLO(model_name)
                yolo_version = "8"
                training_mode = "PRETRAINED"
            use_pretrained = True
        else:
            model_name = f"yolo{yolo_version}{model_size}-obb.yaml"
            model = YOLO(model_name)
            training_mode = "FROM-SCRATCH"
            use_pretrained = False
        
        print(f"\n{'='*70}")
        print(f"Training YOLOv{yolo_version}{model_size.upper()}-OBB - Oriented Detection")
        print(f"  Mode: {training_mode}")
        if freeze_layers > 0:
            print(f"  Frozen layers: {freeze_layers}")
        print(f"{'='*70}")
        
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=output_dir,
            name="yolo_obb",
            exist_ok=True,
            pretrained=use_pretrained,
            resume=bool(resume_from),
            **kwargs
        )
        
        metrics = model.val()
        
        print(f"\n{'='*70}")
        print("Training Complete - YOLOv11-OBB")
        print(f"{'='*70}")
        print(f"  Best weights: {output_dir}/yolo_obb/weights/best.pt")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"{'='*70}\n")
        
        return model, metrics


# ══════════════════════════════════════════════════════════════════════════
# 2. TrackNet Training (Temporal Heatmap Tracking)
# ══════════════════════════════════════════════════════════════════════════

class TrackNetDataset(Dataset):
    """Dataset for TrackNet training (multi-frame input → heatmap output)."""
    
    def __init__(self, split_dir: str, split: str = "train",
                 sequence_length: int = 3, img_size: int = 512):
        """
        Args:
            split_dir: Path to splits/ directory
            split: 'train', 'val', or 'test'
            sequence_length: Number of consecutive frames (default: 3)
            img_size: Input image size
        """
        self.img_dir = os.path.join(split_dir, split, "images")
        self.ann_path = os.path.join(split_dir, split, "annotations.json")
        self.sequence_length = sequence_length
        self.img_size = img_size
        
        with open(self.ann_path) as f:
            self.annotations = json.load(f)
        
        self.frames = sorted(self.annotations.keys())
        
        # Only use frames where we have sequence_length consecutive frames
        self.valid_indices = []
        for i in range(len(self.frames) - sequence_length + 1):
            # Check if frames are consecutive (frame_0001, frame_0002, frame_0003)
            if self._are_consecutive(self.frames[i:i + sequence_length]):
                self.valid_indices.append(i)
        
        print(f"TrackNet {split}: {len(self.valid_indices)} sequences "
              f"(from {len(self.frames)} frames)")
    
    def _are_consecutive(self, frame_list):
        """Check if frame names are sequential."""
        import re
        numbers = []
        for frame in frame_list:
            m = re.findall(r'\d+', frame)
            if not m:
                return False
            numbers.append(int(m[-1]))
        
        for i in range(len(numbers) - 1):
            if numbers[i + 1] != numbers[i] + 1:
                return False
        return True
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        sequence_frames = self.frames[start_idx:start_idx + self.sequence_length]
        
        # Load images
        images = []
        for frame_name in sequence_frames:
            img_path = os.path.join(self.img_dir, frame_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        
        images = np.array(images).transpose(0, 3, 1, 2)  # (T, C, H, W)
        images = images.astype(np.float32) / 255.0
        
        # Generate heatmap from last frame's annotation
        last_frame = sequence_frames[-1]
        ann = self.annotations[last_frame]
        
        heatmap = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        if ann.get("visibility") in ("visible", "occluded") and "x" in ann:
            # Get original image size to scale coordinates
            orig_img = cv2.imread(os.path.join(self.img_dir, last_frame))
            h_orig, w_orig = orig_img.shape[:2]
            
            # Scale to img_size
            x = int(ann["x"] * self.img_size / w_orig)
            y = int(ann["y"] * self.img_size / h_orig)
            
            # Gaussian heatmap (sigma = 5 pixels)
            sigma = 5
            for i in range(max(0, y - 3*sigma), min(self.img_size, y + 3*sigma)):
                for j in range(max(0, x - 3*sigma), min(self.img_size, x + 3*sigma)):
                    dist_sq = (i - y)**2 + (j - x)**2
                    heatmap[i, j] = np.exp(-dist_sq / (2 * sigma**2))
        
        return torch.FloatTensor(images), torch.FloatTensor(heatmap)


class TrackNet(nn.Module):
    """TrackNet architecture (VGG-based encoder-decoder for heatmap prediction)."""
    
    def __init__(self, sequence_length=3):
        super().__init__()
        
        # Encoder (VGG-style)
        self.conv1 = nn.Sequential(
            nn.Conv2d(sequence_length * 3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU()
        )
        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU()
        )
        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU()
        )
        
        # Output heatmap
        self.output = nn.Conv2d(32, 1, 1)
    
    def forward(self, x):
        # x: (B, T, 3, H, W) → flatten temporal + channel dim
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        
        # Encode
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        # Decode
        x = self.deconv1(x3)
        x = self.deconv2(x)
        x = self.deconv3(x)
        
        # Heatmap
        x = self.output(x)
        x = torch.sigmoid(x)
        
        return x.squeeze(1)  # (B, H, W)


class TrackNetTrainer:
    """Train TrackNet for temporal shuttle tracking."""
    
    @staticmethod
    def train(
        split_dir: str,
        output_dir: str,
        sequence_length: int = 3,
        img_size: int = 512,
        epochs: int = 10,
        batch_size: int = 8,
        lr: float = 1e-4,
        device: str = DEVICE,
        resume_from: Optional[str] = None,      # Resume interrupted training
        finetune_from: Optional[str] = None,    # Fine-tune from pre-trained weights
        freeze_encoder: bool = False 
    ):
        """Train TrackNet model.
        
        Args:
            split_dir: Path to rally-aware splits
            output_dir: Where to save model checkpoints
            sequence_length: Number of input frames (default: 3)
            img_size: Input image size
            epochs: Training epochs
            batch_size: Batch size
            lr: Learning rate
            device: 'cuda' or 'cpu'
            resume_from: Resume interrupted training from checkpoint
            finetune_from: Fine-tune from pre-trained model
            freeze_encoder: If True, freeze encoder layers during fine-tuning
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed. Run: pip install torch torchvision")
        
        if device.isdigit():  # If device is "0", "1", etc.
            device = f"cuda:{device}"
            print(f"Using device: {device}")
        elif device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            device = "cpu"
        
        os.makedirs(output_dir, exist_ok=True)

        # Determine training mode
        if resume_from and finetune_from:
            raise ValueError("Cannot use both resume_from and finetune_from simultaneously")
        
        # Datasets
        try:
            train_dataset = TrackNetDataset(split_dir, "train", sequence_length, img_size)
            val_dataset = TrackNetDataset(split_dir, "val", sequence_length, img_size)
        except Exception as e:
            print(f"❌ Failed to load datasets: {str(e)}")
            raise
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=4, pin_memory=True if device == "cuda" else False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=4, pin_memory=True if device == "cuda" else False)
        
        # Model
        model = TrackNet(sequence_length).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        start_epoch = 0
        best_val_loss = float('inf')
        training_mode = "FROM SCRATCH"
        
        # Mode 1: Resume interrupted training
        if resume_from:
            if not os.path.exists(resume_from):
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
            
            print(f"  Loading checkpoint from: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            training_mode = "RESUME"
            print(f"  Resuming from epoch {start_epoch}")
        
        # Mode 2: Fine-tune from pre-trained weights
        elif finetune_from:
            if not os.path.exists(finetune_from):
                raise FileNotFoundError(f"Fine-tune checkpoint not found: {finetune_from}")
            
            print(f"  Loading pre-trained weights from: {finetune_from}")
            checkpoint = torch.load(finetune_from, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            training_mode = "FINE-TUNE"
            
            # Freeze encoder if requested
            if freeze_encoder:
                print(f"  Freezing encoder layers")
                for name, param in model.named_parameters():
                    if 'conv1' in name or 'conv2' in name or 'conv3' in name:
                        param.requires_grad = False
                
                # Re-create optimizer with only trainable parameters
                optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
            
            print(f"  Starting fine-tuning from epoch 0")
        
        output_dir = os.path.join(output_dir, "tracknet")
        os.makedirs(output_dir, exist_ok=True)


        print(f"\n{'='*70}")
        print("Training TrackNet")
        print(f"  Mode: {training_mode}")
        if freeze_encoder:
            print(f"  Encoder: FROZEN")
        print(f"{'='*70}")
        print(f"  Device: {device}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Image size: {img_size}")
        print(f"  Train sequences: {len(train_dataset)}")
        print(f"  Val sequences: {len(val_dataset)}")
        print(f"{'='*70}\n")
        
        for epoch in range(start_epoch, start_epoch + epochs):
            # Train
            model.train()
            train_loss = 0.0
            
            for images, heatmaps in train_loader:
                images = images.to(device)
                heatmaps = heatmaps.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, heatmaps)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

                # Print progress every 10 batches
                
            
            train_loss /= len(train_loader)
            
            # Validate
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, heatmaps in val_loader:
                    images = images.to(device)
                    heatmaps = heatmaps.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, heatmaps)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}  Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}")
            
            # Save checkpoint every epoch
            checkpoint_path = os.path.join(output_dir, f'tracknet_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(output_dir, 'tracknet_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, best_path)
                print(f"  → Saved best model (val_loss: {val_loss:.6f})")
        
        print(f"\n{'='*70}")
        print("TrackNet Training Complete")
        print(f"{'='*70}")
        print(f"  Best weights: {output_dir}/tracknet/tracknet_best.pth")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"{'='*70}\n")
        
        return model


# ══════════════════════════════════════════════════════════════════════════
# 3. Inference on New Videos
# ══════════════════════════════════════════════════════════════════════════

class ShuttleTracker:
    """Unified inference for YOLO, TrackNet, and hybrid tracking."""
    
    def __init__(self, yolo_weights=None, obb_weights=None, tracknet_weights=None,
                 device=DEVICE):
        """
        Args:
            yolo_weights: Path to YOLO standard weights
            obb_weights: Path to YOLO-OBB weights
            tracknet_weights: Path to TrackNet weights
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.yolo_model = None
        self.obb_model = None
        self.tracknet_model = None

        if self.device.isdigit():  # If device is "0", "1", etc.
            self.device = f"cuda:{self.device}"
            print(f"Using device: {self.device}")
        elif self.device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        if yolo_weights and YOLO_AVAILABLE:
            self.yolo_model = YOLO(yolo_weights)
            print(f"✓ Loaded YOLO: {yolo_weights}")
        
        if obb_weights and YOLO_AVAILABLE:
            self.obb_model = YOLO(obb_weights)
            print(f"✓ Loaded YOLO-OBB: {obb_weights}")
        
        if tracknet_weights and TORCH_AVAILABLE:
            checkpoint = torch.load(tracknet_weights, map_location=self.device)
            self.tracknet_model = TrackNet(sequence_length=3).to(self.device)
            self.tracknet_model.load_state_dict(checkpoint['model_state_dict'])
            self.tracknet_model.eval()
            print(f"✓ Loaded TrackNet: {tracknet_weights}")
        
        self.frame_buffer = deque(maxlen=3)  # For TrackNet multi-frame input
    
    def track_video(self, video_path: str, output_path: str,
                    mode: str = "hybrid", conf_threshold: float = 0.25,
                    show_trail: bool = True, trail_length: int = 30):
        """Track shuttle in video and save annotated output.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            mode: 'yolo' | 'obb' | 'tracknet' | 'hybrid'
            conf_threshold: Confidence threshold for YOLO detections
            show_trail: Show shuttle trajectory trail
            trail_length: Number of past positions to show
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        trail = deque(maxlen=trail_length)
        frame_idx = 0
        
        print(f"\nTracking video: {video_path}")
        print(f"  Mode: {mode}")
        print(f"  FPS: {fps}  Resolution: {width}x{height}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detection/prediction
            if mode == "yolo" and self.yolo_model:
                position = self._detect_yolo(frame, self.yolo_model, conf_threshold)
            
            elif mode == "obb" and self.obb_model:
                position = self._detect_yolo(frame, self.obb_model, conf_threshold, is_obb=True)
            
            elif mode == "tracknet" and self.tracknet_model:
                position = self._detect_tracknet(frame)
            
            elif mode == "hybrid":
                # Use YOLO for detection, TrackNet for refinement
                yolo_pos = None
                if self.obb_model:
                    yolo_pos = self._detect_yolo(frame, self.obb_model, conf_threshold, is_obb=True)
                elif self.yolo_model:
                    yolo_pos = self._detect_yolo(frame, self.yolo_model, conf_threshold)
                
                tracknet_pos = self._detect_tracknet(frame) if self.tracknet_model else None
                
                # Fusion: if both agree (within 50px), use TrackNet (more precise)
                # Otherwise use YOLO (more robust)
                if yolo_pos and tracknet_pos:
                    dist = np.sqrt((yolo_pos[0] - tracknet_pos[0])**2 +
                                  (yolo_pos[1] - tracknet_pos[1])**2)
                    position = tracknet_pos if dist < 50 else yolo_pos
                else:
                    position = yolo_pos or tracknet_pos
            else:
                position = None
            
            # Visualize
            if position:
                x, y = int(position[0]), int(position[1])
                trail.append((x, y))
                
                # Draw current position
                cv2.circle(frame, (x, y), 12, (0, 255, 0), 2)
                
            # Info overlay
            cv2.putText(frame, f"Frame: {frame_idx}  Mode: {mode.upper()}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if position:
                cv2.putText(frame, f"Position: ({int(position[0])}, {int(position[1])})",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames...")
        
        cap.release()
        out.release()
        
        print(f"✓ Tracking complete → {output_path}")
        print(f"  Frames processed: {frame_idx}")
    
    def _detect_yolo(self, frame, model, conf_threshold, is_obb=False):
        results = model(frame, conf=conf_threshold, verbose=False)
        r = results[0]

        if is_obb:
            if not hasattr(r, "obb") or r.obb is None or r.obb.conf is None:
                return None

            confs = r.obb.conf.cpu().numpy()
            if len(confs) == 0:
                return None

            boxes = r.obb.xyxyxyxy.cpu().numpy()
            best_idx = np.argmax(confs)
            corners = boxes[best_idx]

            cx = corners[:, 0].mean()
            cy = corners[:, 1].mean()
            return (cx, cy)

        else:
            if r.boxes is None or r.boxes.conf is None:
                return None

            confs = r.boxes.conf.cpu().numpy()
            if len(confs) == 0:
                return None

            boxes = r.boxes.xyxy.cpu().numpy()
            best_idx = np.argmax(confs)
            x1, y1, x2, y2 = boxes[best_idx]
            return ((x1 + x2) / 2, (y1 + y2) / 2)

    
    def _detect_tracknet(self, frame):
        """Run TrackNet prediction on frame."""
        if self.tracknet_model is None:
            return None
        
        # Add to buffer
        self.frame_buffer.append(frame.copy())
        
        if len(self.frame_buffer) < 3:
            return None
        
        # Prepare input
        frames = list(self.frame_buffer)
        img_size = 512
        
        images = []
        for f in frames:
            img = cv2.resize(f, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        
        images = np.array(images).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        images = torch.FloatTensor(images).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            heatmap = self.tracknet_model(images).cpu().numpy()[0]
        
        # Find peak
        y_max, x_max = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        # Scale back to original frame size
        h, w = frame.shape[:2]
        x = x_max * w / img_size
        y = y_max * h / img_size
        
        # Only return if heatmap confidence is high
        if heatmap[y_max, x_max] > 0.5:
            return (x, y)
        return None


# ══════════════════════════════════════════════════════════════════════════
# 4. Main Entry Point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and track shuttle")
    parser.add_argument("--action", choices=["train-yolo", "train-obb", "train-tracknet", "track"],
                       required=True)
    
    # Training args
    parser.add_argument("--split-dir", help="Path to splits/ directory")
    parser.add_argument("--output-dir", help="Where to save trained models")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--yolo-version", default="11", choices=["8", "11"],
                       help="YOLO version to use (default:11, more stable)")
    
    # Fine-tuning args
    parser.add_argument("--resume-from", help="Resume interrupted training from checkpoint")
    parser.add_argument("--finetune-from", help="Fine-tune from pre-trained weights")
    parser.add_argument("--freeze-layers", type=int, default=0,
                       help="Number of layers to freeze for YOLO fine-tuning (0-10)")
    parser.add_argument("--freeze-encoder", action="store_true",
                       help="Freeze encoder layers for TrackNet fine-tuning")
    
    # TrackNet specific
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--tracknet-img-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for TrackNet")
    
    # Inference args
    parser.add_argument("--video", help="Input video for tracking")
    parser.add_argument("--output-video", help="Output tracked video")
    parser.add_argument("--yolo-weights", help="Path to YOLO weights")
    parser.add_argument("--obb-weights", help="Path to OBB weights")
    parser.add_argument("--tracknet-weights", help="Path to TrackNet weights")
    parser.add_argument("--mode", default="hybrid", choices=["yolo", "obb", "tracknet", "hybrid"])
    parser.add_argument("--conf", type=float, default=0.25)
    
    args = parser.parse_args()
    
    try:
        if args.action == "train-yolo":
            YOLOTrainer.train_standard(
                split_dir=args.split_dir,
                output_dir=args.output_dir,
                model_size=args.model_size,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                yolo_version=args.yolo_version,
                resume_from=args.resume_from,
                finetune_from=args.finetune_from,
                freeze_layers=args.freeze_layers
            )
        
        elif args.action == "train-obb":
            YOLOTrainer.train_obb(
                split_dir=args.split_dir,
                output_dir=args.output_dir,
                model_size=args.model_size,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                yolo_version=args.yolo_version,
                resume_from=args.resume_from,
                finetune_from=args.finetune_from,
                freeze_layers=args.freeze_layers
            )
        
        elif args.action == "train-tracknet":
            TrackNetTrainer.train(
                split_dir=args.split_dir,
                output_dir=args.output_dir,
                sequence_length=args.sequence_length,
                img_size=args.tracknet_img_size,
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                device=args.device,
                resume_from=args.resume_from,
                finetune_from=args.finetune_from,
                freeze_encoder=args.freeze_encoder
            )
        
        elif args.action == "track":
            tracker = ShuttleTracker(
                yolo_weights=args.yolo_weights,
                obb_weights=args.obb_weights,
                tracknet_weights=args.tracknet_weights,
                device=args.device
            )
            
            tracker.track_video(
                video_path=args.video,
                output_path=args.output_video,
                mode=args.mode,
                conf_threshold=args.conf
            )
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)