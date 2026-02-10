import os
import yaml
import json
import cv2
import numpy as np
from pathlib import Path
from collections import deque
import pickle
from datetime import datetime

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
    from torch.utils.data import Dataset, DataLoader, ConcatDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch not installed. TrackNet training disabled.")


# ══════════════════════════════════════════════════════════════════════════
# NEW: Data Management System
# ══════════════════════════════════════════════════════════════════════════

class DataRegistry:
    """Manages multiple video datasets and their metadata for incremental training."""
    
    def __init__(self, registry_path: str = "./data_registry.json"):
        """
        Args:
            registry_path: Path to JSON file tracking all datasets
        """
        self.registry_path = registry_path
        self.datasets = self._load_registry()
    
    def _load_registry(self):
        """Load existing registry or create new one."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path) as f:
                return json.load(f)
        return {
            "datasets": {},
            "training_history": [],
            "active_datasets": []
        }
    
    def save(self):
        """Persist registry to disk."""
        with open(self.registry_path, 'w') as f:
            json.dump(self.datasets, f, indent=2)
        print(f"✓ Registry saved to {self.registry_path}")
    
    def add_dataset(self, name: str, split_dir: str, metadata: dict = None):
        """Register a new dataset.
        
        Args:
            name: Unique dataset identifier (e.g., 'indoor_fast', 'outdoor_slow')
            split_dir: Path to the splits/ directory
            metadata: Optional dict with video characteristics:
                - lighting: 'indoor', 'outdoor', 'mixed'
                - speed: 'slow', 'medium', 'fast'
                - resolution: '720p', '1080p', '4k'
                - camera_angle: 'side', 'top', 'diagonal'
                - players: 'singles', 'doubles'
                - court_type: 'wood', 'synthetic', 'concrete'
        """
        if name in self.datasets["datasets"]:
            print(f"⚠ Dataset '{name}' already exists. Use update_dataset() to modify.")
            return
        
        # Validate split_dir exists
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Auto-detect some metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Count frames
        train_img_dir = os.path.join(split_dir, "train", "images")
        if os.path.exists(train_img_dir):
            metadata["train_frames"] = len([f for f in os.listdir(train_img_dir) if f.endswith(('.jpg', '.png'))])
        
        self.datasets["datasets"][name] = {
            "split_dir": os.path.abspath(split_dir),
            "metadata": metadata,
            "added_date": datetime.now().isoformat(),
            "trained_models": []  # Track which models used this data
        }
        
        # Add to active datasets by default
        if name not in self.datasets["active_datasets"]:
            self.datasets["active_datasets"].append(name)
        
        self.save()
        print(f"✓ Added dataset '{name}'")
        print(f"  Location: {split_dir}")
        print(f"  Metadata: {metadata}")
    
    def remove_dataset(self, name: str, delete_files: bool = False):
        """Remove dataset from registry.
        
        Args:
            name: Dataset name
            delete_files: If True, also delete the actual files (dangerous!)
        """
        if name not in self.datasets["datasets"]:
            print(f"⚠ Dataset '{name}' not found")
            return
        
        dataset_info = self.datasets["datasets"][name]
        
        if delete_files:
            import shutil
            split_dir = dataset_info["split_dir"]
            if os.path.exists(split_dir):
                shutil.rmtree(split_dir)
                print(f"✓ Deleted files: {split_dir}")
        
        del self.datasets["datasets"][name]
        
        if name in self.datasets["active_datasets"]:
            self.datasets["active_datasets"].remove(name)
        
        self.save()
        print(f"✓ Removed dataset '{name}' from registry")
    
    def set_active_datasets(self, names: list):
        """Choose which datasets to use for training.
        
        Args:
            names: List of dataset names to activate
        """
        for name in names:
            if name not in self.datasets["datasets"]:
                raise ValueError(f"Dataset '{name}' not found in registry")
        
        self.datasets["active_datasets"] = names
        self.save()
        print(f"✓ Active datasets: {names}")
    
    def get_active_split_dirs(self):
        """Get split directories for all active datasets."""
        return [
            self.datasets["datasets"][name]["split_dir"]
            for name in self.datasets["active_datasets"]
        ]
    
    def get_dataset_info(self, name: str):
        """Get information about a specific dataset."""
        return self.datasets["datasets"].get(name)
    
    def list_datasets(self):
        """Print all registered datasets."""
        print("\n" + "="*70)
        print("REGISTERED DATASETS")
        print("="*70)
        
        if not self.datasets["datasets"]:
            print("  No datasets registered yet.")
            return
        
        for name, info in self.datasets["datasets"].items():
            active = "✓ ACTIVE" if name in self.datasets["active_datasets"] else "  inactive"
            print(f"\n{active} | {name}")
            print(f"  Path: {info['split_dir']}")
            print(f"  Added: {info['added_date']}")
            if info['metadata']:
                print(f"  Metadata: {info['metadata']}")
            if info['trained_models']:
                print(f"  Used in models: {info['trained_models']}")
        
        print("\n" + "="*70 + "\n")
    
    def record_training(self, model_type: str, model_path: str, datasets_used: list, metrics: dict):
        """Record a training session in history.
        
        Args:
            model_type: 'yolo', 'obb', or 'tracknet'
            model_path: Path to saved weights
            datasets_used: List of dataset names used
            metrics: Dict of performance metrics
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "model_path": model_path,
            "datasets": datasets_used,
            "metrics": metrics
        }
        
        self.datasets["training_history"].append(record)
        
        # Update each dataset's trained_models list
        for ds_name in datasets_used:
            if ds_name in self.datasets["datasets"]:
                if model_path not in self.datasets["datasets"][ds_name]["trained_models"]:
                    self.datasets["datasets"][ds_name]["trained_models"].append(model_path)
        
        self.save()


# ══════════════════════════════════════════════════════════════════════════
# UPDATED: Multi-Dataset YOLO Training
# ══════════════════════════════════════════════════════════════════════════

class YOLOTrainerV2:
    """Enhanced YOLO trainer with multi-dataset support and incremental learning."""
    
    @staticmethod
    def create_combined_dataset_yaml(split_dirs: list, output_dir: str, use_obb: bool = False) -> str:
        """Merge multiple datasets into one YOLO-compatible structure.
        
        Args:
            split_dirs: List of paths to splits/ directories
            output_dir: Where to create the combined dataset
            use_obb: Use OBB labels instead of standard
        
        Returns:
            Path to generated dataset.yaml
        """
        import shutil
        
        os.makedirs(output_dir, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            combined_img_dir = os.path.join(output_dir, split, 'images')
            combined_lbl_dir = os.path.join(output_dir, split, 'labels')
            
            os.makedirs(combined_img_dir, exist_ok=True)
            os.makedirs(combined_lbl_dir, exist_ok=True)
            
            # Merge all datasets
            for i, split_dir in enumerate(split_dirs):
                src_img_dir = os.path.join(split_dir, split, 'images')
                src_lbl_dir = os.path.join(split_dir, split, 'obb_labels' if use_obb else 'labels')
                
                if not os.path.exists(src_img_dir):
                    print(f"⚠ Skipping {split_dir}/{split} - not found")
                    continue
                
                # Copy with prefix to avoid name collisions
                dataset_prefix = f"ds{i}_"
                
                for img_file in os.listdir(src_img_dir):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        shutil.copy(
                            os.path.join(src_img_dir, img_file),
                            os.path.join(combined_img_dir, dataset_prefix + img_file)
                        )
                
                if os.path.exists(src_lbl_dir):
                    for lbl_file in os.listdir(src_lbl_dir):
                        if lbl_file.endswith('.txt'):
                            shutil.copy(
                                os.path.join(src_lbl_dir, lbl_file),
                                os.path.join(combined_lbl_dir, dataset_prefix + lbl_file)
                            )
                
                print(f"  ✓ Merged dataset {i+1}/{len(split_dirs)} - {split} split")
        
        # Create YAML
        yaml_path = os.path.join(output_dir, "dataset.yaml")
        config = {
            "path": os.path.abspath(output_dir),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 1,
            "names": ["shuttle"],
        }
        
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Combined dataset created: {output_dir}")
        print(f"  Mode: {'OBB' if use_obb else 'Standard'}")
        return yaml_path
    
    @staticmethod
    def train_incremental(
        registry: DataRegistry,
        output_dir: str,
        model_type: str = "standard",  # 'standard' or 'obb'
        model_size: str = "n",
        epochs: int = 10,
        imgsz: int = 640,
        batch: int = 8,
        device: str = "0",
        pretrained_weights: str = None,  # Path to existing model to continue training
        freeze_layers: int = 0,  # Number of layers to freeze (for transfer learning)
        yolo_version: str = "8",
        **kwargs
    ):
        """Train YOLO on multiple datasets with optional incremental learning.
        
        Args:
            registry: DataRegistry instance with active datasets
            output_dir: Where to save model
            model_type: 'standard' or 'obb'
            pretrained_weights: If provided, continue training from this model
            freeze_layers: Freeze first N layers (useful when adding new data)
        """
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed.")
        
        # Get active datasets
        split_dirs = registry.get_active_split_dirs()
        if not split_dirs:
            raise ValueError("No active datasets! Use registry.set_active_datasets([...])")
        
        print(f"\n{'='*70}")
        print(f"Training YOLO on {len(split_dirs)} dataset(s)")
        print(f"{'='*70}")
        for i, sd in enumerate(split_dirs):
            print(f"  {i+1}. {sd}")
        print(f"{'='*70}\n")
        
        # Combine datasets
        combined_dir = os.path.join(output_dir, "combined_dataset")
        yaml_path = YOLOTrainerV2.create_combined_dataset_yaml(
            split_dirs, combined_dir, use_obb=(model_type == "obb")
        )
        
        # Load model
        if pretrained_weights:
            # Incremental learning from existing model
            model = YOLO(pretrained_weights)
            print(f"✓ Loading existing model: {pretrained_weights}")
            
            if freeze_layers > 0:
                # Freeze early layers to preserve learned features
                for i, (name, param) in enumerate(model.model.named_parameters()):
                    if i < freeze_layers:
                        param.requires_grad = False
                print(f"  Froze first {freeze_layers} layers")
        else:
            # Train from scratch or pretrained YOLO weights
            if model_type == "obb":
                model_name = f"yolo{yolo_version}{model_size}-obb.pt"
            else:
                model_name = f"yolo{yolo_version}{model_size}.pt"
            
            model = YOLO(model_name)
            print(f"✓ Using pretrained {model_name}")
        
        # Train
        run_name = f"yolo_{model_type}_multi" if not pretrained_weights else f"yolo_{model_type}_incremental"
        
        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=output_dir,
            name=run_name,
            exist_ok=True,
            **kwargs
        )
        
        # Validate
        metrics = model.val()
        
        # Record in registry
        best_weights = os.path.join(output_dir, run_name, "weights", "best.pt")
        registry.record_training(
            model_type=f"yolo_{model_type}",
            model_path=best_weights,
            datasets_used=registry.datasets["active_datasets"],
            metrics={
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map)
            }
        )
        
        print(f"\n{'='*70}")
        print(f"✓ Training Complete")
        print(f"{'='*70}")
        print(f"  Weights: {best_weights}")
        print(f"  mAP50: {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        print(f"{'='*70}\n")
        
        return model, metrics


# ══════════════════════════════════════════════════════════════════════════
# UPDATED: Multi-Dataset TrackNet Training
# ══════════════════════════════════════════════════════════════════════════

class TrackNetDatasetV2(Dataset):
    """TrackNet dataset supporting multiple data sources."""
    
    def __init__(self, split_dirs: list, split: str = "train",
                 sequence_length: int = 3, img_size: int = 512):
        """
        Args:
            split_dirs: List of paths to splits/ directories
            split: 'train', 'val', or 'test'
            sequence_length: Number of consecutive frames
            img_size: Input image size
        """
        self.datasets = []
        self.sequence_length = sequence_length
        self.img_size = img_size
        
        # Load all datasets
        for split_dir in split_dirs:
            img_dir = os.path.join(split_dir, split, "images")
            ann_path = os.path.join(split_dir, split, "annotations.json")
            
            if not os.path.exists(ann_path):
                print(f"⚠ Skipping {split_dir} - no annotations")
                continue
            
            with open(ann_path) as f:
                annotations = json.load(f)
            
            self.datasets.append({
                "img_dir": img_dir,
                "annotations": annotations,
                "frames": sorted(annotations.keys())
            })
        
        # Build valid sequences across all datasets
        self.sequences = []
        
        for ds_idx, ds in enumerate(self.datasets):
            frames = ds["frames"]
            for i in range(len(frames) - sequence_length + 1):
                if self._are_consecutive(frames[i:i + sequence_length]):
                    self.sequences.append({
                        "dataset_idx": ds_idx,
                        "start_frame_idx": i
                    })
        
        print(f"TrackNet {split}: {len(self.sequences)} sequences from {len(split_dirs)} dataset(s)")
    
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
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        ds = self.datasets[seq_info["dataset_idx"]]
        
        start_idx = seq_info["start_frame_idx"]
        sequence_frames = ds["frames"][start_idx:start_idx + self.sequence_length]
        
        # Load images
        images = []
        for frame_name in sequence_frames:
            img_path = os.path.join(ds["img_dir"], frame_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        
        images = np.array(images).transpose(0, 3, 1, 2)
        images = images.astype(np.float32) / 255.0
        
        # Generate heatmap
        last_frame = sequence_frames[-1]
        ann = ds["annotations"][last_frame]
        
        heatmap = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        
        if ann.get("visibility") in ("visible", "occluded") and "x" in ann:
            orig_img = cv2.imread(os.path.join(ds["img_dir"], last_frame))
            h_orig, w_orig = orig_img.shape[:2]
            
            x = int(ann["x"] * self.img_size / w_orig)
            y = int(ann["y"] * self.img_size / h_orig)
            
            sigma = 5
            for i in range(max(0, y - 3*sigma), min(self.img_size, y + 3*sigma)):
                for j in range(max(0, x - 3*sigma), min(self.img_size, x + 3*sigma)):
                    dist_sq = (i - y)**2 + (j - x)**2
                    heatmap[i, j] = np.exp(-dist_sq / (2 * sigma**2))
        
        return torch.FloatTensor(images), torch.FloatTensor(heatmap)


class TrackNet(nn.Module):
    """TrackNet architecture (VGG-based encoder-decoder)."""
    
    def __init__(self, sequence_length=3):
        super().__init__()
        
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
        
        self.output = nn.Conv2d(32, 1, 1)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        
        x = self.deconv1(x3)
        x = self.deconv2(x)
        x = self.deconv3(x)
        
        x = self.output(x)
        x = torch.sigmoid(x)
        
        return x.squeeze(1)


class TrackNetTrainerV2:
    """Enhanced TrackNet trainer with multi-dataset and incremental learning."""
    
    @staticmethod
    def train(
        registry: DataRegistry,
        output_dir: str,
        sequence_length: int = 3,
        img_size: int = 512,
        epochs: int = 10,
        batch_size: int = 8,
        lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        pretrained_weights: str = None,  # Continue from existing model
        freeze_encoder: bool = False  # Freeze encoder for transfer learning
    ):
        """Train TrackNet on multiple datasets.
        
        Args:
            registry: DataRegistry with active datasets
            pretrained_weights: Path to existing .pth file to continue training
            freeze_encoder: If True, only train decoder (for domain adaptation)
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        split_dirs = registry.get_active_split_dirs()
        if not split_dirs:
            raise ValueError("No active datasets!")
        
        print(f"\n{'='*70}")
        print(f"Training TrackNet on {len(split_dirs)} dataset(s)")
        print(f"{'='*70}")
        
        # Datasets
        train_dataset = TrackNetDatasetV2(split_dirs, "train", sequence_length, img_size)
        val_dataset = TrackNetDatasetV2(split_dirs, "val", sequence_length, img_size)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Model
        model = TrackNet(sequence_length).to(device)
        
        if pretrained_weights:
            checkpoint = torch.load(pretrained_weights, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded pretrained weights: {pretrained_weights}")
            
            if freeze_encoder:
                # Freeze conv layers, only train decoder
                for param in model.conv1.parameters():
                    param.requires_grad = False
                for param in model.conv2.parameters():
                    param.requires_grad = False
                for param in model.conv3.parameters():
                    param.requires_grad = False
                print("  Froze encoder layers")
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
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
            
            print(f"Epoch {epoch+1}/{epochs}  Train: {train_loss:.6f}  Val: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(output_dir, 'tracknet_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                print(f"  → Saved best model")
        
        # Record in registry
        registry.record_training(
            model_type="tracknet",
            model_path=save_path,
            datasets_used=registry.datasets["active_datasets"],
            metrics={"val_loss": float(best_val_loss)}
        )
        
        print(f"\n{'='*70}")
        print("✓ TrackNet Training Complete")
        print(f"  Weights: {save_path}")
        print(f"  Best val loss: {best_val_loss:.6f}")
        print(f"{'='*70}\n")
        
        return model


# ══════════════════════════════════════════════════════════════════════════
# UPDATED: Adaptive Tracking with Domain Classification
# ══════════════════════════════════════════════════════════════════════════

class AdaptiveShuttleTracker:
    """Intelligent tracker that selects best model based on video conditions."""
    
    def __init__(self, model_registry_path: str = "./model_registry.json"):
        """
        Args:
            model_registry_path: JSON file mapping conditions → best models
        """
        self.model_registry = self._load_model_registry(model_registry_path)
        self.loaded_models = {}  # Cache for loaded models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model_registry(self, path):
        """Load or create model registry."""
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        
        # Default registry template
        return {
            "models": {
                "yolo_indoor": None,
                "yolo_outdoor": None,
                "obb_fast": None,
                "obb_blur": None,
                "tracknet_occlusion": None,
                "tracknet_default": None
            },
            "selection_rules": {
                "indoor_slow": "yolo_indoor",
                "indoor_fast": "obb_fast",
                "outdoor_slow": "yolo_outdoor",
                "outdoor_fast": "obb_blur",
                "occlusion_heavy": "tracknet_occlusion",
                "default": "tracknet_default"
            }
        }
    
    def register_model(self, condition: str, model_path: str, model_type: str):
        """Register a model for specific conditions.
        
        Args:
            condition: e.g., 'indoor_fast', 'outdoor_slow', 'heavy_blur'
            model_path: Path to model weights
            model_type: 'yolo', 'obb', or 'tracknet'
        """
        self.model_registry["models"][condition] = {
            "path": model_path,
            "type": model_type
        }
        print(f"✓ Registered {model_type} for condition: {condition}")
    
    def detect_video_conditions(self, video_path: str, sample_frames: int = 30):
        """Auto-detect video characteristics.
        
        Returns:
            dict with 'lighting', 'speed', 'blur_level', 'occlusion_risk'
        """
        cap = cv2.VideoCapture(video_path)
        
        brightnesses = []
        motion_magnitudes = []
        blur_scores = []
        
        prev_gray = None
        frame_count = 0
        
        while frame_count < sample_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Brightness
            brightnesses.append(gray.mean())
            
            # Motion magnitude (optical flow)
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
                motion_magnitudes.append(mag)
            
            # Blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(laplacian_var)
            
            prev_gray = gray
            frame_count += 1
        
        cap.release()
        
        # Analyze results
        avg_brightness = np.mean(brightnesses)
        avg_motion = np.mean(motion_magnitudes) if motion_magnitudes else 0
        avg_blur = np.mean(blur_scores)
        
        conditions = {
            "lighting": "indoor" if avg_brightness < 120 else "outdoor",
            "speed": "fast" if avg_motion > 5 else "slow",
            "blur_level": "high" if avg_blur < 100 else "low",
            "occlusion_risk": "high" if avg_motion > 10 else "low"
        }
        
        print(f"\n{'='*70}")
        print("AUTO-DETECTED VIDEO CONDITIONS")
        print(f"{'='*70}")
        print(f"  Lighting: {conditions['lighting']} (brightness: {avg_brightness:.1f})")
        print(f"  Speed: {conditions['speed']} (motion: {avg_motion:.2f})")
        print(f"  Blur: {conditions['blur_level']} (sharpness: {avg_blur:.1f})")
        print(f"  Occlusion risk: {conditions['occlusion_risk']}")
        print(f"{'='*70}\n")
        
        return conditions
    
    def select_best_model(self, conditions: dict):
        """Choose best model based on detected conditions.
        
        Args:
            conditions: Output from detect_video_conditions()
        
        Returns:
            (model_path, model_type)
        """
        # Build condition key
        key = f"{conditions['lighting']}_{conditions['speed']}"
        
        # Check if we have a specific model for this
        if key in self.model_registry["selection_rules"]:
            model_name = self.model_registry["selection_rules"][key]
        elif conditions['occlusion_risk'] == 'high':
            model_name = "tracknet_occlusion"
        else:
            model_name = "tracknet_default"
        
        model_info = self.model_registry["models"].get(model_name)
        
        if model_info and model_info["path"]:
            print(f"✓ Selected model: {model_name}")
            print(f"  Type: {model_info['type']}")
            print(f"  Path: {model_info['path']}")
            return model_info["path"], model_info["type"]
        else:
            raise ValueError(f"No model registered for condition: {model_name}")
    
    def track_video_adaptive(self, video_path: str, output_path: str,
                            auto_detect: bool = True, manual_conditions: dict = None,
                            conf_threshold: float = 0.25):
        """Track video with automatic model selection.
        
        Args:
            video_path: Input video
            output_path: Output annotated video
            auto_detect: Auto-detect conditions (recommended)
            manual_conditions: Override auto-detection with manual conditions
            conf_threshold: Detection confidence threshold
        """
        # Detect or use manual conditions
        if auto_detect:
            conditions = self.detect_video_conditions(video_path)
        else:
            conditions = manual_conditions or {"lighting": "indoor", "speed": "medium"}
        
        # Select best model
        model_path, model_type = self.select_best_model(conditions)
        
        # Load model if not cached
        if model_path not in self.loaded_models:
            if model_type in ["yolo", "obb"]:
                if YOLO_AVAILABLE:
                    self.loaded_models[model_path] = YOLO(model_path)
                else:
                    raise RuntimeError("YOLO not available")
            
            elif model_type == "tracknet":
                if TORCH_AVAILABLE:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    model = TrackNet(sequence_length=3).to(self.device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    self.loaded_models[model_path] = model
                else:
                    raise RuntimeError("PyTorch not available")
        
        # Run tracking
        model = self.loaded_models[model_path]
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_buffer = deque(maxlen=3)
        frame_idx = 0
        
        print(f"\nTracking video...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detection logic based on model type
            if model_type in ["yolo", "obb"]:
                position = self._detect_yolo(frame, model, conf_threshold, is_obb=(model_type == "obb"))
            else:  # tracknet
                frame_buffer.append(frame.copy())
                position = self._detect_tracknet(frame, frame_buffer, model)
            
            # Visualize
            if position:
                x, y = int(position[0]), int(position[1])
                cv2.circle(frame, (x, y), 12, (0, 255, 0), 2)
                cv2.putText(frame, f"({x}, {y})", (x+15, y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Model: {model_type.upper()} | Frame: {frame_idx}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"  Processed {frame_idx} frames...")
        
        cap.release()
        out.release()
        
        print(f"✓ Tracking complete → {output_path}")
    
    def _detect_yolo(self, frame, model, conf_threshold, is_obb=False):
        """YOLO detection logic."""
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
            return (corners[:, 0].mean(), corners[:, 1].mean())
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
    
    def _detect_tracknet(self, frame, frame_buffer, model):
        """TrackNet detection logic."""
        if len(frame_buffer) < 3:
            return None
        
        frames = list(frame_buffer)
        img_size = 512
        
        images = []
        for f in frames:
            img = cv2.resize(f, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        
        images = np.array(images).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        images = torch.FloatTensor(images).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            heatmap = model(images).cpu().numpy()[0]
        
        y_max, x_max = np.unravel_index(heatmap.argmax(), heatmap.shape)
        
        h, w = frame.shape[:2]
        x = x_max * w / img_size
        y = y_max * h / img_size
        
        if heatmap[y_max, x_max] > 0.5:
            return (x, y)
        return None


# ══════════════════════════════════════════════════════════════════════════
# CLI Interface
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental shuttle tracking system")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Dataset management
    dataset_parser = subparsers.add_parser("dataset", help="Manage datasets")
    dataset_parser.add_argument("action", choices=["add", "remove", "list", "activate"])
    dataset_parser.add_argument("--name", help="Dataset name")
    dataset_parser.add_argument("--split-dir", help="Path to splits/ directory")
    dataset_parser.add_argument("--metadata", type=json.loads, help="JSON metadata")
    dataset_parser.add_argument("--active", nargs="+", help="List of active dataset names")
    
    # Training
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--model-type", choices=["yolo", "obb", "tracknet"], required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch", type=int, default=8)
    train_parser.add_argument("--pretrained-weights", help="Continue from existing model")
    train_parser.add_argument("--freeze-layers", type=int, default=0, help="Freeze N layers")
    train_parser.add_argument("--device", default="0")
    
    # Tracking
    track_parser = subparsers.add_parser("track", help="Track shuttle in video")
    track_parser.add_argument("--video", required=True)
    track_parser.add_argument("--output", required=True)
    track_parser.add_argument("--adaptive", action="store_true", help="Auto-select best model")
    track_parser.add_argument("--model-path", help="Manual model selection")
    track_parser.add_argument("--model-type", choices=["yolo", "obb", "tracknet"])
    
    args = parser.parse_args()
    
    # Execute commands
    if args.command == "dataset":
        registry = DataRegistry()
        
        if args.action == "add":
            registry.add_dataset(args.name, args.split_dir, args.metadata)
        
        elif args.action == "remove":
            registry.remove_dataset(args.name)
        
        elif args.action == "list":
            registry.list_datasets()
        
        elif args.action == "activate":
            registry.set_active_datasets(args.active)
    
    elif args.command == "train":
        registry = DataRegistry()
        
        if args.model_type == "yolo":
            YOLOTrainerV2.train_incremental(
                registry=registry,
                output_dir=args.output_dir,
                model_type="standard",
                epochs=args.epochs,
                batch=args.batch,
                pretrained_weights=args.pretrained_weights,
                freeze_layers=args.freeze_layers,
                device=args.device
            )
        
        elif args.model_type == "obb":
            YOLOTrainerV2.train_incremental(
                registry=registry,
                output_dir=args.output_dir,
                model_type="obb",
                epochs=args.epochs,
                batch=args.batch,
                pretrained_weights=args.pretrained_weights,
                freeze_layers=args.freeze_layers,
                device=args.device
            )
        
        elif args.model_type == "tracknet":
            TrackNetTrainerV2.train(
                registry=registry,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch,
                pretrained_weights=args.pretrained_weights,
                device=args.device
            )
    
    elif args.command == "track":
        tracker = AdaptiveShuttleTracker()
        
        if args.adaptive:
            tracker.track_video_adaptive(
                video_path=args.video,
                output_path=args.output
            )
        else:
            # Manual tracking (implement similar to original code)
            pass