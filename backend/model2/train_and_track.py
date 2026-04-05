import os
import yaml
import json
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional
from TrackNetV2 import TrackNetV2, TrackNetDataset, TrackNetV2Trainer

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


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "shuttle_config.json")

def load_config() -> dict:
    """Load config from shuttle_config.json if it exists."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

def save_config(config: dict):
    """Save config to shuttle_config.json."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
    print(f"✓ Config saved to {CONFIG_FILE}")

def get_default(config: dict, *keys, fallback=None):
    """Safely traverse nested config keys."""
    val = config
    for key in keys:
        if not isinstance(val, dict) or key not in val:
            return fallback
        val = val[key]
    return val

def get_device():
    """Auto-detect best available device."""
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():  # Works for both NVIDIA and AMD ROCm
            return "0"                 # GPU device 0
        else:
            return "cpu"
    return "cpu"

DEVICE = get_device()


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


class ShuttleTracker:
    """Unified inference for YOLO, TrackNet, and hybrid tracking."""
    
    def __init__(self, yolo_weights=None, obb_weights=None, tracknet_weights=None,
                 device=DEVICE):

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
            self.tracknet_model = TrackNetV2(sequence_length=3).to(self.device)
            self.tracknet_model.load_state_dict(checkpoint['model_state_dict'])
            self.tracknet_model.eval()
            print(f"✓ Loaded TrackNet: {tracknet_weights}")
        
        self.frame_buffer = deque(maxlen=3)  # For TrackNet multi-frame input
    
    def track_video(self, video_path: str, output_path: str,
                    mode: str = "hybrid", conf_threshold: float = 0.25,
                    show_trail: bool = True, trail_length: int = 30):
        
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


if __name__ == "__main__":
    import argparse
    
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description="Train and track shuttle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("--action", choices=["train-yolo", "train-obb", "train-tracknet", "track", "save-config"],
                       required=True)

    # Paths (fall back to config file)
    parser.add_argument("--split-dir",
                        default=get_default(config, "paths", "split_dir"))
    parser.add_argument("--output-dir",
                        default=get_default(config, "paths", "output_dir"))
    parser.add_argument("--yolo-weights",
                        default=get_default(config, "paths", "yolo_weights"))
    parser.add_argument("--obb-weights",
                        default=get_default(config, "paths", "obb_weights"))
    parser.add_argument("--tracknet-weights",
                        default=get_default(config, "paths", "tracknet_weights"))

    # Training args
    parser.add_argument("--model-size",
                        default=get_default(config, "training", "model_size", fallback="n"),
                        choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs",      type=int,
                        default=get_default(config, "training", "epochs",    fallback=10))
    parser.add_argument("--batch",       type=int,
                        default=get_default(config, "training", "batch",     fallback=8))
    parser.add_argument("--imgsz",       type=int,
                        default=get_default(config, "training", "imgsz",     fallback=640))
    parser.add_argument("--device",
                        default=DEVICE)
    parser.add_argument("--yolo-version",
                        default=get_default(config, "training", "yolo_version", fallback="11"),
                        choices=["8", "11"])
    parser.add_argument("--lr",          type=float,
                        default=get_default(config, "training", "lr",        fallback=1e-4))
    parser.add_argument("--sequence-length", type=int,
                        default=get_default(config, "training", "sequence_length", fallback=3))
    parser.add_argument("--tracknet-img-size", type=int,
                        default=get_default(config, "training", "tracknet_img_size", fallback=512))
    
    # Fine-tuning args
    parser.add_argument("--resume-from", help="Resume interrupted training from checkpoint")
    parser.add_argument("--finetune-from", help="Fine-tune from pre-trained weights")
    parser.add_argument("--freeze-layers", type=int, default=0,
                       help="Number of layers to freeze for YOLO fine-tuning (0-10)")
    parser.add_argument("--freeze-encoder", action="store_true",
                       help="Freeze encoder layers for TrackNet fine-tuning")
    

    # Inference args
    parser.add_argument("--video", help="Input video for tracking")
    parser.add_argument("--output-video", help="Output tracked video")
    parser.add_argument("--mode",
                        default=get_default(config, "inference", "mode",     fallback="hybrid"),
                        choices=["yolo", "obb", "tracknet", "hybrid"])
    parser.add_argument("--conf",        type=float,
                        default=get_default(config, "inference", "conf",     fallback=0.25))
    
    args = parser.parse_args()
    
    try:
        if args.action == "save-config":
            new_config = {
                "paths": {
                    "split_dir":         args.split_dir,
                    "output_dir":        args.output_dir,
                    "yolo_weights":      args.yolo_weights,
                    "obb_weights":       args.obb_weights,
                    "tracknet_weights":  args.tracknet_weights,
                },
                "training": {
                    "model_size":        args.model_size,
                    "epochs":            args.epochs,
                    "batch":             args.batch,
                    "imgsz":             args.imgsz,
                    "device":            args.device,
                    "yolo_version":      args.yolo_version,
                    "lr":                args.lr,
                    "sequence_length":   args.sequence_length,
                    "tracknet_img_size": args.tracknet_img_size,
                },
                "inference": {
                    "mode":         args.mode,
                    "conf":         args.conf,
                    "trail_length": 30,
                }
            }
            save_config(new_config)
            exit(0)
        elif args.action == "train-yolo":
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
            TrackNetV2Trainer.train(
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