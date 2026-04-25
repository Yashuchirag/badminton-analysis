import os
import sys
import yaml
import json
import cv2
import threading
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional
from queue import Queue, Empty

# Ensure model2/ is on sys.path so sibling modules (TrackNetV2.py) are found
# regardless of which directory this file is imported from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
INFERENCE_SIZE = 416
TRACKNET_SIZE = 256

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
                 device=DEVICE, use_fp16=True, use_compile=True):

        self.device = device
        self.use_fp16 = use_fp16
        self.yolo_model = None
        self.obb_model = None
        self.tracknet_model = None

        if self.device.isdigit():  # If device is "0", "1", etc.
            self.device = f"cuda:{self.device}"
            print(f"Using device: {self.device}")
        elif self.device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            self.device = "cpu"
            self.use_fp16 = False
        
        use_cuda = "cuda" in self.device
        print(f"Using device: {self.device}  FP16={self.use_fp16 and use_cuda}")

        if yolo_weights and YOLO_AVAILABLE:
            self.yolo_model = YOLO(yolo_weights)
            print(f"✓ Loaded YOLO: {yolo_weights}")
        
        if obb_weights and YOLO_AVAILABLE:
            self.obb_model = YOLO(obb_weights)
            print(f"✓ Loaded YOLO-OBB: {obb_weights}")
        
        if tracknet_weights and TORCH_AVAILABLE:
            checkpoint = torch.load(tracknet_weights, map_location=self.device,
                                    weights_only=False)
            self.tracknet_model = TrackNetV2(sequence_length=3).to(self.device)
            self.tracknet_model.load_state_dict(checkpoint['model_state_dict'])
            self.tracknet_model.eval()

            if use_cuda and self.use_fp16:
                self.tracknet_model.half()

            # torch.compile is disabled: Triton's ptxas subprocess fails when the
            # install path contains spaces (e.g. WSL mount /mnt/d/Personal Projects/…).
            # The eager path is fast enough for real-time inference on GPU.
            # Uncomment the block below only if your install path has no spaces.
            # if use_compile and hasattr(torch, "compile"):
            #     try:
            #         self.tracknet_model = torch.compile(self.tracknet_model)
            #         print("✓ torch.compile applied to TrackNet")
            #     except Exception as e:
            #         print(f"⚠ torch.compile skipped: {e}")

            print(f"✓ Loaded TrackNet: {tracknet_weights}")
        
        self.frame_buffer = deque(maxlen=3)  # For TrackNet multi-frame input

    def predict_frame(
            self,
            frame: np.ndarray,
            mode: str = "hybrid",
            conf_threshold: float = 0.25,
            yolo_conf_skip_tracknet: float = 0.65,
    ) -> tuple:
        """
        Run inference on a single frame.

        Returns:
            (pos, conf, source)
            pos    – (x, y) pixel coordinates or None if not detected
            conf   – detection confidence 0.0–1.0
            source – "obb" | "yolo" | "tracknet" | "hybrid" | "none"
        """
        h, w = frame.shape[:2]
        use_half = "cuda" in self.device and self.use_fp16

        infer = cv2.resize(frame, (INFERENCE_SIZE, INFERENCE_SIZE))
        sx = w / INFERENCE_SIZE
        sy = h / INFERENCE_SIZE

        def _scale(pos):
            return (pos[0] * sx, pos[1] * sy) if pos is not None else None

        # ── YOLO-only ────────────────────────────────────────────────────
        if mode == "yolo" and self.yolo_model is not None:
            r = self.yolo_model([infer], conf=conf_threshold,
                                verbose=False, half=use_half)[0]
            pos, conf = self._extract_box_with_conf(r, is_obb=False)
            return _scale(pos), float(conf), "yolo"

        # ── OBB-only ─────────────────────────────────────────────────────
        if mode == "obb" and self.obb_model is not None:
            r = self.obb_model([infer], conf=conf_threshold,
                               verbose=False, half=use_half)[0]
            pos, conf = self._extract_box_with_conf(r, is_obb=True)
            return _scale(pos), float(conf), "obb"

        # ── TrackNet-only ─────────────────────────────────────────────────
        if mode == "tracknet" and self.tracknet_model is not None:
            pos = self._detect_tracknet(frame, w, h)
            return pos, (0.7 if pos is not None else 0.0), "tracknet"

        # ── Hybrid: YOLO/OBB first, TrackNet as low-confidence fallback ──
        model   = self.obb_model if self.obb_model is not None else self.yolo_model
        is_obb  = self.obb_model is not None
        src_det = "obb" if is_obb else "yolo"

        yolo_pos, yolo_conf = None, 0.0
        if model is not None:
            r = model([infer], conf=conf_threshold,
                      verbose=False, half=use_half)[0]
            yolo_pos, yolo_conf = self._extract_box_with_conf(r, is_obb=is_obb)
            yolo_pos = _scale(yolo_pos)

        # High-confidence YOLO hit — skip TrackNet entirely
        if yolo_conf >= yolo_conf_skip_tracknet or self.tracknet_model is None:
            return yolo_pos, float(yolo_conf), src_det

        tracknet_pos = self._detect_tracknet(frame, w, h)

        if yolo_pos and tracknet_pos:
            dist = np.hypot(yolo_pos[0] - tracknet_pos[0],
                            yolo_pos[1] - tracknet_pos[1])
            pos = tracknet_pos if dist < 50 else yolo_pos
            return pos, float(yolo_conf), "hybrid"

        pos = yolo_pos or tracknet_pos
        conf = float(yolo_conf) if yolo_pos else (0.7 if tracknet_pos else 0.0)
        src  = "hybrid" if pos else "none"
        return pos, conf, src

    def track_video(self, video_path: str, output_path: str,
                    mode: str = "hybrid", conf_threshold: float = 0.25,
                    show_trail: bool = True, trail_length: int = 30, batch_size: int = 8,
                    yolo_conf_skip_tracknet: float = 0.65):
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\nTracking video : {video_path}")
        print(f"  Mode         : {mode}")
        print(f"  Resolution   : {width}x{height}  FPS: {fps}")
        print(f"  Batch size   : {batch_size}")
        print(f"  Infer size   : YOLO={INFERENCE_SIZE}, TrackNet={TRACKNET_SIZE}")
        
        read_queue = Queue(maxsize=batch_size * 4)
        write_queue = Queue(maxsize=batch_size * 4)

        def _reader():
            while True:
                ret, frame = cap.read()
                if not ret:
                    read_queue.put(None)
                    break
                read_queue.put(frame)

        def _writer():
            while True:
                item = write_queue.get()
                if item is None:
                    break
                out.write(item)

        threading.Thread(target=_reader, daemon=True).start()
        writer_t = threading.Thread(target=_writer, daemon=True)
        writer_t.start()

        trail = deque(maxlen=trail_length)
        frame_idx = 0
        done = False

        while not done:
            batch_frames = []
            while len(batch_frames) < batch_size:
                try:
                    frame = read_queue.get(timeout=5)
                except Empty:
                    done = True
                    break
                if frame is None:
                    done = True
                    break
                batch_frames.append(frame)

            if not batch_frames:
                break
            
            positions = self._process_batch(
                batch_frames, mode, conf_threshold, 
                yolo_conf_skip_tracknet, width, height
            )
            
            for frame, position in zip(batch_frames, positions):
                if position:
                    x, y = int(position[0]), int(position[1])
                    trail.append((x, y))
                    cv2.circle(frame, (x, y), 12, (0, 255, 0), 2)

                    # Draw motion trail
                    if show_trail and len(trail) > 1:
                        pts = list(trail)
                        for i in range(1, len(pts)):
                            alpha = i / len(pts)
                            color = (0, int(255 * alpha), int(100 * (1 - alpha)))
                            cv2.line(frame, pts[i - 1], pts[i], color, 2)

                cv2.putText(frame, f"Frame: {frame_idx}  Mode: {mode.upper()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if position:
                    cv2.putText(frame, f"Pos: ({int(position[0])}, {int(position[1])})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                write_queue.put(frame)
                frame_idx += 1

            if frame_idx % 200 == 0 and frame_idx > 0:
                print(f"  Processed {frame_idx} frames…")

        # ── Teardown ──────────────────────────────────────────────────────────
        write_queue.put(None)
        writer_t.join()
        cap.release()
        out.release()

        print(f"✓ Tracking complete → {output_path}")
        print(f"  Total frames : {frame_idx}")
    

    def _process_batch(
                self,
                frames: list,
                mode: str,
                conf_threshold: float,
                width: int,
                height: int,
                yolo_conf_skip_tracknet: float,
            ) -> list:
        
        positions = [None] * len(frames)
        use_half = "cuda" in self.device and self.use_fp16

        # Resize all frames once for inference
        infer_frames = [cv2.resize(f, (INFERENCE_SIZE, INFERENCE_SIZE)) for f in frames]
        scale_x = width / INFERENCE_SIZE
        scale_y = height / INFERENCE_SIZE

        def _scale(pos):
            if pos is None:
                return None
            return (pos[0] * scale_x, pos[1] * scale_y)

        # YOLO-only 
        if mode == "yolo" and self.yolo_model is not None:
            results = self.yolo_model(
                infer_frames, conf=conf_threshold, verbose=False, stream=False, half=use_half
            )
            for i, r in enumerate(results):
                positions[i] = _scale(self._extract_box(r, is_obb=False))

        # OBB-only 
        elif mode == "obb" and self.obb_model is not None:
            results = self.obb_model(
                infer_frames, conf=conf_threshold, verbose=False, stream=False, half=use_half
            )
            for i, r in enumerate(results):
                positions[i] = _scale(self._extract_box(r, is_obb=True))

        # TrackNet-only (frame-by-frame; no batching in TrackNet) 
        elif mode == "tracknet" and self.tracknet_model is not None:
            for i, frame in enumerate(frames):
                positions[i] = self._detect_tracknet(frame, width, height)

        # Hybrid (batched YOLO + selective TrackNet) 
        elif mode == "hybrid":
            model = self.obb_model if self.obb_model is not None else self.yolo_model
            is_obb = self.obb_model is not None

            if model is not None:
                results = model(
                    infer_frames, conf=conf_threshold, verbose=False, stream=False, half=use_half
                )
                yolo_positions = []
                yolo_confs = []
                for r in results:
                    pos, conf = self._extract_box_with_conf(r, is_obb=is_obb)
                    yolo_positions.append(_scale(pos))
                    yolo_confs.append(conf)
            else:
                yolo_positions = [None] * len(frames)
                yolo_confs = [0.0] * len(frames)

            for i, frame in enumerate(frames):
                yolo_pos = yolo_positions[i]
                yolo_conf = yolo_confs[i]
                
                if yolo_conf >= yolo_conf_skip_tracknet or self.tracknet_model is None:
                    positions[i] = yolo_pos
                    continue

                tracknet_pos = self._detect_tracknet(frame, width, height)

                if yolo_pos and tracknet_pos:
                    dist = np.hypot(
                        yolo_pos[0] - tracknet_pos[0],
                        yolo_pos[1] - tracknet_pos[1],
                    )
                    positions[i] = tracknet_pos if dist < 50 else yolo_pos
                else:
                    positions[i] = yolo_pos or tracknet_pos

        return positions


    def _extract_box(self, r, is_obb: bool):
        pos, _ = self._extract_box_with_conf(r, is_obb)
        return pos

    def _extract_box_with_conf(self, r, is_obb: bool):
        """Return ((cx, cy), best_confidence) or (None, 0.0)."""
        if is_obb:
            if not hasattr(r, "obb") or r.obb is None or r.obb.conf is None:
                return None, 0.0
            confs = r.obb.conf.cpu().numpy()
            if len(confs) == 0:
                return None, 0.0
            best_idx = int(np.argmax(confs))
            corners = r.obb.xyxyxyxy.cpu().numpy()[best_idx]
            cx = corners[:, 0].mean()
            cy = corners[:, 1].mean()
            return (cx, cy), float(confs[best_idx])
        else:
            if r.boxes is None or r.boxes.conf is None:
                return None, 0.0
            confs = r.boxes.conf.cpu().numpy()
            if len(confs) == 0:
                return None, 0.0
            best_idx = int(np.argmax(confs))
            x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy()[best_idx]
            return ((x1 + x2) / 2, (y1 + y2) / 2), float(confs[best_idx])

        
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

    
    def _detect_tracknet(self, frame, width: int, height: int):
        """Run TrackNet prediction on frame."""
        if self.tracknet_model is None:
            return None
        
        # Add to buffer
        self.frame_buffer.append(frame.copy())
        
        if len(self.frame_buffer) < 3:
            return None
        
        images = []
        for f in self.frame_buffer:
            img = cv2.resize(f, (TRACKNET_SIZE, TRACKNET_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

        arr = np.array(images).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        tensor = torch.FloatTensor(arr).unsqueeze(0)

        # Non-blocking transfer to GPU
        if "cuda" in self.device:
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
            if self.use_fp16:
                tensor = tensor.half()
        else:
            tensor = tensor.to(self.device)

        with torch.no_grad():
            heatmap = self.tracknet_model(tensor).cpu().float().numpy()[0]

        y_max, x_max = np.unravel_index(heatmap.argmax(), heatmap.shape)

        if heatmap[y_max, x_max] <= 0.5:
            return None

        # Scale to original resolution
        x = x_max * width / TRACKNET_SIZE
        y = y_max * height / TRACKNET_SIZE
        return (x, y)

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