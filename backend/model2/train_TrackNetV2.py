import os
import yaml
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional

from TrackNetV2 import TrackNetV2, TrackNetV2Trainer, TrackNetDataset

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
    print("⚠ PyTorch not installed. TrackNetV2 training disabled.")


CONFIG_FILE = os.path.join(os.path.dirname(__file__), "shuttle_config.json")

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {}

def save_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)
    print(f"✓ Config saved to {CONFIG_FILE}")

def get_default(config: dict, *keys, fallback=None):
    val = config
    for key in keys:
        if not isinstance(val, dict) or key not in val:
            return fallback
        val = val[key]
    return val

def get_device():
    if TORCH_AVAILABLE:
        if torch.cuda.is_available():
            return "0"
        return "cpu"
    return "cpu"

DEVICE = get_device()

class YOLOTrainer:
    """Train YOLOv8/v11 for shuttle detection (standard or OBB mode)."""

    @staticmethod
    def create_dataset_yaml(split_dir: str, use_obb: bool = False) -> str:
        yaml_path = os.path.join(split_dir, "dataset.yaml")

        if use_obb:
            import shutil
            for split in ['train', 'val', 'test']:
                labels_dir     = os.path.join(split_dir, split, 'labels')
                obb_labels_dir = os.path.join(split_dir, split, 'obb_labels')
                if not os.path.exists(obb_labels_dir):
                    print(f"⚠ Warning: {split}/obb_labels not found")
                    continue
                if os.path.exists(labels_dir):
                    if os.path.islink(labels_dir):
                        os.unlink(labels_dir)
                    elif os.path.isdir(labels_dir):
                        if os.path.samefile(labels_dir, obb_labels_dir):
                            continue
                        else:
                            shutil.rmtree(labels_dir)
                try:
                    os.symlink(os.path.abspath(obb_labels_dir),
                               labels_dir, target_is_directory=True)
                    print(f"  ✓ Symlinked {split}/labels → obb_labels")
                except (OSError, NotImplementedError):
                    shutil.copytree(obb_labels_dir, labels_dir)
                    print(f"  ✓ Copied {split}/obb_labels → labels")

        config = {
            "path":  os.path.abspath(split_dir),
            "train": "train/images",
            "val":   "val/images",
            "test":  "test/images",
            "nc":    1,
            "names": ["shuttle"],
        }
        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✓ Created {yaml_path} | Mode: {'OBB' if use_obb else 'Standard'}")
        return yaml_path

    @staticmethod
    def generate_labels_from_json(split_dir: str, box_size: int = 15):
        """Auto-generate YOLO .txt label files from annotations.json only if labels are missing."""
        import cv2
        from tqdm import tqdm

        print("Checking YOLO labels...")
        for split in ["train", "val", "test"]:
            ann_path = Path(split_dir) / split / "annotations.json"
            img_dir  = Path(split_dir) / split / "images"
            lbl_dir  = Path(split_dir) / split / "labels"
            obb_dir  = Path(split_dir) / split / "obb_labels"

            if not ann_path.exists():
                print(f"  ⚠ No annotations.json for {split}, skipping")
                continue

            yolo_exists = lbl_dir.exists() and any(lbl_dir.glob("*.txt"))
            obb_exists  = obb_dir.exists()  and any(obb_dir.glob("*.txt"))

            # Skip if labels already exist 
            if yolo_exists and obb_exists:
                print(f"  ✓ {split}: both labels/ and obb_labels/ exist, skipping")
                continue
            

            lbl_dir.mkdir(parents=True, exist_ok=True)
            obb_dir.mkdir(parents=True, exist_ok=True)

            with open(ann_path) as f:
                annotations = json.load(f)

            converted, skipped, background = 0, 0, 0
            size_cache = {}
            half = box_size / 2

            with tqdm(
                annotations.items(),
                total=len(annotations),
                desc=f"  [{split}] Generating labels",
                unit="label",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ) as pbar:
                for fname, ann in pbar:
                    stem = Path(fname).stem
                    yolo_path = lbl_dir / (stem + ".txt")
                    obb_path  = obb_dir  / (stem + ".txt")

                    if ann.get("visibility") != "visible" or ann.get("x") is None:
                        yolo_path.write_text("")
                        obb_path.write_text("")
                        background += 1
                        pbar.set_postfix({"labels": converted, "bg": background,
                                        "skip": skipped}, refresh=False)
                        continue

                    img_path = str(img_dir / fname)
                    if img_path not in size_cache:
                        img = cv2.imread(img_path)
                        if img is None:
                            skipped += 1
                            pbar.set_postfix({"labels": converted, "bg": background,
                                          "skip": skipped}, refresh=False)
                            continue
                        size_cache[img_path] = img.shape[:2]

                    h, w = size_cache[img_path]
                    ax, ay = float(ann["x"]), float(ann["y"])

                    # ── Standard YOLO: class cx cy bw bh (normalised) ──────
                    cx = max(0.0, min(1.0, ax / w))
                    cy = max(0.0, min(1.0, ay / h))
                    bw = max(0.0, min(1.0, box_size / w))
                    bh = max(0.0, min(1.0, box_size / h))
                    yolo_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

                    # ── OBB: class x1 y1 x2 y2 x3 y3 x4 y4 (normalised) ──
                    # Corners: top-left → top-right → bottom-right → bottom-left
                    x1 = max(0.0, min(1.0, (ax - half) / w))
                    y1 = max(0.0, min(1.0, (ay - half) / h))
                    x2 = max(0.0, min(1.0, (ax + half) / w))
                    y2 = max(0.0, min(1.0, (ay - half) / h))
                    x3 = max(0.0, min(1.0, (ax + half) / w))
                    y3 = max(0.0, min(1.0, (ay + half) / h))
                    x4 = max(0.0, min(1.0, (ax - half) / w))
                    y4 = max(0.0, min(1.0, (ay + half) / h))
                    obb_path.write_text(
                        f"0 {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
                        f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}\n"
                    )
                    converted += 1
                    pbar.set_postfix({"labels": converted, "bg": background, "skip": skipped}, refresh=False)

            print(f"  ✓ {split}: {converted} labels | {background} backgrounds | {skipped} missing images")


    @staticmethod
    def train_standard(
        split_dir: str, output_dir: str,
        model_size: str = "n", epochs: int = 20, imgsz: int = 640,
        batch: int = 32, device: str = DEVICE, pretrained: bool = True,
        yolo_version: str = "8", resume_from: Optional[str] = None,
        finetune_from: Optional[str] = None, freeze_layers: int = 0, **kwargs
    ):
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed.")
        output_dir = os.path.abspath(output_dir)

        # Generate .txt labels from annotations.json before anything else
        YOLOTrainer.generate_labels_from_json(split_dir)

        yaml_path = YOLOTrainer.create_dataset_yaml(split_dir, use_obb=False)

        if resume_from:
            model          = YOLO(resume_from)
            training_mode  = "RESUME"
            use_pretrained = False
        elif finetune_from:
            model          = YOLO(finetune_from)
            training_mode  = "FINE-TUNE"
            use_pretrained = False
            if freeze_layers > 0:
                for i, (name, param) in enumerate(model.model.named_parameters()):
                    if i < freeze_layers:
                        param.requires_grad = False
        elif pretrained:
            try:
                model = YOLO(f"yolo{yolo_version}{model_size}.pt")
                training_mode = "PRETRAINED"
            except Exception:
                model = YOLO(f"yolov8{model_size}.pt")
                yolo_version  = "8"
                training_mode = "PRETRAINED"
            use_pretrained = True
        else:
            model          = YOLO(f"yolo{yolo_version}{model_size}.yaml")
            training_mode  = "FROM-SCRATCH"
            use_pretrained = False

        print(f"\n{'='*70}")
        print(f"Training YOLOv{yolo_version}{model_size.upper()} Standard | Mode: {training_mode}")
        print(f"{'='*70}")

        # ── Clean epoch-level logging ──────────────────────────────────────
        def on_epoch_end(trainer):
            loss = trainer.loss_items
            print(
                f"  Epoch {trainer.epoch + 1}/{trainer.epochs} | "
                f"box: {loss[0]:.4f} | cls: {loss[1]:.4f} | dfl: {loss[2]:.4f}"
            )

        def on_val_end(trainer):
            try:
                map50 = trainer.metrics.box.map50  # correct attribute access
                print(f"  Val mAP50: {map50:.4f}\n")
            except Exception as e:
                print(f"  Val mAP50: unavailable ({e})\n")

        model.add_callback("on_train_epoch_end", on_epoch_end)
        model.add_callback("on_val_end", on_val_end)
        # ──────────────────────────────────────────────────────────────────

        run_name = "yolo_finetune" if finetune_from else "yolo_standard"
        results  = model.train(
            data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch,
            device=device, project=output_dir, name=run_name,
            exist_ok=True, pretrained=use_pretrained,
            resume=bool(resume_from), patience=20,
            verbose=False,
            workers=8,
            cache='ram',
            **kwargs
        )
        metrics = model.val()
        print(f"\n✓ Standard YOLO complete | mAP50: {metrics.box.map50:.4f}")
        return model, metrics


    @staticmethod
    def train_obb(
        split_dir: str, output_dir: str,
        model_size: str = "n", epochs: int = 20, imgsz: int = 640,
        batch: int = 8, device: str = DEVICE, pretrained: bool = True,
        yolo_version: str = "8", resume_from: Optional[str] = None,
        finetune_from: Optional[str] = None, freeze_layers: int = 0, **kwargs
    ):
        if not YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed.")
        output_dir = os.path.abspath(output_dir)

        YOLOTrainer.generate_labels_from_json(split_dir)
        yaml_path  = YOLOTrainer.create_dataset_yaml(split_dir, use_obb=True)

        if resume_from and finetune_from:
            raise ValueError("Cannot use both resume_from and finetune_from.")

        if resume_from:
            model = YOLO(resume_from)
            training_mode = "RESUME"
            use_pretrained = False
        elif finetune_from:
            model = YOLO(finetune_from)
            training_mode = "FINE-TUNE"
            use_pretrained = False
            if freeze_layers > 0:
                for i, (name, param) in enumerate(model.model.named_parameters()):
                    if i < freeze_layers:
                        param.requires_grad = False
        elif pretrained:
            try:
                model = YOLO(f"yolo{yolo_version}{model_size}-obb.pt")
                training_mode = "PRETRAINED"
            except Exception:
                model = YOLO(f"yolov8{model_size}-obb.pt")
                yolo_version  = "8"
                training_mode = "PRETRAINED"
            use_pretrained = True
        else:
            model = YOLO(f"yolo{yolo_version}{model_size}-obb.yaml")
            training_mode = "FROM-SCRATCH"
            use_pretrained = False

        print(f"\n{'='*70}")
        print(f"Training YOLOv{yolo_version}{model_size.upper()}-OBB | Mode: {training_mode}")
        print(f"{'='*70}")

        run_name = "yolo_obb_finetune" if finetune_from else "yolo_obb"
        results  = model.train(
            data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch,
            device=device, project=output_dir, name=run_name,
            exist_ok=True, pretrained=use_pretrained,
            resume=bool(resume_from), patience=20, **kwargs
        )
        metrics = model.val()
        print(f"\n✓ OBB YOLO complete | mAP50: {metrics.box.map50:.4f}")
        return model, metrics


def convert_tracknetv2_dataset(
    tracknetv2_dir: str,
    output_split_dir: str,
    val_ratio: float = 0.1,
    frame_skip: int = 1,
):
    root = Path(tracknetv2_dir)
    out = Path(output_split_dir)
    splits = {"train": {}, "val": {}, "test": {}}

    # Collect train/val matches from amateur + pro
    trainval_matches = []
    for category in ["Amateur", "Professional"]:
        print(f"Processing {category} category...")
        cat_dir = root / category
        print(cat_dir)
        if not cat_dir.exists():
            print(f"⚠ {category}/ not found — skipping")
            continue
        for match_dir in sorted(cat_dir.iterdir()):
            if match_dir.is_dir():
                trainval_matches.append((category, match_dir))

    n_val = max(1, int(len(trainval_matches) * val_ratio))
    val_matches = trainval_matches[-n_val:]
    train_matches = trainval_matches[:-n_val]

    print(f"Dataset split: {len(train_matches)} train | {len(val_matches)} val matches")

    for split_name, match_list in [("train", train_matches), ("val", val_matches)]:
        for category, match_dir in match_list:
            _process_match(match_dir, category, split_name, out, splits, frame_skip)

    # Test folder
    test_dir = root / "Test"
    if test_dir.exists():
        for match_dir in sorted(test_dir.iterdir()):
            if match_dir.is_dir():
                _process_match(match_dir, "test", "test", out, splits, frame_skip)

    for split_name, annotations in splits.items():
        ann_path = out / split_name / "annotations.json"
        ann_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ann_path, "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"✓ {split_name}: {len(annotations)} frames → {ann_path}")

    print("\n✓ Conversion complete!")
    return out


def _process_match(match_dir, category, split_name, out, splits, frame_skip):
    csv_dir = match_dir / "csv"
    vid_dir = match_dir / "video"

    if not csv_dir.exists() or not vid_dir.exists():
        print(f"  ⚠ Skipping {match_dir.name} — missing csv/ or video/ subfolder")
        return

    csv_files = sorted(csv_dir.glob("*_ball.csv"))

    if not csv_files:
        print(f"  ⚠ Skipping {match_dir.name} — no *_ball.csv files found")
        return

    match_name  = f"{category}_{match_dir.name}"
    total_saved = 0

    for csv_path in csv_files:
        # 1_00_01_ball.csv → rally_name = 1_00_01
        rally_name = csv_path.stem.replace("_ball", "")
        mp4_path   = vid_dir / f"{rally_name}.mp4"

        if not mp4_path.exists():
            print(f"  ⚠ No video for rally {rally_name} — skipping")
            continue

        # Parse CSV 
        frame_anns = {}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                try:
                    frame_num  = int(row[0])
                    visibility = int(row[1])
                    x = float(row[2]) if visibility == 1 else None
                    y = float(row[3]) if visibility == 1 else None
                    frame_anns[frame_num] = {
                        "visibility": "visible" if visibility == 1 else "not_visible",
                        "x": x,
                        "y": y,
                    }
                except (ValueError, IndexError):
                    continue

        # Extract frames from MP4
        out_img_dir = out / split_name / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)

        cap       = cv2.VideoCapture(str(mp4_path))
        frame_idx = 0

        print(f"  {match_name}/{rally_name} ...", end=" ", flush=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Unique name includes match + rally + frame to avoid collisions
                unique_name = f"{match_name}_{rally_name}_frame{frame_idx:05d}.jpg"
                out_path    = out_img_dir / unique_name

                if not out_path.exists():
                    cv2.imwrite(str(out_path), frame)

                ann = frame_anns.get(frame_idx, {
                    "visibility": "not_visible", "x": None, "y": None,
                })
                splits[split_name][unique_name] = ann
                total_saved += 1

            frame_idx += 1

        cap.release()
        print(f"{frame_idx} frames")

    print(f"  ✓ {match_name}: {total_saved} total frames across {len(csv_files)} rallies")

def merge_with_existing_data(
    existing_split_dir: str,
    tracknetv2_split_dir: str,
    merged_output_dir: str,
):
    """
    Auto-discovers all version subfolders inside existing_split_dir.
    e.g. dataset/processed/ → finds v1/, v2/, v3/ automatically.
    """
    import shutil
    from tqdm import tqdm

    parent = Path(existing_split_dir)

    # Auto-discover all subdirs that contain at least one split folder
    version_dirs = []
    for subdir in sorted(parent.iterdir()):
        if subdir.is_dir():
            has_split = any((subdir / s / "annotations.json").exists()
                            for s in ["train", "val", "test"])
            if has_split:
                version_dirs.append(subdir)
            else:
                print(f"  ⚠ Skipping {subdir.name} — no train/val/test found inside")

    if not version_dirs:
        print(f"❌ No valid version folders found in {existing_split_dir}")
        return

    print(f"\nFound {len(version_dirs)} version(s): "
          f"{[d.name for d in version_dirs]}")

    # Sources: (prefix, path)
    # TrackNetV2 already has unique names so no prefix needed
    sources = [(d.name, d) for d in version_dirs]
    if Path(tracknetv2_split_dir).exists():
        sources.append(("", Path(tracknetv2_split_dir)))
    else:
        print(f"  ⚠ TrackNetV2 split dir not found — merging existing versions only")

    for split in ["train", "val", "test"]:
        print(f"\n── {split.upper()} ──────────────────────────")
        merged_ann = {}
        out_img_dir = Path(merged_output_dir) / split / "images"
        out_img_dir.mkdir(parents=True, exist_ok=True)

        for prefix, source_dir in sources:
            ann_path = source_dir / split / "annotations.json"
            img_dir = source_dir / split / "images"
            label = prefix if prefix else "tracknetv2"

            if not ann_path.exists():
                print(f"  ⚠ [{label}] No annotations.json — skipping")
                continue

            with open(ann_path) as f:
                anns = json.load(f)

            # tqdm progress bar per source per split
            skipped = 0
            with tqdm(
                anns.items(),
                total=len(anns),
                desc=f"  [{label}]",
                unit="frame",
                dynamic_ncols=True,              # adapts to terminal width
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]",
            ) as pbar:
                for fname, ann in pbar:
                    new_fname = f"{prefix}_{fname}" if prefix else fname
                    src = img_dir / fname
                    dst = out_img_dir / new_fname

                    if src.exists():
                        if not dst.exists():
                            shutil.copy2(src, dst)
                        merged_ann[new_fname] = ann
                    else:
                        skipped += 1

                    # Live stats in the bar suffix
                    pbar.set_postfix({
                        "merged": len(merged_ann),
                        "missing": skipped,
                    }, refresh=False)

        out_ann = Path(merged_output_dir) / split / "annotations.json"
        with open(out_ann, "w") as f:
            json.dump(merged_ann, f, indent=2)
        print(f"  ✓ {split}: {len(merged_ann)} total frames → {out_ann}")

    print("\n✓ Merge complete!")


if __name__ == "__main__":
    import argparse

    config = load_config()

    parser = argparse.ArgumentParser(
        description="Badminton shuttle train and track",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--action", required=True,
                        choices=["train-yolo", "train-obb", "train-tracknet",
                                 "convert-dataset", "merge-datasets", "save-config"])

    # Paths
    parser.add_argument("--split-dir", default=get_default(config, "paths", "split_dir"))
    parser.add_argument("--output-dir", default=get_default(config, "paths", "output_dir"))
    parser.add_argument("--yolo-weights", default=get_default(config, "paths", "yolo_weights"))
    parser.add_argument("--obb-weights", default=get_default(config, "paths", "obb_weights"))
    parser.add_argument("--tracknet-weights", default=get_default(config, "paths", "tracknet_weights"))

    # Dataset conversion
    parser.add_argument("--tracknetv2-dir", help="Root of downloaded TrackNetV2 dataset")
    parser.add_argument("--converted-dir", help="Output dir for converted dataset")
    parser.add_argument("--existing-split-dir", help="Your current annotated splits")
    parser.add_argument("--merged-output-dir", help="Output for merged dataset")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    # Training
    parser.add_argument("--model-size", default=get_default(config, "training", "model_size", fallback="n"),
                        choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=get_default(config, "training", "batch", fallback=8))
    parser.add_argument("--imgsz", type=int, default=get_default(config, "training", "imgsz", fallback=640))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--yolo-version",  default=get_default(config, "training", "yolo_version", fallback="11"),
                        choices=["8", "11"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--tracknet-img-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=3)
    parser.add_argument("--resume-from", help="Resume interrupted training")
    parser.add_argument("--finetune-from", help="Fine-tune from pretrained weights")
    parser.add_argument("--freeze-layers", type=int, default=0)
    parser.add_argument("--freeze-encoder", action="store_true")

    # Inference
    parser.add_argument("--video", help="Input video for tracking")
    parser.add_argument("--output-video", help="Output tracked video")
    parser.add_argument("--mode", default=get_default(config, "inference", "mode", fallback="hybrid"),
                        choices=["yolo", "obb", "tracknet", "hybrid"])
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--gap-seconds", type=float, default=0.5)

    args = parser.parse_args()

    try:
        if args.action == "save-config":
            save_config({
                "paths": {
                    "split_dir":        args.split_dir,
                    "output_dir":       args.output_dir,
                    "yolo_weights":     args.yolo_weights,
                    "obb_weights":      args.obb_weights,
                    "tracknet_weights": args.tracknet_weights,
                },
                "training": {
                    "model_size":    args.model_size,
                    "epochs":        args.epochs,
                    "batch":         args.batch,
                    "imgsz":         args.imgsz,
                    "yolo_version":  args.yolo_version,
                    "lr":            args.lr,
                },
                "inference": {
                    "mode": args.mode,
                    "conf": args.conf,
                },
            })

        elif args.action == "train-yolo":
            YOLOTrainer.train_standard(
                split_dir=args.split_dir, output_dir=args.output_dir,
                model_size=args.model_size, epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=args.device,
                yolo_version=args.yolo_version, resume_from=args.resume_from,
                finetune_from=args.finetune_from, freeze_layers=args.freeze_layers,
            )

        elif args.action == "train-obb":
            YOLOTrainer.train_obb(
                split_dir=args.split_dir, output_dir=args.output_dir,
                model_size=args.model_size, epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=args.device,
                yolo_version=args.yolo_version, resume_from=args.resume_from,
                finetune_from=args.finetune_from, freeze_layers=args.freeze_layers,
            )

        elif args.action == "train-tracknet":
            TrackNetV2Trainer.train(
                split_dir=args.split_dir, output_dir=args.output_dir,
                sequence_length=args.sequence_length, img_size=args.tracknet_img_size,
                epochs=args.epochs, batch_size=args.batch, lr=args.lr,
                device=args.device, patience=args.patience,
                lr_patience=args.lr_patience, resume_from=args.resume_from,
                finetune_from=args.finetune_from, freeze_encoder=args.freeze_encoder,
            )

        elif args.action == "convert-dataset":
            convert_tracknetv2_dataset(
                tracknetv2_dir=args.tracknetv2_dir,
                output_split_dir=args.converted_dir,
                val_ratio=args.val_ratio,
                frame_skip=args.frame_skip,
            )

        elif args.action == "merge-datasets":
            merge_with_existing_data(
                existing_split_dir=args.existing_split_dir,
                tracknetv2_split_dir=args.converted_dir,
                merged_output_dir=args.merged_output_dir,
            )

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)