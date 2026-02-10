import os
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import yaml


class IncrementalTrainingManager:
    """
    Manages incremental dataset updates and training without overwriting previous work.
    
    Key Features:
    - Dataset versioning (v1, v2, v3...)
    - Checkpoint management
    - Merge new videos with existing datasets
    - Preserve training history
    - Support for fine-tuning vs full retraining
    """
    
    def __init__(self, base_dir: str = "./shuttle_tracking_project"):
        """
        Args:
            base_dir: Root directory for all project data
        """
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "datasets"
        self.models_dir = self.base_dir / "models"
        self.config_file = self.base_dir / "training_config.json"
        
        # Create directory structure
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize config
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load or create training configuration."""
        if self.config_file.exists():
            with open(self.config_file) as f:
                return json.load(f)
        else:
            config = {
                "current_version": 0,
                "datasets": {},
                "models": {},
                "video_registry": {},  # Track which videos are in which dataset version
            }
            self._save_config(config)
            return config
    
    def _save_config(self, config: Optional[Dict] = None):
        """Save configuration to disk."""
        if config is None:
            config = self.config
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _compute_video_hash(self, video_path: str) -> str:
        """Compute hash of video file to detect duplicates."""
        hasher = hashlib.md5()
        with open(video_path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def register_new_videos(
        self, 
        video_paths: List[str],
        video_names: Optional[List[str]] = None
    ) -> Tuple[List[str], List[str]]:
        """
        Register new videos and detect duplicates.
        
        Args:
            video_paths: Paths to new video files
            video_names: Optional custom names (defaults to filenames)
        
        Returns:
            (new_videos, duplicate_videos) tuple
        """
        if video_names is None:
            video_names = [Path(p).stem for p in video_paths]
        
        new_videos = []
        duplicates = []
        
        for video_path, video_name in zip(video_paths, video_names):
            video_hash = self._compute_video_hash(video_path)
            
            # Check if this video already exists
            if video_hash in self.config["video_registry"]:
                existing = self.config["video_registry"][video_hash]
                duplicates.append(f"{video_name} (duplicate of {existing['name']})")
            else:
                self.config["video_registry"][video_hash] = {
                    "name": video_name,
                    "original_path": video_path,
                    "added_date": datetime.now().isoformat(),
                    "hash": video_hash
                }
                new_videos.append(video_name)
        
        self._save_config()
        return new_videos, duplicates
    
    def create_dataset_version(
        self,
        new_annotation_dir: str,
        new_images_dir: str,
        version_name: Optional[str] = None,
        merge_with_previous: bool = True,
        split_method: str = "rally",
        **split_kwargs
    ) -> str:
        """
        Create a new dataset version by merging or replacing previous data.
        
        Args:
            new_annotation_dir: Directory with new annotations.json
            new_images_dir: Directory with new frames
            version_name: Optional custom version name
            merge_with_previous: If True, merge with latest version; if False, use only new data
            split_method: 'rally' or 'sequence'
            **split_kwargs: Additional arguments for splitting
        
        Returns:
            Path to new dataset version directory
        """
        # Determine version number
        self.config["current_version"] += 1
        version_num = self.config["current_version"]
        
        if version_name is None:
            version_name = f"v{version_num}"
        
        version_dir = self.datasets_dir / version_name
        raw_dir = version_dir / "raw"
        splits_dir = version_dir / "splits"
        
        raw_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Creating Dataset Version: {version_name}")
        print(f"{'='*70}")
        
        # Merge or copy data
        if merge_with_previous and version_num > 1:
            merged_annotations, merged_images_dir = self._merge_datasets(
                new_annotation_dir, new_images_dir, raw_dir
            )
        else:
            merged_annotations = new_annotation_dir
            merged_images_dir = new_images_dir
            print("  Using only new data (no merge)")
        
        # Store metadata
        dataset_info = {
            "version": version_name,
            "version_number": version_num,
            "created_date": datetime.now().isoformat(),
            "merge_with_previous": merge_with_previous,
            "split_method": split_method,
            "split_kwargs": split_kwargs,
            "raw_data_path": str(raw_dir),
            "splits_path": str(splits_dir),
        }
        
        self.config["datasets"][version_name] = dataset_info
        self._save_config()
        
        # Perform splitting
        print(f"\nSplitting dataset using method: {split_method}")
        self._perform_split(
            merged_images_dir, 
            merged_annotations, 
            splits_dir, 
            split_method, 
            **split_kwargs
        )
        
        print(f"\n✓ Dataset version {version_name} created at: {version_dir}")
        return str(splits_dir)
    
    def _merge_datasets(
        self, 
        new_annotation_dir: str, 
        new_images_dir: str,
        output_raw_dir: Path
    ) -> Tuple[str, str]:
        """
        Merge new dataset with previous version.
        
        Returns:
            (merged_annotation_dir, merged_images_dir)
        """
        print("\n  Merging with previous dataset version...")
        
        # Get previous version
        prev_version_num = self.config["current_version"]
        prev_version = None
        
        for v_name, v_info in self.config["datasets"].items():
            if v_info["version_number"] == prev_version_num:
                prev_version = v_info
                break
        
        if prev_version is None:
            print("  No previous version found, using new data only")
            return new_annotation_dir, new_images_dir
        
        # Setup merge directories
        merged_images_dir = output_raw_dir / "images"
        merged_annotations_dir = output_raw_dir / "annotations"
        merged_images_dir.mkdir(parents=True, exist_ok=True)
        merged_annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Load annotations
        prev_ann_path = Path(prev_version["raw_data_path"]) / "annotations" / "annotations.json"
        new_ann_path = Path(new_annotation_dir) / "annotations.json"
        
        prev_annotations = {}
        if prev_ann_path.exists():
            with open(prev_ann_path) as f:
                prev_annotations = json.load(f)
        
        with open(new_ann_path) as f:
            new_annotations = json.load(f)
        
        # Merge annotations (new data takes precedence for conflicts)
        merged_annotations = prev_annotations.copy()
        
        # Track frame name conflicts and rename if needed
        conflict_count = 0
        for frame_name, ann_data in new_annotations.items():
            if frame_name in merged_annotations:
                # Rename to avoid overwrite
                base, ext = os.path.splitext(frame_name)
                new_frame_name = f"{base}_v{self.config['current_version'] + 1}{ext}"
                merged_annotations[new_frame_name] = ann_data
                conflict_count += 1
                
                # Copy image with new name
                src_img = Path(new_images_dir) / frame_name
                if src_img.exists():
                    shutil.copy2(src_img, merged_images_dir / new_frame_name)
            else:
                merged_annotations[frame_name] = ann_data
                
                # Copy image
                src_img = Path(new_images_dir) / frame_name
                if src_img.exists():
                    shutil.copy2(src_img, merged_images_dir / frame_name)
        
        # Copy previous images
        prev_images_dir = Path(prev_version["raw_data_path"]) / "images"
        if prev_images_dir.exists():
            for img_file in prev_images_dir.iterdir():
                if img_file.is_file() and img_file.name in prev_annotations:
                    shutil.copy2(img_file, merged_images_dir / img_file.name)
        
        # Save merged annotations
        merged_ann_path = merged_annotations_dir / "annotations.json"
        with open(merged_ann_path, 'w') as f:
            json.dump(merged_annotations, f, indent=2)
        
        print(f"  Previous frames: {len(prev_annotations)}")
        print(f"  New frames: {len(new_annotations)}")
        print(f"  Frame name conflicts resolved: {conflict_count}")
        print(f"  Total merged frames: {len(merged_annotations)}")
        
        return str(merged_annotations_dir), str(merged_images_dir)
    
    def _perform_split(
        self, 
        images_dir: str, 
        annotation_dir: str, 
        output_dir: Path,
        split_method: str,
        **kwargs
    ):
        """Perform dataset splitting using rally-aware splitter."""
        from data_splitter import RallyAwareDatasetSplitter
        
        if split_method == "rally":
            RallyAwareDatasetSplitter.split_by_rally(
                images_dir=images_dir,
                annotation_output_dir=annotation_dir,
                output_base_dir=str(output_dir),
                **kwargs
            )
        elif split_method == "sequence":
            RallyAwareDatasetSplitter.split_by_sequence(
                images_dir=images_dir,
                annotation_output_dir=annotation_dir,
                output_base_dir=str(output_dir),
                **kwargs
            )
        else:
            raise ValueError(f"Unknown split method: {split_method}")
    
    def train_model(
        self,
        dataset_version: str,
        model_type: str,  # 'yolo', 'obb', 'tracknet'
        training_mode: str = "full",  # 'full' or 'finetune'
        base_weights: Optional[str] = None,  # For fine-tuning
        model_name_suffix: Optional[str] = None,
        **training_kwargs
    ) -> str:
        """
        Train a model on a specific dataset version.
        
        Args:
            dataset_version: Version name (e.g., 'v1', 'v2')
            model_type: 'yolo', 'obb', or 'tracknet'
            training_mode: 'full' (train from scratch/pretrained) or 'finetune' (from your previous model)
            base_weights: Path to weights for fine-tuning (uses latest model if None)
            model_name_suffix: Optional suffix for model name
            **training_kwargs: Additional training arguments
        
        Returns:
            Path to trained model weights
        """
        if dataset_version not in self.config["datasets"]:
            raise ValueError(f"Dataset version {dataset_version} not found")
        
        dataset_info = self.config["datasets"][dataset_version]
        splits_dir = dataset_info["splits_path"]
        
        # Setup model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{model_name_suffix}" if model_name_suffix else ""
        model_dir_name = f"{model_type}_{dataset_version}_{training_mode}_{timestamp}{suffix}"
        model_output_dir = self.models_dir / model_dir_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()} - {training_mode.upper()} mode")
        print(f"Dataset: {dataset_version}")
        print(f"{'='*70}")
        
        # Import trainers
        from model_trainer import YOLOTrainer, TrackNetTrainer
        
        # Handle fine-tuning
        if training_mode == "finetune":
            if base_weights is None:
                # Find latest model of this type
                base_weights = self._find_latest_model(model_type)
                if base_weights:
                    print(f"  Using base weights: {base_weights}")
                else:
                    print(f"  No previous {model_type} model found, training from pretrained")
                    training_mode = "full"
        
        # Train
        model = None
        if model_type == "yolo":
            if training_mode == "finetune" and base_weights:
                training_kwargs['model'] = base_weights
                training_kwargs['pretrained'] = False
            
            model, metrics = YOLOTrainer.train_standard(
                split_dir=splits_dir,
                output_dir=str(model_output_dir),
                **training_kwargs
            )
            weights_path = model_output_dir / "yolo_standard" / "weights" / "best.pt"
        
        elif model_type == "obb":
            if training_mode == "finetune" and base_weights:
                training_kwargs['model'] = base_weights
                training_kwargs['pretrained'] = False
            
            model, metrics = YOLOTrainer.train_obb(
                split_dir=splits_dir,
                output_dir=str(model_output_dir),
                **training_kwargs
            )
            weights_path = model_output_dir / "yolo_obb" / "weights" / "best.pt"
        
        elif model_type == "tracknet":
            # TrackNet fine-tuning - load checkpoint and continue training
            model = TrackNetTrainer.train(
                split_dir=splits_dir,
                output_dir=str(model_output_dir),
                resume_from=base_weights if training_mode == "finetune" else None,
                **training_kwargs
            )
            weights_path = model_output_dir / "tracknet_best.pth"
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Store model metadata
        model_info = {
            "model_type": model_type,
            "dataset_version": dataset_version,
            "training_mode": training_mode,
            "base_weights": base_weights,
            "weights_path": str(weights_path),
            "output_dir": str(model_output_dir),
            "training_kwargs": training_kwargs,
            "created_date": datetime.now().isoformat(),
        }
        
        model_id = f"{model_type}_{dataset_version}_{timestamp}"
        self.config["models"][model_id] = model_info
        self._save_config()
        
        print(f"\n✓ Model saved: {model_id}")
        print(f"  Weights: {weights_path}")
        
        return str(weights_path)
    
    def _find_latest_model(self, model_type: str) -> Optional[str]:
        """Find the latest trained model of a specific type."""
        latest_model = None
        latest_date = None
        
        for model_id, model_info in self.config["models"].items():
            if model_info["model_type"] == model_type:
                model_date = datetime.fromisoformat(model_info["created_date"])
                if latest_date is None or model_date > latest_date:
                    latest_date = model_date
                    latest_model = model_info["weights_path"]
        
        return latest_model
    
    def list_datasets(self):
        """List all dataset versions."""
        print(f"\n{'='*70}")
        print("Dataset Versions")
        print(f"{'='*70}")
        
        for version_name in sorted(self.config["datasets"].keys()):
            info = self.config["datasets"][version_name]
            print(f"\n{version_name}:")
            print(f"  Created: {info['created_date']}")
            print(f"  Method: {info['split_method']}")
            print(f"  Path: {info['splits_path']}")
    
    def list_models(self):
        """List all trained models."""
        print(f"\n{'='*70}")
        print("Trained Models")
        print(f"{'='*70}")
        
        for model_id in sorted(self.config["models"].keys()):
            info = self.config["models"][model_id]
            print(f"\n{model_id}:")
            print(f"  Type: {info['model_type']}")
            print(f"  Dataset: {info['dataset_version']}")
            print(f"  Mode: {info['training_mode']}")
            print(f"  Weights: {info['weights_path']}")
            print(f"  Created: {info['created_date']}")
    
    def export_dataset_yaml(self, dataset_version: str, output_path: Optional[str] = None) -> str:
        """
        Export a YOLO-compatible dataset.yaml for a specific version.
        Useful for external training scripts.
        """
        if dataset_version not in self.config["datasets"]:
            raise ValueError(f"Dataset version {dataset_version} not found")
        
        dataset_info = self.config["datasets"][dataset_version]
        splits_dir = Path(dataset_info["splits_path"])
        
        if output_path is None:
            output_path = splits_dir / "dataset.yaml"
        
        config = {
            "path": str(splits_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 1,
            "names": ["shuttle"],
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✓ Exported dataset.yaml to: {output_path}")
        return str(output_path)


# ══════════════════════════════════════════════════════════════════════════
# Example Usage
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental Training Manager")
    parser.add_argument("--action", required=True, choices=[
        "register-videos", "create-dataset", "train", "list-datasets", 
        "list-models", "export-yaml"
    ])
    
    # Common args
    parser.add_argument("--base-dir", default="./shuttle_tracking_project")
    
    # Register videos
    parser.add_argument("--videos", nargs='+', help="Video file paths")
    parser.add_argument("--video-names", nargs='+', help="Custom video names")
    
    # Create dataset
    parser.add_argument("--annotation-dir", help="Directory with annotations.json")
    parser.add_argument("--images-dir", help="Directory with frame images")
    parser.add_argument("--version-name", help="Custom version name")
    parser.add_argument("--no-merge", action="store_true", help="Don't merge with previous")
    parser.add_argument("--split-method", default="rally", choices=["rally", "sequence"])
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    
    # Train
    parser.add_argument("--dataset-version", help="Dataset version to use")
    parser.add_argument("--model-type", choices=["yolo", "obb", "tracknet"])
    parser.add_argument("--mode", default="full", choices=["full", "finetune"])
    parser.add_argument("--base-weights", help="Weights for fine-tuning")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"])
    
    args = parser.parse_args()
    
    manager = IncrementalTrainingManager(args.base_dir)
    
    if args.action == "register-videos":
        new, dupes = manager.register_new_videos(args.videos, args.video_names)
        print(f"\nNew videos: {len(new)}")
        for v in new:
            print(f"  - {v}")
        if dupes:
            print(f"\nDuplicates detected: {len(dupes)}")
            for d in dupes:
                print(f"  - {d}")
    
    elif args.action == "create-dataset":
        splits_dir = manager.create_dataset_version(
            new_annotation_dir=args.annotation_dir,
            new_images_dir=args.images_dir,
            version_name=args.version_name,
            merge_with_previous=not args.no_merge,
            split_method=args.split_method,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
        print(f"Dataset created at: {splits_dir}")
    
    elif args.action == "train":
        weights_path = manager.train_model(
            dataset_version=args.dataset_version,
            model_type=args.model_type,
            training_mode=args.mode,
            base_weights=args.base_weights,
            epochs=args.epochs,
            batch=args.batch,
            model_size=args.model_size
        )
        print(f"Model trained: {weights_path}")
    
    elif args.action == "list-datasets":
        manager.list_datasets()
    
    elif args.action == "list-models":
        manager.list_models()
    
    elif args.action == "export-yaml":
        yaml_path = manager.export_dataset_yaml(args.dataset_version)
        print(f"YAML exported: {yaml_path}")