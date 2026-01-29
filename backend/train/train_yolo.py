import os
from ultralytics import YOLO
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def prepare_yolo_dataset(video_folder, output_folder):
    """
    Helper to organize dataset in YOLO format
    
    Expected structure:
    output_folder/
        images/
            train/
            val/
        labels/
            train/
            val/
        dataset.yaml
    """
    os.makedirs(f"{output_folder}/images/train", exist_ok=True)
    os.makedirs(f"{output_folder}/images/val", exist_ok=True)
    os.makedirs(f"{output_folder}/labels/train", exist_ok=True)
    os.makedirs(f"{output_folder}/labels/val", exist_ok=True)
    
    # Create dataset.yaml
    dataset_config = {
        'path': os.path.abspath(output_folder),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'shuttle'
        },
        'nc': 1  # number of classes
    }
    
    with open(f"{output_folder}/dataset.yaml", 'w') as f:
        yaml.dump(dataset_config, f)
    
    print(f"Dataset structure created at {output_folder}")
    print("\nNext steps:")
    print("1. Extract frames from your videos")
    print("2. Annotate shuttles using a tool like Roboflow, CVAT, or LabelImg")
    print("3. Export in YOLO format")
    print("4. Place images in images/train and images/val")
    print("5. Place labels in labels/train and labels/val")


def train_yolo_shuttle_detector(data_yaml, epochs=100, img_size=640, batch_size=16):
    """
    Train YOLOv8n for shuttle detection
    
    Args:
        data_yaml: Path to dataset.yaml file
        epochs: Number of training epochs
        img_size: Image size for training
        batch_size: Batch size
    
    Returns:
        Trained model
    """
    # Load pre-trained YOLOv8n model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=str(SCRIPT_DIR / "runs"),
        name='shuttle_detector',
        patience=20,  # Early stopping patience
        save=True,
        device=0,  # Use GPU 0, or 'cpu' for CPU training
        
        # Augmentation parameters (important for small objects like shuttles)
        degrees=10.0,  # Rotation
        translate=0.1,  # Translation
        scale=0.5,  # Scale
        flipud=0.0,  # No vertical flip (shuttles have orientation)
        fliplr=0.5,  # Horizontal flip
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.1,  # Mixup augmentation
        
        # Optimization
        optimizer='Adam',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # Other settings
        exist_ok=True,
        pretrained=True,
        verbose=True
    )
    
    print("\nTraining complete!")
    print(f"Best model saved at: runs/detect/shuttle_detector/weights/best.pt")
    
    return model


def validate_model(model_path, data_yaml):
    """Validate trained model"""
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    
    print("\nValidation Metrics:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall: {metrics.box.mr:.3f}")
    
    return metrics


def export_model(model_path, format='onnx'):
    """Export model for deployment"""
    model = YOLO(model_path)
    model.export(format=format)
    print(f"Model exported to {format} format")


if __name__ == "__main__":
    import argparse
    DEFAULT_MODEL = SCRIPT_DIR / "runs/detect/shuttle_detector/weights/best.pt"

    parser = argparse.ArgumentParser(description='Train YOLO for shuttle detection')
    parser.add_argument('--prepare', action='store_true', help='Prepare dataset structure')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--validate', action='store_true', help='Validate model')
    parser.add_argument('--export', action='store_true', help='Export model')
    
    parser.add_argument('--data', type=str, default='dataset/dataset.yaml', help='Path to dataset.yaml')
    parser.add_argument('--output', type=str, default='dataset', help='Output folder for dataset')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL))
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    
    args = parser.parse_args()
    
    if args.prepare:
        prepare_yolo_dataset('videos', args.output)
    
    if args.train:
        train_yolo_shuttle_detector(
            data_yaml=args.data,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch
        )
    
    if args.validate:
        validate_model(args.model, args.data)
    
    if args.export:
        export_model(args.model)
