import os
import yaml


def create_dataset_yaml(dataset_folder='dataset'):
    """
    Create dataset.yaml file for YOLO training
    
    Args:
        dataset_folder: Root folder of your dataset (default: 'dataset')
    """
    
    # Ensure required folders exist
    required_folders = [
        f"{dataset_folder}/images/train",
        f"{dataset_folder}/images/val",
        f"{dataset_folder}/labels/train",
        f"{dataset_folder}/labels/val",
    ]
    
    print("Checking dataset structure...")
    for folder in required_folders:
        if not os.path.exists(folder):
            print(f"  Creating: {folder}")
            os.makedirs(folder, exist_ok=True)
        else:
            print(f"  ✓ Exists: {folder}")
    
    # Create the dataset.yaml configuration
    dataset_config = {
        'path': os.path.abspath(dataset_folder),  # Absolute path to dataset root
        'train': 'images/train',                   # Path to training images (relative to 'path')
        'val': 'images/val',                       # Path to validation images (relative to 'path')
        'names': {                                 # Class names
            0: 'shuttle'                           # Class 0 is 'shuttle'
        },
        'nc': 1  # Number of classes (we only have 1: shuttle)
    }
    
    # Save to YAML file
    yaml_path = f"{dataset_folder}/dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"\n✓ Created: {yaml_path}")
    print("\nContents of dataset.yaml:")
    print("-" * 50)
    with open(yaml_path, 'r') as f:
        print(f.read())
    print("-" * 50)
    
    # Show summary
    print("\nDataset structure:")
    print(f"""
{dataset_folder}/
├── dataset.yaml          ← Configuration file (just created!)
├── images/
│   ├── train/           ← Put training images here
│   └── val/             ← Put validation images here
└── labels/
    ├── train/           ← Put training labels here (.txt files)
    └── val/             ← Put validation labels here (.txt files)
    """)
    
    # Count files
    train_images = len([f for f in os.listdir(f"{dataset_folder}/images/train") 
                       if f.endswith(('.jpg', '.png'))]) if os.path.exists(f"{dataset_folder}/images/train") else 0
    val_images = len([f for f in os.listdir(f"{dataset_folder}/images/val") 
                     if f.endswith(('.jpg', '.png'))]) if os.path.exists(f"{dataset_folder}/images/val") else 0
    train_labels = len([f for f in os.listdir(f"{dataset_folder}/labels/train") 
                       if f.endswith('.txt')]) if os.path.exists(f"{dataset_folder}/labels/train") else 0
    val_labels = len([f for f in os.listdir(f"{dataset_folder}/labels/val") 
                     if f.endswith('.txt')]) if os.path.exists(f"{dataset_folder}/labels/val") else 0
    
    print("\nCurrent dataset status:")
    print(f"  Training images:   {train_images}")
    print(f"  Training labels:   {train_labels}")
    print(f"  Validation images: {val_images}")
    print(f"  Validation labels: {val_labels}")
    
    if train_images == 0:
        print("\n⚠️  No training images found!")
        print("Next step: Extract frames from videos")
        print(f"  python data_preparation.py --action extract --video YOUR_VIDEO.mp4 --output {dataset_folder}/images/train")
    elif train_labels == 0:
        print("\n⚠️  No training labels found!")
        print("Next step: Annotate the images")
        print(f"  python data_preparation.py --action annotate --images {dataset_folder}/images/train --output {dataset_folder}/labels/train")
    elif train_images == train_labels and val_images == val_labels and val_images > 0:
        print("\n✓ Dataset looks ready for training!")
        print("Next step: Train the model")
        print(f"  python train_yolo.py --train --data {dataset_folder}/dataset.yaml --epochs 100")
    else:
        print("\n⚠️  Dataset incomplete:")
        if train_images != train_labels:
            print(f"  Training: {train_images} images but {train_labels} labels (should match!)")
        if val_images != val_labels:
            print(f"  Validation: {val_images} images but {val_labels} labels (should match!)")
        if val_images == 0:
            print("  No validation data (need at least some validation images)")
    
    return yaml_path




if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create dataset.yaml for YOLO training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dataset.yaml in default location (dataset/)
  python create_dataset_yaml.py
  
  # Create dataset.yaml in custom location
  python create_dataset_yaml.py --folder my_dataset

        """
    )
    
    parser.add_argument('--folder', default='dataset',
                       help='Dataset folder path (default: dataset)')
    
    args = parser.parse_args()
    print("="*70)
    print("DATASET.YAML CREATOR")
    print("="*70)
    create_dataset_yaml(args.folder)
    print("\n" + "="*70)
    
        