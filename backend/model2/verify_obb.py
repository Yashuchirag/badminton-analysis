"""
Verify and fix OBB label format for YOLO training
"""

import os
import sys

def check_obb_labels(split_dir):
    """Check if OBB labels are correctly formatted."""
    
    issues = []
    checked = 0
    
    for split in ['train', 'val', 'test']:
        obb_dir = os.path.join(split_dir, split, 'obb_labels')
        
        if not os.path.exists(obb_dir):
            issues.append(f"Missing directory: {obb_dir}")
            continue
        
        label_files = [f for f in os.listdir(obb_dir) if f.endswith('.txt')]
        
        for label_file in label_files:
            label_path = os.path.join(obb_dir, label_file)
            checked += 1
            
            # Check if file is empty (valid for not_visible frames)
            if os.path.getsize(label_path) == 0:
                continue
            
            # Check format
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    
                    # OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    # Should have 9 values total (1 class + 8 coordinates)
                    if len(parts) != 9:
                        issues.append(
                            f"{split}/{label_file}:{line_num} - "
                            f"Expected 9 values (class + 8 coords), got {len(parts)}"
                        )
                        continue
                    
                    # Check class ID is 0
                    try:
                        class_id = int(parts[0])
                        if class_id != 0:
                            issues.append(
                                f"{split}/{label_file}:{line_num} - "
                                f"Class ID should be 0, got {class_id}"
                            )
                    except ValueError:
                        issues.append(
                            f"{split}/{label_file}:{line_num} - "
                            f"Invalid class ID: {parts[0]}"
                        )
                    
                    # Check coordinates are normalized (0-1)
                    try:
                        coords = [float(x) for x in parts[1:]]
                        for i, coord in enumerate(coords):
                            if coord < 0 or coord > 1:
                                issues.append(
                                    f"{split}/{label_file}:{line_num} - "
                                    f"Coordinate {i+1} out of range [0,1]: {coord:.6f}"
                                )
                    except ValueError as e:
                        issues.append(
                            f"{split}/{label_file}:{line_num} - "
                            f"Invalid coordinate: {e}"
                        )
    
    # Report
    print(f"\n{'='*70}")
    print("OBB LABEL VERIFICATION")
    print(f"{'='*70}")
    print(f"Checked: {checked} label files")
    print(f"Issues found: {len(issues)}")
    
    if issues:
        print("\nISSUES:")
        for issue in issues[:20]:  # Show first 20
            print(f"  ✗ {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
        print(f"\n{'='*70}")
        return False
    else:
        print("  ✓ All labels are correctly formatted")
        print(f"{'='*70}")
        return True


def show_sample_labels(split_dir, n=3):
    """Show sample OBB labels for inspection."""
    print(f"\n{'='*70}")
    print(f"SAMPLE OBB LABELS (first {n} non-empty files)")
    print(f"{'='*70}")
    
    shown = 0
    for split in ['train', 'val']:
        if shown >= n:
            break
        
        obb_dir = os.path.join(split_dir, split, 'obb_labels')
        if not os.path.exists(obb_dir):
            continue
        
        for label_file in sorted(os.listdir(obb_dir)):
            if shown >= n:
                break
            if not label_file.endswith('.txt'):
                continue
            
            label_path = os.path.join(obb_dir, label_file)
            
            # Skip empty files
            if os.path.getsize(label_path) == 0:
                continue
            
            with open(label_path, 'r') as f:
                content = f.read().strip()
                if content:
                    print(f"\n{split}/{label_file}:")
                    print(f"  {content}")
                    shown += 1
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_obb_labels.py <split_dir>")
        print("Example: python verify_obb_labels.py ./dataset/processed/match2")
        sys.exit(1)
    
    split_dir = sys.argv[1]
    
    if not os.path.exists(split_dir):
        print(f"Error: Directory not found: {split_dir}")
        sys.exit(1)
    
    # Show samples first
    show_sample_labels(split_dir)
    
    # Then verify
    if check_obb_labels(split_dir):
        print("\n✓ Ready for YOLO-OBB training")
    else:
        print("\n✗ Fix issues before training")
        print("\nCommon fixes:")
        print("  1. Re-run the annotation tool's save step")
        print("  2. Check that rally_aware_splitter copied obb_labels correctly")
        print("  3. Verify annotations.json has 'angle' field for oriented boxes")