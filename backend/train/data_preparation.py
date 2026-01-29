"""
Data preparation utilities for shuttle tracking
Includes frame extraction, auto-labeling, and annotation helpers
"""

import cv2
import os
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


class VideoFrameExtractor:
    """Extract frames from videos for training"""
    
    @staticmethod
    def extract_frames(video_path, output_dir, sample_rate=5, max_frames=None):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            sample_rate: Extract every Nth frame
            max_frames: Maximum number of frames to extract
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        extracted = 0
        frame_num = 0
        
        print(f"Extracting frames from {video_path}")
        
        with tqdm(total=min(total_frames // sample_rate, max_frames or float('inf'))) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_num % sample_rate == 0:
                    output_path = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
                    cv2.imwrite(output_path, frame)
                    extracted += 1
                    pbar.update(1)
                    
                    if max_frames and extracted >= max_frames:
                        break
                
                frame_num += 1
        
        cap.release()
        print(f"Extracted {extracted} frames to {output_dir}")
        
        return extracted


class ShuttleAnnotationTool:
    """Simple annotation tool for labeling shuttle positions"""
    
    def __init__(self, image_dir, output_dir):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.display_width = 800  # or 1280 if Monitor
        self.scale = None
        os.makedirs(output_dir, exist_ok=True)
        
        self.images = sorted([f for f in os.listdir(image_dir) 
                            if f.endswith(('.jpg', '.png'))])
        print(f"Total images: {len(self.images)}")
        self.current_idx = 0
        self.annotations = {}
        
    def annotate(self):
        """Interactive annotation interface"""
        print("Annotation Tool")
        print("-" * 50)
        print("Instructions:")
        print("  - Click on shuttle location")
        print("  - Press 'space' for next image")
        print("  - Press 'a' for previous image")
        print("  - Press 's' to skip (no shuttle visible)")
        print("  - Press 'q' to quit and save")
        print("-" * 50)
        
        cv2.namedWindow('Annotate Shuttle')
        cv2.setMouseCallback('Annotate Shuttle', self._mouse_callback)
        
        self.current_point = None
        
        while self.current_idx < len(self.images):
            image_name = self.images[self.current_idx]
            image_path = os.path.join(self.image_dir, image_name)
            
            img = cv2.imread(image_path)
            h, w = img.shape[:2]
            self.scale = self.display_width / w
            new_w = int(w * self.scale)
            new_h = int(h * self.scale)

            display_img = cv2.resize(img, (new_w, new_h))
            
            # Show existing annotation if available
            if image_name in self.annotations:
                x, y = self.annotations[image_name]
                sx, sy = int(x * self.scale), int(y * self.scale)
                cv2.circle(display_img, (sx, sy), 5, (0, 255, 0), -1)
                cv2.putText(display_img, 'Annotated', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show current point being annotated
            if self.current_point:
                cv2.circle(display_img, (int(self.current_point[0] * self.scale),
                         int(self.current_point[1] * self.scale)),
                         5, (255, 0, 0), -1)
            
            # Display info
            cv2.putText(display_img, f'Image {self.current_idx + 1}/{len(self.images)}',
                       (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Annotate Shuttle', display_img)

            key = cv2.waitKey(1) & 0xFF
            
            
            if key == ord(' '):  # Next
                if self.current_point:
                    self.annotations[image_name] = self.current_point
                self.current_point = None
                self.current_idx += 1
                
            elif key == ord('a'):  # Previous
                if self.current_idx > 0:
                    self.current_idx -= 1
                self.current_point = None
                
            elif key == ord('s'):  # Skip
                self.annotations[image_name] = None
                self.current_point = None
                self.current_idx += 1
                
            elif key == ord('q'):  # Quit
                break
        
        cv2.destroyAllWindows()
        self._save_annotations()
    
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            self.current_point = (orig_x, orig_y)
    
    def _save_annotations(self):
        """Save annotations in YOLO format"""
        for image_name, point in self.annotations.items():
            label_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = os.path.join(self.output_dir, label_name)

            if point is None:
                # create empty YOLO file for skipped frame
                open(label_path, 'w').close()
            else:
                x, y = point
                img = cv2.imread(os.path.join(self.image_dir, image_name))
                h, w = img.shape[:2]
                x_center = x / w
                y_center = y / h
                width = 20 / w
                height = 20 / h
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center} {y_center} {width} {height}\n")

        
        print(f"Saved {len(self.annotations)} annotations to {self.output_dir}")
        
        # Also save raw annotations
        json_path = os.path.join(self.output_dir, 'annotations.json')
        with open(json_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)


class TrackNetDataPreparator:
    """Prepare data for TrackNet training"""
    
    @staticmethod
    def video_to_tracknet_format(video_path, annotations_json, output_path):
        """
        Convert video and frame annotations to TrackNet training format
        
        Args:
            video_path: Path to video
            annotations_json: JSON file with annotations {frame_num: [x, y]}
            output_path: Output pickle file path
        """
        with open(annotations_json, 'r') as f:
            annotations = json.load(f)
        
        # Convert frame names to numbers and normalize coordinates
        processed_annotations = {}
        
        for frame_name, (x, y) in annotations.items():
            # Extract frame number from name
            frame_num = int(''.join(filter(str.isdigit, frame_name)))
            
            # Get video dimensions for normalization
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                processed_annotations[frame_num] = (x / w, y / h)
            cap.release()
        
        # Save processed annotations
        with open(output_path, 'w') as f:
            json.dump(processed_annotations, f, indent=2)
        
        print(f"Processed {len(processed_annotations)} annotations")
        print(f"Saved to {output_path}")
        
        return processed_annotations


def create_sample_dataset():
    """Create a sample dataset structure with example videos"""
    print("Creating sample dataset structure...")
    
    structure = {
        'dataset': {
            'videos': ['Place your training videos here (.mp4, .avi, etc.)'],
            'images': {
                'train': ['Extracted training frames will go here'],
                'val': ['Extracted validation frames will go here']
            },
            'labels': {
                'train': ['YOLO format labels (.txt) for training'],
                'val': ['YOLO format labels (.txt) for validation']
            },
            'annotations': ['TrackNet annotations (.json) will go here']
        }
    }
    
    def create_structure(base_path, struct):
        for key, value in struct.items():
            path = os.path.join(base_path, key)
            if isinstance(value, dict):
                os.makedirs(path, exist_ok=True)
                create_structure(path, value)
            else:
                os.makedirs(path, exist_ok=True)
                readme_path = os.path.join(path, 'README.txt')
                with open(readme_path, 'w') as f:
                    f.write('\n'.join(value))
    
    create_structure('.', structure)
    print("Dataset structure created!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Data preparation utilities')
    parser.add_argument('--action', choices=['extract', 'annotate', 'prepare-tracknet', 'create-structure'],
                       required=True, help='Action to perform')
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--images', type=str, help='Images directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--annotations', type=str, help='Annotations JSON file')
    parser.add_argument('--sample-rate', type=int, default=5, help='Frame sampling rate')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to extract')
    
    args = parser.parse_args()
    
    if args.action == 'extract':
        VideoFrameExtractor.extract_frames(
            args.video, args.output, args.sample_rate, args.max_frames
        )
    
    elif args.action == 'annotate':
        tool = ShuttleAnnotationTool(args.images, args.output)
        tool.annotate()
    
    elif args.action == 'prepare-tracknet':
        TrackNetDataPreparator.video_to_tracknet_format(
            args.video, args.annotations, args.output
        )
    
    elif args.action == 'create-structure':
        create_sample_dataset()
