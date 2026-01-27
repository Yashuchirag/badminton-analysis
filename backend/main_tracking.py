import cv2
import torch
import asyncio
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

# Import the hybrid tracking function
# Make sure the previous artifact code is saved as 'hybrid_tracker.py'
from hybrid_tracker import shuttle_tracking_hybrid


def load_tracknet_model(model_path, device):
    """
    Load TrackNet model
    
    If you have a TrackNet checkpoint, load it here.
    If not, this returns None and the system will use YOLO + Color detection only.
    """
    try:
        if Path(model_path).exists():
            print(f"Loading TrackNet model from {model_path}...")
            model = torch.load(model_path, map_location=device)
            model.eval()
            print("✓ TrackNet model loaded successfully")
            return model
        else:
            print(f"⚠ TrackNet model not found at {model_path}")
            print("  Continuing with YOLO + Color detection only")
            return None
    except Exception as e:
        print(f"⚠ Error loading TrackNet: {e}")
        print("  Continuing with YOLO + Color detection only")
        return None


class DummyTrackNet:
    """Dummy TrackNet that always returns no detection (for when TrackNet is unavailable)"""
    def __call__(self, x):
        # Return zero heatmap
        return torch.zeros(1, 1, 540, 960)


async def process_video(video_path, tracknet_path=None, output_name=None):
    """
    Main processing function
    
    Args:
        video_path: Path to input video
        tracknet_path: Path to TrackNet model (optional)
        output_name: Custom output name (optional)
    """
    
    # Check if video exists
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"BADMINTON TRACKING SYSTEM - HYBRID SHUTTLE DETECTION")
    print(f"{'='*70}\n")
    print(f"Input Video: {video_path}")
    print(f"File Size: {video_path.stat().st_size / 1024 / 1024:.2f} MB\n")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print()
    
    # Load models
    print("Loading AI Models...")
    print("-" * 70)
    
    # 1. Person detection & tracking (YOLOv8)
    print("1. Loading YOLO Person Tracker...")
    try:
        person_model = YOLO("yolov8n.pt")
        print("   ✓ YOLOv8n loaded (person detection & tracking)")
    except Exception as e:
        print(f"   ❌ Error loading YOLO: {e}")
        return
    
    # 2. Pose estimation (YOLOv8-pose)
    print("2. Loading YOLO Pose Model...")
    try:
        pose_model = YOLO("yolov8n-pose.pt")
        print("   ✓ YOLOv8n-pose loaded (wrist tracking)")
    except Exception as e:
        print(f"   ❌ Error loading pose model: {e}")
        return
    
    # 3. TrackNet (shuttle detection)
    print("3. Loading TrackNet Model...")
    if tracknet_path:
        tracknet = load_tracknet_model(tracknet_path, device)
    else:
        tracknet = None
    
    if tracknet is None:
        print("   ℹ Using DummyTrackNet (YOLO + Color detection only)")
        tracknet = DummyTrackNet()
    
    print("\n" + "="*70)
    print("All models loaded successfully!")
    print("="*70 + "\n")
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file")
        return
    
    # Get video info
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print("Video Information:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print()
    
    # Determine output name
    if output_name is None:
        output_name = video_path.stem
    
    # Process video
    print("Starting video processing...")
    print("="*70 + "\n")
    
    last_progress = 0
    
    try:
        async for update in shuttle_tracking_hybrid(
            cap, person_model, pose_model, tracknet, device, output_name
        ):
            
            if update["type"] == "started":
                print("✓ Processing started")
                print(f"  Output will be saved to: outputs/{output_name}_annotated.mp4\n")
            
            elif update["type"] == "progress":
                progress = update["progress_percent"]
                
                # Print progress every 10%
                if progress - last_progress >= 10:
                    print(f"Progress: {progress}% | "
                          f"Frame {update['frame']}/{update['total_frames']} | "
                          f"People: {update['people_count']} | "
                          f"Shuttle: {'✓' if update['shuttle_detected'] else '✗'} ({update['detection_method']})")
                    last_progress = progress
            
            elif update["type"] == "complete":
                print("\n" + "="*70)
                print("PROCESSING COMPLETE!")
                print("="*70)
                print(f"\nResults:")
                print(f"  Total Frames Processed: {update['total_frames']}")
                print(f"  Unique People Tracked: {update['unique_people']}")
                print(f"  Shuttle Detection Rate: {update['summary']['detection_rate']}%")
                print(f"\nDetection Breakdown:")
                print(f"  TrackNet: {update['detection_breakdown']['tracknet']} ({update['summary']['tracknet_rate']}%)")
                print(f"  YOLO: {update['detection_breakdown']['yolo']} ({update['summary']['yolo_rate']}%)")
                print(f"  Color: {update['detection_breakdown']['color']} ({update['summary']['color_rate']}%)")
                print(f"\nOutput Video: {update['output_video']}")
                print("\n" + "="*70)
            
            elif update["type"] == "error":
                print(f"\n❌ Error: {update['message']}")
                return
    
    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCleanup complete.")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Hybrid Badminton Shuttle Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with TrackNet model
  python main_tracking.py video.mp4 --tracknet tracknet_model.pth
  
  # Process video without TrackNet (YOLO + Color only)
  python main_tracking.py video.mp4
  
  # Process with custom output name
  python main_tracking.py video.mp4 --output my_game
        """
    )
    
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--tracknet", "-t", help="Path to TrackNet model file (.pth)", default=None)
    parser.add_argument("--output", "-o", help="Custom output name (without extension)", default=None)
    
    args = parser.parse_args()
    
    # Run async processing
    asyncio.run(process_video(args.video, args.tracknet, args.output))


if __name__ == "__main__":
    main()