
import cv2
import torch
import asyncio
import argparse
from pathlib import Path
from ultralytics import YOLO

# Import the manual tracking system
from manual_tracking import (
    ShuttleInitializer,
    shuttle_tracking_with_manual_init
)


async def process_video_with_manual_init(video_path, tracking_method='template', 
                                         num_init_frames=10, output_name=None):
    """
    Main processing function with manual initialization
    
    Args:
        video_path: Path to input video
        tracking_method: 'template' (default) or 'optical_flow'
        num_init_frames: Number of frames to show for initialization (default: 5)
        output_name: Custom output name
    """
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"MANUAL SHUTTLE INITIALIZATION + AUTOMATIC TRACKING")
    print(f"{'='*70}\n")
    print(f"Input Video: {video_path}")
    print(f"Tracking Method: {tracking_method}")
    print(f"Initialization Frames: {num_init_frames}")
    print()
    
    # ==========================================
    # STEP 1: MANUAL INITIALIZATION
    # ==========================================
    
    print("STEP 1: Manual Shuttle Initialization")
    print("-" * 70)
    
    initializer = ShuttleInitializer(str(video_path), num_init_frames=num_init_frames)
    manual_positions = initializer.run()
    
    if manual_positions is None or len(manual_positions) == 0:
        print("❌ No shuttle positions marked. Exiting.")
        return
    
    print(f"\n✓ Collected {len(manual_positions)} manual positions:")
    for i, pos in enumerate(manual_positions):
        print(f"  {i+1}. Frame {pos['frame_idx']}: ({pos['x']}, {pos['y']})")
    print()
    
    # ==========================================
    # STEP 2: LOAD MODELS
    # ==========================================
    
    print("STEP 2: Loading AI Models")
    print("-" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    
    try:
        print("Loading YOLO Person Tracker...")
        person_model = YOLO("yolov8n.pt")
        print("✓ Person tracker loaded")
        
        print("Loading YOLO Pose Model...")
        pose_model = YOLO("yolov8n-pose.pt")
        print("✓ Pose model loaded")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    print()
    
    # ==========================================
    # STEP 3: AUTOMATIC TRACKING
    # ==========================================
    
    print("STEP 3: Automatic Tracking")
    print("-" * 70)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file")
        return
    
    # Determine output name
    if output_name is None:
        output_name = video_path.stem
    
    # Process video
    last_progress = 0
    
    try:
        async for update in shuttle_tracking_with_manual_init(
            cap, person_model, pose_model, device,
            video_filename=output_name,
            manual_positions=manual_positions,
            tracking_method=tracking_method
        ):
            
            if update["type"] == "started":
                print("✓ Tracking started")
                print(f"  Total frames: {update['total_frames']}")
                print(f"  Manual positions: {update['manual_positions_count']}")
                print()
            
            elif update["type"] == "progress":
                progress = update["progress_percent"]
                
                # Print progress every 10%
                if progress - last_progress >= 10:
                    print(f"Progress: {progress:.1f}% | "
                          f"Frame {update['frame']}/{update['total_frames']} | "
                          f"Detections: {update['detections']} ({update['detection_rate']:.1f}%)")
                    last_progress = progress
            
            elif update["type"] == "complete":
                print("\n" + "="*70)
                print("TRACKING COMPLETE!")
                print("="*70)
                print(f"\nResults:")
                print(f"  Total Frames: {update['total_frames']}")
                print(f"  Shuttle Detections: {update['detections']}")
                print(f"  Detection Rate: {update['detection_rate']:.1f}%")
                print(f"\nOutput Video: {update['output_video']}")
                print("="*70)
            
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
        description="Manual Shuttle Initialization + Automatic Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (template matching, 5 init frames)
  python main_manual_tracking.py video.mp4
  
  # Use optical flow tracking (more robust)
  python main_manual_tracking.py video.mp4 --method optical_flow
  
  # Use more frames for initialization (better template)
  python main_manual_tracking.py video.mp4 --frames 10
  
  # Custom output name
  python main_manual_tracking.py video.mp4 --output my_game

Instructions:
  1. The video will pause on first few frames
  2. Click on the shuttle in each frame (left mouse button)
  3. Press SPACE to go to next frame
  4. Press BACKSPACE to go back
  5. Press ENTER when done marking (minimum 1 position)
  6. System will automatically track shuttle through entire video
        """
    )
    
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--method", "-m", 
                       choices=['template', 'optical_flow'],
                       default='template',
                       help="Tracking method (default: template)")
    parser.add_argument("--frames", "-f", type=int, default=5,
                       help="Number of initialization frames (default: 5)")
    parser.add_argument("--output", "-o", help="Custom output name", default=None)
    
    args = parser.parse_args()
    
    # Run processing
    asyncio.run(process_video_with_manual_init(
        args.video,
        tracking_method=args.method,
        num_init_frames=args.frames,
        output_name=args.output
    ))


if __name__ == "__main__":
    main()