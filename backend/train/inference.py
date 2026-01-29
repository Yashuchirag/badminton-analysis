import cv2
import numpy as np
import argparse
from shuttle_tracker import ShuttleTracker
import pandas as pd


def track_shuttle_simple(video_path, output_video=None, output_csv=None, 
                         yolo_model='yolov8n.pt', tracknet_model=None,
                         confidence_threshold=0.3, use_pose=False):
    """
    Simple function to track shuttle in a video
    
    Args:
        video_path: Path to input video
        output_video: Path to save output video with tracking visualization
        output_csv: Path to save trajectory data as CSV
        yolo_model: Path to trained YOLO model
        tracknet_model: Path to trained TrackNet model (optional)
        confidence_threshold: Minimum confidence for detections
        use_pose: Whether to detect player poses
    """
    
    print("=" * 70)
    print("SHUTTLE TRACKING")
    print("=" * 70)
    print(f"Input video: {video_path}")
    print(f"YOLO model: {yolo_model}")
    print(f"TrackNet model: {tracknet_model if tracknet_model else 'Not used'}")
    print(f"Player pose detection: {'Enabled' if use_pose else 'Disabled'}")
    print("-" * 70)
    
    # Initialize tracker
    tracker = ShuttleTracker(
        yolo_model_path=yolo_model,
        tracknet_model_path=tracknet_model,
        use_pose=use_pose
    )
    
    # Track video
    trajectory = tracker.track_video(
        video_path=video_path,
        output_path=output_video,
        show_visualization=True
    )
    
    print("\n" + "=" * 70)
    print("TRACKING COMPLETE")
    print("=" * 70)
    print(f"Total frames processed: {len(trajectory)}")
    print(f"Shuttle detected in: {len([t for t in trajectory if t[3] > confidence_threshold])} frames")
    
    # Save trajectory to CSV
    if output_csv:
        df = pd.DataFrame(trajectory, columns=['frame', 'x', 'y', 'confidence'])
        df.to_csv(output_csv, index=False)
        print(f"Trajectory saved to: {output_csv}")
    
    if output_video:
        print(f"Output video saved to: {output_video}")
    
    return trajectory


def analyze_trajectory(trajectory, fps=30):
    """
    Analyze shuttle trajectory for insights
    
    Args:
        trajectory: List of (frame, x, y, confidence) tuples
        fps: Video frame rate
    """
    if len(trajectory) < 2:
        print("Not enough data for analysis")
        return
    
    df = pd.DataFrame(trajectory, columns=['frame', 'x', 'y', 'confidence'])
    
    # Calculate velocities
    df['vx'] = df['x'].diff() * fps
    df['vy'] = df['y'].diff() * fps
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
    
    # Calculate accelerations
    df['ax'] = df['vx'].diff() * fps
    df['ay'] = df['vy'].diff() * fps
    
    print("\n" + "=" * 70)
    print("TRAJECTORY ANALYSIS")
    print("=" * 70)
    
    print(f"\nPosition Statistics:")
    print(f"  X range: {df['x'].min():.1f} to {df['x'].max():.1f} pixels")
    print(f"  Y range: {df['y'].min():.1f} to {df['y'].max():.1f} pixels")
    
    print(f"\nSpeed Statistics:")
    print(f"  Average speed: {df['speed'].mean():.1f} pixels/sec")
    print(f"  Max speed: {df['speed'].max():.1f} pixels/sec")
    print(f"  Min speed: {df['speed'].min():.1f} pixels/sec")
    
    print(f"\nDetection Quality:")
    print(f"  Average confidence: {df['confidence'].mean():.3f}")
    print(f"  Frames with confidence > 0.5: {(df['confidence'] > 0.5).sum()} / {len(df)}")
    
    # Detect rallies (periods of continuous tracking)
    gaps = df['frame'].diff()
    rally_starts = df[gaps > 10].index.tolist()
    rally_starts.insert(0, 0)
    
    print(f"\nRally Detection:")
    print(f"  Number of rallies detected: {len(rally_starts)}")
    
    return df


def batch_process_videos(video_folder, output_folder, **kwargs):
    """
    Process multiple videos in a folder
    
    Args:
        video_folder: Folder containing videos
        output_folder: Folder to save outputs
        **kwargs: Additional arguments for track_shuttle_simple
    """
    import os
    from pathlib import Path
    
    os.makedirs(output_folder, exist_ok=True)
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    videos = [f for f in os.listdir(video_folder) 
              if Path(f).suffix.lower() in video_extensions]
    
    print(f"Found {len(videos)} videos to process")
    
    for i, video in enumerate(videos, 1):
        print(f"\n\nProcessing video {i}/{len(videos)}: {video}")
        print("=" * 70)
        
        video_path = os.path.join(video_folder, video)
        output_video = os.path.join(output_folder, f"tracked_{video}")
        output_csv = os.path.join(output_folder, f"{Path(video).stem}_trajectory.csv")
        
        try:
            trajectory = track_shuttle_simple(
                video_path=video_path,
                output_video=output_video,
                output_csv=output_csv,
                **kwargs
            )
            
            # Quick analysis
            analyze_trajectory(trajectory)
            
        except Exception as e:
            print(f"Error processing {video}: {e}")
            continue
    
    print(f"\n\nBatch processing complete! Outputs saved to {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description='Track shuttle in badminton videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Track single video with pre-trained YOLO
  python inference.py --video match.mp4 --output tracked.mp4
  
  # Track with custom trained models
  python inference.py --video match.mp4 --output tracked.mp4 \\
      --yolo-model runs/detect/shuttle_detector/weights/best.pt \\
      --tracknet-model tracknet_shuttle.pth
  
  # Track with player pose detection
  python inference.py --video match.mp4 --output tracked.mp4 --use-pose
  
  # Batch process folder of videos
  python inference.py --batch --input-folder videos/ --output-folder outputs/
  
  # Save trajectory and analyze
  python inference.py --video match.mp4 --csv trajectory.csv --analyze
        """
    )
    
    # Input/output
    parser.add_argument('--video', type=str, help='Input video path')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--csv', type=str, help='Output CSV path for trajectory')
    
    # Batch processing
    parser.add_argument('--batch', action='store_true', help='Process multiple videos')
    parser.add_argument('--input-folder', type=str, help='Folder with videos (for batch)')
    parser.add_argument('--output-folder', type=str, help='Output folder (for batch)')
    
    # Model paths
    parser.add_argument('--yolo-model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model (default: yolov8n.pt)')
    parser.add_argument('--tracknet-model', type=str, default=None,
                       help='Path to TrackNet model (optional)')
    
    # Options
    parser.add_argument('--use-pose', action='store_true',
                       help='Enable player pose detection')
    parser.add_argument('--confidence', type=float, default=0.3,
                       help='Confidence threshold (default: 0.3)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze trajectory after tracking')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display video during processing')
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.input_folder or not args.output_folder:
            parser.error("--batch requires --input-folder and --output-folder")
        
        batch_process_videos(
            video_folder=args.input_folder,
            output_folder=args.output_folder,
            yolo_model=args.yolo_model,
            tracknet_model=args.tracknet_model,
            confidence_threshold=args.confidence,
            use_pose=args.use_pose
        )
    else:
        if not args.video:
            parser.error("--video is required for single video processing")
        
        trajectory = track_shuttle_simple(
            video_path=args.video,
            output_video=args.output,
            output_csv=args.csv,
            yolo_model=args.yolo_model,
            tracknet_model=args.tracknet_model,
            confidence_threshold=args.confidence,
            use_pose=args.use_pose
        )
        
        if args.analyze:
            analyze_trajectory(trajectory)


if __name__ == "__main__":
    main()
