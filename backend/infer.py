import cv2
import torch
import numpy as np
from pathlib import Path


# -------------------------
# CONFIG
# -------------------------
INPUT_W, INPUT_H = 960, 540
HEATMAP_THRESH = 0.4

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def shuttle_tracking_combined(cap, model, pose_model, tracknet, device, video_filename="output"):
    """Process shuttle + human tracking in single pass"""
    
    
    print(f"Starting tracking on {device}...")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = OUTPUT_DIR / f"{video_filename}_annotated.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("Error: Could not open video writer")
        return None, None, None, None
    
    frame_buffer = []
    shuttle_positions = []
    wrist_data = []
    unique_ids = set()
    frame_idx = 0

    shuttle_trail = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        if frame_idx % 30 == 0:
            print(f"Processing frame {frame_idx}...")
        
        # Create a copy for visualization
        vis_frame = frame.copy()
        
        # Resize for shuttle tracking
        frame_resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        frame_buffer.append(frame_resized)
        
        # Human tracking
        results = model.track(frame, persist=True, conf=0.4, iou=0.5, tracker="bytetrack.yaml")
        pose_results = pose_model(frame)
        
        # Draw YOLO detections
        vis_frame = results[0].plot()
        
        # Process and draw wrists
        for r in pose_results:
            if r.keypoints is None:
                continue
            for kp in r.keypoints.xy.cpu().numpy():
                left_wrist = kp[9]
                right_wrist = kp[10]
                
                wrist_data.append({
                    "frame": frame_idx,
                    "left_wrist": {"x": float(left_wrist[0]), "y": float(left_wrist[1])},
                    "right_wrist": {"x": float(right_wrist[0]), "y": float(right_wrist[1])}
                })
                
                # Draw wrists with labels
                left_x, left_y = int(left_wrist[0]), int(left_wrist[1])
                right_x, right_y = int(right_wrist[0]), int(right_wrist[1])
                
                # Left wrist (green)
                cv2.circle(vis_frame, (left_x, left_y), 8, (0, 255, 0), -1)
                cv2.circle(vis_frame, (left_x, left_y), 10, (0, 255, 0), 2)
                cv2.putText(vis_frame, "L", (left_x + 12, left_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Right wrist (blue)
                cv2.circle(vis_frame, (right_x, right_y), 8, (255, 0, 0), -1)
                cv2.circle(vis_frame, (right_x, right_y), 10, (255, 0, 0), 2)
                cv2.putText(vis_frame, "R", (right_x + 12, right_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Shuttle tracking
        current_shuttle_pos = None
        if len(frame_buffer) == 3:
            input_img = np.concatenate(frame_buffer, axis=2)
            input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                heatmap = tracknet(input_tensor)
            
            heatmap = heatmap.squeeze().cpu().numpy()
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            
            if heatmap[y, x] > HEATMAP_THRESH:
                # Scale shuttle position back to original frame size
                scale_x = width / INPUT_W
                scale_y = height / INPUT_H
                shuttle_x = int(x * scale_x)
                shuttle_y = int(y * scale_y)
                
                current_shuttle_pos = (shuttle_x, shuttle_y)
                shuttle_positions.append((int(x), int(y)))  # Store original scale
                shuttle_trail.append((shuttle_x, shuttle_y, frame_idx))
            else:
                shuttle_positions.append(None)
            
            frame_buffer.pop(0)
        
        # Draw shuttle trail (last 15 frames)
        trail_length = 15
        recent_trail = [s for s in shuttle_trail if frame_idx - s[2] <= trail_length]
        
        for i in range(1, len(recent_trail)):
            # Fade effect
            alpha = i / len(recent_trail)
            thickness = max(1, int(3 * alpha))
            cv2.line(vis_frame, 
                    (recent_trail[i-1][0], recent_trail[i-1][1]),
                    (recent_trail[i][0], recent_trail[i][1]),
                    (0, 255, 255), thickness)  # Yellow trail
        
        # Draw current shuttle position
        if current_shuttle_pos:
            sx, sy = current_shuttle_pos
            # Outer circle (yellow)
            cv2.circle(vis_frame, (sx, sy), 12, (0, 255, 255), 3)
            # Inner circle (red)
            cv2.circle(vis_frame, (sx, sy), 6, (0, 0, 255), -1)
            # Label
            cv2.putText(vis_frame, "SHUTTLE", (sx + 15, sy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add frame info
        cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"People: {len(unique_ids)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Track unique IDs
        for r in results:
            if r.boxes.id is not None:
                for track_id, cls_id in zip(r.boxes.id, r.boxes.cls):
                    if int(cls_id) == 0:
                        unique_ids.add(int(track_id))
        
        # Write frame to output video
        out.write(vis_frame)
    
    # Release video writer
    out.release()
    
    print(f"✅ Tracking complete! Processed {frame_idx} frames")
    print(f"✅ Found {len(unique_ids)} unique people")
    print(f"✅ Detected shuttle in {sum(1 for pos in shuttle_positions if pos is not None)} frames")
    print(f"✅ Annotated video saved to: {output_path}")
    
    return shuttle_positions, wrist_data, unique_ids, str(output_path)