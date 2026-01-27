import cv2
import torch
import base64
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# -------------------------
# HYBRID SHUTTLE DETECTOR
# -------------------------

class HybridShuttleDetector:
    """
    Combined TrackNet + YOLO + Color shuttle detector for maximum reliability
    """
    
    def __init__(self, tracknet_model, device, use_yolo_fallback=True):
        self.tracknet = tracknet_model
        self.device = device
        self.use_yolo_fallback = use_yolo_fallback
        
        # Configuration
        self.INPUT_W = 960
        self.INPUT_H = 540
        self.HEATMAP_THRESH = 0.4
        
        # Initialize YOLO for fallback (lightweight model)
        if use_yolo_fallback:
            try:
                self.yolo = YOLO("yolov8n.pt")
                print("✓ YOLO fallback detector loaded")
            except Exception as e:
                print(f"⚠ YOLO not available: {e}")
                print("  Continuing with TrackNet only")
                self.yolo = None
        else:
            self.yolo = None
    
    def detect_tracknet(self, frame_buffer, original_size):
        """Primary detection using TrackNet (temporal heatmap approach)"""
        if len(frame_buffer) != 3:
            return None, 0.0
        
        # Prepare input (concatenate 3 frames)
        input_img = np.concatenate(frame_buffer, axis=2)
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # Run TrackNet inference
        with torch.no_grad():
            heatmap = self.tracknet(input_tensor)
        
        # Process heatmap
        heatmap = heatmap.squeeze().cpu().numpy()
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        confidence = float(heatmap[y, x])
        
        if confidence > self.HEATMAP_THRESH:
            # Scale to original frame size
            width, height = original_size
            scale_x = width / self.INPUT_W
            scale_y = height / self.INPUT_H
            shuttle_x = int(x * scale_x)
            shuttle_y = int(y * scale_y)
            
            return (shuttle_x, shuttle_y), confidence
        
        return None, confidence
    
    def detect_yolo_sports_ball(self, frame):
        """YOLO detection using sports ball class"""
        if self.yolo is None:
            return None, 0.0
        
        try:
            results = self.yolo(frame, verbose=False, conf=0.3)
            
            for r in results:
                if r.boxes is None:
                    continue
                
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    # Class 32 is 'sports ball' in COCO dataset
                    if int(cls) == 32 and conf > 0.3:
                        x1, y1, x2, y2 = box.cpu().numpy()
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        return (center_x, center_y), float(conf)
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return None, 0.0
    
    def detect_color(self, frame):
        """Color-based white shuttle detection (final fallback)"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # White shuttle detection range
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Morphological operations to clean up noise
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None, 0.0
            
            # Find small bright objects (shuttle characteristics)
            candidates = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 50 < area < 500:  # Shuttle size range in pixels
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        # Check brightness
                        x, y, w, h = cv2.boundingRect(cnt)
                        roi = frame[y:y+h, x:x+w]
                        brightness = np.mean(roi)
                        candidates.append((cx, cy, brightness))
            
            if not candidates:
                return None, 0.0
            
            # Choose brightest candidate
            best = max(candidates, key=lambda x: x[2])
            confidence = min(best[2] / 255.0, 1.0) * 0.6  # Scale down confidence for color detection
            
            return (best[0], best[1]), confidence
        except Exception as e:
            print(f"Color detection error: {e}")
            return None, 0.0
    
    def detect(self, frame, frame_buffer, original_size):
        """
        Hybrid detection strategy with intelligent fallback
        
        Args:
            frame: Current frame (original size)
            frame_buffer: List of 3 resized frames for TrackNet
            original_size: (width, height) of original frame
        
        Returns:
            position: (x, y) or None
            confidence: float 0-1
            method: 'tracknet', 'yolo', or 'color'
        """
        # Strategy 1: Try TrackNet first (primary method - best for shuttlecocks)
        pos_tracknet, conf_tracknet = self.detect_tracknet(frame_buffer, original_size)
        
        # If TrackNet has high confidence, use it
        if pos_tracknet and conf_tracknet > 0.5:
            return pos_tracknet, conf_tracknet, 'tracknet'
        
        # Strategy 2: If TrackNet has low/no confidence, try YOLO
        if self.use_yolo_fallback:
            pos_yolo, conf_yolo = self.detect_yolo_sports_ball(frame)
            
            # Use YOLO if it has higher confidence than TrackNet
            if pos_yolo and conf_yolo > max(conf_tracknet, 0.4):
                return pos_yolo, conf_yolo, 'yolo'
        
        # Strategy 3: Try color detection as final fallback
        if self.use_yolo_fallback:  # Only if fallback enabled
            pos_color, conf_color = self.detect_color(frame)
            
            # Use color if it has higher confidence than both previous methods
            if pos_color and conf_color > max(conf_tracknet, 0.35):
                return pos_color, conf_color, 'color'
        
        # Return TrackNet result even if low confidence (better than nothing)
        if pos_tracknet:
            return pos_tracknet, conf_tracknet, 'tracknet'
        
        return None, 0.0, 'none'


# -------------------------
# MAIN TRACKING FUNCTION
# -------------------------

async def shuttle_tracking_hybrid(cap, model, pose_model, tracknet, device, video_filename="output"):
    """
    Complete tracking system with hybrid shuttle detection
    Tracks: People, Poses (wrists), and Shuttle with multi-method detection
    """
    
    print(f"\n{'='*60}")
    print(f"Starting Hybrid Shuttle Tracking System")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Initialize hybrid detector
    shuttle_detector = HybridShuttleDetector(tracknet, device, use_yolo_fallback=True)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    INPUT_W, INPUT_H = 960, 540
    
    # Setup output
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{video_filename}_annotated.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        yield {"type": "error", "message": "Could not open video writer"}
        return
    
    print(f"✓ Video Writer initialized")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Output: {output_path}\n")
    
    yield {
        "type": "started",
        "total_frames": total_frames,
        "fps": fps,
        "resolution": {"width": width, "height": height},
        "output_path": str(output_path)
    }
    
    # Tracking state
    frame_buffer = []
    shuttle_trail = []
    unique_ids = set()
    wrist_data = []
    
    # Statistics
    detection_stats = {'tracknet': 0, 'yolo': 0, 'color': 0, 'none': 0}
    frame_idx = 0
    
    # Processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Progress logging
        if frame_idx % 30 == 0:
            print(f"Processing frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        # ========================
        # 1. HUMAN TRACKING (YOLO)
        # ========================
        results = model.track(frame, persist=True, conf=0.4, iou=0.5, tracker="bytetrack.yaml")
        vis_frame = results[0].plot()
        
        # Track unique IDs
        frame_people = 0
        for r in results:
            if r.boxes.id is not None:
                for track_id, cls_id in zip(r.boxes.id, r.boxes.cls):
                    if int(cls_id) == 0:  # Person class
                        unique_ids.add(int(track_id))
                        frame_people += 1
        
        # ========================
        # 2. POSE DETECTION (Wrists)
        # ========================
        pose_results = pose_model(frame)
        frame_wrists = []
        
        for r in pose_results:
            if r.keypoints is None:
                continue
            
            for kp in r.keypoints.xy.cpu().numpy():
                left_wrist = kp[9]   # Left wrist keypoint
                right_wrist = kp[10]  # Right wrist keypoint
                
                left_x, left_y = int(left_wrist[0]), int(left_wrist[1])
                right_x, right_y = int(right_wrist[0]), int(right_wrist[1])
                
                # Store wrist data
                wrist_data.append({
                    "frame": frame_idx,
                    "left_wrist": {"x": float(left_wrist[0]), "y": float(left_wrist[1])},
                    "right_wrist": {"x": float(right_wrist[0]), "y": float(right_wrist[1])}
                })
                
                frame_wrists.append({
                    "left_wrist": {"x": float(left_wrist[0]), "y": float(left_wrist[1])},
                    "right_wrist": {"x": float(right_wrist[0]), "y": float(right_wrist[1])}
                })
                
                # Draw left wrist (green)
                cv2.circle(vis_frame, (left_x, left_y), 8, (0, 255, 0), -1)
                cv2.circle(vis_frame, (left_x, left_y), 10, (0, 255, 0), 2)
                cv2.putText(vis_frame, "L", (left_x + 12, left_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw right wrist (blue)
                cv2.circle(vis_frame, (right_x, right_y), 8, (255, 0, 0), -1)
                cv2.circle(vis_frame, (right_x, right_y), 10, (255, 0, 0), 2)
                cv2.putText(vis_frame, "R", (right_x + 12, right_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # ========================
        # 3. SHUTTLE DETECTION (Hybrid)
        # ========================
        # Prepare frame buffer for TrackNet
        frame_resized = cv2.resize(frame, (INPUT_W, INPUT_H))
        frame_buffer.append(frame_resized)
        
        current_shuttle_pos = None
        detection_method = 'none'
        shuttle_confidence = 0.0
        
        if len(frame_buffer) == 3:
            # Run hybrid detection
            position, confidence, method = shuttle_detector.detect(
                frame, frame_buffer, (width, height)
            )
            
            # Update statistics and trail
            if position and confidence > 0.3:
                current_shuttle_pos = position
                detection_method = method
                shuttle_confidence = confidence
                detection_stats[method] += 1
                shuttle_trail.append((*position, frame_idx))
            else:
                detection_stats['none'] += 1
            
            # Remove oldest frame
            frame_buffer.pop(0)
        
        # ========================
        # 4. VISUALIZATION
        # ========================
        
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
                    (0, 255, 255), thickness)
        
        # Draw current shuttle with method-specific color
        if current_shuttle_pos:
            sx, sy = current_shuttle_pos
            
            # Color code by detection method
            if detection_method == 'tracknet':
                color = (0, 255, 255)  # Yellow - TrackNet
                label = f"SHUTTLE (TrackNet {shuttle_confidence:.2f})"
            elif detection_method == 'yolo':
                color = (0, 165, 255)  # Orange - YOLO
                label = f"SHUTTLE (YOLO {shuttle_confidence:.2f})"
            else:  # color detection
                color = (255, 20, 147)  # Pink - Color
                label = f"SHUTTLE (Color {shuttle_confidence:.2f})"
            
            # Draw shuttle marker
            cv2.circle(vis_frame, (sx, sy), 15, color, 3)
            cv2.circle(vis_frame, (sx, sy), 8, (0, 0, 255), -1)
            cv2.putText(vis_frame, label, (sx + 20, sy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # ========================
        # 5. INFO OVERLAYS
        # ========================
        
        # Frame counter
        cv2.putText(vis_frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # People counter
        cv2.putText(vis_frame, f"People: {frame_people} (Total: {len(unique_ids)})", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Detection stats
        total_detections = sum(detection_stats.values())
        cv2.putText(vis_frame, f"TrackNet: {detection_stats['tracknet']}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(vis_frame, f"YOLO: {detection_stats['yolo']}", (10, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(vis_frame, f"Color: {detection_stats['color']}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 20, 147), 2)
        
        # Progress bar
        bar_width = 300
        bar_height = 20
        bar_x = width - bar_width - 20
        bar_y = 20
        progress = frame_idx / total_frames
        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
        cv2.rectangle(vis_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Legend box
        legend_x = width - 220
        legend_y = height - 160
        cv2.rectangle(vis_frame, (legend_x - 10, legend_y - 20), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (legend_x - 10, legend_y - 20), (width - 10, height - 10), (255, 255, 255), 2)
        
        cv2.putText(vis_frame, "Legend:", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Legend items
        y_offset = legend_y + 20
        cv2.circle(vis_frame, (legend_x + 10, y_offset), 5, (0, 255, 0), -1)
        cv2.putText(vis_frame, "Left Wrist", (legend_x + 25, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.circle(vis_frame, (legend_x + 10, y_offset), 5, (255, 0, 0), -1)
        cv2.putText(vis_frame, "Right Wrist", (legend_x + 25, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.circle(vis_frame, (legend_x + 10, y_offset), 5, (0, 255, 255), -1)
        cv2.putText(vis_frame, "Shuttle (TN)", (legend_x + 25, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.circle(vis_frame, (legend_x + 10, y_offset), 5, (0, 165, 255), -1)
        cv2.putText(vis_frame, "Shuttle (YL)", (legend_x + 25, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        y_offset += 20
        cv2.line(vis_frame, (legend_x + 5, y_offset), (legend_x + 15, y_offset), (0, 255, 255), 2)
        cv2.putText(vis_frame, "Trail", (legend_x + 25, y_offset + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Write frame to video
        out.write(vis_frame)
        
        # ========================
        # 6. YIELD PROGRESS UPDATE
        # ========================
        if frame_idx % 5 == 0:
            # Create preview image
            preview_frame = cv2.resize(vis_frame, (480, 270))
            _, buffer = cv2.imencode('.jpg', preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            yield {
                "type": "progress",
                "frame": frame_idx,
                "total_frames": total_frames,
                "progress_percent": round((frame_idx / total_frames) * 100, 1),
                "people_count": frame_people,
                "unique_people": len(unique_ids),
                "shuttle_detected": current_shuttle_pos is not None,
                "shuttle_position": current_shuttle_pos,
                "detection_method": detection_method,
                "detection_stats": detection_stats,
                "wrists": frame_wrists,
                "preview_image": f"data:image/jpeg;base64,{frame_b64}"
            }
    
    # ========================
    # 7. CLEANUP & FINAL REPORT
    # ========================
    out.release()
    
    total_detections = detection_stats['tracknet'] + detection_stats['yolo'] + detection_stats['color']
    detection_rate = round((total_detections / frame_idx) * 100, 1) if frame_idx > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Total Frames: {frame_idx}")
    print(f"Unique People Tracked: {len(unique_ids)}")
    print(f"Shuttle Detections: {total_detections} ({detection_rate}%)")
    print(f"  - TrackNet: {detection_stats['tracknet']}")
    print(f"  - YOLO: {detection_stats['yolo']}")
    print(f"  - Color: {detection_stats['color']}")
    print(f"Wrist Data Points: {len(wrist_data)}")
    print(f"Output Video: {output_path}")
    print(f"{'='*60}\n")
    
    yield {
        "type": "complete",
        "total_frames": frame_idx,
        "unique_people": len(unique_ids),
        "shuttle_detections": total_detections,
        "detection_breakdown": detection_stats,
        "wrist_data_points": len(wrist_data),
        "output_video": str(output_path),
        "summary": {
            "detection_rate": detection_rate,
            "tracknet_rate": round((detection_stats['tracknet'] / frame_idx) * 100, 1),
            "yolo_rate": round((detection_stats['yolo'] / frame_idx) * 100, 1),
            "color_rate": round((detection_stats['color'] / frame_idx) * 100, 1)
        }
    }


# -------------------------
# USAGE EXAMPLE
# -------------------------

"""
# In your main script, use it like this:

import asyncio
from ultralytics import YOLO

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")  # Person tracking
pose_model = YOLO("yolov8n-pose.pt")  # Pose detection
tracknet = torch.load("tracknet_model.pth").to(device)  # Your TrackNet model

# Open video
cap = cv2.VideoCapture("your_video.mp4")

# Run tracking
async def main():
    async for update in shuttle_tracking_hybrid(cap, model, pose_model, tracknet, device, "output"):
        if update["type"] == "progress":
            print(f"Progress: {update['progress_percent']}%")
        elif update["type"] == "complete":
            print(f"Done! Output: {update['output_video']}")

asyncio.run(main())
cap.release()
"""