# ==========================================
# INTERACTIVE INITIALIZATION WINDOW
# ==========================================

import cv2
import numpy as np
from pathlib import Path

class ShuttleInitializer:
    """
    Interactive window for manual shuttle initialization
    """
    
    def __init__(self, video_path, num_init_frames=10):
        self.video_path = video_path
        self.num_init_frames = num_init_frames
        print(f"Loading first {num_init_frames} frames from video... This is from init")
        self.manual_positions = []
        self.frames = []
        self.current_frame_idx = 0
        self.window_name = "Click Shuttle Positions (Press SPACE for next frame, ENTER when done)"
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add clicked position
            self.manual_positions.append({
                'frame_idx': self.current_frame_idx,
                'x': x,
                'y': y
            })
            print(f"✓ Shuttle marked at ({x}, {y}) in frame {self.current_frame_idx}")
            
            # Redraw with marker
            self.draw_current_frame()
    
    def draw_current_frame(self):
        """Draw current frame with markers"""
        if self.current_frame_idx >= len(self.frames):
            return
        
        frame = self.frames[self.current_frame_idx].copy()
        
        # Draw all positions marked in this frame
        for pos in self.manual_positions:
            if pos['frame_idx'] == self.current_frame_idx:
                x, y = pos['x'], pos['y']
                # Draw crosshair
                cv2.drawMarker(frame, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.circle(frame, (x, y), 15, (0, 255, 0), 2)
                cv2.putText(frame, "SHUTTLE", (x + 20, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Instructions
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, h-100), (w-10, h-10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, h-100), (w-10, h-10), (0, 255, 0), 2)
        
        instructions = [
            f"Frame {self.current_frame_idx + 1}/{len(self.frames)} | Marked: {len(self.manual_positions)}",
            "LEFT CLICK: Mark shuttle position",
            "SPACE: Next frame | BACKSPACE: Previous | ENTER: Start tracking"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (20, h - 80 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow(self.window_name, frame)
    
    def run(self):
        """Run interactive initialization"""
        print("\n" + "="*70)
        print("MANUAL SHUTTLE INITIALIZATION")
        print("="*70)
        print(f"Loading first {self.num_init_frames} frames from video...")
        
        # Load initialization frames
        cap = cv2.VideoCapture(self.video_path)
        print("Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))


        for i in range(self.num_init_frames):
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
        
        cap.release()
        
        if len(self.frames) == 0:
            print("❌ Error: Could not load frames from video")
            return None
        
        print(f"✓ Loaded {len(self.frames)} frames")
        print("\nInstructions:")
        print("  1. Click on the shuttle in each frame")
        print("  2. Press SPACE to go to next frame")
        print("  3. Press BACKSPACE to go back")
        print("  4. Press ENTER when you're done marking")
        print("  5. Minimum 3 positions recommended for best results")
        print()
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Show first frame
        self.draw_current_frame()
        
        # Main loop
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - next frame
                if self.current_frame_idx < len(self.frames) - 1:
                    self.current_frame_idx += 1
                    self.draw_current_frame()
                else:
                    print("⚠ Already at last frame")
            
            elif key == 8:  # Backspace - previous frame
                if self.current_frame_idx > 0:
                    self.current_frame_idx -= 1
                    self.draw_current_frame()
                else:
                    print("⚠ Already at first frame")
            
            elif key == 13 or key == 10:  # Enter - done
                if len(self.manual_positions) >= 1:
                    print(f"\n✓ Initialization complete with {len(self.manual_positions)} positions")
                    cv2.destroyAllWindows()
                    return self.manual_positions
                else:
                    print("⚠ Please mark at least 1 shuttle position")
            
            elif key == 27:  # ESC - cancel
                print("\n⚠ Initialization cancelled")
                cv2.destroyAllWindows()
                return None


class ManualShuttleTracker:
    """
    Interactive shuttle tracker with manual initialization
    1. User clicks shuttle in first few frames
    2. System learns shuttle appearance
    3. Automatic tracking begins
    """
    
    def __init__(self, template_size=(30, 30)):
        self.template_size = template_size
        self.shuttle_template = None
        self.manual_positions = []
        self.tracking_started = False
        self.current_position = None
        
        # Tracking parameters
        self.search_radius = 150  # How far to search from last position
        self.confidence_threshold = 0.6
        
        # Kalman filter for smooth tracking
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state, 2 measurement
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman_initialized = False
    
    def add_manual_position(self, x, y, frame):
        """Store manually clicked position and extract template"""
        self.manual_positions.append((x, y))
        
        # Extract shuttle template from this frame
        h, w = self.template_size
        y1 = max(0, y - h//2)
        y2 = min(frame.shape[0], y + h//2)
        x1 = max(0, x - w//2)
        x2 = min(frame.shape[1], x + w//2)
        
        template = frame[y1:y2, x1:x2].copy()
        
        # Store or update template (average multiple clicks)
        if self.shuttle_template is None:
            self.shuttle_template = template.astype(np.float32)
        else:
            # Average with previous templates
            self.shuttle_template = (self.shuttle_template + template.astype(np.float32)) / 2
        
        self.current_position = (x, y)
        
        # Initialize Kalman filter with this position
        if not self.kalman_initialized:
            self.kalman.statePre = np.array([x, y, 0, 0], np.float32)
            self.kalman.statePost = np.array([x, y, 0, 0], np.float32)
            self.kalman_initialized = True
        
        return template
    
    def start_tracking(self):
        """Begin automatic tracking after manual initialization"""
        if len(self.manual_positions) >= 1 and self.shuttle_template is not None:
            self.tracking_started = True
            print(f"✓ Tracking started with {len(self.manual_positions)} manual points")
            return True
        return False
    
    def track_shuttle(self, frame):
        """
        Automatically track shuttle using multi-scale template matching + Kalman filter
        Returns: (x, y), confidence
        """
        if not self.tracking_started or self.shuttle_template is None:
            return None, 0.0
        
        # Predict next position using Kalman filter
        prediction = self.kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])
        
        # Define search region around predicted position
        h, w = frame.shape[:2]
        search_radius = self.search_radius
        
        # Expand search if confidence was low previously
        search_x1 = max(0, pred_x - search_radius)
        search_x2 = min(w, pred_x + search_radius)
        search_y1 = max(0, pred_y - search_radius)
        search_y2 = min(h, pred_y + search_radius)
        
        search_region = frame[search_y1:search_y2, search_x1:search_x2]
        
        if search_region.size == 0:
            return self.current_position, 0.0
        
        # Multi-scale matching
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        best_val = -1
        best_match = None
        best_size = None
        
        base_h, base_w = self.shuttle_template.shape[:2]
        
        for scale in scales:
            # Resize template
            new_w, new_h = int(base_w * scale), int(base_h * scale)
            if new_w >= search_region.shape[1] or new_h >= search_region.shape[0] or new_w == 0 or new_h == 0:
                continue
                
            resized_template = cv2.resize(self.shuttle_template.astype(np.uint8), (new_w, new_h))
            
            # Match template
            result = cv2.matchTemplate(search_region, resized_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_val:
                best_val = max_val
                # Convert to global coords
                # max_loc is top-left in search region
                center_x = search_x1 + max_loc[0] + new_w//2
                center_y = search_y1 + max_loc[1] + new_h//2
                best_match = (center_x, center_y)
                best_size = (new_w, new_h)
        
        # Update state if match found
        if best_match and best_val > self.confidence_threshold:
            match_x, match_y = best_match
            self.current_position = (match_x, match_y)
            
            # Update Kalman filter
            measurement = np.array([match_x, match_y], np.float32)
            self.kalman.correct(measurement)
            
            # Adaptive Template Update
            # Extract the actual matched patch from the frame
            bw, bh = best_size
            x1 = max(0, match_x - bw//2)
            y1 = max(0, match_y - bh//2)
            x2 = min(w, x1 + bw)
            y2 = min(h, y1 + bh)
            
            if x2 > x1 and y2 > y1:
                current_patch = frame[y1:y2, x1:x2]
                if current_patch.shape[:2] == (bh, bw):
                    # Resize back to base template size to maintain consistency
                    patch_canonical = cv2.resize(current_patch, (base_w, base_h))
                    
                    # Update template (exponential moving average)
                    alpha = 0.1
                    self.shuttle_template = cv2.addWeighted(
                        self.shuttle_template, 1 - alpha,
                        patch_canonical.astype(np.float32), alpha,
                        0
                    )
            
            return (match_x, match_y), best_val
        else:
            # Use Kalman prediction if match is poor
            return (pred_x, pred_y), best_val * 0.5


class OpticalFlowShuttleTracker:
    """
    Alternative: Track shuttle using optical flow after manual initialization
    More robust to appearance changes
    """
    
    def __init__(self):
        self.tracking_points = None
        self.prev_gray = None
        self.tracking_started = False
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
    
    def initialize(self, x, y, frame):
        """Initialize tracking point"""
        self.tracking_points = np.array([[[x, y]]], dtype=np.float32)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.tracking_started = True
        return True
    
    def track(self, frame):
        """Track using optical flow"""
        if not self.tracking_started or self.tracking_points is None:
            return None, 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        new_points, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.tracking_points, None, **self.lk_params
        )
        
        if new_points is not None and status[0][0] == 1:
            x, y = new_points[0][0]
            self.tracking_points = new_points
            self.prev_gray = gray
            
            # Confidence based on tracking error
            confidence = 1.0 / (1.0 + error[0][0] / 100.0)
            return (int(x), int(y)), confidence
        
        return None, 0.0
