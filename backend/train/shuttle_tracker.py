import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
from collections import deque
from filterpy.kalman import KalmanFilter


# ============================================================================
# TrackNet Model for Temporal Refinement
# ============================================================================

class TrackNet(nn.Module):
    """
    TrackNet model for temporal refinement of shuttle detection
    Takes 3 consecutive frames and outputs a heatmap of shuttle location
    """
    def __init__(self):
        super(TrackNet, self).__init__()
        
        # Encoder (VGG-like architecture)
        self.conv1 = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=3, padding=1),  # 3 frames * 3 channels
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x = self.pool1(x1)
        
        x2 = self.conv2(x)
        x = self.pool2(x2)
        
        x3 = self.conv3(x)
        x = self.pool3(x3)
        
        # Decoder
        x = self.upsample1(x)
        x = self.conv4(x)
        
        x = self.upsample2(x)
        x = self.conv5(x)
        
        x = self.upsample3(x)
        x = self.conv6(x)
        
        return x


# ============================================================================
# Kalman Filter for Tracking
# ============================================================================

class ShuttleKalmanFilter:
    """Kalman filter for smooth shuttle trajectory tracking"""
    
    def __init__(self):
        # State: [x, y, vx, vy]
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix
        dt = 1.0
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise
        self.kf.R = np.eye(2) * 10
        
        # Process noise
        self.kf.Q = np.eye(4) * 0.1
        
        # Initial covariance
        self.kf.P *= 100
        
        self.initialized = False
        
    def update(self, measurement):
        """Update filter with new measurement (x, y)"""
        if not self.initialized:
            self.kf.x = np.array([measurement[0], measurement[1], 0, 0])
            self.initialized = True
        else:
            self.kf.predict()
            self.kf.update(measurement)
        
        return self.kf.x[:2]  # Return position only


# ============================================================================
# Training Dataset for TrackNet
# ============================================================================

class ShuttleTrackingDataset(torch.utils.data.Dataset):
    """Dataset for training TrackNet on shuttle videos"""
    
    def __init__(self, video_paths, annotations, img_size=(360, 640)):
        """
        Args:
            video_paths: List of paths to training videos
            annotations: List of dictionaries with frame-by-frame shuttle positions
                        Format: {frame_num: (x, y)} or None if shuttle not visible
            img_size: (height, width) for resizing
        """
        self.video_paths = video_paths
        self.annotations = annotations
        self.img_size = img_size
        self.samples = []
        
        self._prepare_samples()
    
    def _prepare_samples(self):
        """Extract frame triplets and generate heatmaps"""
        for vid_path, annot in zip(self.video_paths, self.annotations):
            cap = cv2.VideoCapture(vid_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frames = []
            for _ in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (self.img_size[1], self.img_size[0]))
                frames.append(frame)
            
            cap.release()
            
            # Create triplets
            for i in range(1, len(frames) - 1):
                frame_triplet = [frames[i-1], frames[i], frames[i+1]]
                
                # Create heatmap for middle frame
                heatmap = np.zeros(self.img_size, dtype=np.float32)
                if i in annot and annot[i] is not None:
                    x, y = annot[i]
                    # Scale coordinates to image size
                    x = int(x * self.img_size[1])
                    y = int(y * self.img_size[0])
                    # Create Gaussian heatmap
                    heatmap = self._create_gaussian_heatmap(heatmap, x, y, sigma=5)
                
                self.samples.append((frame_triplet, heatmap))
    
    def _create_gaussian_heatmap(self, heatmap, x, y, sigma=5):
        """Create Gaussian heatmap centered at (x, y)"""
        h, w = heatmap.shape
        tmp_size = sigma * 3
        
        # Generate Gaussian
        x_range = np.arange(max(0, x - tmp_size), min(w, x + tmp_size + 1))
        y_range = np.arange(max(0, y - tmp_size), min(h, y + tmp_size + 1))
        
        for yi in y_range:
            for xi in x_range:
                heatmap[yi, xi] = np.exp(-((xi - x) ** 2 + (yi - y) ** 2) / (2 * sigma ** 2))
        
        return heatmap
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        frame_triplet, heatmap = self.samples[idx]
        
        # Convert frames to tensor and concatenate
        frames_tensor = []
        for frame in frame_triplet:
            frame = frame.astype(np.float32) / 255.0
            frame = torch.from_numpy(frame).permute(2, 0, 1)  # HWC to CHW
            frames_tensor.append(frame)
        
        input_tensor = torch.cat(frames_tensor, dim=0)  # 9 channels
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)  # Add channel dim
        
        return input_tensor, heatmap_tensor


# ============================================================================
# Training Function for TrackNet
# ============================================================================

def train_tracknet(video_paths, annotations, epochs=50, batch_size=4, lr=0.001):
    """
    Train TrackNet model on shuttle videos
    
    Args:
        video_paths: List of training video paths
        annotations: List of annotation dictionaries
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        Trained TrackNet model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and dataloader
    dataset = ShuttleTrackingDataset(video_paths, annotations)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    # Initialize model
    model = TrackNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}] completed, Average Loss: {avg_loss:.4f}')
    
    return model


# ============================================================================
# Complete Shuttle Tracking Pipeline
# ============================================================================

class ShuttleTracker:
    """Complete shuttle tracking system combining YOLO, TrackNet, and Kalman filter"""
    
    def __init__(self, yolo_model_path, tracknet_model_path=None, use_pose=False):
        """
        Initialize shuttle tracker
        
        Args:
            yolo_model_path: Path to trained YOLOv8n model
            tracknet_model_path: Path to trained TrackNet model (optional)
            use_pose: Whether to use YOLO pose estimation
        """
        # Load YOLO model
        self.yolo = YOLO(yolo_model_path)
        
        # Load pose model if requested
        self.pose_model = None
        if use_pose:
            self.pose_model = YOLO('yolov8n-pose.pt')
        
        # Load TrackNet if provided
        self.tracknet = None
        if tracknet_model_path:
            self.tracknet = TrackNet()
            self.tracknet.load_state_dict(torch.load(tracknet_model_path))
            self.tracknet.eval()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.tracknet.to(self.device)
        
        # Initialize Kalman filter
        self.kalman = ShuttleKalmanFilter()
        
        # Frame buffer for TrackNet
        self.frame_buffer = deque(maxlen=3)
        
    def detect_shuttle_yolo(self, frame):
        """Detect shuttle using YOLO"""
        results = self.yolo(frame, verbose=False)
        
        if len(results[0].boxes) > 0:
            # Get the most confident detection
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            best_idx = np.argmax(scores)
            
            x1, y1, x2, y2 = boxes[best_idx]
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            return (center_x, center_y), scores[best_idx]
        
        return None, 0.0
    
    def refine_with_tracknet(self, frame):
        """Refine detection using TrackNet temporal model"""
        if self.tracknet is None:
            return None
        
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) < 3:
            return None
        
        # Prepare input (3 consecutive frames)
        frames = list(self.frame_buffer)
        h, w = frames[0].shape[:2]
        resized_frames = [cv2.resize(f, (640, 360)) for f in frames]
        
        # Convert to tensor
        frames_tensor = []
        for f in resized_frames:
            f = f.astype(np.float32) / 255.0
            f_tensor = torch.from_numpy(f).permute(2, 0, 1)
            frames_tensor.append(f_tensor)
        
        input_tensor = torch.cat(frames_tensor, dim=0).unsqueeze(0).to(self.device)
        
        # Get heatmap
        with torch.no_grad():
            heatmap = self.tracknet(input_tensor)
            heatmap = heatmap.squeeze().cpu().numpy()
        
        # Find peak in heatmap
        if heatmap.max() > 0.5:  # Confidence threshold
            y_peak, x_peak = np.unravel_index(heatmap.argmax(), heatmap.shape)
            # Scale back to original size
            x = int(x_peak * w / 640)
            y = int(y_peak * h / 360)
            return (x, y)
        
        return None
    
    def track_video(self, video_path, output_path=None, show_visualization=True):
        """
        Track shuttle in video
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show_visualization: Whether to display tracking in real-time
        
        Returns:
            List of tracked positions: [(frame_num, x, y, confidence), ...]
        """
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        trajectory = []
        frame_num = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect shuttle with YOLO
            yolo_pos, yolo_conf = self.detect_shuttle_yolo(frame)
            
            # Refine with TrackNet if available
            tracknet_pos = self.refine_with_tracknet(frame)
            
            # Choose best detection
            final_pos = None
            if tracknet_pos is not None:
                final_pos = tracknet_pos
            elif yolo_pos is not None:
                final_pos = yolo_pos
            
            # Apply Kalman filter if we have a detection
            if final_pos is not None:
                smoothed_pos = self.kalman.update(final_pos)
                trajectory.append((frame_num, smoothed_pos[0], smoothed_pos[1], yolo_conf))
                
                # Draw on frame
                x, y = int(smoothed_pos[0]), int(smoothed_pos[1])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f'Shuttle: {yolo_conf:.2f}', (x + 10, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw trajectory
                if len(trajectory) > 1:
                    points = [(int(t[1]), int(t[2])) for t in trajectory[-30:]]
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], (255, 0, 0), 2)
            
            # Detect players if using pose
            if self.pose_model is not None:
                pose_results = self.pose_model(frame, verbose=False)
                if len(pose_results[0].keypoints) > 0:
                    for kp in pose_results[0].keypoints:
                        keypoints = kp.xy.cpu().numpy()[0]
                        for x, y in keypoints:
                            if x > 0 and y > 0:
                                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
            
            if output_path:
                out.write(frame)
            
            if show_visualization:
                cv2.imshow('Shuttle Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_num += 1
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        return trajectory


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Shuttle Tracking System")
    print("=" * 50)
    
    # Example: Training TrackNet
    print("\n1. Training TrackNet (if you have annotated data)")
    print("-" * 50)
    """
    # Prepare your training data
    training_videos = ['video1.mp4', 'video2.mp4']
    annotations = [
        {0: (0.5, 0.3), 1: (0.52, 0.31), 2: None, ...},  # Frame: (x, y) normalized
        {0: (0.4, 0.5), 1: (0.42, 0.51), ...}
    ]
    
    # Train TrackNet
    tracknet_model = train_tracknet(training_videos, annotations, epochs=50)
    torch.save(tracknet_model.state_dict(), 'tracknet_shuttle.pth')
    """
    
    # Example: Using the tracker
    print("\n2. Using the Tracker")
    print("-" * 50)
    """
    # Initialize tracker
    tracker = ShuttleTracker(
        yolo_model_path='yolov8n.pt',  # Or your custom trained model
        tracknet_model_path='tracknet_shuttle.pth',  # Optional
        use_pose=True  # Set to True to track players
    )
    
    # Track shuttle in video
    trajectory = tracker.track_video(
        video_path='badminton_match.mp4',
        output_path='tracked_output.mp4',
        show_visualization=True
    )
    
    # Save trajectory data
    np.savetxt('shuttle_trajectory.csv', trajectory, 
               delimiter=',', header='frame,x,y,confidence', comments='')
    """
    
    print("\nSetup complete! See code comments for usage examples.")
