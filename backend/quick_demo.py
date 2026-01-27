import cv2
import numpy as np
from pathlib import Path


class QuickShuttleDemo:
    """Simple demo for testing manual initialization"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.clicks = []
        self.current_frame = None
        self.frame_idx = 0
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((x, y))
            print(f"✓ Click {len(self.clicks)}: ({x}, {y})")
            self.draw_frame()
    
    def draw_frame(self):
        """Draw frame with click markers"""
        if self.current_frame is None:
            return
        
        display = self.current_frame.copy()
        
        # Draw all clicks
        for i, (x, y) in enumerate(self.clicks):
            # Draw crosshair
            cv2.drawMarker(display, (x, y), (0, 255, 0), 
                          cv2.MARKER_CROSS, 30, 3)
            # Draw circle
            cv2.circle(display, (x, y), 20, (0, 255, 0), 2)
            # Draw number
            cv2.putText(display, str(i+1), (x+25, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw extracted templates
        if len(self.clicks) > 0:
            self.draw_templates(display)
        
        # Instructions
        h, w = display.shape[:2]
        
        # Black background for instructions
        cv2.rectangle(display, (10, h-120), (w-10, h-10), (0, 0, 0), -1)
        cv2.rectangle(display, (10, h-120), (w-10, h-10), (0, 255, 0), 2)
        
        instructions = [
            f"Frame: {self.frame_idx + 1} | Clicks: {len(self.clicks)}",
            "LEFT CLICK: Mark shuttle",
            "SPACE: Next frame | 'R': Reset | 'Q': Quit"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(display, text, (20, h - 90 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Quick Demo - Click Shuttle Positions", display)
    
    def draw_templates(self, display):
        """Show extracted shuttle templates"""
        h, w = display.shape[:2]
        
        # Show templates in top-right corner
        template_y = 10
        
        for i, (x, y) in enumerate(self.clicks):
            # Extract template
            size = 30
            y1 = max(0, y - size//2)
            y2 = min(self.current_frame.shape[0], y + size//2)
            x1 = max(0, x - size//2)
            x2 = min(self.current_frame.shape[1], x + size//2)
            
            template = self.current_frame[y1:y2, x1:x2].copy()
            
            if template.size > 0:
                # Resize for display
                template_display = cv2.resize(template, (60, 60))
                
                # Place in corner
                tx = w - 80
                ty = template_y + i * 70
                
                if ty + 60 < h - 130:  # Don't overlap instructions
                    # Black background
                    cv2.rectangle(display, (tx-5, ty-5), 
                                (tx+65, ty+65), (0, 0, 0), -1)
                    # Template
                    display[ty:ty+60, tx:tx+60] = template_display
                    # Border
                    cv2.rectangle(display, (tx, ty), 
                                (tx+60, ty+60), (0, 255, 0), 2)
                    # Label
                    cv2.putText(display, f"#{i+1}", (tx, ty-8),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def run(self, num_frames=5):
        """Run the demo"""
        print(f"\n{'='*60}")
        print(f"Quick Demo: Manual Shuttle Initialization")
        print(f"{'='*60}\n")
        print(f"Video: {self.video_path}")
        print(f"Frames to show: {num_frames}\n")
        
        cap = cv2.VideoCapture(str(self.video_path))
        
        if not cap.isOpened():
            print("❌ Error: Could not open video")
            return
        
        print("Instructions:")
        print("  1. Click on the shuttle (the small white object)")
        print("  2. Press SPACE to see next frame")
        print("  3. Press 'R' to reset current frame clicks")
        print("  4. Press 'Q' when done\n")
        print("Starting demo...\n")
        
        cv2.namedWindow("Quick Demo - Click Shuttle Positions", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Quick Demo - Click Shuttle Positions", self.mouse_callback)
        
        frames = []
        
        # Load frames
        for i in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            print("❌ Error: Could not load frames")
            return
        
        print(f"✓ Loaded {len(frames)} frames\n")
        
        # Show first frame
        self.current_frame = frames[0]
        self.draw_frame()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                break
            
            elif key == ord(' '):  # SPACE - next frame
                if self.frame_idx < len(frames) - 1:
                    self.frame_idx += 1
                    self.current_frame = frames[self.frame_idx]
                    self.clicks = []  # Reset clicks for new frame
                    self.draw_frame()
                    print(f"→ Frame {self.frame_idx + 1}/{len(frames)}")
                else:
                    print("⚠ Last frame reached")
            
            elif key == ord('r'):  # R - reset current frame
                self.clicks = []
                self.draw_frame()
                print("↺ Clicks reset")
            
            elif key == 8:  # BACKSPACE - previous frame
                if self.frame_idx > 0:
                    self.frame_idx -= 1
                    self.current_frame = frames[self.frame_idx]
                    self.clicks = []
                    self.draw_frame()
                    print(f"← Frame {self.frame_idx + 1}/{len(frames)}")
                else:
                    print("⚠ First frame reached")
        
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print(f"Demo Complete!")
        print(f"{'='*60}\n")
        
        if len(self.clicks) > 0:
            print("Summary:")
            print(f"  Frames shown: {len(frames)}")
            print(f"  Total clicks: {len(self.clicks)}")
            print(f"\nClick positions:")
            for i, (x, y) in enumerate(self.clicks):
                print(f"  {i+1}. ({x}, {y})")
            print("\n✓ You're ready to use the full tracking system!")
        else:
            print("No clicks recorded. Try again!")
        print()


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quick_demo.py <video_file> [num_frames]")
        print("\nExample:")
        print("  python quick_demo.py badminton.mp4")
        print("  python quick_demo.py badminton.mp4 10")
        print("\nThis demo lets you practice clicking shuttle positions")
        print("before running the full tracking system.")
        sys.exit(1)
    
    video_path = sys.argv[1]
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    demo = QuickShuttleDemo(video_path)
    demo.run(num_frames=num_frames)


if __name__ == "__main__":
    main()