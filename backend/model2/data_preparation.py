import cv2
import os
import re
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from data_splitter import *
import copy



class VideoFrameExtractor:
    """Extract frames from videos for training"""
    
    @staticmethod
    def extract_frames(video_path, output_dir, sample_rate=1, max_frames=None, 
                      preserve_aspect=True, target_size=None):
        """
        Extract frames from video with proper aspect ratio handling
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            sample_rate: Extract every Nth frame
            max_frames: Maximum number of frames to extract
            preserve_aspect: If True, maintain aspect ratio with padding
            target_size: (width, height) or None to keep original
        """
        version = 1
        while True:
            versioned_dir = os.path.join(output_dir, f"match{version}")
            if not os.path.exists(versioned_dir):
                break
            version += 1
        os.makedirs(versioned_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Save video metadata
        metadata = {
            'total_frames': total_frames,
            'fps': fps,
            'sample_rate': sample_rate,
            'original_size': None
        }
        
        extracted = 0
        frame_num = 0
        
        print(f"Extracting frames from {video_path}")
        print(f"FPS: {fps}, Total frames: {total_frames}")
        
        with tqdm(total=min(total_frames // sample_rate, max_frames or float('inf'))) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Store original size from first frame
                if metadata['original_size'] is None:
                    metadata['original_size'] = frame.shape[:2]  # (height, width)
                
                if frame_num % sample_rate == 0:
                    # Optionally resize with aspect ratio preservation
                    if target_size is not None:
                        if preserve_aspect:
                            frame = VideoFrameExtractor._resize_with_padding(
                                frame, target_size
                            )
                        else:
                            frame = cv2.resize(frame, target_size)
                    
                    output_path = os.path.join(versioned_dir, f"frame_{frame_num:06d}.jpg")
                    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    extracted += 1
                    pbar.update(1)
                    
                    if max_frames and extracted >= max_frames:
                        break
                
                frame_num += 1
        
        cap.release()
        
        # Save metadata
        metadata_path = os.path.join(versioned_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Extracted {extracted} frames to {versioned_dir}")
        print(f"Metadata saved to {metadata_path}")
        
        return extracted, versioned_dir
    
    @staticmethod
    def _resize_with_padding(image, target_size):
        """Resize image preserving aspect ratio with padding"""
        target_w, target_h = target_size
        h, w = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return padded


class ShuttleAnnotationTool:
    """Enhanced annotation tool with motion blur and elongation support"""

    def __init__(self, image_dir, output_dir, display_height=960):
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.display_height = display_height
        os.makedirs(output_dir, exist_ok=True)

        # Find existing version folders (v1, v2, v3...)
        existing_versions = [
            d for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d)) and re.match(r"v\d+", d)
        ]

        # Extract version numbers
        version_numbers = [
            int(re.search(r"\d+", v).group()) for v in existing_versions
        ]

        # Determine next version
        next_version = max(version_numbers) + 1 if version_numbers else 1

        # Create new version folder
        self.output_dir = os.path.join(output_dir, f"v{next_version}")
        os.makedirs(self.output_dir, exist_ok=True)


        self.images = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        )


        if not self.images:
            raise ValueError(f"No images found in {image_dir}")

        print(f"Total images: {len(self.images)}")

        self.current_idx = 0
        self.annotations = self._load_existing_annotations()
        self.annotation_history = []  # For undo

        # Multi-frame view
        self.show_previous = True
        self.show_next = True

        # Zoom state
        self.zoom_level = 1.0
        self.zoom_center = None                                          
        self._base_scale  = 1.0                                          
        self._zoom_offset = (0, 0)                                       
        self._zoom_scale  = 1.0     

        # Annotation state
        self.current_annotation = None
        self.box_start = None
        self.box_end = None

        # Visibility flag
        self.visibility_states = {
            "visible": 0,       # Clear shuttle
            "occluded": 1,      # Partially occluded
            "not_visible": 2,   # Not in frame / fully occluded
        }
        self.current_visibility = "visible"

        # Enhanced annotation modes
        self.annotation_mode = "point"  # 'point', 'box', 'line', 'oriented_box'

        # For oriented bounding boxes (handles elongation)
        self.obb_points = []  # Store multiple points for oriented box

        # Blur handling
        self.blur_states = {
            "clear": 0,          # Sharp, visible shuttle
            "slight_blur": 1,    # Slightly blurred
            "motion_blur": 2,    # Clear motion blur streak
            "severe_blur": 3,    # Very blurred / barely visible
        }
        self.current_blur = "clear"

    def annotate(self):
        """Interactive annotation loop with enhanced features"""
        self._print_banner()

        cv2.namedWindow("Annotate Shuttle", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Annotate Shuttle", self._mouse_callback)

        while self.current_idx < len(self.images):
            self._display_current_frame()

            key = cv2.waitKey(1) & 0xFF

            if key == ord("m"):
                self._cycle_annotation_mode()

            elif key == ord("l"):
                self._cycle_blur_level()

            elif key == ord("e"):  # Quick elongated mode
                self.annotation_mode = "line"
                print("Elongated shuttle mode: Draw a line along the blur")

            elif key == ord(" "):  # Next
                self._save_current_annotation()
                self.current_idx = min(self.current_idx + 1, len(self.images) - 1)

            elif key == ord("a"):  # Previous
                self._save_current_annotation()
                self.current_idx = max(self.current_idx - 1, 0)

            elif key == ord("s"):  # Skip
                self._save_skip()
                self.current_idx = min(self.current_idx + 1, len(self.images) - 1)

            elif key == ord("u"):
                self._undo_annotation()

            elif key == ord("d"):
                self._delete_current_annotation()

            elif key == ord("v"):
                self._cycle_visibility()

            elif key == ord("p"):
                self.show_previous = not self.show_previous

            elif key == ord("n"):
                self.show_next = not self.show_next

            elif key in (ord("+"), ord("=")):  # Zoom in
                if self.zoom_center is None:
                    img_path = os.path.join(self.image_dir, self.images[self.current_idx])
                    tmp = cv2.imread(img_path)
                    if tmp is not None:
                        self.zoom_center = (tmp.shape[1] // 2, tmp.shape[0] // 2)
                self.zoom_level = min(self.zoom_level * 1.2, 5.0)

            elif key in (ord("-"), ord("_")):  # Zoom out
                self.zoom_level = max(self.zoom_level / 1.2, 1.0)
                if self.zoom_level <= 1.0:
                    self.zoom_level = 1.0
                    self.zoom_center = None

            elif key == ord("i"):
                self._interpolate_annotations()

            elif key == ord("h"):
                self._show_help()

            elif key == ord("q"):  # Quit
                self._save_current_annotation()
                break

        cv2.destroyAllWindows()
        self._save_all_annotations()
        self._print_statistics()

    def _load_existing_annotations(self):
        """Load previously saved annotations if they exist"""
        json_path = os.path.join(self.output_dir, "annotations.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                print(f"Loaded {len(data)} existing annotations")
                return data
        return {}

    def _display_current_frame(self):
        """Display current frame with annotations and overlays"""
        image_name = self.images[self.current_idx]
        image_path = os.path.join(self.image_dir, image_name)

        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading {image_path}")
            return

        h, w = img.shape[:2]

        # Base scale: fit height to display_height
        self._base_scale = self.display_height / h
        display_w = int(w * self._base_scale)
        display_img = cv2.resize(img, (display_w, self.display_height))

        # ── Overlay previous / next frames (independent blends) ──
        # Each neighbour is blended against the *original* resized current
        # frame so their weights don't compound.
        if self.show_previous and self.current_idx > 0:
            prev_img = self._load_and_resize_frame(
                self.current_idx - 1, (display_w, self.display_height)
            )
            if prev_img is not None:
                display_img = cv2.addWeighted(display_img, 0.85, prev_img, 0.15, 0)

        if self.show_next and self.current_idx < len(self.images) - 1:
            next_img = self._load_and_resize_frame(
                self.current_idx + 1, (display_w, self.display_height)
            )
            if next_img is not None:
                display_img = cv2.addWeighted(display_img, 0.85, next_img, 0.15, 0)

        # ── Apply zoom (after blending, before drawing annotations) ──
        if self.zoom_level > 1.0:
            display_img = self._apply_zoom(display_img)

        # ── Draw saved annotation for this frame ──
        if image_name in self.annotations:
            self._draw_annotation(
                display_img, self.annotations[image_name], (0, 255, 0)
            )

        # ── Draw in-progress annotation ──
        if self.current_annotation:
            self._draw_annotation(display_img, self.current_annotation, (255, 0, 0))

        # ── Draw rubber-band line / box while dragging ──
        if self.box_start and self.box_end:
            p1 = self._orig_to_display(self.box_start)
            p2 = self._orig_to_display(self.box_end)
            if self.annotation_mode == "box":
                cv2.rectangle(display_img, p1, p2, (255, 255, 0), 2)
            else:  # line mode rubber band
                cv2.line(display_img, p1, p2, (255, 255, 0), 2)

        # ── Draw OBB guide points while placing oriented box ──
        if self.annotation_mode == "oriented_box" and self.obb_points:
            for i, pt in enumerate(self.obb_points):
                dp = self._orig_to_display(pt)
                cv2.circle(display_img, dp, 6, (255, 255, 0), -1)
                cv2.putText(
                    display_img,
                    str(i + 1),
                    (dp[0] + 8, dp[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )
            if len(self.obb_points) == 2:
                p1 = self._orig_to_display(self.obb_points[0])
                p2 = self._orig_to_display(self.obb_points[1])
                cv2.line(display_img, p1, p2, (255, 255, 0), 2)

        # ── HUD ──
        self._draw_ui(display_img)

        cv2.imshow("Annotate Shuttle", display_img)

    def _orig_to_display(self, orig_pt):                                 
        """Map a point in original-image space → current display pixel space."""
        if self.zoom_level > 1.0:                                        
            dx = (orig_pt[0] - self._zoom_offset[0]) * self._zoom_scale  
            dy = (orig_pt[1] - self._zoom_offset[1]) * self._zoom_scale  
        else:
            dx = orig_pt[0] * self._base_scale
            dy = orig_pt[1] * self._base_scale
        return (int(dx), int(dy))

    def _display_to_orig(self, disp_x, disp_y):                         
        """Map a display pixel → original-image coordinates (zoom-aware)."""
        if self.zoom_level > 1.0:                                        
            orig_x = disp_x / self._zoom_scale + self._zoom_offset[0]    
            orig_y = disp_y / self._zoom_scale + self._zoom_offset[1]    
        else:
            orig_x = disp_x / self._base_scale
            orig_y = disp_y / self._base_scale
        return (max(0.0, orig_x), max(0.0, orig_y))

    def _apply_zoom(self, display_img):                                  
        """Crop & upscale the display image around zoom_center."""
        disp_h, disp_w = display_img.shape[:2]                           

        # --- visible rectangle in ORIGINAL-image coords ---             
        orig_w = disp_w / self._base_scale                               
        orig_h = disp_h / self._base_scale                               

        cx, cy = self.zoom_center if self.zoom_center else (orig_w / 2, orig_h / 2)  

        vis_w = orig_w / self.zoom_level                                 
        vis_h = orig_h / self.zoom_level                                 

        ox = max(0, min(cx - vis_w / 2, orig_w - vis_w))                 
        oy = max(0, min(cy - vis_h / 2, orig_h - vis_h))                 

        # Cache for coordinate helpers (original-image space)
        self._zoom_offset = (ox, oy)                                     
        self._zoom_scale  = self._base_scale * self.zoom_level           

        # --- map orig window → display pixels, then crop ---            
        dx1 = int(round(ox * self._base_scale))                          
        dy1 = int(round(oy * self._base_scale))                          
        dx2 = int(round((ox + vis_w) * self._base_scale))                
        dy2 = int(round((oy + vis_h) * self._base_scale))                

        dx1, dy1 = max(0, dx1), max(0, dy1)                              
        dx2, dy2 = min(disp_w, dx2), min(disp_h, dy2)                    

        cropped = display_img[dy1:dy2, dx1:dx2]                          
        return cv2.resize(cropped, (disp_w, disp_h))

    def _mouse_callback(self, event, x, y, flags, param):
        """Route mouse events to the active annotation-mode handler."""
        # Right-click sets zoom centre (in pre-zoom display coords)
        if event == cv2.EVENT_RBUTTONDOWN:                               
            ox, oy = self._display_to_orig(x, y)                         
            img_path = os.path.join(self.image_dir, self.images[self.current_idx])
            tmp = cv2.imread(img_path)
            if tmp is not None:
                self.zoom_center = (                                     
                    int(max(0, min(ox, tmp.shape[1] - 1))),
                    int(max(0, min(oy, tmp.shape[0] - 1))),
                )
            return

        # Convert screen click → original image pixel (zoom-aware)
        orig_x, orig_y = self._display_to_orig(x, y)

        # Clamp to actual image dimensions
        image_path = os.path.join(self.image_dir, self.images[self.current_idx])
        img = cv2.imread(image_path)
        if img is None:
            return
        img_h, img_w = img.shape[:2]
        orig_x = max(0, min(orig_x, img_w - 1))
        orig_y = max(0, min(orig_y, img_h - 1))

        if self.annotation_mode == "point":
            self._handle_point_annotation(event, orig_x, orig_y)
        elif self.annotation_mode == "box":
            self._handle_box_annotation(event, orig_x, orig_y)
        elif self.annotation_mode == "line":
            self._handle_line_annotation(event, orig_x, orig_y, img_w, img_h)
        elif self.annotation_mode == "oriented_box":
            self._handle_oriented_box_annotation(event, orig_x, orig_y, img_w, img_h)

    def _handle_point_annotation(self, event, x, y):
        """Single-click point annotation for slow / clear shuttles."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_annotation = {
                "x": x,
                "y": y,
                "visibility": self.current_visibility,
                "blur": self.current_blur,
                "type": "point",
            }
            print(f"Point annotation: ({x}, {y})")

    def _handle_box_annotation(self, event, x, y):
        """Drag to draw an axis-aligned bounding box."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.box_start = (x, y)
            self.box_end = None
            self.current_annotation = None  # discard any previous pending

        elif event == cv2.EVENT_MOUSEMOVE and self.box_start is not None:
            self.box_end = (x, y)  # rubber-band

        elif event == cv2.EVENT_LBUTTONUP and self.box_start is not None:
            self.box_end = (x, y)

            x1, y1 = self.box_start
            x2, y2 = self.box_end

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = abs(x2 - x1)
            bh = abs(y2 - y1)

            self.current_annotation = {
                "x": cx,
                "y": cy,
                "w": bw,
                "h": bh,
                "visibility": self.current_visibility,
                "blur": self.current_blur,
                "type": "box",
            }

            print(f"Box annotation: center=({cx:.0f},{cy:.0f}) size={bw}x{bh}")

            self.box_start = None
            self.box_end = None

    def _handle_line_annotation(self, event, x, y, img_w, img_h):
        """Drag along the blur streak → auto-generates an oriented box."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.box_start = (x, y)
            self.box_end = None

        elif event == cv2.EVENT_MOUSEMOVE and self.box_start is not None:
            self.box_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP and self.box_start is not None:
            self.box_end = (x, y)

            x1, y1 = self.box_start
            x2, y2 = self.box_end

            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 2:  # ignore accidental micro-drags
                self.box_start = None
                self.box_end = None
                return

            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # Perpendicular width scales with blur severity
            width_multipliers = {
                "clear": 0.15,
                "slight_blur": 0.12,
                "motion_blur": 0.08,
                "severe_blur": 0.05,
            }
            width = max(8, length * width_multipliers[self.current_blur])

            self.current_annotation = {
                "x": center_x,
                "y": center_y,
                "w": length,
                "h": width,
                "angle": angle,
                "visibility": self.current_visibility,
                "blur": self.current_blur,
                "type": "oriented_box",
                "is_elongated": True,
                "motion_direction": angle,
            }

            print(
                f"Elongated shuttle: length={length:.1f}px, "
                f"angle={angle:.1f}°, blur={self.current_blur}"
            )

            self.box_start = None
            self.box_end = None

    def _handle_oriented_box_annotation(self, event, x, y, img_w, img_h):
        """Three-click oriented bounding box.
        Click 1 & 2 → length edge.  Click 3 → width (perpendicular offset).
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        self.obb_points.append((x, y))
        print(f"OBB point {len(self.obb_points)}/3: ({x}, {y})")

        if len(self.obb_points) == 3:
            p1, p2, p3 = self.obb_points

            length = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            if length < 1:
                print("Points too close – try again")
                self.obb_points = []
                return

            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180.0 / np.pi
            center_x = (p1[0] + p2[0]) / 2.0
            center_y = (p1[1] + p2[1]) / 2.0

            # Perpendicular distance from p3 to the line p1→p2
            width = (
                abs(
                    (p2[0] - p1[0]) * (p1[1] - p3[1])
                    - (p1[0] - p3[0]) * (p2[1] - p1[1])
                )
                / length
            )
            width *= 2  # p3 gives half-width

            self.current_annotation = {
                "x": center_x,
                "y": center_y,
                "w": length,
                "h": width,
                "angle": angle,
                "visibility": self.current_visibility,
                "blur": self.current_blur,
                "type": "oriented_box",
                "is_elongated": True,
                "motion_direction": angle,
            }

            print(f"Oriented box: {length:.1f}×{width:.1f}px @ {angle:.1f}°")
            self.obb_points = []

    def _draw_annotation(self, img, annotation, color):
        """Draw a single annotation onto the display image (already zoomed)."""
        if annotation is None:
            return
        if annotation.get("visibility") == "not_visible":
            return
        if "x" not in annotation or "y" not in annotation:
            return

        # Map original coords → current display coords
        dx, dy = self._orig_to_display((annotation["x"], annotation["y"]))

        ann_type = annotation.get("type", "point")

        if ann_type == "oriented_box" and "angle" in annotation:
            # ── rotated rectangle ──
            dw = int(annotation["w"] * self._base_scale)
            dh = int(annotation["h"] * self._base_scale)
            # boxPoints expects ((cx,cy),(w,h),angle)
            rect = ((dx, dy), (dw, dh), annotation["angle"])
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(img, [box], 0, color, 2)

            # Motion-direction arrow
            arrow_len = dw // 2
            angle_rad = annotation["angle"] * np.pi / 180.0
            end_x = int(dx + arrow_len * np.cos(angle_rad))
            end_y = int(dy + arrow_len * np.sin(angle_rad))
            cv2.arrowedLine(img, (dx, dy), (end_x, end_y), (255, 0, 255), 2)

            # Blur-level dot
            blur_colors = {
                "clear": (0, 255, 0),
                "slight_blur": (0, 255, 255),
                "motion_blur": (0, 165, 255),
                "severe_blur": (0, 0, 255),
            }
            dot_color = blur_colors.get(annotation.get("blur", "clear"), color)
            cv2.circle(img, (dx, dy), 5, dot_color, -1)

        elif ann_type == "box":
            # ── axis-aligned box ──
            dw = int(annotation["w"] * self._base_scale)
            dh = int(annotation["h"] * self._base_scale)
            x1 = dx - dw // 2
            y1 = dy - dh // 2
            cv2.rectangle(img, (x1, y1), (x1 + dw, y1 + dh), color, 2)
            cv2.circle(img, (dx, dy), 4, color, -1)

        else:
            # ── point ──
            cv2.circle(img, (dx, dy), 5, color, -1)
            cv2.circle(img, (dx, dy), 9, color, 2)

        # Label
        label = f"{annotation.get('blur', 'clear')} | {annotation.get('visibility', 'visible')}"
        cv2.putText(
            img, label, (dx + 12, dy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
        )

    def _draw_ui(self, img):
        """HUD overlay with current mode / blur / frame info."""
        h, w = img.shape[:2]

        # Semi-transparent banner
        banner_h = 140
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        cv2.rectangle(img, (0, 0), (w, banner_h), (0, 255, 0), 2)

        image_name = self.images[self.current_idx]
        lines = [
            f"Frame: {self.current_idx + 1}/{len(self.images)}  –  {image_name}",
            f"Annotated: {len(self.annotations)}/{len(self.images)}",
            f"Mode: {self.annotation_mode.upper()}  |  Vis: {self.current_visibility.upper()}",
            f"Blur: {self.current_blur.upper()}  |  Zoom: {self.zoom_level:.1f}x",
            "M: mode  L: blur  E: elongated  H: help  Q: quit",
        ]

        for i, text in enumerate(lines):
            cv2.putText(
                img, text, (10, 22 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )

    def _save_current_annotation(self):
        """Commit the pending annotation to the annotations dict."""
        if self.current_annotation is None:
            return

        image_name = self.images[self.current_idx]

        # History entry for undo
        self.annotation_history.append(
            {
                "image": image_name,
                "old_value": self.annotations.get(image_name),
                "new_value": copy.deepcopy(self.current_annotation),
            }
        )

        self.annotations[image_name] = self.current_annotation
        self.current_annotation = None
        self.box_start = None
        self.box_end = None

    def _save_skip(self):
        """Mark frame as having no visible shuttle."""
        image_name = self.images[self.current_idx]

        self.annotation_history.append(
            {
                "image": image_name,
                "old_value": self.annotations.get(image_name),
                "new_value": {"visibility": "not_visible", "type": "skip"},
            }
        )

        self.annotations[image_name] = {"visibility": "not_visible", "type": "skip"}
        self.current_annotation = None
        print("Frame marked as no shuttle visible")

    def _undo_annotation(self):
        """Revert the most recent annotation change."""
        if not self.annotation_history:
            print("Nothing to undo")
            return

        last = self.annotation_history.pop()
        image_name = last["image"]

        if last["old_value"] is None:
            self.annotations.pop(image_name, None)
        else:
            self.annotations[image_name] = last["old_value"]

        # If we undid the current frame, also clear pending state
        if image_name == self.images[self.current_idx]:
            self.current_annotation = None

        print(f"Undone annotation for {image_name}")

    def _delete_current_annotation(self):
        """Delete the saved annotation for the current frame."""
        image_name = self.images[self.current_idx]
        if image_name in self.annotations:
            self.annotation_history.append(
                {
                    "image": image_name,
                    "old_value": self.annotations[image_name],
                    "new_value": None,
                }
            )
            del self.annotations[image_name]
            print(f"Deleted annotation for {image_name}")

        self.current_annotation = None

    def _cycle_annotation_mode(self):
        modes = ["point", "box", "line", "oriented_box"]
        descriptions = {
            "point": "Single point (slow shuttle)",
            "box": "Axis-aligned box (medium speed)",
            "line": "Line along motion blur (fast shuttle)",
            "oriented_box": "Rotated box – 3 clicks (precise elongation)",
        }
        idx = modes.index(self.annotation_mode)
        self.annotation_mode = modes[(idx + 1) % len(modes)]
        self.obb_points = []  # reset partial OBB
        print(f"Annotation mode → {descriptions[self.annotation_mode]}")

    def _cycle_blur_level(self):
        levels = list(self.blur_states.keys())
        idx = levels.index(self.current_blur)
        self.current_blur = levels[(idx + 1) % len(levels)]
        print(f"Blur level → {self.current_blur}")

    def _cycle_visibility(self):
        states = list(self.visibility_states.keys())
        idx = states.index(self.current_visibility)
        self.current_visibility = states[(idx + 1) % len(states)]
        print(f"Visibility → {self.current_visibility}")

    def _interpolate_annotations(self):
        """Linear interpolation of position (and orientation) between two annotated frames."""
        print("\n── Interpolation ──")
        try:
            start_idx = int(input(f"Start frame (0–{len(self.images)-1}): "))
            end_idx   = int(input(f"End   frame (0–{len(self.images)-1}): "))
        except (ValueError, KeyboardInterrupt, EOFError):
            print("Interpolation cancelled")
            return

        if start_idx >= end_idx or start_idx < 0 or end_idx >= len(self.images):
            print("Invalid frame range")
            return

        start_name = self.images[start_idx]
        end_name   = self.images[end_idx]

        if start_name not in self.annotations or end_name not in self.annotations:
            print("Both start and end frames must already be annotated")
            return

        start_ann = self.annotations[start_name]
        end_ann   = self.annotations[end_name]

        if "x" not in start_ann or "x" not in end_ann:
            print("Can only interpolate annotations that have position data")
            return

        num_frames = end_idx - start_idx
        interpolated = 0

        for i in range(1, num_frames):
            frame_idx  = start_idx + i
            frame_name = self.images[frame_idx]

            if frame_name in self.annotations:
                continue  # don't overwrite existing

            t = i / num_frames  # 0 < t < 1

            ann = {
                "x": start_ann["x"] + t * (end_ann["x"] - start_ann["x"]),
                "y": start_ann["y"] + t * (end_ann["y"] - start_ann["y"]),
                "visibility": "visible",
                "blur": start_ann.get("blur", "clear"),
                "type": "interpolated",
            }

            # Interpolate box dimensions if present on both ends
            for key in ("w", "h"):
                if key in start_ann and key in end_ann:
                    ann[key] = start_ann[key] + t * (end_ann[key] - start_ann[key])

            # Interpolate oriented-box fields if present on both ends
            if "angle" in start_ann and "angle" in end_ann:
                ann["angle"] = start_ann["angle"] + t * (end_ann["angle"] - start_ann["angle"])
                ann["type"] = "oriented_box"
                ann["is_elongated"] = True
                ann["motion_direction"] = ann["angle"]

            self.annotations[frame_name] = ann
            interpolated += 1

        print(f"Interpolated {interpolated} frames between {start_idx} and {end_idx}")

    def _save_all_annotations(self):
        """Persist annotations in JSON + YOLO formats."""
        # Raw JSON
        json_path = os.path.join(self.output_dir, "annotations.json")
        with open(json_path, "w") as f:
            json.dump(self.annotations, f, indent=2)
        print(f"Saved raw annotations → {json_path}")

        self._save_yolo_format()
        self._save_validation_info()

    def _save_yolo_format(self):
        """Write standard YOLO and YOLO-OBB label files for every annotated image."""
        yolo_dir     = os.path.join(self.output_dir, "yolo_labels")
        yolo_obb_dir = os.path.join(self.output_dir, "yolo_obb_labels")
        os.makedirs(yolo_dir, exist_ok=True)
        os.makedirs(yolo_obb_dir, exist_ok=True)

        saved_standard = 0
        saved_obb      = 0

        for image_name, annotation in self.annotations.items():
            label_name = os.path.splitext(image_name)[0] + ".txt"

            # ── not_visible → empty label files ──
            if annotation.get("visibility") == "not_visible":
                open(os.path.join(yolo_dir, label_name), "w").close()
                open(os.path.join(yolo_obb_dir, label_name), "w").close()
                continue

            if "x" not in annotation or "y" not in annotation:
                continue

            # Need image dimensions for normalisation
            img = cv2.imread(os.path.join(self.image_dir, image_name))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            cx = annotation["x"]
            cy = annotation["y"]

            # ── Determine axis-aligned w/h ──
            if annotation.get("type") == "oriented_box" and "angle" in annotation:
                bw     = annotation["w"]
                bh     = annotation["h"]
                angle  = annotation["angle"] * np.pi / 180.0
                cos_a  = abs(np.cos(angle))
                sin_a  = abs(np.sin(angle))
                aa_w   = bw * cos_a + bh * sin_a
                aa_h   = bw * sin_a + bh * cos_a
            else:
                aa_w = annotation.get("w", max(20, int(img_w * 0.02)))
                aa_h = annotation.get("h", max(20, int(img_h * 0.02)))

            # Normalise & clamp
            nx  = np.clip(cx / img_w, 0.0, 1.0)
            ny  = np.clip(cy / img_h, 0.0, 1.0)
            nw  = np.clip(aa_w / img_w, 0.0, 1.0)
            nh  = np.clip(aa_h / img_h, 0.0, 1.0)

            # ── Standard YOLO label ──
            with open(os.path.join(yolo_dir, label_name), "w") as f:
                f.write(f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}\n")
            saved_standard += 1

            # ── OBB label ──
            if annotation.get("type") == "oriented_box" and "angle" in annotation:
                corners = self._obb_corners(annotation)
                # Normalise
                corners[:, 0] /= img_w
                corners[:, 1] /= img_h
                corners = np.clip(corners, 0.0, 1.0)

                with open(os.path.join(yolo_obb_dir, label_name), "w") as f:
                    pts = " ".join(
                        f"{corners[i, 0]:.6f} {corners[i, 1]:.6f}" for i in range(4)
                    )
                    f.write(f"0 {pts}\n")
                saved_obb += 1
            else:
                # Write axis-aligned corners as a degenerate OBB so every
                # image gets a matching label file.
                half_w = aa_w / 2.0
                half_h = aa_h / 2.0
                corners = np.array(
                    [
                        [cx - half_w, cy - half_h],
                        [cx + half_w, cy - half_h],
                        [cx + half_w, cy + half_h],
                        [cx - half_w, cy + half_h],
                    ]
                )
                corners[:, 0] /= img_w
                corners[:, 1] /= img_h
                corners = np.clip(corners, 0.0, 1.0)

                with open(os.path.join(yolo_obb_dir, label_name), "w") as f:
                    pts = " ".join(
                        f"{corners[i, 0]:.6f} {corners[i, 1]:.6f}" for i in range(4)
                    )
                    f.write(f"0 {pts}\n")
                saved_obb += 1

        print(f"Saved {saved_standard} standard YOLO labels → {yolo_dir}")
        print(f"Saved {saved_obb}      OBB labels          → {yolo_obb_dir}")

    @staticmethod
    def _obb_corners(annotation):
        """Return the 4 corners (N×2 float array) of an oriented-box annotation."""
        cx, cy = annotation["x"], annotation["y"]
        w, h   = annotation["w"], annotation["h"]
        angle  = annotation["angle"] * np.pi / 180.0

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Half-extents along local axes
        dx_w =  (w / 2) * cos_a
        dy_w =  (w / 2) * sin_a
        dx_h = -(h / 2) * sin_a
        dy_h =  (h / 2) * cos_a

        return np.array(
            [
                [cx - dx_w - dx_h, cy - dy_w - dy_h],  # top-left
                [cx + dx_w - dx_h, cy + dy_w - dy_h],  # top-right
                [cx + dx_w + dx_h, cy + dy_w + dy_h],  # bottom-right
                [cx - dx_w + dx_h, cy - dy_w + dy_h],  # bottom-left
            ],
            dtype=np.float64,
        )

    def _save_validation_info(self):
        """Write a JSON report of annotation coverage and potential issues."""
        total     = len(self.images)
        annotated = len(self.annotations)

        vis_counts  = defaultdict(int)
        type_counts = defaultdict(int)
        for ann in self.annotations.values():
            vis_counts[ann.get("visibility", "unknown")]  += 1
            type_counts[ann.get("type", "unknown")]       += 1

        # Detect large gaps
        annotated_indices = sorted(
            self.images.index(name) for name in self.annotations
        )
        issues = []
        for i in range(len(annotated_indices) - 1):
            gap = annotated_indices[i + 1] - annotated_indices[i]
            if gap > 30:
                issues.append(
                    f"Large gap ({gap} frames) between frames "
                    f"{annotated_indices[i]} and {annotated_indices[i+1]}"
                )

        recommendations = []
        if total > 0 and annotated / total < 0.3:
            recommendations.append(
                "Low annotation rate – consider annotating more frames or using interpolation"
            )
        if vis_counts.get("occluded", 0) == 0:
            recommendations.append(
                "No occluded samples – consider adding occluded examples for robustness"
            )

        info = {
            "total_frames": total,
            "annotated_frames": annotated,
            "annotation_rate": annotated / total if total > 0 else 0,
            "visibility_distribution": dict(vis_counts),
            "annotation_types": dict(type_counts),
            "potential_issues": issues,
            "recommended_actions": recommendations,
        }

        path = os.path.join(self.output_dir, "validation_info.json")
        with open(path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"Saved validation info → {path}")

    def _print_banner(self):
        sep = "=" * 70
        print(f"\n{sep}")
        print("SHUTTLE ANNOTATION TOOL – MOTION BLUR SUPPORT")
        print(sep)
        print("MODES:  1-Point  2-Box  3-Line  4-Oriented Box")
        print("\nKEYS:")
        print("  SPACE  Next                    A      Previous")
        print("  S      Skip                    U      Undo                       D   Delete")
        print("  M      Cycle mode              L      Cycle blur                 E   Quick elongated")
        print("  V      Visibility              P/N    Prev/Next overlay")
        print("  +/-    Zoom                    I      Interpolate                Q   Quit")
        print("  H      Help")
        print(f"{sep}\n")

    def _show_help(self):
        self._print_banner()

    def _print_statistics(self):
        total     = len(self.images)
        annotated = len(self.annotations)

        vis_counts = defaultdict(int)
        for ann in self.annotations.values():
            vis_counts[ann.get("visibility", "unknown")] += 1

        sep = "=" * 70
        print(f"\n{sep}")
        print("ANNOTATION STATISTICS")
        print(sep)
        print(f"Total frames     : {total}")
        print(f"Annotated frames : {annotated}")
        print(f"Annotation rate  : {annotated / total * 100:.1f}%" if total else "N/A")
        print("\nVisibility breakdown:")
        for vis, count in sorted(vis_counts.items()):
            pct = count / annotated * 100 if annotated else 0
            print(f"  {vis:15s} {count:>5d}  ({pct:.1f} %)")
        print(f"{sep}\n")

    def _load_and_resize_frame(self, idx, size):
        """Load and resize a neighbour frame for overlay."""
        if idx < 0 or idx >= len(self.images):
            return None
        img = cv2.imread(os.path.join(self.image_dir, self.images[idx]))
        if img is None:
            return None
        return cv2.resize(img, size)



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced data preparation utilities')
    parser.add_argument('--action', 
                       choices=['extract', 'annotate', 'split', 'create-structure'],
                       required=True, help='Action to perform')

    # Common arguments
    parser.add_argument('--video', type=str, help='Video file path')
    parser.add_argument('--images', type=str, help='Images directory')
    parser.add_argument('--labels', type=str, help='Labels directory')
    parser.add_argument('--output', type=str, help='Output directory')

    # Extract arguments
    parser.add_argument('--sample-rate', type=int, default=1, help='Frame sampling rate')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to extract')


    # Annotate argument
    parser.add_argument('--display-height', type=int, default=960, help='Display height for annotation')

    # Split arguments
    parser.add_argument('--annotation-output', type=str,
                        help="[split] Annotator's output_dir (contains annotations.json, yolo_labels/, yolo_obb_labels/)")
    parser.add_argument("--method", choices=["rally", "sequence"], required=False)
    parser.add_argument('--train-ratio', type=float, default=0.7, help='[split] Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='[split] Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='[split] Test set ratio')
    parser.add_argument("--stratify-by", default="difficulty",
                        choices=["difficulty", "length", "none"])
    parser.add_argument('--temporal-gap', type=int, default=1,
                        help='[split] Temporal gap between chunks (default: 1)')
    parser.add_argument('--max-chunk-size', type=int, default=10,
                        help='[split] Maximum chunk size (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='[split] Random seed for reproducibility (default: 42)')
    

    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--sequence-stride", type=int, default=8)
    parser.add_argument("--min-rally-length", type=int, default=10)
    parser.add_argument("--max-gap", type=int, default=5)
    

    args = parser.parse_args()
    
    if args.action == 'extract':
        if not args.video or not args.output:
            print("Error: --video and --output required for extraction")
        else:
            VideoFrameExtractor.extract_frames(
                args.video, args.output, args.sample_rate, args.max_frames
            )
    
    elif args.action == 'annotate':
        if not args.images or not args.output:
            print("Error: --images and --output required for annotation")
        else:
            tool = ShuttleAnnotationTool(args.images, args.output, args.display_height)
            tool.annotate()
    
    elif args.action == 'split':
        if not args.images or not args.annotation_output or not args.output:
            parser.error("--action split requires --images, --annotation-output, and --output")
        
        # Validate ratios
        if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
            parser.error("--train-ratio, --val-ratio, and --test-ratio must sum to 1.0")
        
        if args.method == "rally":
            RallyAwareDatasetSplitter.split_by_rally(
                images_dir=args.images,
                annotation_output_dir=args.annotation_output,
                output_base_dir=args.output,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                stratify_by=args.stratify_by,
                min_rally_length=args.min_rally_length,
                max_gap=args.max_gap,
                seed=args.seed,
            )
        else:  # sequence
            RallyAwareDatasetSplitter.split_by_sequence(
                images_dir=args.images,
                annotation_output_dir=args.annotation_output,
                output_base_dir=args.output,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                sequence_length=args.sequence_length,
                sequence_stride=args.sequence_stride,
                min_rally_length=args.min_rally_length,
                max_gap=args.max_gap,
                seed=args.seed,
            )
    
    elif args.action == 'create-structure':
        # Create enhanced dataset structure
        print("Creating dataset structure...")
        os.makedirs('dataset/videos', exist_ok=True)
        os.makedirs('dataset/raw_frames', exist_ok=True)
        os.makedirs('dataset/annotations', exist_ok=True)
        os.makedirs('dataset/processed', exist_ok=True)
        
        readme = """
Enhanced Badminton Shuttle Dataset Structure

1. Place videos in dataset/videos/
2. Extract frames: python data_preparation.py --action extract --video dataset/videos/match1.mp4 --output dataset/raw_frames --sample-rate 2
3. Annotate frames: python data_preparation.py --action annotate --images dataset/raw_frames --output dataset/annotations
4. Split dataset: python data_preparation.py --action split --images dataset/raw_frames --labels dataset/annotations --output dataset/processed

The annotation tool includes:
- Undo/redo functionality
- Zoom and pan
- Previous/next frame overlay
- Interpolation between frames
- Visibility states (visible/occluded/not_visible)
- Both point and bounding box annotation modes
- Automatic YOLO format export
- Quality validation metrics
"""
        
        with open('dataset/README.txt', 'w') as f:
            f.write(readme)
        
        print("Dataset structure created! See dataset/README.txt for usage instructions")