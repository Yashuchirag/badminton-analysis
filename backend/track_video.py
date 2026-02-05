import cv2
import torch
import numpy as np
import argparse
from tqdm import tqdm
from model.tracknet_code import TrackNet

def normalize_frame(frame, input_h, input_w):
    """Resize and normalize frame"""
    frame_resized = cv2.resize(frame, (input_w, input_h))
    frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_resized = frame_resized.astype(np.float32) / 255.0
    return frame_resized

def overlay_heatmap(frame, heatmap, alpha=0.5):
    """Overlay heatmap on frame"""
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 1.0, heatmap_color, alpha, 0)
    return overlay

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = TrackNet()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    # Video capture / output
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    input_h, input_w = args.input_h, args.input_w
    prev_frames = []

    print(f"Processing video: {args.video} | Total frames: {total_frames}")

    trail_points = []
    for _ in tqdm(range(total_frames), desc="Frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_disp = frame.copy()
        frame_norm = normalize_frame(frame, input_h, input_w)
        prev_frames.append(frame_norm)

        # Keep 3-frame stack
        if len(prev_frames) < 3:
            continue
        elif len(prev_frames) > 3:
            prev_frames.pop(0)

        x_stack = np.concatenate(prev_frames, axis=2)  # [H, W, 9]
        x_stack = torch.from_numpy(x_stack).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            heatmap = model(x_stack)[0, 0].cpu().numpy()  # [H, W]

        # Resize heatmap to original frame
        heatmap_resized = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # ---------------- Normalize heatmap ----------------
        heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
        heatmap_norm = np.clip(heatmap_norm * 1.5, 0, 1)

        # Overlay heatmap
        heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlayed = cv2.addWeighted(frame_disp, 0.7, heatmap_color, 0.5, 0)  # adjust alpha if needed

        y_coord, x_coord = np.unravel_index(heatmap_norm.argmax(), heatmap_norm.shape)
        trail_points.append((x_coord, y_coord))
        # Draw trail
        for i, point in enumerate(trail_points[-30:]):  # last 30 points
            # Older points are more transparent
            alpha = (i + 1) / 30
            color = (0, int(255 * alpha), 0)  # green fading
            cv2.circle(overlayed, point, 5, color, -1)
        
        # Draw circle at max point
        cv2.circle(overlayed, (x_coord, y_coord), 10, (0, 255, 0), -1)

        print(f"Max heatmap value: {heatmap_resized.max():.4f}, Min: {heatmap_resized.min():.4f}")
        
        out.write(overlayed)
        cv2.imshow("TrackNet Overlay", overlayed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved annotated video to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--model", type=str, required=True, help="Trained TrackNet model path")
    parser.add_argument("--output", type=str, default="annotated.mp4", help="Output video path")
    parser.add_argument("--input_h", type=int, default=960, help="Model input height")
    parser.add_argument("--input_w", type=int, default=544, help="Model input width")
    args = parser.parse_args()

    main(args)
