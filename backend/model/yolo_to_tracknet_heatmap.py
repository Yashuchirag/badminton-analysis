import cv2
import numpy as np
from pathlib import Path
import argparse


def draw_gaussian(heatmap, center, sigma=4):
    """Draw a 2D Gaussian on heatmap"""
    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape

    size = int(6 * sigma + 1)
    x0 = max(0, x - size // 2)
    y0 = max(0, y - size // 2)
    x1 = min(w, x + size // 2 + 1)
    y1 = min(h, y + size // 2 + 1)

    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)[:, None]

    gaussian = np.exp(
        -((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2)
    )

    heatmap[y0:y1, x0:x1] = np.maximum(
        heatmap[y0:y1, x0:x1],
        gaussian
    )


def yolo_to_heatmap(
    image_dir,
    label_dir,
    output_dir,
    sigma=4
):
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

        images = list((image_dir / split).glob("*.jpg"))

        print(f"\nProcessing {split}: {len(images)} images")

        for img_path in images:
            label_path = label_dir / split / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]

            heatmap = np.zeros((h, w), dtype=np.float32)

            with open(label_path, "r") as f:
                for line in f:
                    cls, cx, cy, bw, bh = map(float, line.split())

                    # YOLO normalized → pixel coords
                    x = cx * w
                    y = cy * h

                    draw_gaussian(heatmap, (x, y), sigma=sigma)

            heatmap = np.clip(heatmap * 255, 0, 255).astype(np.uint8)

            # Save heatmap
            out_label = output_dir / split / "labels" / f"{img_path.stem}.png"
            cv2.imwrite(str(out_label), heatmap)

            # Copy image
            out_img = output_dir / split / "images" / img_path.name
            if not out_img.exists():
                cv2.imwrite(str(out_img), img)

    print("\n✅ YOLO → TrackNet heatmap conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert YOLO labels to TrackNet heatmaps"
    )
    parser.add_argument("--images", required=True, help="YOLO images directory")
    parser.add_argument("--labels", required=True, help="YOLO labels directory")
    parser.add_argument("--output", required=True, help="TrackNet output directory")
    parser.add_argument("--sigma", type=int, default=4, help="Gaussian sigma")

    args = parser.parse_args()

    yolo_to_heatmap(
        image_dir=args.images,
        label_dir=args.labels,
        output_dir=args.output,
        sigma=args.sigma
    )