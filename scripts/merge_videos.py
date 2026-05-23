"""
Merge all videos in a folder into a single output video.
Output is saved in the same folder and never overwrites existing files.

Usage:
    python scripts/merge_videos.py <folder_path>
    python scripts/merge_videos.py .            # current directory
"""

import os
import sys
import subprocess
import tempfile
from datetime import datetime

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}


def find_videos(folder: str) -> list[str]:
    files = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            files.append(os.path.join(folder, name))
    return files


def unique_output_path(folder: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(folder, f"merged_{timestamp}.mp4")
    if not os.path.exists(base):
        return base
    # Append counter if timestamp collides
    i = 1
    while True:
        candidate = os.path.join(folder, f"merged_{timestamp}_{i}.mp4")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def merge(folder: str):
    videos = find_videos(folder)

    if not videos:
        print("No video files found in:", folder)
        sys.exit(1)

    if len(videos) == 1:
        print("Only one video found. Nothing to merge.")
        print(" ", videos[0])
        sys.exit(0)

    print(f"Found {len(videos)} videos:")
    for v in videos:
        print(" ", os.path.basename(v))

    output = unique_output_path(folder)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = f.name
        for v in videos:
            # ffmpeg concat requires escaped single quotes in paths
            safe_path = v.replace("'", "'\\''")
            f.write(f"file '{safe_path}'\n")

    try:
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output,
        ]
        print(f"\nMerging → {os.path.basename(output)}\n")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print("\nffmpeg failed. Check the output above for details.")
            sys.exit(1)

        print("\nDone:", output)

    finally:
        os.unlink(concat_file)


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    folder = os.path.abspath(folder)

    if not os.path.isdir(folder):
        print("Not a valid directory:", folder)
        sys.exit(1)

    merge(folder)
