"""
Extract keyframes (I-frames) from all videos in a folder using FFmpeg.

Example (Windows paths):
  python data_process/extract_keyframes.py ^
      --video_dir "E:\\CODE\\dataset\\meld\\MELD.Raw\\train_splits" ^
      --out_dir "E:\\CODE\\dataset\\meld\\MELD.Raw\\keyframes\\train"

Each video gets its own subfolder under out_dir, named after the video (without extension),
and frames are saved as keyframe-001.jpg, keyframe-002.jpg, ...
"""

import argparse
import subprocess
from pathlib import Path


def extract_for_video(video_path: Path, dest_dir: Path, overwrite: bool = True):
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(dest_dir / "keyframe-%03d.jpg")
    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(video_path),
        "-vf", "select='eq(pict_type,I)'",
        "-vsync", "vfr",
        "-q:v", "2",
        out_pattern,
    ]
    print(f"[keyframe] {video_path.name} -> {dest_dir}")
    subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True, help="Folder containing input videos (mp4/mkv/avi)")
    ap.add_argument("--out_dir", required=True, help="Root folder to save extracted keyframes")
    ap.add_argument("--no_overwrite", action="store_true", help="Do not overwrite existing frames")
    args = ap.parse_args()

    video_dir = Path(args.video_dir)
    out_root = Path(args.out_dir)
    exts = [".mp4", ".mkv", ".avi"]
    vids = [p for p in video_dir.glob("*") if p.suffix.lower() in exts]

    if not vids:
        print(f"No videos found under {video_dir}")
        return

    for v in vids:
        dest = out_root / v.stem
        extract_for_video(v, dest, overwrite=not args.no_overwrite)


if __name__ == "__main__":
    main()
