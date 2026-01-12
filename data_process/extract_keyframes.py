"""
Extract a fixed number of uniformly sampled frames from all videos in a folder using FFmpeg.

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


def get_video_duration_seconds(video_path: Path) -> float | None:
    """Return video duration in seconds using ffprobe, or None if unavailable."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if p.returncode != 0:
            return None
        s = (p.stdout or "").strip()
        if not s:
            return None
        dur = float(s)
        if dur <= 0:
            return None
        return dur
    except Exception:
        return None


def extract_for_video(video_path: Path, dest_dir: Path, overwrite: bool = True, num_frames: int = 32):
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(dest_dir / "keyframe-%03d.jpg")
    duration = get_video_duration_seconds(video_path)
    # Aim for exactly num_frames images by choosing an FPS based on duration and capping output.
    # If duration is unavailable, fall back to extracting the first num_frames frames.
    vf = None
    if duration is not None:
        fps = num_frames / duration
        # Keep FPS in a reasonable range to avoid weird edge cases.
        if fps <= 0:
            vf = None
        else:
            vf = f"fps={fps}"  # uniform sampling over time

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i", str(video_path),
    ]
    if vf is not None:
        cmd += ["-vf", vf]
    cmd += [
        "-frames:v", str(num_frames),
        "-vsync", "0",
        "-q:v", "2",
        out_pattern,
    ]
    print(f"[frames] {video_path.name} -> {dest_dir} (num_frames={num_frames})")
    subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir", required=True, help="Folder containing input videos (mp4/mkv/avi)")
    ap.add_argument("--out_dir", required=True, help="Root folder to save extracted keyframes")
    ap.add_argument("--no_overwrite", action="store_true", help="Do not overwrite existing frames")
    ap.add_argument("--num_frames", type=int, default=32, help="Number of frames to sample per video")
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
        extract_for_video(v, dest, overwrite=not args.no_overwrite, num_frames=args.num_frames)


if __name__ == "__main__":
    main()
