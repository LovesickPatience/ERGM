"""
Cut MELD video clips by start/end times from the CSV.

Usage (example on Windows paths):
  python data_process/cut_meld_clips.py ^
      --csv "E:\\CODE\\dataset\\meld\\MELD.Raw\\train_sent_emo.csv" ^
      --video_dir "E:\\CODE\\dataset\\meld\\MELD.Raw\\train_splits" ^
      --out_dir "E:\\CODE\\dataset\\meld\\clips\\train" ^
      --name_pattern "dia{dialogue_id}.mp4"

If your CSV already has a column with the video filename (e.g., "VideoFile"),
set --video_col VideoFile and leave --name_pattern unused.
"""

import argparse
import csv
import subprocess
from pathlib import Path


def cut_clips(csv_path: Path, video_dir: Path, out_dir: Path,
              name_pattern: str = "dia{dialogue_id}.mp4",
              video_col: str = None,
              overwrite: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open(newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                did = int(row["Dialogue_ID"])
                uid = int(row["Utterance_ID"])
                start = float(row["StartTime"])
                end = float(row["EndTime"])
            except Exception as e:
                raise RuntimeError(f"Failed to parse row: {row}") from e

            # Resolve source video name
            if video_col and row.get(video_col):
                src_name = row[video_col]
            else:
                src_name = name_pattern.format(dialogue_id=did, utterance_id=uid)

            src_path = video_dir / src_name
            if not src_path.exists():
                print(f"[skip] source not found: {src_path}")
                continue

            out_name = f"dia{did}_utt{uid}.mp4"
            out_path = out_dir / out_name

            if out_path.exists() and not overwrite:
                print(f"[skip] exists: {out_path}")
                continue

            cmd = [
                "ffmpeg",
                "-y" if overwrite else "-n",
                "-i", str(src_path),
                "-ss", str(start),
                "-to", str(end),
                "-c", "copy",
                str(out_path),
            ]
            print(f"Cutting {src_path.name} -> {out_path.name} [{start}, {end}]")
            subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to MELD *_sent_emo.csv")
    ap.add_argument("--video_dir", required=True, help="Folder containing source mp4s")
    ap.add_argument("--out_dir", required=True, help="Where to save clipped mp4s")
    ap.add_argument("--name_pattern", default="dia{dialogue_id}.mp4",
                    help="Pattern to resolve source video if no video filename column is provided")
    ap.add_argument("--video_col", default=None,
                    help="CSV column that directly gives the video filename (optional)")
    ap.add_argument("--no_overwrite", action="store_true", help="Do not overwrite existing outputs")
    args = ap.parse_args()

    cut_clips(
        csv_path=Path(args.csv),
        video_dir=Path(args.video_dir),
        out_dir=Path(args.out_dir),
        name_pattern=args.name_pattern,
        video_col=args.video_col,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
