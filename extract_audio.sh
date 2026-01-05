#!/bin/bash
# Batch-extract 16 kHz mono WAV audio from all video files in a folder.
# Usage: ./data_process/extract_audio.sh <video_dir> [out_dir]
# Example: ./data_process/extract_audio.sh /path/to/MELD.Raw/train_splits data/audio/train

set -euo pipefail

VID_DIR=${1:-}
OUT_DIR=${2:-audio}

if [[ -z "$VID_DIR" ]]; then
    echo "Usage: $0 <video_dir> [out_dir]" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

# Find videos (mp4/mkv/avi) under VID_DIR and extract audio.
find "$VID_DIR" -type f \( -iname "*.mp4" -o -iname "*.mkv" -o -iname "*.avi" \) -print0 |
while IFS= read -r -d '' v; do
    base="$(basename "${v%.*}")"
    echo "Extracting audio from: $v"
    ffmpeg -y -i "$v" -vn -ar 16000 -ac 1 -c:a pcm_s16le "$OUT_DIR/${base}.wav"
done

echo "Done. WAV files saved to: $OUT_DIR"
