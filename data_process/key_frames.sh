#!/bin/bash


# Please install FFmpeg in advance
# git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg


INPUT_VIDEO="$1"
BASENAME=$(basename -- "$INPUT_VIDEO")
FILENAME="${BASENAME%.*}"
OUTPUT_DIR="keyframes_${FILENAME}"


# check if files exist
if [ ! -f "$INPUT_VIDEO" ]; then
    echo " '$INPUT_VIDEO' dose not exit!"
    exit 1
fi


# check output dir
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi


echo "Processing '$INPUT_VIDEO' ..."


ffmpeg -i "$INPUT_VIDEO" -vf "select='eq(pict_type,I)'" -vsync vfr -q:v 2 -f image2 "$OUTPUT_DIR/keyframe_%03d.jpg"


echo "Finished! Saved in '$OUTPUT_DIR' "