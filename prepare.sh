#!/usr/bin/env bash
#
# Get the first and second command line arguments
DATA_ROOT="$1"
ID="$2"

BUILDTYPE=RelWithDebInfo

# Check if the arguments are empty
if [ -z "$DATA_ROOT" ] || [ -z "$ID" ]; then
    echo "Usage: $0 <DATA_ROOT> <ID>"
    exit 1
fi

# check if build exists
if [ ! -d "build/$BUILDTYPE" ]; then
    echo "Build directory not found. Please build the project first."
    exit 1
fi

source .venv/bin/activate
# run video_to_png
./build/$BUILDTYPE/examples/video_to_png -f $DATA_ROOT/$ID -i $ID -o $DATA_ROOT/$ID/inputs/images/
if [ $? -ne 0 ]; then
    echo "Error: video_to_png failed"
    exit 1
fi

# run aligner
if [ -z "$HF_ENDPOINT" ]; then
    export HF_ENDPOINT=https://hf-mirror.com
fi

python scripts/aligner.py --root "$DATA_ROOT" --id "$ID" --out "$DATA_ROOT/$ID/aligned_points/"
if [ $? -ne 0 ]; then
    echo "Error: Aligner script failed"
    exit 1
fi
