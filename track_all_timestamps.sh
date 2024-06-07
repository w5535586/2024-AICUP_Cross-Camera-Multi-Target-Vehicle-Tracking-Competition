#!/bin/bash

DEFAULT_WEIGHTS="pretrained/yolov7-e6e.pt"
DEFAULT_SOURCE_DIR="/mnt/datasets/AI_CUP_MCMOT_dataset/train/images"
DEFAULT_DEVICE="1"
DEFAULT_FAST_REID_CONFIG="fast_reid/configs/AICUP/bagtricks_R50-ibn.yml"
DEFAULT_FAST_REID_WEIGHTS="logs/AICUP/bagtricks_R50-ibn/model_0058.pth"

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --weights)
            WEIGHTS="$2"
            shift
            shift
            ;;
        --source-dir)
            SOURCE_DIR="$2"
            shift
            shift
            ;;
        --device)
            DEVICE="$2"
            shift
            shift
            ;;
        --fast-reid-config)
            FAST_REID_CONFIG="$2"
            shift
            shift
            ;;
        --fast-reid-weights)
            FAST_REID_WEIGHTS="$2"
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Using default values if not provided
WEIGHTS="${WEIGHTS:-$DEFAULT_WEIGHTS}"
SOURCE_DIR="${SOURCE_DIR:-$DEFAULT_SOURCE_DIR}"
DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
FAST_REID_CONFIG="${FAST_REID_CONFIG:-$DEFAULT_FAST_REID_CONFIG}"
FAST_REID_WEIGHTS="${FAST_REID_WEIGHTS:-$DEFAULT_FAST_REID_WEIGHTS}"

for folder in "$SOURCE_DIR"/*; do
    timestamp=${folder##*/}
    python3 mc_demo_yolov9.py --weights "$WEIGHTS" --source "$folder" --device "$DEVICE" --name "$timestamp" --fuse-score --agnostic-nms --with-reid
done

