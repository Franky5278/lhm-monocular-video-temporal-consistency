#!/bin/bash

MODEL_NAME=LHM-1B
IMAGE_INPUT="./train_data/example_imgs/"

MODEL_NAME=${1:-$MODEL_NAME}
IMAGE_INPUT=${2:-$IMAGE_INPUT}
MOTION_SEQS_DIR=None

echo "MODEL_NAME: $MODEL_NAME"
echo "IMAGE_INPUT: $IMAGE_INPUT"
echo "INFERENCE MESH"

MOTION_IMG_DIR=None
VIS_MOTION=true
MOTION_IMG_NEED_MASK=true
EXPORT_MESH=True

python -m LHM.launch infer.human_lrm model_name=$MODEL_NAME \
        image_input=$IMAGE_INPUT \
        export_mesh=$EXPORT_MESH \
        motion_seqs_dir=$MOTION_SEQS_DIR motion_img_dir=$MOTION_IMG_DIR  \
        vis_motion=$VIS_MOTION motion_img_need_mask=$MOTION_IMG_NEED_MASK \
        render_fps=$RENDER_FPS motion_video_read_fps=$MOTION_VIDEO_READ_FPS