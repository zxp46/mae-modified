#!/bin/bash
IMAGENET_DIR="~/data/imagenet"
PRETRAIN_CHKPT="./exp3/checkpoint-799.pth"

python main_linprobe.py \
    --batch_size 512 \
    --model vit_large_patch16 --cls_token \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --output_dir ./exp_linp \
    --log_dir ./exp_linp > out_linp.txt