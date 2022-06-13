#!/bin/bash
IMAGENET_DIR="~/autodl-tmp/imagenet"
PRETRAIN_CHKPT="./exp/checkpoint-590.pth"

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 distributed_finetune.py \
    --accum_iter 4 \
    --batch_size 64 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT}\
    --epochs 110 --resume ./exp_train/checkpoint-99.pth \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --output_dir ./exp_train \
    --log_dir ./exp_train > out_train.txt