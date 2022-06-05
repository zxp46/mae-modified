#!/bin/bash
IMAGENET_DIR="~/autodl-tmp/imagenet"

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 distributed_train.py \
    --batch_size 128 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 --resume ./exp/checkpoint-550.pth\
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ./exp \
    --log_dir ./exp > out.txt
