#!/bin/bash
IMAGENET_DIR="~/data/imagenet"

python main_pretrain.py \
    --batch_size 128 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 805 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR} \
    --output_dir ./exp \
    --log_dir ./exp > out.txt \
    --resume ./exp/checkpoint-306.pth

