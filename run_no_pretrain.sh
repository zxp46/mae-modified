#!/bin/bash
IMAGENET_DIR="~/data/imagenet"

python main_finetune.py \
    --accum_iter 4 \
    --batch_size 8 \
    --model vit_large_patch16 \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --output_dir ./exp_no_pretrain \
    --log_dir ./exp_no_pretrain > out_finetune_no_pretrain.txt