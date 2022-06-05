#!/bin/bash
IMAGENET_DIR="~/data/imagenet"
PRETRAIN_CHKPT="./exp3/checkpoint-799.pth"

python main_finetune.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_large_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 105 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --output_dir ./exp_train_2 \
    --resume ./exp_train_3/checkpoint-99.pth\
    --log_dir ./exp_train_3 > out_train.txt