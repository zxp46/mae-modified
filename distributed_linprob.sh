#!/bin/bash
IMAGENET_DIR="~/autodl-tmp/imagenet"
PRETRAIN_CHKPT="./exp/checkpoint-590.pth"

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 distributed_linprob.py \
    --batch_size 256 \
    --model vit_base_patch16 --cls_token \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --output_dir ~/autodl-tmp/exp_linp \
    --log_dir ~/autodl-tmp/exp_linp > out_linp.txt
