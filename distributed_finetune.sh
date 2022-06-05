PRETRAIN_CHKPT=./exp/c

OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=4 distributed_finetune.py \
    --accum_iter 4 \
    --batch_size 64 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR}