#!/usr/bin/env bash
set -e

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_DISTRIBUTED_DEBUG=INFO \
CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port $((RANDOM % 30000 + 20000)) \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root data/3eed \
    --split_dir data/3eed/splits \
    --val_freq 1 --batch_size 16 --save_freq 1 --print_freq 100 \
    --max_epoch 23 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --dataset quad drone --test_dataset quad drone \
    --detect_intermediate --joint_det \
    --lr_decay_epochs 25 26 \
    --use_soft_token_loss --use_contrastive_align \
    --use_enclosing_aligned_gt_as_box_target \
    --log_dir logs \
    --self_attend \
    --checkpoint_path logs/Train_quad_drone_Val_quad_drone/0618_0857/ckpt_epoch_20.pth
