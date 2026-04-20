TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node 1 --master_port $((RANDOM % 30000 + 20000)) \
    train_dist_mod.py --num_decoder_layers 6 \
    --use_color \
    --weight_decay 0.0005 \
    --data_root data/3eed \
    --split_dir data/3eed/splits \
    --val_freq 1 --batch_size 48 --save_freq 5 --print_freq 1000 \
    --max_epoch 100 \
    --lr_backbone=1e-3 --lr=1e-4 \
    --test_dataset quad drone waymo \
    --detect_intermediate --joint_det \
    --lr_decay_epochs 25 26 \
    --use_soft_token_loss --use_contrastive_align \
    --log_dir logs \
    --self_attend \
    --eval \
    --checkpoint_path logs/Train_quad_drone_waymo_Val_quad_drone_waymo/ckpt_epoch_last.pth