#!/usr/bin/env bash
set -e

MODE=${1:-pce_per}
GPU=${GPU:-0}
PORT=${PORT:-$((RANDOM % 30000 + 20000))}

COMMON_ARGS=(
    train_dist_mod.py --num_decoder_layers 6
    --use_color
    --weight_decay 0.0005
    --data_root data/3eed
    --split_dir data/3eed/splits
    --val_freq 5 --batch_size 24 --save_freq 10 --print_freq 1000
    --max_epoch 100
    --lr_backbone=1e-3 --lr=1e-4
    --dataset drone quad --test_dataset drone quad
    --detect_intermediate --joint_det
    --lr_decay_epochs 25 26
    --use_soft_token_loss --use_contrastive_align
    --log_dir logs
    --self_attend
)

PROTO_ARGS=()
case "${MODE}" in
    baseline)
        ;;
    pce)
        PROTO_ARGS=(
            --use_platform_proto
            --proto_use_pce
            --proto_pce_weight 0.1
            --proto_per_weight 0.0
        )
        ;;
    per)
        PROTO_ARGS=(
            --use_platform_proto
            --proto_use_per
            --proto_pce_weight 0.0
            --proto_per_weight 0.01
            --proto_warmup_epoch 5
        )
        ;;
    pce_per)
        PROTO_ARGS=(
            --use_platform_proto
            --proto_use_pce
            --proto_use_per
            --proto_pce_weight 0.1
            --proto_per_weight 0.01
            --proto_warmup_epoch 5
        )
        ;;
    *)
        echo "Unknown MODE=${MODE}. Use baseline, pce, per, or pce_per."
        exit 1
        ;;
esac

TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=${GPU} \
python -m torch.distributed.launch --nproc_per_node 1 --master_port "${PORT}" \
    "${COMMON_ARGS[@]}" "${PROTO_ARGS[@]}"
