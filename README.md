# MyGroundingProject

本仓库是基于 3EED / BeaUTyDETR baseline 的研究扩展项目，当前最终方法主线聚焦 drone + quad 联合 3D grounding。仓库不是 3EED 官方仓库，仍沿用 3EED 数据、训练入口和官方评估协议。

当前最终方法由三部分组成：

1. 基础 3EED 模型；
2. 最小轴对齐外接框监督，即训练时将 rotated GT 转换为 axis-aligned enclosing GT 作为 box target；
3. 第二阶段候选框级原型分类排序损失 prop-proto，用于 proposal ranking finetuning。

PCE/PER/platform prototype 代码仍保留，便于复现实验和对照，但不再作为最终方法主线。difficulty/reweight loss、box refine head、refine-only enclosing loss、proto_pce_difficulty_aware，以及 platform probe 的工具/脚本输出已从 clean 实验路径中清理或归档。

## 1. 项目状态

保留并推荐使用的最终方法模块：

- `train_dist_mod.py`：训练主入口。
- `main_utils.py`：参数解析、criterion 构建、训练/日志流程。
- `models/bdetr.py`：基础 3EED / BeaUTyDETR 主干与 query 特征输出。
- `models/losses.py`：Hungarian loss、soft token loss、contrastive align loss、最小轴对齐 target 分支、prop-proto 接入。
- `models/proposal_proto_loss.py`：proposal-level positive prototype ranking loss。
- `src/grounding_evaluator.py`：官方评估路径，保持 rotated GT vs axis-aligned prediction 的 IoU 口径。

已清理或归档的旧实验模块：

- difficulty / reweight loss；
- box refine / refined_boxes / box_refine_head；
- refine-only enclosing loss：`--use_enclosing_aligned_gt_loss`；
- PCE 内部 difficulty gate：`--proto_pce_difficulty_aware`；
- platform probe 工具与旧脚本。

相关归档目录：

```text
archive_experiments/removed_module_artifacts/
```

重要说明：

- `--use_enclosing_aligned_gt_as_box_target` 是最终方法主线，必须保留。
- `--use_enclosing_aligned_gt_loss` 是旧 refine-only loss，已清理，不应再使用。
- `--use_prop_proto` 是第二阶段 prop-proto finetune 入口，必须保留。
- PCE/PER 仍可通过 `--use_platform_proto --proto_use_pce --proto_use_per` 启用，但不是当前最终方法主线。

## 2. 环境配置

本项目沿用原始 3EED 的环境依赖。请使用能正常运行 3EED baseline 的 Python、PyTorch、CUDA 和自定义 CUDA 算子环境。

| 组件 | 建议 |
|---|---|
| Python | 3.10 或 3.11 |
| PyTorch | 与 CUDA 匹配 |
| CUDA | 与编译环境一致 |
| transformers | 支持 RoBERTa |
| numpy / scipy / tqdm / tensorboard | 常规版本即可 |

编译自定义 CUDA 算子：

```bash
cd ops/teed_pointnet/pointnet2_batch
python setup.py develop

cd ../roiaware_pool3d
python setup.py develop
```

RoBERTa 权重默认路径：

```text
data/roberta_base/
```

## 3. 数据准备

数据组织沿用 3EED：

```text
data/3eed/
├── drone/
├── quad/
├── waymo/
├── splits/
│   ├── drone_train.txt
│   ├── drone_val.txt
│   ├── quad_train.txt
│   ├── quad_val.txt
│   ├── waymo_train.txt
│   └── waymo_val.txt
└── roberta_base/
```

当前最终实验主要使用：

```bash
--dataset quad drone
--test_dataset quad drone
```

## 4. 推荐训练路径

### 4.1 Baseline 3EED

基础 drone + quad 训练可以直接使用 `train_dist_mod.py`：

```bash
TORCH_DISTRIBUTED_DEBUG=INFO CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node 1 \
  --master_port $((RANDOM % 30000 + 20000)) \
  train_dist_mod.py --num_decoder_layers 6 \
  --use_color \
  --weight_decay 0.0005 \
  --data_root data/3eed \
  --split_dir data/3eed/splits \
  --val_freq 5 --batch_size 16 --save_freq 5 --print_freq 100 \
  --max_epoch 20 \
  --lr_backbone=1e-3 --lr=1e-4 \
  --dataset quad drone --test_dataset quad drone \
  --detect_intermediate --joint_det \
  --lr_decay_epochs 25 26 \
  --use_soft_token_loss --use_contrastive_align \
  --log_dir logs \
  --self_attend
```

### 4.2 Geometry-only：最小轴对齐监督

推荐脚本：

```bash
bash scripts/train_enclosing_only_clean_20e.sh
```

核心差异是启用：

```bash
--use_enclosing_aligned_gt_as_box_target
```

该分支只改变训练时 box target，将 rotated GT 转为最小 axis-aligned enclosing GT；评估仍保持官方方式，不改 IoU 口径。

### 4.3 Geometry + Prop-Proto Finetune

推荐从 geometry checkpoint 启动第二阶段 finetune：

```bash
bash scripts/finetune_prop_proto_from_enclosing20_w001_23e.sh
```

核心参数：

```bash
--use_enclosing_aligned_gt_as_box_target
--use_prop_proto
--prop_proto_weight 0.001
--prop_proto_tau 0.07
--prop_pos_iou_thr 0.5
--prop_neg_iou_thr 0.25
--prop_hn_topk 5
```

继续训练脚本：

```bash
bash scripts/resume_prop_proto_23to100.sh
```

## 5. Prop-Proto 方法说明

`models/proposal_proto_loss.py` 定义 `ProposalPrototypeRankingLoss`。该损失不是直接复用 PCE/PER，而是借鉴“以 prototype 作为判别参照”的思想，并重构为候选框级排序约束。

简要流程：

- 对每个 proposal 计算与 GT box 的 IoU；
- IoU 高于阈值的 proposal 作为 positive；
- 若没有 positive，则使用 IoU 最大的 proposal 作为 fallback positive；
- 使用 positive proposal features 按 IoU 加权聚合出动态 positive prototype；
- 从低 IoU proposals 中选择语言分数高或与正原型相似的 hard negatives；
- 使用 cosine/dot-product similarity 和 temperature `prop_proto_tau`；
- 通过 `softplus((sim_neg - sim_pos) / tau)` 约束正候选框比 hard negative 更接近正原型。

没有显式 negative prototype，也没有跨 batch/global prototype bank。

## 6. PCE/PER 与旧实验模块

PCE/PER/platform prototype 仍保留在代码中：

- `models/prototype_rebalance.py`
- `models/losses.py` 中 `use_platform_proto` 分支

启用方式：

```bash
--use_platform_proto
--proto_use_pce
--proto_use_per
```

但它们不是当前最终方法主线。旧的 PCE/PER 脚本已归档：

```text
archive_experiments/removed_module_artifacts/pceper_box_difficulty_scripts/
```

不要把第二阶段 prop-proto 写成“直接使用 PCE/PER loss”。更准确的表述是：

```text
借鉴 PCE/PER 的原型分类思想，并将其重构为候选框级原型分类排序损失。
```

## 7. 评估

评估脚本仍保留：

```bash
bash scripts/val_3eed.sh
bash scripts/val_drone.sh
bash scripts/val_quad.sh
bash scripts/val_waymo.sh
```

评估前请确认脚本中的 `--checkpoint_path` 指向目标 checkpoint。

评估逻辑保持官方口径：

- prediction 为 axis-aligned box；
- GT 仍使用 rotated GT；
- IoU 使用 rotated GT vs axis-aligned prediction。

## 8. 诊断与检查

清理后建议运行：

```bash
python3 -m py_compile main_utils.py train_dist_mod.py models/losses.py models/bdetr.py src/grounding_evaluator.py models/ap_helper.py
bash scripts/check_no_difficulty_box_refine.sh
bash scripts/check_removed_module_outputs.sh
```

smoke 日志检查脚本：

```bash
bash scripts/check_smoke_test_logs.sh
```

几何诊断工具保留在 `tools/` 下，包括：

- `analyze_official_aligned_oracle.py`
- `analyze_parking2_rotation_mismatch.py`
- `analyze_platform_rotation_mismatch.py`
- `debug_official_iou_consistency.py`
- `evaluate_baseline_ignore_yaw.py`
- `compare_enclosing20_vs_prop_proto_finetune23.py`
- `compare_three_way_enclosing_prop_proto.py`
- `extract_best_eval_epochs.py`

## 9. 当前注意事项

- 不要提交 `data/`、`logs/`、checkpoints、prediction JSON 或大型输出文件。
- `outputs/code_cleanup_review/` 中保留小型清理报告。
- `platform_probe` 主代码入口暂时保留，后续如需完全清理应单独处理。
- `proto_status_mode="box_difficulty"` 仍作为 PCE/PER 旧实验状态输入存在，不属于已删除的独立 difficulty/reweight loss。

## 10. License

本仓库基于原始 3EED / BeaUTyDETR 代码进行研究扩展。请同时遵守原始项目、数据集和依赖库的许可协议。
