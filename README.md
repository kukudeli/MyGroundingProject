# 面向多平台 3D Grounding 的平台条件原型重平衡方法

本仓库是基于 3EED baseline 的研究扩展项目，面向多平台户外 3D grounding 场景，探索 drone、quadruped、vehicle 等不同平台在联合训练中的学习不平衡问题。项目在原始 3EED / BeaUTyDETR 主干基础上，加入一个训练阶段可插拔的平台条件原型重平衡模块，用于动态识别强势平台与弱势平台，并通过 PCE 和 PER 调节平台间学习状态。

需要说明：

- 本仓库不是 3EED 官方仓库。
- 本项目基于 3EED 数据集、baseline 代码和评估协议进行研究扩展。
- 原始 3EED 的任务是根据 LiDAR 点云、RGB 图像和自然语言表达预测目标 3D 边界框。
- 本项目目前重点关注 drone 与 quad 的联合训练，并保留 waymo 平台兼容性。

## 1. 项目概述

任务输入：

- LiDAR 点云
- RGB 图像
- 自然语言表达
- 平台标签，例如 waymo / drone / quad

任务输出：

- 语言所指目标的 3D bounding box

研究问题：

在多平台联合训练中，不同平台由于视角高度、点云稀疏程度、目标尺度、遮挡情况和场景覆盖范围不同，可能出现学习速度不一致。学习更快的平台可能主导共享表示空间，学习较慢的平台可能难以形成稳定类别特征。

本项目方法：

在 3EED baseline 的基础上增加 Platform-conditioned Prototype Rebalancing 模块。该模块：

- 只在训练阶段启用；
- 不改变原始 3D box 推理路径；
- 维护平台条件类别原型；
- 使用 EMA 动态估计平台学习状态；
- 使用 PCE 增强弱势平台的类别聚类；
- 使用 PER 抑制强势平台过早过度自信；
- 支持 baseline / PCE / PER / PCE+PER 消融。

## 2. 当前代码特性

- 保留 3EED / BeaUTyDETR 主干结构。
- 在 `models/bdetr.py` 中额外导出 `proto_features` 作为原型模块输入。
- 在 `models/prototype_rebalance.py` 中实现平台条件原型平衡模块。
- 在 `models/losses.py` 中以可选方式追加 `loss_proto`。
- 在 `main_utils.py` 中加入原型模块参数、日志和训练开关。
- 在 `src/joint_det_dataset.py` 中使用平台标签支持平台级原型重平衡。
- 支持通过命令行开关启用或关闭原型模块。

## 3. 环境配置

本项目沿用 3EED 的环境依赖。请优先使用与原始 3EED baseline 一致的 Python、PyTorch、CUDA 和自定义 CUDA 算子环境。

| 组件 | 推荐版本 |
|---|---|
| Python | 3.10 或 3.11 |
| PyTorch | 与 CUDA 匹配的版本 |
| CUDA | 11.1 或 12.4 |
| torchvision | 与 PyTorch 匹配 |
| transformers | 支持 RoBERTa |
| numpy / scipy / tqdm / tensorboard | 常规版本即可 |

如果服务器已经能运行原始 3EED baseline，则通常不需要额外配置大量依赖，只需要确认新增代码所需的 PyTorch、TensorBoard 等基础包可用。

### 3.1 编译自定义 CUDA 算子

```bash
cd ops/teed_pointnet/pointnet2_batch
python setup.py develop

cd ../roiaware_pool3d
python setup.py develop
```

如果编译失败，优先检查 CUDA 版本、PyTorch 版本和 `CUDA_HOME` 是否正确。

### 3.2 RoBERTa 权重

本项目使用 RoBERTa 作为文本编码器。请下载 RoBERTa-base 权重，并放置到：

```text
data/roberta_base/
```

代码中默认从以下路径加载：

```text
./data/roberta_base/
```

## 4. 数据准备

本项目使用 3EED 数据集，数据组织方式沿用原始 3EED。

```text
data/3eed/
├── drone/
│   ├── scene-xxxx/
│   │   ├── frame/
│   │   │   ├── image.jpg
│   │   │   ├── lidar.bin
│   │   │   └── meta_info.json
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

当前原型平衡实验主要使用：

```bash
--dataset drone quad
--test_dataset drone quad
```

也可以使用原始 3EED 脚本训练单平台或全平台模型。

## 5. 代码使用方式

### 5.1 原始 3EED-style 训练

```bash
# 全平台训练
bash scripts/train_3eed.sh

# 单平台训练
bash scripts/train_waymo.sh
bash scripts/train_drone.sh
bash scripts/train_quad.sh
```

这些脚本主要用于复现或对比原始 baseline 行为。

### 5.2 drone + quad 原型平衡实验

`scripts/train_proto_drone_quad.sh` 是当前项目的主要实验入口。它支持四种模式：

```bash
# 1. baseline：不启用原型平衡模块
bash scripts/train_proto_drone_quad.sh baseline

# 2. pce：只启用平台条件 Prototype Cross-Entropy
bash scripts/train_proto_drone_quad.sh pce

# 3. per：只启用 Prototype Entropy Regularization
bash scripts/train_proto_drone_quad.sh per

# 4. pce_per：启用完整原型平衡模块
bash scripts/train_proto_drone_quad.sh pce_per
```

- `baseline`：原始 3EED-style drone + quad 联合训练，不启用原型平衡。
- `pce`：启用平台条件原型交叉熵，主要验证原型聚类约束是否有效。
- `per`：只启用强势平台熵正则，主要用于消融。
- `pce_per`：启用完整平台原型重平衡模块。

### 5.3 关键参数说明

```text
--use_platform_proto          启用平台原型平衡模块
--proto_use_pce               启用 PCE
--proto_use_per               启用 PER
--proto_pce_weight            PCE 损失权重
--proto_per_weight            PER 损失权重
--proto_score_momentum        平台状态 EMA 动量
--proto_gap_threshold         强弱平台差距阈值
--proto_warmup_epoch          动态重平衡 warmup epoch
--proto_min_platform_seen     平台参与强弱判断所需的最小累计样本数
--proto_weak_pce_boost        弱势平台 PCE 权重增强系数
--proto_max_pce_boost         弱势平台 PCE 最大增强上限
```

### 5.4 推荐 smoke test

正式训练前建议先跑小规模测试，确认数据、平台标签、loss 和日志都正常。

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port 29501 \
train_dist_mod.py \
--num_decoder_layers 6 \
--use_color \
--data_root data/3eed \
--split_dir data/3eed/splits \
--dataset drone quad \
--test_dataset drone quad \
--batch_size 2 \
--max_epoch 1 \
--print_freq 1 \
--save_freq 1 \
--val_freq 1 \
--detect_intermediate \
--joint_det \
--use_soft_token_loss \
--use_contrastive_align \
--self_attend \
--debug \
--use_platform_proto \
--proto_use_pce \
--proto_use_per \
--flag smoke_pce_per
```

如果 smoke test 通过，再使用脚本进行正式训练。

## 6. 评估

```bash
# 全平台评估
bash scripts/val_3eed.sh

# 单平台评估
bash scripts/val_waymo.sh
bash scripts/val_drone.sh
bash scripts/val_quad.sh
```

运行评估前需要在脚本中确认 `--checkpoint_path` 指向正确 checkpoint。

对于原型平衡实验，建议不仅观察整体 Acc@25 / Acc@50，也要关注 drone 和 quad 的平台级性能差距。

## 7. 日志与诊断

训练时重点关注以下日志：

```text
loss_proto
loss_pce
loss_per
platform_gap
status_ready
pce_rebalance_active
weak_pce_weight
weak_platform
strong_platform
platform_score_ema_0 / 1 / 2
platform_seen_count_0 / 1 / 2
platform_batch_score_0 / 1 / 2
```

平台编号：

```text
0 = waymo
1 = drone
2 = quad
```

这些指标用于判断原型模块是否真的检测到平台学习差异，以及 PCE/PER 是否按预期激活。

## 8. 当前状态

当前项目仍处于研究开发阶段。已完成：

- 可插拔平台原型平衡模块；
- 平台条件原型库；
- 全局原型与 fallback 原型；
- EMA-based 平台状态估计；
- 弱势平台加权 PCE；
- 强势平台 PER；
- baseline / PCE / PER / PCE+PER 运行入口。

完整实验结果待补充。
