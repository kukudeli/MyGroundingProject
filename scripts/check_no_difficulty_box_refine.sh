#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

keywords=(
  "use_difficulty_loss_weight"
  "difficulty_loss_mode"
  "get_difficulty_loss_sample_weights"
  "get_mid_iou_difficulty_loss_match_weights"
  "use_box_refine_head"
  "box_refine_use_at_eval"
  "loss_refined_boxes"
  "box_refine_head"
  "use_enclosing_aligned_gt_loss"
  "--proto_pce_difficulty_aware"
  "proto_pce_difficulty_aware"
  "pce_difficulty_aware"
)

search_paths=(
  "main_utils.py"
  "train_dist_mod.py"
  "models"
  "src"
  "scripts"
)

found=0
for keyword in "${keywords[@]}"; do
  if rg -n --fixed-strings \
    --glob '!scripts/check_no_difficulty_box_refine.sh' \
    --glob '!scripts/check_removed_module_outputs.sh' \
    --glob '!scripts/check_smoke_test_logs.sh' \
    --glob '!archive_experiments/**' \
    --glob '!logs/**' \
    --glob '!outputs/**' \
    --glob '!code_backup/**' \
    -e "$keyword" "${search_paths[@]}"; then
    found=1
  fi
done

if [[ "$found" -ne 0 ]]; then
  echo "检查失败：主代码路径仍残留 difficulty / box refine 关键词。"
  exit 1
fi

echo "检查通过：主代码路径未发现 difficulty / box refine 残留关键词。"
