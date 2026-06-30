#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

keywords=(
  "difficulty"
  "reweight"
  "box_refine"
  "refined_boxes"
  "refine_head"
  "use_enclosing_aligned_gt_loss"
  "proto_pce_difficulty_aware"
  "pce_difficulty_aware"
  "platform_probe"
  "enable_platform_probe"
)

protected_keywords=(
  "rotation_mismatch"
  "ignore_yaw"
  "official_aligned_oracle"
  "parking2"
  "diagnostic"
  "enclosing"
  "prop_proto"
  "three_way"
  "best_eval"
  "iou_consistency"
  "baseline_ignore_yaw"
)

protected_paths=(
  "outputs/prop_proto_loss_diagnostics/"
  "outputs/code_cleanup_review/"
  "outputs/smoke_tests/"
  "tools/analyze_official_aligned_oracle.py"
  "tools/analyze_parking2_c_failure_detail.py"
  "tools/analyze_parking2_c_iou_decomposition.py"
  "tools/analyze_parking2_rotation_mismatch.py"
  "tools/analyze_platform_rotation_mismatch.py"
  "tools/analyze_scene_parking2_failure.py"
  "tools/build_diagnostic_subset.py"
  "tools/debug_official_iou_consistency.py"
  "tools/evaluate_baseline_ignore_yaw.py"
  "tools/compare_enclosing20_vs_prop_proto_finetune23.py"
  "tools/compare_three_way_enclosing_prop_proto.py"
  "tools/extract_best_eval_epochs.py"
  "scripts/check_no_difficulty_box_refine.sh"
  "scripts/check_smoke_test_logs.sh"
  "scripts/check_removed_module_outputs.sh"
)

search_roots=(
  "tools"
  "statistics"
  "scripts"
  "outputs"
  "logs"
)

if [[ -d logs ]]; then
  while IFS= read -r -d '' config_path; do
    run_dir="$(dirname "$config_path")"
    log_path="$run_dir/log.txt"
    protected_args=()
    [[ -f "$config_path" ]] && protected_args+=("$config_path")
    [[ -f "$log_path" ]] && protected_args+=("$log_path")

    if [[ "${#protected_args[@]}" -gt 0 ]]; then
      for keyword in "${protected_keywords[@]}"; do
        if rg -q --fixed-strings -i -e "$keyword" "${protected_args[@]}"; then
          protected_paths+=("$run_dir/")
          break
        fi
      done
    fi
  done < <(find logs -mindepth 2 -maxdepth 4 -name config.json -print0 2>/dev/null)
fi

is_protected_path() {
  local path="$1"
  [[ "$path" == archive_experiments/* ]] && return 0
  [[ "$path" == code_backup/* ]] && return 0
  [[ "$path" == *"/code_backup/"* ]] && return 0

  local protected
  for protected in "${protected_paths[@]}"; do
    [[ "$path" == "$protected" || "$path" == "$protected"* ]] && return 0
  done
  return 1
}

has_protected_content() {
  local path="$1"
  local keyword
  for keyword in "${protected_keywords[@]}"; do
    if rg -q --fixed-strings -i -e "$keyword" "$path"; then
      return 0
    fi
  done
  return 1
}

found=0

for root in "${search_roots[@]}"; do
  [[ -e "$root" ]] || continue
  while IFS= read -r -d '' path; do
    if is_protected_path "$path"; then
      continue
    fi

    matched=()
    for keyword in "${keywords[@]}"; do
      if rg -q --fixed-strings -i -e "$keyword" "$path"; then
        matched+=("$keyword")
      fi
    done

    if [[ "${#matched[@]}" -eq 0 ]]; then
      continue
    fi

    if has_protected_content "$path"; then
      continue
    fi

    printf "%s\t%s\n" "$path" "$(IFS=,; echo "${matched[*]}")"
    found=1
  done < <(find "$root" -type f -print0 2>/dev/null)
done

if [[ "$found" -ne 0 ]]; then
  echo "检查失败：active tools/scripts/outputs/logs 中仍有已移除模块关键词。"
  exit 1
fi

echo "检查通过：active tools/scripts/outputs/logs 未发现未保护的已移除模块关键词。"
