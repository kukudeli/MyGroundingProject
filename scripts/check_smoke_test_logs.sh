#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="outputs/smoke_tests"
SUMMARY="$OUT_DIR/smoke_summary.tsv"
mkdir -p "$OUT_DIR"

FORBIDDEN_PATTERN='use_difficulty_loss_weight|difficulty_loss_mode|difficulty_loss_active|difficulty_loss_bbox_contrib|difficulty_loss_giou_contrib|get_difficulty_loss_sample_weights|get_mid_iou_difficulty_loss_match_weights|use_box_refine_head|box_refine|loss_refined_boxes|refined_boxes|use_enclosing_aligned_gt_loss|proto_pce_difficulty_aware|pce_difficulty_aware'
ERROR_PATTERN='Traceback|RuntimeError|ImportError|ModuleNotFoundError|AttributeError|KeyError|AssertionError|FileNotFoundError|ChildFailedError|CUDA out of memory|unrecognized arguments|unexpected key|missing key'

write_header() {
  printf "config\tstatus\texit_code\tdataset\tmodel_train\tbasic_loss\texpected_keys\tforbidden_old_keys\treason\n" > "$SUMMARY"
}

contains() {
  local log_file="$1"
  local pattern="$2"
  rg -q "$pattern" "$log_file"
}

last_match() {
  local log_file="$1"
  local pattern="$2"
  rg -n "$pattern" "$log_file" | tail -n 1 | cut -c 1-240
}

check_one() {
  local name="$1"
  local log_file="$2"
  local expected_pattern="$3"
  local expected_label="$4"
  local exit_code="NA"
  local dataset="no"
  local model_train="no"
  local basic_loss="no"
  local expected_keys="no"
  local forbidden="no"
  local status="FAIL"
  local reason=""

  if [[ ! -f "$log_file" ]]; then
    printf "%s\tFAIL\tNA\tno\tno\tno\tno\tno\tmissing log file\n" "$name" >> "$SUMMARY"
    echo "$name: FAIL - missing log file"
    return
  fi

  exit_code="$(awk -F= '/^SMOKE_EXIT_CODE=/{value=$2} END{print value}' "$log_file")"
  [[ -n "$exit_code" ]] || exit_code="NA"

  if contains "$log_file" 'length of training dataset|length of testing dataset'; then
    dataset="yes"
  fi
  if contains "$log_file" 'Train epoch|Train: \['; then
    model_train="yes"
  fi
  if contains "$log_file" 'loss_bbox|loss_ce|loss_constrastive_align|query_points_generation_loss|[[:space:]]loss[[:space:]]'; then
    basic_loss="yes"
  fi
  if contains "$log_file" "$expected_pattern"; then
    expected_keys="yes:${expected_label}"
  else
    expected_keys="no:${expected_label}"
  fi
  if contains "$log_file" "$FORBIDDEN_PATTERN"; then
    forbidden="yes"
  fi

  if [[ "$forbidden" == "yes" ]]; then
    status="FAIL"
    reason="forbidden old-module keyword: $(last_match "$log_file" "$FORBIDDEN_PATTERN")"
  elif [[ "$basic_loss" == "yes" && "$expected_keys" == yes:* && "$exit_code" == "0" ]]; then
    status="PASS"
    reason="completed command and observed expected training keys"
  elif [[ "$basic_loss" == "yes" && "$expected_keys" == yes:* && "$exit_code" == "124" ]]; then
    status="PARTIAL"
    reason="timeout after entering training and observing expected keys"
  elif [[ "$basic_loss" == "yes" && "$expected_keys" == yes:* ]]; then
    status="PARTIAL"
    reason="entered training and observed expected keys; process exited ${exit_code}: $(last_match "$log_file" "$ERROR_PATTERN")"
  elif contains "$log_file" "$ERROR_PATTERN"; then
    status="FAIL"
    reason="$(last_match "$log_file" "$ERROR_PATTERN")"
  else
    status="FAIL"
    reason="did not observe required training loss or expected keys"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$status" "$exit_code" "$dataset" "$model_train" "$basic_loss" "$expected_keys" "$forbidden" "$reason" >> "$SUMMARY"
  echo "$name: $status - $reason"
}

write_header
check_one "baseline" "$OUT_DIR/baseline.log" 'loss_bbox|loss_ce|loss_constrastive_align|query_points_generation_loss' "basic losses"
check_one "baseline+pceper" "$OUT_DIR/pceper.log" 'loss_proto|loss_pce|loss_per|proto_active|pce_active|per_active' "PCE/PER proto losses"
check_one "geometry-only" "$OUT_DIR/geometry.log" 'enclosing_box_target_active' "enclosing box target stats"
check_one "geometry+prop-proto" "$OUT_DIR/geometry_prop_proto.log" 'loss_prop_proto|loss_prop_proto_raw|prop_proto_active|prop_proto_pos_count|prop_proto_neg_count' "prop-proto losses/stats"

echo "summary: $SUMMARY"
