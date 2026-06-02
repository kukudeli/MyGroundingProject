#!/usr/bin/env python3
"""Official-IoU oracle analysis for aligned 6D prediction boxes.

This script asks whether center/size refinement alone can push C samples over
official IoU 0.5 when predictions remain axis-aligned 6D boxes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_ALL_RECORDS_CSV = "outputs/diagnostic_baseline20_dq/all_records.csv"
DEFAULT_TARGETS = "drone:Outdoor_Day_penno_parking_2,quad:Outdoor_Day_penno_short_loop"
DEFAULT_OUT_DIR = "outputs/official_aligned_oracle_analysis"
C_LABEL = "C_coarse_success_precise_fail"
EPS = 1e-9


try:
    from utils.eval_det import iou3d_rotated_vs_aligned

    OFFICIAL_IMPORT_STATUS = "ok"
except Exception as exc:  # pragma: no cover - reported in findings on failure.
    iou3d_rotated_vs_aligned = None
    OFFICIAL_IMPORT_STATUS = repr(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze official-IoU center/size oracle limits for aligned 6D prediction boxes."
    )
    parser.add_argument("--all_records_csv", default=DEFAULT_ALL_RECORDS_CSV)
    parser.add_argument("--targets", default=DEFAULT_TARGETS, help="Comma-separated platform:scene targets.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def parse_targets(raw: str) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid target {item!r}; expected platform:scene")
        platform, scene = item.split(":", 1)
        platform = platform.strip()
        scene = scene.strip()
        if not platform or not scene:
            raise ValueError(f"Invalid target {item!r}; expected non-empty platform and scene")
        targets.append((platform, scene))
    if not targets:
        raise ValueError("--targets produced no valid platform:scene entries")
    return targets


def repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt(value: Any, digits: int = 4) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def numeric_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    return arr


def read_prediction(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    return data if isinstance(data, dict) else None


def first_box(value: Any, min_dims: int = 6) -> np.ndarray | None:
    arr = numeric_array(value)
    if arr is None:
        return None
    if arr.ndim == 1 and arr.shape[0] >= min_dims:
        return arr.astype(np.float64)
    if arr.ndim >= 2 and arr.shape[-1] >= min_dims:
        return arr.reshape(-1, arr.shape[-1])[0].astype(np.float64)
    return None


def pick_pred_box(prediction: dict[str, Any]) -> tuple[np.ndarray | None, str]:
    top1 = first_box(prediction.get("top1_pred_box"), min_dims=6)
    if top1 is not None:
        return top1[:6], "top1_pred_box"
    top10 = first_box(prediction.get("top10_pred_boxes"), min_dims=6)
    if top10 is not None:
        return top10[:6], "top10_pred_boxes[0]"
    return None, "missing"


def official_iou(gt_box: np.ndarray | None, aligned_pred_box: np.ndarray | None) -> float:
    if iou3d_rotated_vs_aligned is None:
        raise RuntimeError(f"Cannot compute official IoU; import_status={OFFICIAL_IMPORT_STATUS}")
    if gt_box is None or aligned_pred_box is None:
        return math.nan
    if gt_box.shape[0] < 7 or aligned_pred_box.shape[0] < 6:
        return math.nan
    try:
        gt_tensor = torch.as_tensor(gt_box.reshape(1, -1), dtype=torch.float32)
        pred_tensor = torch.as_tensor(aligned_pred_box[:6].reshape(1, 6), dtype=torch.float32)
        ious, _ = iou3d_rotated_vs_aligned(gt_tensor, pred_tensor)
        return safe_float(ious[0, 0].detach().cpu().item())
    except Exception:
        return math.nan


def center_size_to_xyzxyz(box: np.ndarray | None) -> np.ndarray | None:
    if box is None or box.shape[0] < 6:
        return None
    size = box[3:6]
    if np.any(size <= 0):
        return None
    half = size / 2.0
    return np.concatenate([box[:3] - half, box[:3] + half])


def axis_aligned_iou_center_size(gt_box: np.ndarray | None, pred_box: np.ndarray | None) -> float:
    gt_xyz = center_size_to_xyzxyz(gt_box[:6] if gt_box is not None and gt_box.shape[0] >= 6 else None)
    pred_xyz = center_size_to_xyzxyz(pred_box[:6] if pred_box is not None and pred_box.shape[0] >= 6 else None)
    if gt_xyz is None or pred_xyz is None:
        return math.nan
    inter_min = np.maximum(gt_xyz[:3], pred_xyz[:3])
    inter_max = np.minimum(gt_xyz[3:6], pred_xyz[3:6])
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_size))
    gt_vol = float(np.prod(np.maximum(gt_xyz[3:6] - gt_xyz[:3], 0.0)))
    pred_vol = float(np.prod(np.maximum(pred_xyz[3:6] - pred_xyz[:3], 0.0)))
    union = gt_vol + pred_vol - inter_vol
    if union <= EPS:
        return math.nan
    return inter_vol / union


def normalize_yaw_mod_pi(yaw: float) -> float:
    return yaw % math.pi if math.isfinite(yaw) else math.nan


def yaw_to_axis_angle_deg(yaw: float) -> float:
    yaw_mod = normalize_yaw_mod_pi(yaw)
    if not math.isfinite(yaw_mod):
        return math.nan
    rad = min(abs(yaw_mod), abs(yaw_mod - math.pi / 2.0), abs(yaw_mod - math.pi))
    return math.degrees(rad)


def box_volume(box: np.ndarray | None) -> float:
    if box is None or box.shape[0] < 6 or np.any(box[3:6] <= 0):
        return math.nan
    return float(np.prod(box[3:6]))


def rotated_gt_to_enclosing_aligned_box(gt_box: np.ndarray | None) -> np.ndarray | None:
    """Convert a 7D yaw-rotated GT box to its minimal global aligned AABB."""
    if gt_box is None or gt_box.shape[0] < 7:
        return None
    cx, cy, cz, sx, sy, sz, yaw = gt_box[:7]
    if not np.isfinite(gt_box[:7]).all() or sx <= 0 or sy <= 0 or sz <= 0:
        return None

    x = np.array([sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2])
    y = np.array([sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2])
    z = np.array([sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2])

    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    x_rot = cos_yaw * x - sin_yaw * y + cx
    y_rot = sin_yaw * x + cos_yaw * y + cy
    z_global = z + cz
    corners = np.stack([x_rot, y_rot, z_global], axis=1)
    mins = corners.min(axis=0)
    maxs = corners.max(axis=0)
    center = (mins + maxs) / 2.0
    size = np.maximum(maxs - mins, 1e-6)
    return np.concatenate([center, size]).astype(np.float64)


def finite_values(values: list[Any]) -> list[float]:
    return [value for value in (safe_float(item) for item in values) if math.isfinite(value)]


def mean(values: list[Any]) -> float:
    vals = finite_values(values)
    return sum(vals) / len(vals) if vals else math.nan


def median(values: list[Any]) -> float:
    vals = sorted(finite_values(values))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def ratio_bool(rows: list[dict[str, Any]], key: str) -> float:
    vals = [row.get(key) for row in rows if row.get(key) not in ("", None)]
    if not vals:
        return math.nan
    return sum(1 for value in vals if bool(value)) / len(vals)


def oracle_best_type(gains: dict[str, float]) -> str:
    valid = {name: value for name, value in gains.items() if math.isfinite(value)}
    if not valid:
        return "none"
    return max(valid.items(), key=lambda item: (item[1], item[0]))[0]


def oracle_interpretation(row: dict[str, Any]) -> str:
    current = safe_float(row["official_current_iou"])
    center = bool(row["center_oracle_pass_05"])
    size = bool(row["size_oracle_pass_05"])
    center_size = bool(row["center_size_oracle_pass_05"])
    enclosing = bool(row["enclosing_aligned_gt_pass_05"])
    if enclosing and not center_size:
        return "gt 前 6 维轴对齐框不能过 0.5，但旋转真实框的最小轴对齐外接框可以过 0.5，说明 6D 轴对齐预测仍有理论空间，但 BoxRefine 应该学习外接框形式，而不是直接学习 gt_box 的原始尺寸。"
    if not enclosing:
        return "即使使用旋转真实框的最小轴对齐外接框，官方 IoU 仍不能达到 0.5，说明不预测方向角可能是结构性瓶颈，应考虑带方向角的旋转框预测。"
    if enclosing and center_size:
        return "普通轴对齐框理论上可过 0.5，中心点和尺寸修正仍有空间。"
    if not center_size:
        return "即使中心点和尺寸都理想化，但由于预测框仍然不带方向角，官方 IoU 仍不能达到 0.5，说明轴对齐框表达本身可能是瓶颈。"
    if center and not size:
        return "只修中心点理论上即可过 0.5，说明中心点修正可能有效。"
    if size and not center:
        return "只修尺寸理论上即可过 0.5，说明尺寸修正可能有效。"
    if not center and not size and center_size:
        return "单独修中心点或尺寸不够，需要中心点和尺寸联合修正。"
    if center_size and math.isfinite(current) and current >= 0.45:
        return "普通轴对齐框仍有提升空间，可作为 BoxRefine 候选样本。"
    return "中心点和尺寸 oracle 显示普通轴对齐框仍有一定理论空间。"


def analyze_record(row: dict[str, str], target_set: set[tuple[str, str]]) -> dict[str, Any] | None:
    platform = row.get("platform", "")
    scene = row.get("scene", "")
    if (platform, scene) not in target_set:
        return None

    prediction = read_prediction(repo_path(row.get("json_path", "")))
    missing_reasons: list[str] = []
    if prediction is None:
        prediction = {}
        missing_reasons.append("prediction_json")

    gt_box = first_box(prediction.get("gt_box"), min_dims=7)
    if gt_box is None:
        missing_reasons.append("gt_box")
    pred_box, pred_source = pick_pred_box(prediction)
    if pred_box is None:
        missing_reasons.append("pred_box")

    current_box = pred_box[:6] if pred_box is not None else None
    center_oracle_box = (
        np.concatenate([gt_box[:3], pred_box[3:6]]) if gt_box is not None and pred_box is not None else None
    )
    size_oracle_box = (
        np.concatenate([pred_box[:3], gt_box[3:6]]) if gt_box is not None and pred_box is not None else None
    )
    center_size_oracle_box = gt_box[:6] if gt_box is not None else None
    enclosing_aligned_gt_box = rotated_gt_to_enclosing_aligned_box(gt_box)

    official_current = official_iou(gt_box, current_box)
    official_center = official_iou(gt_box, center_oracle_box)
    official_size = official_iou(gt_box, size_oracle_box)
    official_center_size = official_iou(gt_box, center_size_oracle_box)
    official_enclosing = official_iou(gt_box, enclosing_aligned_gt_box)

    top1_recorded = safe_float(prediction.get("top1_iou", row.get("top1_iou")))
    current_recorded_abs_diff = (
        abs(official_current - top1_recorded)
        if math.isfinite(official_current) and math.isfinite(top1_recorded)
        else math.nan
    )

    axis_iou = axis_aligned_iou_center_size(gt_box, pred_box)
    rotation_penalty = (
        axis_iou - official_current if math.isfinite(axis_iou) and math.isfinite(official_current) else math.nan
    )
    rotation_penalty_ratio = (
        rotation_penalty / axis_iou
        if math.isfinite(rotation_penalty) and math.isfinite(axis_iou) and axis_iou > EPS
        else math.nan
    )

    center_gain = official_center - official_current if math.isfinite(official_center) and math.isfinite(official_current) else math.nan
    size_gain = official_size - official_current if math.isfinite(official_size) and math.isfinite(official_current) else math.nan
    center_size_gain = (
        official_center_size - official_current
        if math.isfinite(official_center_size) and math.isfinite(official_current)
        else math.nan
    )
    enclosing_gain = (
        official_enclosing - official_current
        if math.isfinite(official_enclosing) and math.isfinite(official_current)
        else math.nan
    )

    gt_volume = box_volume(gt_box)
    pred_volume = box_volume(pred_box)
    enclosing_volume = box_volume(enclosing_aligned_gt_box)
    volume_ratio = pred_volume / gt_volume if math.isfinite(pred_volume) and math.isfinite(gt_volume) and gt_volume > EPS else math.nan
    enclosing_volume_ratio = (
        enclosing_volume / gt_volume
        if math.isfinite(enclosing_volume) and math.isfinite(gt_volume) and gt_volume > EPS
        else math.nan
    )
    enclosing_size_ratios = (
        enclosing_aligned_gt_box[3:6] / gt_box[3:6]
        if enclosing_aligned_gt_box is not None and gt_box is not None and np.all(gt_box[3:6] > EPS)
        else np.array([math.nan, math.nan, math.nan])
    )
    center_error = float(np.linalg.norm(gt_box[:3] - pred_box[:3])) if gt_box is not None and pred_box is not None else math.nan
    size_l1_error = float(np.abs(gt_box[3:6] - pred_box[3:6]).sum()) if gt_box is not None and pred_box is not None else math.nan
    yaw_deg = yaw_to_axis_angle_deg(safe_float(gt_box[6]) if gt_box is not None else math.nan)

    top10_arr = numeric_array(prediction.get("top10_ious"))
    top10_vals = finite_values(top10_arr.reshape(-1).tolist()) if top10_arr is not None else []
    top10_best = max(top10_vals, default=safe_float(row.get("max_top10_iou")))
    top1_gap = top10_best - top1_recorded if math.isfinite(top10_best) and math.isfinite(top1_recorded) else math.nan

    out = {
        "id": row.get("id", prediction.get("id", "")),
        "platform": platform,
        "scene": scene,
        "primary_label": row.get("primary_label", ""),
        "utterance": row.get("utterance", prediction.get("utterance", "")),
        "json_path": row.get("json_path", ""),
        "pred_box_source": pred_source,
        "missing_reason": ";".join(missing_reasons),
        "top1_iou_recorded": top1_recorded,
        "official_current_iou": official_current,
        "official_center_oracle_iou": official_center,
        "official_size_oracle_iou": official_size,
        "official_center_size_oracle_iou": official_center_size,
        "official_enclosing_aligned_gt_iou": official_enclosing,
        "current_recorded_abs_diff": current_recorded_abs_diff,
        "center_oracle_gain": center_gain,
        "size_oracle_gain": size_gain,
        "center_size_oracle_gain": center_size_gain,
        "enclosing_aligned_gt_gain": enclosing_gain,
        "current_pass_05": math.isfinite(official_current) and official_current >= 0.5,
        "center_oracle_pass_05": math.isfinite(official_center) and official_center >= 0.5,
        "size_oracle_pass_05": math.isfinite(official_size) and official_size >= 0.5,
        "center_size_oracle_pass_05": math.isfinite(official_center_size) and official_center_size >= 0.5,
        "enclosing_aligned_gt_pass_05": math.isfinite(official_enclosing) and official_enclosing >= 0.5,
        "enclosing_aligned_gt_still_fail_05": math.isfinite(official_enclosing) and official_enclosing < 0.5,
        "center_oracle_pass_025": math.isfinite(official_center) and official_center >= 0.25,
        "size_oracle_pass_025": math.isfinite(official_size) and official_size >= 0.25,
        "center_size_oracle_pass_025": math.isfinite(official_center_size) and official_center_size >= 0.25,
        "gt_center_x": safe_float(gt_box[0]) if gt_box is not None else math.nan,
        "gt_center_y": safe_float(gt_box[1]) if gt_box is not None else math.nan,
        "gt_center_z": safe_float(gt_box[2]) if gt_box is not None else math.nan,
        "gt_size_x": safe_float(gt_box[3]) if gt_box is not None else math.nan,
        "gt_size_y": safe_float(gt_box[4]) if gt_box is not None else math.nan,
        "gt_size_z": safe_float(gt_box[5]) if gt_box is not None else math.nan,
        "gt_yaw": safe_float(gt_box[6]) if gt_box is not None else math.nan,
        "pred_center_x": safe_float(pred_box[0]) if pred_box is not None else math.nan,
        "pred_center_y": safe_float(pred_box[1]) if pred_box is not None else math.nan,
        "pred_center_z": safe_float(pred_box[2]) if pred_box is not None else math.nan,
        "pred_size_x": safe_float(pred_box[3]) if pred_box is not None else math.nan,
        "pred_size_y": safe_float(pred_box[4]) if pred_box is not None else math.nan,
        "pred_size_z": safe_float(pred_box[5]) if pred_box is not None else math.nan,
        "center_error": center_error,
        "size_l1_error": size_l1_error,
        "gt_volume": gt_volume,
        "pred_volume": pred_volume,
        "volume_ratio": volume_ratio,
        "enclosing_aligned_gt_box_cx": safe_float(enclosing_aligned_gt_box[0]) if enclosing_aligned_gt_box is not None else math.nan,
        "enclosing_aligned_gt_box_cy": safe_float(enclosing_aligned_gt_box[1]) if enclosing_aligned_gt_box is not None else math.nan,
        "enclosing_aligned_gt_box_cz": safe_float(enclosing_aligned_gt_box[2]) if enclosing_aligned_gt_box is not None else math.nan,
        "enclosing_aligned_gt_box_sx": safe_float(enclosing_aligned_gt_box[3]) if enclosing_aligned_gt_box is not None else math.nan,
        "enclosing_aligned_gt_box_sy": safe_float(enclosing_aligned_gt_box[4]) if enclosing_aligned_gt_box is not None else math.nan,
        "enclosing_aligned_gt_box_sz": safe_float(enclosing_aligned_gt_box[5]) if enclosing_aligned_gt_box is not None else math.nan,
        "enclosing_size_x_ratio": safe_float(enclosing_size_ratios[0]),
        "enclosing_size_y_ratio": safe_float(enclosing_size_ratios[1]),
        "enclosing_size_z_ratio": safe_float(enclosing_size_ratios[2]),
        "enclosing_volume_ratio": enclosing_volume_ratio,
        "yaw_to_axis_angle_deg": yaw_deg,
        "axis_aligned_iou_ignore_yaw": axis_iou,
        "rotation_penalty": rotation_penalty,
        "rotation_penalty_ratio": rotation_penalty_ratio,
        "top10_best_iou_recorded": top10_best,
        "top1_to_best_top10_gap": top1_gap,
    }
    out["oracle_best_type"] = oracle_best_type({
        "center": center_gain,
        "size": size_gain,
        "center_size": center_size_gain,
        "enclosing_aligned_gt": enclosing_gain,
    })
    out["oracle_interpretation"] = oracle_interpretation(out)
    return out


RECORD_FIELDS = [
    "id", "platform", "scene", "primary_label", "utterance", "json_path", "pred_box_source", "missing_reason",
    "top1_iou_recorded", "official_current_iou", "official_center_oracle_iou",
    "official_size_oracle_iou", "official_center_size_oracle_iou", "official_enclosing_aligned_gt_iou",
    "current_recorded_abs_diff",
    "center_oracle_gain", "size_oracle_gain", "center_size_oracle_gain", "enclosing_aligned_gt_gain",
    "current_pass_05", "center_oracle_pass_05", "size_oracle_pass_05", "center_size_oracle_pass_05",
    "enclosing_aligned_gt_pass_05", "enclosing_aligned_gt_still_fail_05",
    "center_oracle_pass_025", "size_oracle_pass_025", "center_size_oracle_pass_025",
    "gt_center_x", "gt_center_y", "gt_center_z", "gt_size_x", "gt_size_y", "gt_size_z", "gt_yaw",
    "pred_center_x", "pred_center_y", "pred_center_z", "pred_size_x", "pred_size_y", "pred_size_z",
    "center_error", "size_l1_error", "gt_volume", "pred_volume", "volume_ratio",
    "enclosing_aligned_gt_box_cx", "enclosing_aligned_gt_box_cy", "enclosing_aligned_gt_box_cz",
    "enclosing_aligned_gt_box_sx", "enclosing_aligned_gt_box_sy", "enclosing_aligned_gt_box_sz",
    "enclosing_size_x_ratio", "enclosing_size_y_ratio", "enclosing_size_z_ratio", "enclosing_volume_ratio",
    "yaw_to_axis_angle_deg", "axis_aligned_iou_ignore_yaw", "rotation_penalty", "rotation_penalty_ratio",
    "top10_best_iou_recorded", "top1_to_best_top10_gap", "oracle_best_type", "oracle_interpretation",
]


SUMMARY_FIELDS = [
    "platform", "scene", "primary_label", "num_records",
    "mean_current_iou", "median_current_iou",
    "mean_center_oracle_iou", "median_center_oracle_iou",
    "mean_size_oracle_iou", "median_size_oracle_iou",
    "mean_center_size_oracle_iou", "median_center_size_oracle_iou",
    "mean_enclosing_aligned_gt_iou", "median_enclosing_aligned_gt_iou",
    "mean_center_oracle_gain", "mean_size_oracle_gain", "mean_center_size_oracle_gain",
    "mean_enclosing_aligned_gt_gain",
    "ratio_current_pass_05", "ratio_center_oracle_pass_05", "ratio_size_oracle_pass_05",
    "ratio_center_size_oracle_pass_05", "ratio_center_size_oracle_still_fail_05",
    "ratio_enclosing_aligned_gt_pass_05", "ratio_enclosing_aligned_gt_still_fail_05",
    "mean_enclosing_size_x_ratio", "mean_enclosing_size_y_ratio", "mean_enclosing_size_z_ratio",
    "mean_enclosing_volume_ratio",
    "mean_yaw_to_axis_angle_deg", "mean_rotation_penalty", "mean_rotation_penalty_ratio",
    "mean_current_recorded_abs_diff", "max_current_recorded_abs_diff",
]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    still_fail = [
        row for row in rows
        if row.get("center_size_oracle_pass_05") is not None and not bool(row.get("center_size_oracle_pass_05"))
    ]
    enclosing_still_fail = [
        row for row in rows
        if row.get("enclosing_aligned_gt_pass_05") is not None and not bool(row.get("enclosing_aligned_gt_pass_05"))
    ]
    diffs = finite_values([row.get("current_recorded_abs_diff") for row in rows])
    return {
        "num_records": len(rows),
        "mean_current_iou": mean([row["official_current_iou"] for row in rows]),
        "median_current_iou": median([row["official_current_iou"] for row in rows]),
        "mean_center_oracle_iou": mean([row["official_center_oracle_iou"] for row in rows]),
        "median_center_oracle_iou": median([row["official_center_oracle_iou"] for row in rows]),
        "mean_size_oracle_iou": mean([row["official_size_oracle_iou"] for row in rows]),
        "median_size_oracle_iou": median([row["official_size_oracle_iou"] for row in rows]),
        "mean_center_size_oracle_iou": mean([row["official_center_size_oracle_iou"] for row in rows]),
        "median_center_size_oracle_iou": median([row["official_center_size_oracle_iou"] for row in rows]),
        "mean_enclosing_aligned_gt_iou": mean([row["official_enclosing_aligned_gt_iou"] for row in rows]),
        "median_enclosing_aligned_gt_iou": median([row["official_enclosing_aligned_gt_iou"] for row in rows]),
        "mean_center_oracle_gain": mean([row["center_oracle_gain"] for row in rows]),
        "mean_size_oracle_gain": mean([row["size_oracle_gain"] for row in rows]),
        "mean_center_size_oracle_gain": mean([row["center_size_oracle_gain"] for row in rows]),
        "mean_enclosing_aligned_gt_gain": mean([row["enclosing_aligned_gt_gain"] for row in rows]),
        "ratio_current_pass_05": ratio_bool(rows, "current_pass_05"),
        "ratio_center_oracle_pass_05": ratio_bool(rows, "center_oracle_pass_05"),
        "ratio_size_oracle_pass_05": ratio_bool(rows, "size_oracle_pass_05"),
        "ratio_center_size_oracle_pass_05": ratio_bool(rows, "center_size_oracle_pass_05"),
        "ratio_center_size_oracle_still_fail_05": len(still_fail) / len(rows) if rows else math.nan,
        "ratio_enclosing_aligned_gt_pass_05": ratio_bool(rows, "enclosing_aligned_gt_pass_05"),
        "ratio_enclosing_aligned_gt_still_fail_05": len(enclosing_still_fail) / len(rows) if rows else math.nan,
        "mean_enclosing_size_x_ratio": mean([row["enclosing_size_x_ratio"] for row in rows]),
        "mean_enclosing_size_y_ratio": mean([row["enclosing_size_y_ratio"] for row in rows]),
        "mean_enclosing_size_z_ratio": mean([row["enclosing_size_z_ratio"] for row in rows]),
        "mean_enclosing_volume_ratio": mean([row["enclosing_volume_ratio"] for row in rows]),
        "mean_yaw_to_axis_angle_deg": mean([row["yaw_to_axis_angle_deg"] for row in rows]),
        "mean_rotation_penalty": mean([row["rotation_penalty"] for row in rows]),
        "mean_rotation_penalty_ratio": mean([row["rotation_penalty_ratio"] for row in rows]),
        "mean_current_recorded_abs_diff": sum(diffs) / len(diffs) if diffs else math.nan,
        "max_current_recorded_abs_diff": max(diffs) if diffs else math.nan,
    }


def group_by(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def build_summary_by_target_label(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (platform, scene, label), group in sorted(group_by(rows, ("platform", "scene", "primary_label")).items()):
        item = {"platform": platform, "scene": scene, "primary_label": label}
        item.update(summarize(group))
        output.append(item)
    return output


def build_summary_by_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (platform, scene), group in sorted(group_by(rows, ("platform", "scene")).items()):
        item = {"platform": platform, "scene": scene, "primary_label": "ALL"}
        item.update(summarize(group))
        output.append(item)
    return output


def build_c_focus_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (platform, scene), group in sorted(group_by([r for r in rows if r["primary_label"] == C_LABEL], ("platform", "scene")).items()):
        summary = summarize(group)
        output.append({
            "platform": platform,
            "scene": scene,
            "num_C_samples": summary["num_records"],
            "mean_current_iou": summary["mean_current_iou"],
            "mean_center_oracle_iou": summary["mean_center_oracle_iou"],
            "mean_size_oracle_iou": summary["mean_size_oracle_iou"],
            "mean_center_size_oracle_iou": summary["mean_center_size_oracle_iou"],
            "mean_enclosing_aligned_gt_iou": summary["mean_enclosing_aligned_gt_iou"],
            "ratio_current_pass_05": summary["ratio_current_pass_05"],
            "ratio_center_oracle_pass_05": summary["ratio_center_oracle_pass_05"],
            "ratio_size_oracle_pass_05": summary["ratio_size_oracle_pass_05"],
            "ratio_center_size_oracle_pass_05": summary["ratio_center_size_oracle_pass_05"],
            "ratio_center_size_oracle_still_fail_05": summary["ratio_center_size_oracle_still_fail_05"],
            "ratio_enclosing_aligned_gt_pass_05": summary["ratio_enclosing_aligned_gt_pass_05"],
            "ratio_enclosing_aligned_gt_still_fail_05": summary["ratio_enclosing_aligned_gt_still_fail_05"],
            "mean_enclosing_aligned_gt_gain": summary["mean_enclosing_aligned_gt_gain"],
            "mean_enclosing_volume_ratio": summary["mean_enclosing_volume_ratio"],
            "mean_rotation_penalty": summary["mean_rotation_penalty"],
            "mean_yaw_to_axis_angle_deg": summary["mean_yaw_to_axis_angle_deg"],
        })
    return output


C_FOCUS_FIELDS = [
    "platform", "scene", "num_C_samples",
    "mean_current_iou", "mean_center_oracle_iou", "mean_size_oracle_iou",
    "mean_center_size_oracle_iou", "mean_enclosing_aligned_gt_iou",
    "ratio_current_pass_05", "ratio_center_oracle_pass_05",
    "ratio_size_oracle_pass_05", "ratio_center_size_oracle_pass_05",
    "ratio_center_size_oracle_still_fail_05",
    "ratio_enclosing_aligned_gt_pass_05", "ratio_enclosing_aligned_gt_still_fail_05",
    "mean_enclosing_aligned_gt_gain", "mean_enclosing_volume_ratio",
    "mean_rotation_penalty", "mean_yaw_to_axis_angle_deg",
]

CASE_FIELDS = [
    "id", "platform", "scene", "top1_iou_recorded", "official_current_iou",
    "official_center_size_oracle_iou", "center_size_oracle_gain", "yaw_to_axis_angle_deg",
    "rotation_penalty", "utterance", "json_path",
]

ENCLOSING_CASE_FIELDS = [
    "id", "platform", "scene", "top1_iou_recorded", "official_current_iou",
    "official_center_size_oracle_iou", "official_enclosing_aligned_gt_iou",
    "center_size_oracle_pass_05", "enclosing_aligned_gt_pass_05",
    "yaw_to_axis_angle_deg", "rotation_penalty", "enclosing_volume_ratio",
    "utterance", "json_path",
]


def target_label(rows: list[dict[str, Any]], platform: str, scene: str, label: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["platform"] == platform and row["scene"] == scene and row["primary_label"] == label]


def one_line(name: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"{name}: no records"
    summary = summarize(rows)
    return (
        f"{name}: n={summary['num_records']}, current={fmt(summary['mean_current_iou'])}, "
        f"center={fmt(summary['mean_center_oracle_iou'])}, size={fmt(summary['mean_size_oracle_iou'])}, "
        f"center+size={fmt(summary['mean_center_size_oracle_iou'])}, "
        f"enclosing={fmt(summary['mean_enclosing_aligned_gt_iou'])}, "
        f"center+size pass@0.5={fmt(summary['ratio_center_size_oracle_pass_05'])}, "
        f"enclosing pass@0.5={fmt(summary['ratio_enclosing_aligned_gt_pass_05'])}, "
        f"enclosing still_fail={fmt(summary['ratio_enclosing_aligned_gt_still_fail_05'])}, "
        f"enclosing_vol_ratio={fmt(summary['mean_enclosing_volume_ratio'])}, "
        f"rotation_penalty={fmt(summary['mean_rotation_penalty'])}"
    )


def build_findings(
    args: argparse.Namespace,
    targets: list[tuple[str, str]],
    rows: list[dict[str, Any]],
    missing_counts: Counter[str],
) -> list[str]:
    c_groups = {(platform, scene): target_label(rows, platform, scene, C_LABEL) for platform, scene in targets}
    c_summaries = {key: summarize(value) for key, value in c_groups.items() if value}
    diffs = finite_values([row["current_recorded_abs_diff"] for row in rows])
    diff_bad = sum(1 for value in diffs if value > 1e-4)

    lines = [
        "official aligned oracle findings",
        "",
        "本次新增：旋转真实框的最小轴对齐外接框 oracle。",
        "该 oracle 先把 rotated GT 转成能包住 8 个角点的最小 6D AABB，再用 official rotated-GT-vs-aligned-pred IoU 计算理论上限。",
        "",
        f"输入文件: {args.all_records_csv}",
        f"输出目录: {args.out_dir}",
        f"分析 targets: {', '.join(f'{p}:{s}' for p, s in targets)}",
        f"官方 IoU import_status: {OFFICIAL_IMPORT_STATUS}",
        f"总样本数: {len(rows)}",
        f"missing counts: {dict(missing_counts)}",
        f"top1_iou_recorded vs official_current_iou mean_abs_diff={fmt(mean(diffs), 8)}, max_abs_diff={fmt(max(diffs) if diffs else math.nan, 8)}, num_diff_gt_1e-4={diff_bad}",
        "",
        "重点 C 类 oracle 统计:",
    ]
    for platform, scene in targets:
        lines.append("  " + one_line(f"{platform} {scene} C", c_groups.get((platform, scene), [])))

    lines.extend(["", "全部 target 样本数:"])
    for platform, scene in targets:
        group = [row for row in rows if row["platform"] == platform and row["scene"] == scene]
        lines.append(f"  {platform} {scene}: {len(group)}")

    drone_key = ("drone", "Outdoor_Day_penno_parking_2")
    quad_key = ("quad", "Outdoor_Day_penno_short_loop")
    drone_summary = c_summaries.get(drone_key)
    quad_summary = c_summaries.get(quad_key)
    if drone_summary and quad_summary:
        lines.extend([
            "",
            "Drone parking_2 C vs Quad penno_short_loop C:",
            f"  Drone current={fmt(drone_summary['mean_current_iou'])}, center+size={fmt(drone_summary['mean_center_size_oracle_iou'])}, enclosing={fmt(drone_summary['mean_enclosing_aligned_gt_iou'])}",
            f"  Drone center+size pass@0.5={fmt(drone_summary['ratio_center_size_oracle_pass_05'])}, enclosing pass@0.5={fmt(drone_summary['ratio_enclosing_aligned_gt_pass_05'])}, enclosing still_fail={fmt(drone_summary['ratio_enclosing_aligned_gt_still_fail_05'])}, enclosing volume ratio={fmt(drone_summary['mean_enclosing_volume_ratio'])}",
            f"  Quad current={fmt(quad_summary['mean_current_iou'])}, center+size={fmt(quad_summary['mean_center_size_oracle_iou'])}, enclosing={fmt(quad_summary['mean_enclosing_aligned_gt_iou'])}",
            f"  Quad center+size pass@0.5={fmt(quad_summary['ratio_center_size_oracle_pass_05'])}, enclosing pass@0.5={fmt(quad_summary['ratio_enclosing_aligned_gt_pass_05'])}, enclosing still_fail={fmt(quad_summary['ratio_enclosing_aligned_gt_still_fail_05'])}, enclosing volume ratio={fmt(quad_summary['mean_enclosing_volume_ratio'])}",
        ])
        if drone_summary["mean_enclosing_aligned_gt_iou"] < quad_summary["mean_enclosing_aligned_gt_iou"]:
            lines.append("  Drone parking_2 C 的 enclosing aligned oracle 上限更低。")
        elif drone_summary["mean_enclosing_aligned_gt_iou"] > quad_summary["mean_enclosing_aligned_gt_iou"]:
            lines.append("  Quad penno_short_loop C 的 enclosing aligned oracle 上限更低。")
        else:
            lines.append("  两者 enclosing aligned oracle 上限接近。")

    enclosing_pass_rates = [
        summary["ratio_enclosing_aligned_gt_pass_05"]
        for summary in c_summaries.values()
        if math.isfinite(summary["ratio_enclosing_aligned_gt_pass_05"])
    ]
    center_size_pass_rates = [
        summary["ratio_center_size_oracle_pass_05"]
        for summary in c_summaries.values()
        if math.isfinite(summary["ratio_center_size_oracle_pass_05"])
    ]
    mean_enclosing_pass = sum(enclosing_pass_rates) / len(enclosing_pass_rates) if enclosing_pass_rates else math.nan
    mean_center_size_pass = sum(center_size_pass_rates) / len(center_size_pass_rates) if center_size_pass_rates else math.nan

    lines.extend(["", "方法上限判断:"])
    if math.isfinite(mean_enclosing_pass) and math.isfinite(mean_center_size_pass):
        if mean_enclosing_pass >= 0.7 and mean_center_size_pass < 0.4:
            lines.append("  enclosing aligned GT oracle pass@0.5 高，而 gt 前 6 维 center+size oracle pass@0.5 低。")
            lines.append("  说明普通 6D 轴对齐框仍有理论空间，但 BoxRefine 的目标不应是 gt_box 前 6 维，而应学习旋转框的最小轴对齐外接框。")
            lines.append("  方法建议：改为学习旋转框外接轴对齐框，或把它作为 heading-aware refinement 前的较稳妥过渡目标。")
        elif mean_enclosing_pass < 0.4:
            lines.append("  enclosing aligned GT oracle pass@0.5 仍然较低。")
            lines.append("  说明即使使用最优轴对齐外接框，不预测方向角的表达上限仍不足。")
            lines.append("  方法建议：优先转向 heading-aware / rotated box refinement。")
        else:
            lines.append("  enclosing aligned GT oracle pass@0.5 有一定空间，但并非压倒性充分。")
            lines.append("  方法建议：可以比较学习外接轴对齐框与 heading-aware / rotated box refinement 两条路线，普通 gt 前 6 维 BoxRefine 不再是最佳目标。")
    else:
        lines.append("  enclosing aligned GT oracle 统计不足，无法可靠判断方法上限。")
    lines.append("  这个 oracle 是离线理论分析，不是模型真实结果；它的作用是判断方法上限。")
    return lines

def write_import_failure(args: argparse.Namespace) -> None:
    out_dir = repo_path(args.out_dir)
    write_text(out_dir / "official_aligned_oracle_findings.txt", [
        "official aligned oracle findings",
        "",
        f"官方 IoU import_status: {OFFICIAL_IMPORT_STATUS}",
        "无法导入 utils.eval_det.iou3d_rotated_vs_aligned，因此停止分析；不能静默改用非官方 IoU。",
    ])


def main() -> None:
    args = parse_args()
    targets = parse_targets(args.targets)
    if iou3d_rotated_vs_aligned is None:
        write_import_failure(args)
        raise RuntimeError(f"Cannot import official IoU function: {OFFICIAL_IMPORT_STATUS}")

    all_rows = read_csv(repo_path(args.all_records_csv))
    target_set = set(targets)
    records: list[dict[str, Any]] = []
    missing_counts: Counter[str] = Counter()
    for row in all_rows:
        analyzed = analyze_record(row, target_set)
        if analyzed is None:
            continue
        records.append(analyzed)
        for reason in str(analyzed.get("missing_reason", "")).split(";"):
            if reason:
                missing_counts[reason] += 1
    if not records:
        raise ValueError(f"No records matched targets: {args.targets}")

    summary_by_target_label = build_summary_by_target_label(records)
    summary_by_target = build_summary_by_target(records)
    c_focus_summary = build_c_focus_summary(records)
    c_rows = [row for row in records if row["primary_label"] == C_LABEL]
    fail_cases = [row for row in c_rows if not bool(row["center_size_oracle_pass_05"])]
    pass_cases = [row for row in c_rows if bool(row["center_size_oracle_pass_05"])]
    enclosing_fail_cases = [row for row in c_rows if not bool(row["enclosing_aligned_gt_pass_05"])]
    enclosing_pass_cases = [row for row in c_rows if bool(row["enclosing_aligned_gt_pass_05"])]

    out_dir = repo_path(args.out_dir)
    write_csv(out_dir / "official_aligned_oracle_records.csv", records, RECORD_FIELDS)
    write_csv(out_dir / "official_aligned_oracle_summary_by_target_label.csv", summary_by_target_label, SUMMARY_FIELDS)
    write_csv(out_dir / "official_aligned_oracle_summary_by_target.csv", summary_by_target, SUMMARY_FIELDS)
    write_csv(out_dir / "official_aligned_oracle_c_focus_summary.csv", c_focus_summary, C_FOCUS_FIELDS)
    write_csv(out_dir / "center_size_oracle_fail_cases.csv", fail_cases, CASE_FIELDS)
    write_csv(out_dir / "center_size_oracle_pass_cases.csv", pass_cases, CASE_FIELDS)
    write_csv(out_dir / "enclosing_aligned_gt_fail_cases.csv", enclosing_fail_cases, ENCLOSING_CASE_FIELDS)
    write_csv(out_dir / "enclosing_aligned_gt_pass_cases.csv", enclosing_pass_cases, ENCLOSING_CASE_FIELDS)
    write_text(out_dir / "official_aligned_oracle_findings.txt", build_findings(args, targets, records, missing_counts))

    print(f"Saved official aligned oracle analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
