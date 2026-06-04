#!/usr/bin/env python3
"""Offline baseline evaluation with GT yaw ignored.

This diagnostic script compares the original official IoU
(rotated GT vs aligned prediction) with a simple ignore-yaw aligned IoU
computed from the same prediction.json files. It does not load models,
checkpoints, or alter the project's evaluation logic.
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
DEFAULT_OUT_DIR = "outputs/baseline_ignore_yaw_eval"
DEFAULT_PLATFORMS = ("drone", "quad")
EPS = 1e-9

LABEL_A = "A_recall_fail_25"
LABEL_B = "B_ranking_fail_25"
LABEL_C = "C_coarse_success_precise_fail"
LABEL_D = "D_ranking_fail_50"
LABEL_E = "E_strict_success"
LABEL_ORDER = (LABEL_A, LABEL_B, LABEL_C, LABEL_D, LABEL_E)

KEY_SCENES = (
    ("drone", "Outdoor_Day_penno_parking_2"),
    ("drone", "Outdoor_Day_penno_plaza"),
    ("quad", "Outdoor_Day_penno_short_loop"),
    ("quad", "Outdoor_Day_skatepark_1"),
    ("quad", "Outdoor_Day_srt_under_bridge_1"),
)


try:
    from utils.eval_det import iou3d_rotated_vs_aligned

    OFFICIAL_IMPORT_STATUS = "ok"
except Exception as exc:  # pragma: no cover - reported in findings on failure.
    iou3d_rotated_vs_aligned = None
    OFFICIAL_IMPORT_STATUS = repr(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline predictions with GT yaw ignored.")
    parser.add_argument("--all_records_csv", default=DEFAULT_ALL_RECORDS_CSV)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--platforms", nargs="+", default=list(DEFAULT_PLATFORMS))
    return parser.parse_args()


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


def box_list(value: Any, min_dims: int = 6) -> list[np.ndarray]:
    arr = numeric_array(value)
    if arr is None:
        return []
    if arr.ndim == 1 and arr.shape[0] >= min_dims:
        return [arr[:min_dims].astype(np.float64)]
    if arr.ndim >= 2 and arr.shape[-1] >= min_dims:
        return [row[:min_dims].astype(np.float64) for row in arr.reshape(-1, arr.shape[-1])]
    return []


def pick_top1_pred(prediction: dict[str, Any], top10_boxes: list[np.ndarray]) -> tuple[np.ndarray | None, str]:
    top1 = first_box(prediction.get("top1_pred_box"), min_dims=6)
    if top1 is not None:
        return top1[:6], "top1_pred_box"
    if top10_boxes:
        return top10_boxes[0], "top10_pred_boxes[0]"
    return None, "missing"


def official_iou_list(gt_box: np.ndarray | None, pred_boxes: list[np.ndarray]) -> list[float]:
    if iou3d_rotated_vs_aligned is None:
        raise RuntimeError(f"Cannot compute official IoU; import_status={OFFICIAL_IMPORT_STATUS}")
    if gt_box is None or gt_box.shape[0] < 7 or not pred_boxes:
        return []
    try:
        gt_tensor = torch.as_tensor(gt_box.reshape(1, -1), dtype=torch.float32)
        pred_tensor = torch.as_tensor(np.stack([box[:6] for box in pred_boxes], axis=0), dtype=torch.float32)
        ious, _ = iou3d_rotated_vs_aligned(gt_tensor, pred_tensor)
        return [safe_float(v) for v in ious[0].detach().cpu().numpy().reshape(-1).tolist()]
    except Exception:
        return [math.nan for _ in pred_boxes]


def center_size_to_xyzxyz(box: np.ndarray | None) -> np.ndarray | None:
    if box is None or box.shape[0] < 6:
        return None
    size = box[3:6]
    if np.any(size <= 0) or not np.isfinite(box[:6]).all():
        return None
    half = size / 2.0
    return np.concatenate([box[:3] - half, box[:3] + half])


def aligned_iou(box_a: np.ndarray | None, box_b: np.ndarray | None) -> float:
    a = center_size_to_xyzxyz(box_a)
    b = center_size_to_xyzxyz(box_b)
    if a is None or b is None:
        return math.nan
    inter_min = np.maximum(a[:3], b[:3])
    inter_max = np.minimum(a[3:6], b[3:6])
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(np.prod(inter_size))
    vol_a = float(np.prod(np.maximum(a[3:6] - a[:3], 0.0)))
    vol_b = float(np.prod(np.maximum(b[3:6] - b[:3], 0.0)))
    union = vol_a + vol_b - inter_vol
    if union <= EPS:
        return math.nan
    return inter_vol / union


def ignore_yaw_iou_list(gt_box: np.ndarray | None, pred_boxes: list[np.ndarray]) -> list[float]:
    if gt_box is None or gt_box.shape[0] < 6:
        return []
    gt_aligned = gt_box[:6]
    return [aligned_iou(gt_aligned, pred_box[:6]) for pred_box in pred_boxes]


def clean_float_list(value: Any) -> list[float]:
    arr = numeric_array(value)
    if arr is None:
        return []
    return [safe_float(v) for v in arr.reshape(-1).tolist()]


def finite_values(values: list[Any]) -> list[float]:
    return [value for value in (safe_float(item) for item in values) if math.isfinite(value)]


def mean(values: list[Any]) -> float:
    vals = finite_values(values)
    return sum(vals) / len(vals) if vals else math.nan


def ratio_bool(rows: list[dict[str, Any]], key: str) -> float:
    vals = [row.get(key) for row in rows if row.get(key) not in ("", None)]
    if not vals:
        return math.nan
    return sum(1 for value in vals if bool(value)) / len(vals)


def max_finite(values: list[Any]) -> float:
    vals = finite_values(values)
    return max(vals) if vals else math.nan


def label_from_ious(top1_iou: float, max_top10_iou: float) -> str:
    if math.isfinite(top1_iou) and top1_iou >= 0.5:
        return LABEL_E
    if math.isfinite(top1_iou) and 0.25 <= top1_iou < 0.5:
        return LABEL_C
    if math.isfinite(max_top10_iou) and max_top10_iou >= 0.5:
        return LABEL_D
    if math.isfinite(max_top10_iou) and max_top10_iou >= 0.25:
        return LABEL_B
    return LABEL_A


def yaw_to_axis_angle_deg(yaw: float) -> float:
    if not math.isfinite(yaw):
        return math.nan
    yaw_mod = yaw % math.pi
    rad = min(abs(yaw_mod), abs(yaw_mod - math.pi / 2.0), abs(yaw_mod - math.pi))
    return math.degrees(rad)


def rotation_effect_interpretation(official_top1: float, ignore_top1: float) -> str:
    gain = ignore_top1 - official_top1 if math.isfinite(official_top1) and math.isfinite(ignore_top1) else math.nan
    if math.isfinite(official_top1) and math.isfinite(ignore_top1) and official_top1 < 0.5 <= ignore_top1:
        return "忽略旋转角后从失败变为严格成功，说明该样本主要受旋转角影响。"
    if math.isfinite(official_top1) and math.isfinite(ignore_top1) and official_top1 < 0.25 <= ignore_top1:
        return "忽略旋转角后从召回/排序失败变为粗定位成功，说明旋转角影响较明显。"
    if math.isfinite(gain) and gain < 0.05:
        return "忽略旋转角后提升很小，低 IoU 主要不是旋转角造成。"
    return "忽略旋转角后有一定提升，但仍需结合候选质量、中心偏移和遮挡判断。"


def top10_diff(recorded: list[float], recomputed: list[float]) -> float:
    diffs = [abs(a - b) for a, b in zip(recorded, recomputed) if math.isfinite(a) and math.isfinite(b)]
    return max(diffs) if diffs else math.nan


def analyze_record(row: dict[str, str]) -> dict[str, Any]:
    prediction = read_prediction(repo_path(row.get("json_path", "")))
    missing: list[str] = []
    if prediction is None:
        prediction = {}
        missing.append("prediction_json")

    gt_box = first_box(prediction.get("gt_box"), min_dims=7)
    if gt_box is None:
        missing.append("gt_box")
    top10_boxes = box_list(prediction.get("top10_pred_boxes"), min_dims=6)
    top1_box, top1_source = pick_top1_pred(prediction, top10_boxes)
    if top1_box is None:
        missing.append("top1_pred_box")
    if not top10_boxes and top1_box is not None:
        top10_boxes = [top1_box]
    if not top10_boxes:
        missing.append("top10_pred_boxes")

    official_top10 = official_iou_list(gt_box, top10_boxes)
    official_top1 = official_iou_list(gt_box, [top1_box])[0] if top1_box is not None else math.nan
    official_max_top10 = max_finite(official_top10)

    ignore_top10 = ignore_yaw_iou_list(gt_box, top10_boxes)
    ignore_top1 = ignore_yaw_iou_list(gt_box, [top1_box])[0] if top1_box is not None else math.nan
    ignore_max_top10 = max_finite(ignore_top10)

    recorded_top1 = safe_float(prediction.get("top1_iou", row.get("top1_iou")))
    recorded_top10 = clean_float_list(prediction.get("top10_ious"))
    official_label = row.get("primary_label", label_from_ious(official_top1, official_max_top10))
    ignore_label = label_from_ious(ignore_top1, ignore_max_top10)
    yaw_raw = safe_float(gt_box[6]) if gt_box is not None and gt_box.shape[0] >= 7 else math.nan

    out = {
        "id": row.get("id", prediction.get("id", "")),
        "platform": row.get("platform", prediction.get("platform", "")),
        "scene": row.get("scene", ""),
        "primary_label": row.get("primary_label", ""),
        "utterance": row.get("utterance", prediction.get("utterance", "")),
        "json_path": row.get("json_path", ""),
        "top1_pred_source": top1_source,
        "missing_reason": ";".join(missing),
        "official_top1_iou": official_top1,
        "official_max_top10_iou": official_max_top10,
        "official_acc25_top1": math.isfinite(official_top1) and official_top1 >= 0.25,
        "official_acc50_top1": math.isfinite(official_top1) and official_top1 >= 0.5,
        "official_acc25_top10": math.isfinite(official_max_top10) and official_max_top10 >= 0.25,
        "official_acc50_top10": math.isfinite(official_max_top10) and official_max_top10 >= 0.5,
        "ignore_yaw_top1_iou": ignore_top1,
        "ignore_yaw_max_top10_iou": ignore_max_top10,
        "ignore_yaw_acc25_top1": math.isfinite(ignore_top1) and ignore_top1 >= 0.25,
        "ignore_yaw_acc50_top1": math.isfinite(ignore_top1) and ignore_top1 >= 0.5,
        "ignore_yaw_acc25_top10": math.isfinite(ignore_max_top10) and ignore_max_top10 >= 0.25,
        "ignore_yaw_acc50_top10": math.isfinite(ignore_max_top10) and ignore_max_top10 >= 0.5,
        "top1_iou_gain_ignore_yaw": ignore_top1 - official_top1 if math.isfinite(ignore_top1) and math.isfinite(official_top1) else math.nan,
        "max_top10_iou_gain_ignore_yaw": ignore_max_top10 - official_max_top10 if math.isfinite(ignore_max_top10) and math.isfinite(official_max_top10) else math.nan,
        "official_primary_label": official_label,
        "ignore_yaw_primary_label": ignore_label,
        "yaw_raw": yaw_raw,
        "yaw_to_axis_angle_deg": yaw_to_axis_angle_deg(yaw_raw),
        "gt_center_x": safe_float(gt_box[0]) if gt_box is not None else math.nan,
        "gt_center_y": safe_float(gt_box[1]) if gt_box is not None else math.nan,
        "gt_center_z": safe_float(gt_box[2]) if gt_box is not None else math.nan,
        "gt_size_x": safe_float(gt_box[3]) if gt_box is not None else math.nan,
        "gt_size_y": safe_float(gt_box[4]) if gt_box is not None else math.nan,
        "gt_size_z": safe_float(gt_box[5]) if gt_box is not None else math.nan,
        "gt_yaw": yaw_raw,
        "pred_center_x": safe_float(top1_box[0]) if top1_box is not None else math.nan,
        "pred_center_y": safe_float(top1_box[1]) if top1_box is not None else math.nan,
        "pred_center_z": safe_float(top1_box[2]) if top1_box is not None else math.nan,
        "pred_size_x": safe_float(top1_box[3]) if top1_box is not None else math.nan,
        "pred_size_y": safe_float(top1_box[4]) if top1_box is not None else math.nan,
        "pred_size_z": safe_float(top1_box[5]) if top1_box is not None else math.nan,
        "top1_iou_recorded": recorded_top1,
        "top1_recorded_official_abs_diff": abs(recorded_top1 - official_top1) if math.isfinite(recorded_top1) and math.isfinite(official_top1) else math.nan,
        "top10_recorded_official_max_abs_diff": top10_diff(recorded_top10, official_top10),
    }
    out["rotation_effect_interpretation"] = rotation_effect_interpretation(official_top1, ignore_top1)
    return out


RECORD_FIELDS = [
    "id", "platform", "scene", "primary_label", "utterance", "json_path", "top1_pred_source", "missing_reason",
    "official_top1_iou", "official_max_top10_iou", "official_acc25_top1", "official_acc50_top1",
    "official_acc25_top10", "official_acc50_top10",
    "ignore_yaw_top1_iou", "ignore_yaw_max_top10_iou", "ignore_yaw_acc25_top1", "ignore_yaw_acc50_top1",
    "ignore_yaw_acc25_top10", "ignore_yaw_acc50_top10",
    "top1_iou_gain_ignore_yaw", "max_top10_iou_gain_ignore_yaw",
    "official_primary_label", "ignore_yaw_primary_label",
    "yaw_raw", "yaw_to_axis_angle_deg",
    "gt_center_x", "gt_center_y", "gt_center_z", "gt_size_x", "gt_size_y", "gt_size_z", "gt_yaw",
    "pred_center_x", "pred_center_y", "pred_center_z", "pred_size_x", "pred_size_y", "pred_size_z",
    "rotation_effect_interpretation",
    "top1_iou_recorded", "top1_recorded_official_abs_diff", "top10_recorded_official_max_abs_diff",
]

SUMMARY_FIELDS = [
    "platform", "scene", "num_records",
    "official_acc25_top1", "official_acc50_top1", "official_acc25_top10", "official_acc50_top10",
    "official_miou_top1", "official_mean_max_top10_iou",
    "ignore_yaw_acc25_top1", "ignore_yaw_acc50_top1", "ignore_yaw_acc25_top10", "ignore_yaw_acc50_top10",
    "ignore_yaw_miou_top1", "ignore_yaw_mean_max_top10_iou",
    "gain_acc25_top1", "gain_acc50_top1", "gain_acc25_top10", "gain_acc50_top10",
    "gain_miou_top1", "gain_mean_max_top10_iou",
    "ratio_official_fail50_to_ignore_yaw_success50",
    "mean_top1_iou_gain_ignore_yaw", "mean_max_top10_iou_gain_ignore_yaw",
    "mean_top1_recorded_official_abs_diff", "max_top1_recorded_official_abs_diff",
]


def summarize(rows: list[dict[str, Any]], platform: str, scene: str = "ALL") -> dict[str, Any]:
    official_acc25_top1 = ratio_bool(rows, "official_acc25_top1")
    official_acc50_top1 = ratio_bool(rows, "official_acc50_top1")
    official_acc25_top10 = ratio_bool(rows, "official_acc25_top10")
    official_acc50_top10 = ratio_bool(rows, "official_acc50_top10")
    ignore_acc25_top1 = ratio_bool(rows, "ignore_yaw_acc25_top1")
    ignore_acc50_top1 = ratio_bool(rows, "ignore_yaw_acc50_top1")
    ignore_acc25_top10 = ratio_bool(rows, "ignore_yaw_acc25_top10")
    ignore_acc50_top10 = ratio_bool(rows, "ignore_yaw_acc50_top10")
    official_miou = mean([row["official_top1_iou"] for row in rows])
    ignore_miou = mean([row["ignore_yaw_top1_iou"] for row in rows])
    official_top10 = mean([row["official_max_top10_iou"] for row in rows])
    ignore_top10 = mean([row["ignore_yaw_max_top10_iou"] for row in rows])
    fail_to_success = [
        row for row in rows
        if safe_float(row.get("official_top1_iou")) < 0.5 and safe_float(row.get("ignore_yaw_top1_iou")) >= 0.5
    ]
    diffs = finite_values([row.get("top1_recorded_official_abs_diff") for row in rows])
    return {
        "platform": platform,
        "scene": scene,
        "num_records": len(rows),
        "official_acc25_top1": official_acc25_top1,
        "official_acc50_top1": official_acc50_top1,
        "official_acc25_top10": official_acc25_top10,
        "official_acc50_top10": official_acc50_top10,
        "official_miou_top1": official_miou,
        "official_mean_max_top10_iou": official_top10,
        "ignore_yaw_acc25_top1": ignore_acc25_top1,
        "ignore_yaw_acc50_top1": ignore_acc50_top1,
        "ignore_yaw_acc25_top10": ignore_acc25_top10,
        "ignore_yaw_acc50_top10": ignore_acc50_top10,
        "ignore_yaw_miou_top1": ignore_miou,
        "ignore_yaw_mean_max_top10_iou": ignore_top10,
        "gain_acc25_top1": ignore_acc25_top1 - official_acc25_top1 if math.isfinite(ignore_acc25_top1) and math.isfinite(official_acc25_top1) else math.nan,
        "gain_acc50_top1": ignore_acc50_top1 - official_acc50_top1 if math.isfinite(ignore_acc50_top1) and math.isfinite(official_acc50_top1) else math.nan,
        "gain_acc25_top10": ignore_acc25_top10 - official_acc25_top10 if math.isfinite(ignore_acc25_top10) and math.isfinite(official_acc25_top10) else math.nan,
        "gain_acc50_top10": ignore_acc50_top10 - official_acc50_top10 if math.isfinite(ignore_acc50_top10) and math.isfinite(official_acc50_top10) else math.nan,
        "gain_miou_top1": ignore_miou - official_miou if math.isfinite(ignore_miou) and math.isfinite(official_miou) else math.nan,
        "gain_mean_max_top10_iou": ignore_top10 - official_top10 if math.isfinite(ignore_top10) and math.isfinite(official_top10) else math.nan,
        "ratio_official_fail50_to_ignore_yaw_success50": len(fail_to_success) / len(rows) if rows else math.nan,
        "mean_top1_iou_gain_ignore_yaw": mean([row["top1_iou_gain_ignore_yaw"] for row in rows]),
        "mean_max_top10_iou_gain_ignore_yaw": mean([row["max_top10_iou_gain_ignore_yaw"] for row in rows]),
        "mean_top1_recorded_official_abs_diff": sum(diffs) / len(diffs) if diffs else math.nan,
        "max_top1_recorded_official_abs_diff": max(diffs) if diffs else math.nan,
    }


def group_by(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def build_overall_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [summarize(group, platform=platform) for (platform,), group in sorted(group_by(rows, ("platform",)).items())]


def build_scene_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        summarize(group, platform=platform, scene=scene)
        for (platform, scene), group in sorted(group_by(rows, ("platform", "scene")).items())
    ]


def build_label_transition(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = group_by(rows, ("platform", "scene", "official_primary_label", "ignore_yaw_primary_label"))
    official_totals: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        official_totals[(row["platform"], row["scene"], row["official_primary_label"])] += 1
    output = []
    for (platform, scene, official_label, ignore_label), group in sorted(grouped.items()):
        denom = official_totals[(platform, scene, official_label)]
        output.append({
            "platform": platform,
            "scene": scene,
            "official_primary_label": official_label,
            "ignore_yaw_primary_label": ignore_label,
            "count": len(group),
            "ratio_within_official_label": len(group) / denom if denom else math.nan,
        })
    return output


def build_c_focus_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (platform, scene), group in sorted(group_by([r for r in rows if r["official_primary_label"] == LABEL_C], ("platform", "scene")).items()):
        c_to_e = [row for row in group if row["ignore_yaw_primary_label"] == LABEL_E]
        c_still_c = [row for row in group if row["ignore_yaw_primary_label"] == LABEL_C]
        c_to_a_or_b = [row for row in group if row["ignore_yaw_primary_label"] in (LABEL_A, LABEL_B)]
        output.append({
            "platform": platform,
            "scene": scene,
            "num_C_records": len(group),
            "official_mean_top1_iou": mean([row["official_top1_iou"] for row in group]),
            "ignore_yaw_mean_top1_iou": mean([row["ignore_yaw_top1_iou"] for row in group]),
            "mean_top1_iou_gain": mean([row["top1_iou_gain_ignore_yaw"] for row in group]),
            "official_acc50_top1": ratio_bool(group, "official_acc50_top1"),
            "ignore_yaw_acc50_top1": ratio_bool(group, "ignore_yaw_acc50_top1"),
            "ratio_C_to_E_after_ignore_yaw": len(c_to_e) / len(group) if group else math.nan,
            "ratio_C_still_C_after_ignore_yaw": len(c_still_c) / len(group) if group else math.nan,
            "ratio_C_to_A_or_B_after_ignore_yaw": len(c_to_a_or_b) / len(group) if group else math.nan,
        })
    return output


C_FOCUS_FIELDS = [
    "platform", "scene", "num_C_records",
    "official_mean_top1_iou", "ignore_yaw_mean_top1_iou", "mean_top1_iou_gain",
    "official_acc50_top1", "ignore_yaw_acc50_top1",
    "ratio_C_to_E_after_ignore_yaw", "ratio_C_still_C_after_ignore_yaw", "ratio_C_to_A_or_B_after_ignore_yaw",
]

CASE_FIELDS = [
    "id", "platform", "scene", "official_primary_label", "ignore_yaw_primary_label",
    "official_top1_iou", "ignore_yaw_top1_iou", "top1_iou_gain_ignore_yaw",
    "official_max_top10_iou", "ignore_yaw_max_top10_iou", "max_top10_iou_gain_ignore_yaw",
    "yaw_to_axis_angle_deg", "utterance", "json_path", "rotation_effect_interpretation",
]


def find_summary(summary_rows: list[dict[str, Any]], platform: str, scene: str | None = None) -> dict[str, Any] | None:
    for row in summary_rows:
        if row["platform"] == platform and (scene is None or row["scene"] == scene):
            return row
    return None


def summary_line(name: str, row: dict[str, Any] | None) -> str:
    if row is None:
        return f"{name}: no records"
    return (
        f"{name}: n={row['num_records']}, "
        f"official Acc@0.25/0.5={fmt(row['official_acc25_top1'])}/{fmt(row['official_acc50_top1'])}, "
        f"ignore-yaw Acc@0.25/0.5={fmt(row['ignore_yaw_acc25_top1'])}/{fmt(row['ignore_yaw_acc50_top1'])}, "
        f"mIoU {fmt(row['official_miou_top1'])}->{fmt(row['ignore_yaw_miou_top1'])}, "
        f"gain Acc@0.5={fmt(row['gain_acc50_top1'])}"
    )


def c_transition_line(name: str, c_rows: list[dict[str, Any]], platform: str, scene: str | None = None) -> str:
    for row in c_rows:
        if row["platform"] == platform and (scene is None or row["scene"] == scene):
            return (
                f"{name}: C n={row['num_C_records']}, "
                f"C->E={fmt(row['ratio_C_to_E_after_ignore_yaw'])}, "
                f"C still C={fmt(row['ratio_C_still_C_after_ignore_yaw'])}, "
                f"mean IoU {fmt(row['official_mean_top1_iou'])}->{fmt(row['ignore_yaw_mean_top1_iou'])}"
            )
    return f"{name}: no C records"


def build_findings(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    overall: list[dict[str, Any]],
    scenes: list[dict[str, Any]],
    c_focus: list[dict[str, Any]],
    missing_counts: Counter[str],
) -> list[str]:
    diffs = finite_values([row["top1_recorded_official_abs_diff"] for row in rows])
    lines = [
        "baseline ignore-yaw evaluation findings",
        "",
        f"输入文件: {args.all_records_csv}",
        f"输出目录: {args.out_dir}",
        f"分析平台: {', '.join(args.platforms)}",
        f"样本数: {len(rows)}",
        f"official IoU import_status: {OFFICIAL_IMPORT_STATUS}",
        f"missing counts: {dict(missing_counts)}",
        f"top1_iou_recorded vs official_top1_iou mean_abs_diff={fmt(mean(diffs), 8)}, max_abs_diff={fmt(max(diffs) if diffs else math.nan, 8)}",
        "",
        "整体对比:",
    ]
    for platform in args.platforms:
        lines.append("  " + summary_line(platform, find_summary(overall, platform)))

    lines.extend(["", "重点场景对比:"])
    for platform, scene in KEY_SCENES:
        if platform in args.platforms:
            lines.append("  " + summary_line(f"{platform} {scene}", find_summary(scenes, platform, scene)))

    lines.extend(["", "官方 C 类忽略 yaw 后转为 E 的比例:"])
    for platform in args.platforms:
        platform_c = [row for row in rows if row["platform"] == platform and row["official_primary_label"] == LABEL_C]
        if platform_c:
            c_to_e = sum(1 for row in platform_c if row["ignore_yaw_primary_label"] == LABEL_E) / len(platform_c)
            lines.append(f"  {platform}: C->E={fmt(c_to_e)} (n={len(platform_c)})")
    for platform, scene in KEY_SCENES:
        if platform in args.platforms:
            lines.append("  " + c_transition_line(f"{platform} {scene}", c_focus, platform, scene))

    lines.extend(["", "判断:"])
    large_gain = [
        row for row in overall
        if math.isfinite(row["gain_acc50_top1"]) and row["gain_acc50_top1"] >= 0.1
    ]
    if large_gain:
        lines.append("  忽略旋转角后 Acc@0.5 有明显提升，说明旋转角是官方 Acc@0.5 偏低的重要原因之一。")
    else:
        lines.append("  忽略旋转角后整体 Acc@0.5 提升不算大，部分失败仍主要来自候选召回、中心偏移、遮挡或相似目标。")
    lines.append("  这个实验是诊断实验，不是新的官方 benchmark 结果。")
    lines.append("  因为忽略旋转角改变了 IoU 计算方式，所以不能直接和原始 3EED 官方指标混为一谈。")
    lines.append("  如果某些场景提升明显，可以继续做“旋转真实框的最小轴对齐外接框监督”；如果提升不明显，则应继续分析候选召回和场景难度。")
    return lines


def write_import_failure(args: argparse.Namespace) -> None:
    out_dir = repo_path(args.out_dir)
    write_text(out_dir / "baseline_ignore_yaw_findings.txt", [
        "baseline ignore-yaw evaluation findings",
        "",
        f"official IoU import_status: {OFFICIAL_IMPORT_STATUS}",
        "无法导入 utils.eval_det.iou3d_rotated_vs_aligned，因此停止分析；不能静默改用非官方 IoU。",
    ])


def main() -> None:
    args = parse_args()
    if iou3d_rotated_vs_aligned is None:
        write_import_failure(args)
        raise RuntimeError(f"Cannot import official IoU function: {OFFICIAL_IMPORT_STATUS}")

    platforms = set(args.platforms)
    all_rows = read_csv(repo_path(args.all_records_csv))
    selected_rows = [row for row in all_rows if row.get("platform") in platforms]
    if not selected_rows:
        raise ValueError(f"No rows matched platforms: {args.platforms}")

    records = [analyze_record(row) for row in selected_rows]
    missing_counts: Counter[str] = Counter()
    for record in records:
        for reason in str(record.get("missing_reason", "")).split(";"):
            if reason:
                missing_counts[reason] += 1

    overall = build_overall_summary(records)
    scenes = build_scene_summary(records)
    transitions = build_label_transition(records)
    c_focus = build_c_focus_summary(records)

    drone_parking_cases = [
        row for row in records
        if row["platform"] == "drone"
        and row["scene"] == "Outdoor_Day_penno_parking_2"
        and safe_float(row["official_top1_iou"]) < 0.5
        and safe_float(row["ignore_yaw_top1_iou"]) >= 0.5
    ]
    quad_short_cases = [
        row for row in records
        if row["platform"] == "quad"
        and row["scene"] == "Outdoor_Day_penno_short_loop"
        and safe_float(row["official_top1_iou"]) < 0.5
        and safe_float(row["ignore_yaw_top1_iou"]) >= 0.5
    ]
    low_gain_cases = [
        row for row in records
        if safe_float(row["official_top1_iou"]) < 0.5
        and safe_float(row["top1_iou_gain_ignore_yaw"]) < 0.05
    ]

    out_dir = repo_path(args.out_dir)
    write_csv(out_dir / "baseline_ignore_yaw_records.csv", records, RECORD_FIELDS)
    write_csv(out_dir / "baseline_ignore_yaw_summary_overall.csv", overall, [f for f in SUMMARY_FIELDS if f != "scene"])
    write_csv(out_dir / "baseline_ignore_yaw_summary_by_scene.csv", scenes, SUMMARY_FIELDS)
    write_csv(
        out_dir / "baseline_ignore_yaw_label_transition.csv",
        transitions,
        ["platform", "scene", "official_primary_label", "ignore_yaw_primary_label", "count", "ratio_within_official_label"],
    )
    write_csv(out_dir / "baseline_ignore_yaw_c_focus_summary.csv", c_focus, C_FOCUS_FIELDS)
    write_csv(out_dir / "drone_parking2_official_fail_ignore_yaw_success_cases.csv", drone_parking_cases, CASE_FIELDS)
    write_csv(out_dir / "quad_short_loop_official_fail_ignore_yaw_success_cases.csv", quad_short_cases, CASE_FIELDS)
    write_csv(out_dir / "low_gain_cases.csv", low_gain_cases, CASE_FIELDS)
    write_text(out_dir / "baseline_ignore_yaw_findings.txt", build_findings(args, records, overall, scenes, c_focus, missing_counts))

    print(f"Saved baseline ignore-yaw evaluation to {args.out_dir}")


if __name__ == "__main__":
    main()
