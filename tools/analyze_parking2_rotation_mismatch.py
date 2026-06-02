#!/usr/bin/env python3
"""Analyze whether parking_2 C failures are driven by rotation mismatch.

The official grounding evaluation compares rotated GT boxes against aligned
predicted boxes. This offline script measures how much IoU is lost when GT yaw
is respected, compared with an ignore-yaw aligned IoU approximation.
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
DEFAULT_OUT_DIR = "outputs/scene_parking2_rotation_analysis"
SCENES = ("Outdoor_Day_penno_parking_2", "Outdoor_Day_penno_plaza")
PARKING2 = "Outdoor_Day_penno_parking_2"
PLAZA = "Outdoor_Day_penno_plaza"
PARKING2_C = "C_coarse_success_precise_fail"
STRICT_E = "E_strict_success"
EPS = 1e-9
YAW_BINS = (
    "0_5",
    "5_10",
    "10_15",
    "15_20",
    "20_25",
    "25_30",
    "30_35",
    "35_40",
    "40_45",
)


try:
    from utils.eval_det import iou3d_rotated_vs_aligned

    OFFICIAL_IMPORT_STATUS = "ok"
except Exception as exc:  # pragma: no cover - reported in findings.
    iou3d_rotated_vs_aligned = None
    OFFICIAL_IMPORT_STATUS = repr(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze GT yaw and rotation penalty for parking_2 and plaza prediction records."
    )
    parser.add_argument("--all_records_csv", default=DEFAULT_ALL_RECORDS_CSV)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
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
        if not data:
            return None
        data = data[0]
    if not isinstance(data, dict):
        return None
    return data


def first_box(value: Any, min_dims: int = 6) -> np.ndarray | None:
    arr = numeric_array(value)
    if arr is None:
        return None
    if arr.ndim == 1 and arr.shape[0] >= min_dims:
        return arr.astype(np.float64)
    if arr.ndim >= 2 and arr.shape[-1] >= min_dims:
        flat = arr.reshape(-1, arr.shape[-1])
        return flat[0].astype(np.float64)
    return None


def pick_pred_box(prediction: dict[str, Any]) -> tuple[np.ndarray | None, str]:
    top1_box = first_box(prediction.get("top1_pred_box"), min_dims=6)
    if top1_box is not None:
        return top1_box[:6], "top1_pred_box"
    top10_box = first_box(prediction.get("top10_pred_boxes"), min_dims=6)
    if top10_box is not None:
        return top10_box[:6], "top10_pred_boxes[0]"
    pred_box = first_box(prediction.get("pred_boxes"), min_dims=6)
    if pred_box is not None:
        return pred_box[:6], "pred_boxes[0]"
    return None, "missing"


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


def official_iou(gt_box: np.ndarray | None, pred_box: np.ndarray | None) -> float:
    if iou3d_rotated_vs_aligned is None or gt_box is None or pred_box is None:
        return math.nan
    if gt_box.shape[0] < 7 or pred_box.shape[0] < 6:
        return math.nan
    try:
        gt_tensor = torch.as_tensor(gt_box.reshape(1, -1), dtype=torch.float32)
        pred_tensor = torch.as_tensor(pred_box[:6].reshape(1, 6), dtype=torch.float32)
        ious, _ = iou3d_rotated_vs_aligned(gt_tensor, pred_tensor)
        return safe_float(ious[0, 0].detach().cpu().item())
    except Exception:
        return math.nan


def normalize_yaw_mod_pi(yaw: float) -> float:
    if not math.isfinite(yaw):
        return math.nan
    return yaw % math.pi


def yaw_to_axis_angle_rad(yaw: float) -> float:
    yaw_mod = normalize_yaw_mod_pi(yaw)
    if not math.isfinite(yaw_mod):
        return math.nan
    distances = (
        abs(yaw_mod - 0.0),
        abs(yaw_mod - math.pi / 2.0),
        abs(yaw_mod - math.pi),
    )
    return min(distances)


def yaw_bin(angle_deg: float) -> str:
    if not math.isfinite(angle_deg):
        return "missing"
    clamped = max(0.0, min(45.0, angle_deg))
    index = int(clamped // 5)
    if index >= len(YAW_BINS):
        index = len(YAW_BINS) - 1
    return YAW_BINS[index]


def box_volume(box: np.ndarray | None) -> float:
    if box is None or box.shape[0] < 6 or np.any(box[3:6] <= 0):
        return math.nan
    return float(np.prod(box[3:6]))


def mean(values: list[Any]) -> float:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def median(values: list[Any]) -> float:
    vals = [safe_float(v) for v in values]
    vals = sorted(v for v in vals if math.isfinite(v))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def ratio(rows: list[dict[str, Any]], key: str, threshold: float) -> float:
    vals = [safe_float(row.get(key)) for row in rows]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return math.nan
    return sum(1 for v in vals if v > threshold) / len(vals)


def label_prefix(label: str) -> str:
    if label.startswith("A_"):
        return "A"
    if label.startswith("C_"):
        return "C"
    if label.startswith("E_"):
        return "E"
    if label.startswith("B_"):
        return "B"
    if label.startswith("D_"):
        return "D"
    return "other"


def label_ratio(rows: list[dict[str, Any]], prefix: str) -> float:
    if not rows:
        return math.nan
    return sum(1 for row in rows if label_prefix(str(row.get("primary_label", ""))) == prefix) / len(rows)


def rotation_interpretation(official: float, penalty: float, yaw_deg: float, missing: str) -> str:
    if missing:
        return "缺少 gt_box 或 pred box，无法判断。"
    if math.isfinite(yaw_deg) and yaw_deg > 25.0 and math.isfinite(penalty) and penalty > 0.15:
        return "GT 车辆朝向明显偏离坐标轴，axis-aligned prediction 在 official IoU 下存在明显旋转惩罚。"
    if math.isfinite(penalty) and penalty > 0.2:
        return "忽略 yaw 时 IoU 明显更高，official IoU 主要受到 rotated GT 与 aligned prediction 不匹配影响。"
    if math.isfinite(penalty) and penalty < 0.05 and math.isfinite(official) and official < 0.35:
        return "低 IoU 不能主要由 rotation 解释，可能仍是 center/size/candidate quality 问题。"
    return "rotation penalty 有一定影响，但不足以单独解释该样本。"


def analyze_record(row: dict[str, str]) -> dict[str, Any]:
    json_path = repo_path(row.get("json_path", ""))
    prediction = read_prediction(json_path)
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

    yaw_raw = safe_float(gt_box[6]) if gt_box is not None and gt_box.shape[0] >= 7 else math.nan
    yaw_abs = abs(yaw_raw) if math.isfinite(yaw_raw) else math.nan
    yaw_mod_pi = normalize_yaw_mod_pi(yaw_raw)
    yaw_mod_pi_over_2 = yaw_raw % (math.pi / 2.0) if math.isfinite(yaw_raw) else math.nan
    yaw_axis_rad = yaw_to_axis_angle_rad(yaw_raw)
    yaw_axis_deg = math.degrees(yaw_axis_rad) if math.isfinite(yaw_axis_rad) else math.nan

    official = official_iou(gt_box, pred_box)
    axis_iou = axis_aligned_iou_center_size(gt_box, pred_box)
    penalty = axis_iou - official if math.isfinite(axis_iou) and math.isfinite(official) else math.nan
    penalty_ratio = (
        penalty / axis_iou
        if math.isfinite(penalty) and math.isfinite(axis_iou) and axis_iou > EPS
        else math.nan
    )

    gt_volume = box_volume(gt_box)
    pred_volume = box_volume(pred_box)
    volume_ratio = pred_volume / max(gt_volume, EPS) if math.isfinite(gt_volume) and math.isfinite(pred_volume) else math.nan
    center_error = (
        float(np.linalg.norm(gt_box[:3] - pred_box[:3]))
        if gt_box is not None and pred_box is not None else math.nan
    )
    size_l1_error = (
        float(np.abs(gt_box[3:6] - pred_box[3:6]).sum())
        if gt_box is not None and pred_box is not None else math.nan
    )

    top10_ious = numeric_array(prediction.get("top10_ious"))
    if top10_ious is not None:
        top10_values = [safe_float(v) for v in top10_ious.reshape(-1).tolist()]
    else:
        top10_values = []
    top10_best = max([v for v in top10_values if math.isfinite(v)], default=safe_float(row.get("max_top10_iou")))
    top1_recorded = safe_float(prediction.get("top1_iou", row.get("top1_iou")))
    top1_to_best_gap = top10_best - top1_recorded if math.isfinite(top10_best) and math.isfinite(top1_recorded) else math.nan

    out = {
        "id": row.get("id", prediction.get("id", "")),
        "scene": row.get("scene", ""),
        "primary_label": row.get("primary_label", ""),
        "utterance": row.get("utterance", prediction.get("utterance", "")),
        "json_path": row.get("json_path", ""),
        "pred_box_source": pred_source,
        "missing_reason": ";".join(missing_reasons),
        "top1_iou_recorded": top1_recorded,
        "official_iou_recomputed": official,
        "axis_aligned_iou_ignore_yaw": axis_iou,
        "rotation_penalty": penalty,
        "rotation_penalty_ratio": penalty_ratio,
        "yaw_raw": yaw_raw,
        "yaw_abs": yaw_abs,
        "yaw_mod_pi": yaw_mod_pi,
        "yaw_mod_pi_over_2": yaw_mod_pi_over_2,
        "yaw_to_axis_angle_rad": yaw_axis_rad,
        "yaw_to_axis_angle_deg": yaw_axis_deg,
        "yaw_bin": yaw_bin(yaw_axis_deg),
        "gt_center_x": safe_float(gt_box[0]) if gt_box is not None else math.nan,
        "gt_center_y": safe_float(gt_box[1]) if gt_box is not None else math.nan,
        "gt_center_z": safe_float(gt_box[2]) if gt_box is not None else math.nan,
        "gt_size_x": safe_float(gt_box[3]) if gt_box is not None else math.nan,
        "gt_size_y": safe_float(gt_box[4]) if gt_box is not None else math.nan,
        "gt_size_z": safe_float(gt_box[5]) if gt_box is not None else math.nan,
        "gt_yaw": yaw_raw,
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
        "top10_best_iou_recorded": top10_best,
        "top1_to_best_top10_gap": top1_to_best_gap,
    }
    out["rotation_interpretation"] = rotation_interpretation(
        official=official,
        penalty=penalty,
        yaw_deg=yaw_axis_deg,
        missing=out["missing_reason"],
    )
    return out


SUMMARY_FIELDS = [
    "scene",
    "primary_label",
    "yaw_bin",
    "num_records",
    "mean_top1_iou_recorded",
    "median_top1_iou_recorded",
    "mean_axis_aligned_iou_ignore_yaw",
    "median_axis_aligned_iou_ignore_yaw",
    "mean_rotation_penalty",
    "median_rotation_penalty",
    "mean_rotation_penalty_ratio",
    "median_rotation_penalty_ratio",
    "mean_yaw_to_axis_angle_deg",
    "median_yaw_to_axis_angle_deg",
    "ratio_yaw_gt_15",
    "ratio_yaw_gt_25",
    "ratio_yaw_gt_35",
    "ratio_rotation_penalty_gt_0_1",
    "ratio_rotation_penalty_gt_0_2",
    "ratio_rotation_penalty_gt_0_3",
    "E_ratio",
    "C_ratio",
    "A_ratio",
]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_records": len(rows),
        "mean_top1_iou_recorded": mean([row["top1_iou_recorded"] for row in rows]),
        "median_top1_iou_recorded": median([row["top1_iou_recorded"] for row in rows]),
        "mean_axis_aligned_iou_ignore_yaw": mean([row["axis_aligned_iou_ignore_yaw"] for row in rows]),
        "median_axis_aligned_iou_ignore_yaw": median([row["axis_aligned_iou_ignore_yaw"] for row in rows]),
        "mean_rotation_penalty": mean([row["rotation_penalty"] for row in rows]),
        "median_rotation_penalty": median([row["rotation_penalty"] for row in rows]),
        "mean_rotation_penalty_ratio": mean([row["rotation_penalty_ratio"] for row in rows]),
        "median_rotation_penalty_ratio": median([row["rotation_penalty_ratio"] for row in rows]),
        "mean_yaw_to_axis_angle_deg": mean([row["yaw_to_axis_angle_deg"] for row in rows]),
        "median_yaw_to_axis_angle_deg": median([row["yaw_to_axis_angle_deg"] for row in rows]),
        "ratio_yaw_gt_15": ratio(rows, "yaw_to_axis_angle_deg", 15.0),
        "ratio_yaw_gt_25": ratio(rows, "yaw_to_axis_angle_deg", 25.0),
        "ratio_yaw_gt_35": ratio(rows, "yaw_to_axis_angle_deg", 35.0),
        "ratio_rotation_penalty_gt_0_1": ratio(rows, "rotation_penalty", 0.1),
        "ratio_rotation_penalty_gt_0_2": ratio(rows, "rotation_penalty", 0.2),
        "ratio_rotation_penalty_gt_0_3": ratio(rows, "rotation_penalty", 0.3),
        "E_ratio": label_ratio(rows, "E"),
        "C_ratio": label_ratio(rows, "C"),
        "A_ratio": label_ratio(rows, "A"),
    }


def group_by(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def build_scene_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (scene,), group in sorted(group_by(rows, ("scene",)).items()):
        summary = summarize_rows(group)
        summary.update({"scene": scene})
        output.append(summary)
    return output


def build_scene_label_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (scene, label), group in sorted(group_by(rows, ("scene", "primary_label")).items()):
        summary = summarize_rows(group)
        summary.update({"scene": scene, "primary_label": label})
        output.append(summary)
    return output


def build_yaw_bin_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    labels_by_scene = defaultdict(set)
    for row in rows:
        labels_by_scene[row["scene"]].add(row["primary_label"])
    grouped = group_by(rows, ("scene", "primary_label", "yaw_bin"))
    for scene in sorted(labels_by_scene):
        for label in sorted(labels_by_scene[scene]):
            for ybin in YAW_BINS:
                group = grouped.get((scene, label, ybin), [])
                summary = summarize_rows(group)
                summary.update({"scene": scene, "primary_label": label, "yaw_bin": ybin})
                output.append(summary)
    return output


RECORD_FIELDS = [
    "id",
    "scene",
    "primary_label",
    "utterance",
    "json_path",
    "pred_box_source",
    "missing_reason",
    "top1_iou_recorded",
    "official_iou_recomputed",
    "axis_aligned_iou_ignore_yaw",
    "rotation_penalty",
    "rotation_penalty_ratio",
    "yaw_raw",
    "yaw_abs",
    "yaw_mod_pi",
    "yaw_mod_pi_over_2",
    "yaw_to_axis_angle_rad",
    "yaw_to_axis_angle_deg",
    "yaw_bin",
    "gt_center_x",
    "gt_center_y",
    "gt_center_z",
    "gt_size_x",
    "gt_size_y",
    "gt_size_z",
    "gt_yaw",
    "pred_center_x",
    "pred_center_y",
    "pred_center_z",
    "pred_size_x",
    "pred_size_y",
    "pred_size_z",
    "center_error",
    "size_l1_error",
    "gt_volume",
    "pred_volume",
    "volume_ratio",
    "top10_best_iou_recorded",
    "top1_to_best_top10_gap",
    "rotation_interpretation",
]


HIGH_CASE_FIELDS = [
    "id",
    "top1_iou_recorded",
    "axis_aligned_iou_ignore_yaw",
    "rotation_penalty",
    "rotation_penalty_ratio",
    "yaw_to_axis_angle_deg",
    "utterance",
    "json_path",
]


def select_parking2_c(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["scene"] == PARKING2 and row["primary_label"] == PARKING2_C
    ]


def rows_for(rows: list[dict[str, Any]], scene: str, label: str | None = None) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if row["scene"] == scene and (label is None or row["primary_label"] == label)
    ]


def summary_line(name: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"{name}: no records"
    return (
        f"{name}: n={len(rows)}, "
        f"official={fmt(mean([r['top1_iou_recorded'] for r in rows]))}, "
        f"ignore_yaw={fmt(mean([r['axis_aligned_iou_ignore_yaw'] for r in rows]))}, "
        f"penalty={fmt(mean([r['rotation_penalty'] for r in rows]))}, "
        f"penalty_ratio={fmt(mean([r['rotation_penalty_ratio'] for r in rows]))}, "
        f"yaw_axis_deg={fmt(mean([r['yaw_to_axis_angle_deg'] for r in rows]))}"
    )


def build_findings(args: argparse.Namespace, rows: list[dict[str, Any]], missing_counts: Counter[str]) -> list[str]:
    parking_rows = rows_for(rows, PARKING2)
    plaza_rows = rows_for(rows, PLAZA)
    parking_c = rows_for(rows, PARKING2, PARKING2_C)
    parking_e = rows_for(rows, PARKING2, STRICT_E)
    plaza_c = rows_for(rows, PLAZA, PARKING2_C)
    plaza_e = rows_for(rows, PLAZA, STRICT_E)

    parking_c_penalty_gt_01 = ratio(parking_c, "rotation_penalty", 0.1)
    parking_c_penalty_gt_02 = ratio(parking_c, "rotation_penalty", 0.2)
    parking_c_penalty_gt_03 = ratio(parking_c, "rotation_penalty", 0.3)
    parking_c_yaw_gt_15 = ratio(parking_c, "yaw_to_axis_angle_deg", 15.0)
    parking_c_yaw_gt_25 = ratio(parking_c, "yaw_to_axis_angle_deg", 25.0)
    parking_c_yaw_gt_35 = ratio(parking_c, "yaw_to_axis_angle_deg", 35.0)
    parking_c_mean_penalty = mean([row["rotation_penalty"] for row in parking_c])
    parking_c_ratio_penalty = mean([row["rotation_penalty_ratio"] for row in parking_c])

    if parking_c and (parking_c_mean_penalty > 0.1 or parking_c_penalty_gt_02 > 0.25):
        decision = (
            "parking_2 C 类存在明显 rotation mismatch：ignore-yaw IoU 明显高于 official IoU，"
            "后续应优先考虑 heading-aware / rotated box refinement。"
        )
    else:
        decision = (
            "parking_2 C 类 rotation penalty 不足以单独解释低 IoU，"
            "后续仍需继续分析 center/size/candidate/ranking。"
        )

    lines = [
        "parking_2 rotation mismatch findings",
        "",
        f"Input all_records_csv: {args.all_records_csv}",
        f"Output records: {len(rows)}",
        f"Scenes: {', '.join(SCENES)}",
        f"Official IoU import_status: {OFFICIAL_IMPORT_STATUS}",
        f"Missing counts: {dict(missing_counts)}",
        "rotation_penalty_ratio is computed only when axis_aligned_iou_ignore_yaw > eps; otherwise it is NaN.",
        "",
        "Scene-level comparison:",
        f"  {summary_line('parking_2', parking_rows)}",
        f"  {summary_line('plaza', plaza_rows)}",
        "",
        "parking_2 vs plaza yaw_to_axis_angle_deg:",
        f"  parking_2 mean/median={fmt(mean([r['yaw_to_axis_angle_deg'] for r in parking_rows]))}/{fmt(median([r['yaw_to_axis_angle_deg'] for r in parking_rows]))}",
        f"  plaza mean/median={fmt(mean([r['yaw_to_axis_angle_deg'] for r in plaza_rows]))}/{fmt(median([r['yaw_to_axis_angle_deg'] for r in plaza_rows]))}",
        "",
        "parking_2 vs plaza rotation_penalty:",
        f"  parking_2 mean/median={fmt(mean([r['rotation_penalty'] for r in parking_rows]))}/{fmt(median([r['rotation_penalty'] for r in parking_rows]))}",
        f"  plaza mean/median={fmt(mean([r['rotation_penalty'] for r in plaza_rows]))}/{fmt(median([r['rotation_penalty'] for r in plaza_rows]))}",
        "",
        "parking_2 C details:",
        f"  num_records={len(parking_c)}",
        f"  mean official/top1 IoU={fmt(mean([r['top1_iou_recorded'] for r in parking_c]))}",
        f"  mean axis_aligned_iou_ignore_yaw={fmt(mean([r['axis_aligned_iou_ignore_yaw'] for r in parking_c]))}",
        f"  mean rotation_penalty={fmt(parking_c_mean_penalty)}",
        f"  mean rotation_penalty_ratio={fmt(parking_c_ratio_penalty)}",
        f"  ratio rotation_penalty > 0.1: {fmt(parking_c_penalty_gt_01)}",
        f"  ratio rotation_penalty > 0.2: {fmt(parking_c_penalty_gt_02)}",
        f"  ratio rotation_penalty > 0.3: {fmt(parking_c_penalty_gt_03)}",
        f"  ratio yaw_to_axis_angle_deg > 15: {fmt(parking_c_yaw_gt_15)}",
        f"  ratio yaw_to_axis_angle_deg > 25: {fmt(parking_c_yaw_gt_25)}",
        f"  ratio yaw_to_axis_angle_deg > 35: {fmt(parking_c_yaw_gt_35)}",
        "",
        "C/E comparison:",
        f"  {summary_line('parking_2 C', parking_c)}",
        f"  {summary_line('parking_2 E', parking_e)}",
        f"  {summary_line('plaza C', plaza_c)}",
        f"  {summary_line('plaza E', plaza_e)}",
        "",
        "Interpretation:",
        f"  {decision}",
        "  由于 official IoU 是 rotated GT vs aligned prediction，普通 axis-aligned center/size decomposition 不能直接解释 Acc@0.5。",
    ]
    if parking_c and (parking_c_mean_penalty > 0.1 or parking_c_penalty_gt_02 > 0.25):
        lines.append("  建议后续方向从普通 center/size BoxRefine 调整为 heading-aware / rotated box refinement。")
    else:
        lines.append("  建议继续分析 center/size/candidate/ranking，并谨慎看待 rotation 对单样本的局部影响。")
    return lines


def main() -> None:
    args = parse_args()
    all_records_path = repo_path(args.all_records_csv)
    out_dir = repo_path(args.out_dir)
    all_rows = read_csv(all_records_path)
    selected = [
        row for row in all_rows
        if row.get("platform") == "drone" and row.get("scene") in SCENES
    ]
    if not selected:
        raise ValueError(f"No Drone records for scenes {SCENES} in {all_records_path}")

    records = [analyze_record(row) for row in selected]
    missing_counts: Counter[str] = Counter()
    for record in records:
        if record["missing_reason"]:
            for reason in record["missing_reason"].split(";"):
                missing_counts[reason] += 1

    scene_summary = build_scene_summary(records)
    scene_label_summary = build_scene_label_summary(records)
    yaw_bin_summary = build_yaw_bin_summary(records)
    parking2_c = sorted(select_parking2_c(records), key=lambda row: safe_float(row["rotation_penalty"], -999.0), reverse=True)
    high_penalty = parking2_c[:50]
    low_penalty_low_iou = [
        row for row in parking2_c
        if safe_float(row["rotation_penalty"]) < 0.05 and safe_float(row["top1_iou_recorded"]) < 0.35
    ]

    write_csv(out_dir / "rotation_mismatch_records.csv", records, RECORD_FIELDS)
    write_csv(out_dir / "rotation_summary_by_scene.csv", scene_summary, [field for field in SUMMARY_FIELDS if field not in {"primary_label", "yaw_bin", "E_ratio", "C_ratio", "A_ratio"}])
    write_csv(out_dir / "rotation_summary_by_scene_label.csv", scene_label_summary, [field for field in SUMMARY_FIELDS if field != "yaw_bin"])
    write_csv(out_dir / "rotation_summary_by_yaw_bin.csv", yaw_bin_summary, SUMMARY_FIELDS)
    write_csv(out_dir / "parking2_c_rotation_records.csv", parking2_c, RECORD_FIELDS)
    write_csv(out_dir / "parking2_c_high_rotation_penalty_cases.csv", high_penalty, HIGH_CASE_FIELDS)
    write_csv(out_dir / "parking2_c_low_rotation_penalty_low_iou_cases.csv", low_penalty_low_iou, HIGH_CASE_FIELDS)
    write_text(out_dir / "parking2_rotation_findings.txt", build_findings(args, records, missing_counts))

    print(f"Saved parking2 rotation mismatch analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
