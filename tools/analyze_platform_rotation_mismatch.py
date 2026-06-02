#!/usr/bin/env python3
"""Generic offline analysis for rotated-GT vs aligned-pred IoU loss.

The grounding evaluator records official IoU with rotated GT boxes and aligned
predicted boxes. This script compares that official IoU against an ignore-yaw
axis-aligned IoU approximation for a chosen platform.
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
DEFAULT_PLATFORM = "quad"
DEFAULT_OUT_DIR = "outputs/platform_rotation_analysis_quad"
DRONE_PARKING2 = "Outdoor_Day_penno_parking_2"
PARKING2_C = "C_coarse_success_precise_fail"
STRICT_E = "E_strict_success"
EPS = 1e-9
YAW_BINS = ("0_5", "5_10", "10_15", "15_20", "20_25", "25_30", "30_35", "35_40", "40_45")

try:
    from utils.eval_det import iou3d_rotated_vs_aligned

    OFFICIAL_IMPORT_STATUS = "ok"
except Exception as exc:  # pragma: no cover - reported in findings.
    iou3d_rotated_vs_aligned = None
    OFFICIAL_IMPORT_STATUS = repr(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze platform-level rotation mismatch in prediction exports.")
    parser.add_argument("--all_records_csv", default=DEFAULT_ALL_RECORDS_CSV)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
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
    for key, source in (
        ("top1_pred_box", "top1_pred_box"),
        ("top10_pred_boxes", "top10_pred_boxes[0]"),
        ("pred_boxes", "pred_boxes[0]"),
    ):
        box = first_box(prediction.get(key), min_dims=6)
        if box is not None:
            return box[:6], source
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
    return yaw % math.pi if math.isfinite(yaw) else math.nan


def yaw_to_axis_angle_rad(yaw: float) -> float:
    yaw_mod = normalize_yaw_mod_pi(yaw)
    if not math.isfinite(yaw_mod):
        return math.nan
    return min(abs(yaw_mod), abs(yaw_mod - math.pi / 2.0), abs(yaw_mod - math.pi))


def yaw_bin(angle_deg: float) -> str:
    if not math.isfinite(angle_deg):
        return "missing"
    idx = int(max(0.0, min(45.0, angle_deg)) // 5)
    return YAW_BINS[min(idx, len(YAW_BINS) - 1)]


def box_volume(box: np.ndarray | None) -> float:
    if box is None or box.shape[0] < 6 or np.any(box[3:6] <= 0):
        return math.nan
    return float(np.prod(box[3:6]))


def finite_values(values: list[Any]) -> list[float]:
    return [v for v in (safe_float(value) for value in values) if math.isfinite(v)]


def mean(values: list[Any]) -> float:
    vals = finite_values(values)
    return sum(vals) / len(vals) if vals else math.nan


def median(values: list[Any]) -> float:
    vals = sorted(finite_values(values))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def ratio(rows: list[dict[str, Any]], key: str, threshold: float) -> float:
    vals = finite_values([row.get(key) for row in rows])
    return sum(1 for value in vals if value > threshold) / len(vals) if vals else math.nan


def rotation_interpretation(official: float, penalty: float, yaw_deg: float, missing: str) -> str:
    if missing:
        return "缺少 gt_box 或 pred box，无法判断。"
    if math.isfinite(yaw_deg) and yaw_deg > 25.0 and math.isfinite(penalty) and penalty > 0.15:
        return "GT 朝向明显偏离坐标轴，aligned prediction 在 official IoU 下存在明显旋转惩罚。"
    if math.isfinite(penalty) and penalty > 0.2:
        return "忽略 yaw 时 IoU 明显更高，official IoU 主要受到 rotated GT 与 aligned prediction 不匹配影响。"
    if math.isfinite(penalty) and penalty < 0.05 and math.isfinite(official) and official < 0.35:
        return "低 IoU 不能主要由 rotation mismatch 解释，可能仍是 center/size/candidate quality 问题。"
    return "rotation penalty 有一定影响，但不足以单独解释该样本。"


def analyze_record(row: dict[str, str]) -> dict[str, Any]:
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

    top1_recorded = safe_float(prediction.get("top1_iou", row.get("top1_iou")))
    official = official_iou(gt_box, pred_box)
    axis_iou = axis_aligned_iou_center_size(gt_box, pred_box)
    penalty = axis_iou - official if math.isfinite(axis_iou) and math.isfinite(official) else math.nan
    penalty_ratio = penalty / axis_iou if math.isfinite(penalty) and math.isfinite(axis_iou) and axis_iou > EPS else math.nan

    yaw_raw = safe_float(gt_box[6]) if gt_box is not None and gt_box.shape[0] >= 7 else math.nan
    yaw_axis_rad = yaw_to_axis_angle_rad(yaw_raw)
    yaw_axis_deg = math.degrees(yaw_axis_rad) if math.isfinite(yaw_axis_rad) else math.nan

    gt_volume = box_volume(gt_box)
    pred_volume = box_volume(pred_box)
    center_error = float(np.linalg.norm(gt_box[:3] - pred_box[:3])) if gt_box is not None and pred_box is not None else math.nan
    size_l1_error = float(np.abs(gt_box[3:6] - pred_box[3:6]).sum()) if gt_box is not None and pred_box is not None else math.nan
    volume_ratio = pred_volume / gt_volume if math.isfinite(gt_volume) and gt_volume > EPS and math.isfinite(pred_volume) else math.nan

    top10_arr = numeric_array(prediction.get("top10_ious"))
    top10_values = finite_values(top10_arr.reshape(-1).tolist()) if top10_arr is not None else []
    top10_best = max(top10_values, default=safe_float(row.get("max_top10_iou")))
    top1_gap = top10_best - top1_recorded if math.isfinite(top10_best) and math.isfinite(top1_recorded) else math.nan

    out = {
        "id": row.get("id", prediction.get("id", "")),
        "platform": row.get("platform", prediction.get("platform", "")),
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
        "top1_to_best_top10_gap": top1_gap,
    }
    out["rotation_interpretation"] = rotation_interpretation(
        official=official,
        penalty=penalty,
        yaw_deg=yaw_axis_deg,
        missing=out["missing_reason"],
    )
    return out


RECORD_FIELDS = [
    "id", "platform", "scene", "primary_label", "utterance", "json_path", "pred_box_source", "missing_reason",
    "top1_iou_recorded", "official_iou_recomputed", "axis_aligned_iou_ignore_yaw", "rotation_penalty",
    "rotation_penalty_ratio", "yaw_raw", "yaw_to_axis_angle_deg", "yaw_bin",
    "gt_center_x", "gt_center_y", "gt_center_z", "gt_size_x", "gt_size_y", "gt_size_z", "gt_yaw",
    "pred_center_x", "pred_center_y", "pred_center_z", "pred_size_x", "pred_size_y", "pred_size_z",
    "center_error", "size_l1_error", "gt_volume", "pred_volume", "volume_ratio",
    "top10_best_iou_recorded", "top1_to_best_top10_gap", "rotation_interpretation",
]

SUMMARY_METRIC_FIELDS = [
    "num_records", "mean_top1_iou_recorded", "median_top1_iou_recorded",
    "mean_axis_aligned_iou_ignore_yaw", "median_axis_aligned_iou_ignore_yaw",
    "mean_rotation_penalty", "median_rotation_penalty",
    "mean_rotation_penalty_ratio", "median_rotation_penalty_ratio",
    "mean_yaw_to_axis_angle_deg", "median_yaw_to_axis_angle_deg",
    "ratio_yaw_gt_15", "ratio_yaw_gt_25", "ratio_yaw_gt_35",
    "ratio_rotation_penalty_gt_0_1", "ratio_rotation_penalty_gt_0_2", "ratio_rotation_penalty_gt_0_3",
]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    }


def group_by(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    return grouped


def build_scene_summary(platform: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (scene,), group in sorted(group_by(rows, ("scene",)).items()):
        item = {"platform": platform, "scene": scene}
        item.update(summarize(group))
        output.append(item)
    return output


def build_scene_label_summary(platform: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (scene, label), group in sorted(group_by(rows, ("scene", "primary_label")).items()):
        item = {"platform": platform, "scene": scene, "primary_label": label}
        item.update(summarize(group))
        output.append(item)
    return output


def build_label_summary(platform: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for (label,), group in sorted(group_by(rows, ("primary_label",)).items()):
        item = {"platform": platform, "primary_label": label}
        item.update(summarize(group))
        output.append(item)
    return output


def rows_for(rows: list[dict[str, Any]], **filters: str) -> list[dict[str, Any]]:
    return [row for row in rows if all(str(row.get(key, "")) == value for key, value in filters.items())]


def summary_line(name: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return f"{name}: no records"
    return (
        f"{name}: n={len(rows)}, official={fmt(mean([r['top1_iou_recorded'] for r in rows]))}, "
        f"ignore_yaw={fmt(mean([r['axis_aligned_iou_ignore_yaw'] for r in rows]))}, "
        f"penalty={fmt(mean([r['rotation_penalty'] for r in rows]))}, "
        f"penalty_ratio={fmt(mean([r['rotation_penalty_ratio'] for r in rows]))}, "
        f"yaw_axis_deg={fmt(mean([r['yaw_to_axis_angle_deg'] for r in rows]))}"
    )


def scene_counts(rows: list[dict[str, Any]]) -> str:
    counts = Counter(str(row.get("scene", "")) for row in rows)
    return ", ".join(f"{scene}:{count}" for scene, count in counts.most_common())


def build_findings(
    args: argparse.Namespace,
    platform_rows: list[dict[str, Any]],
    label_summary: list[dict[str, Any]],
    scene_summary: list[dict[str, Any]],
    drone_parking_rows: list[dict[str, Any]],
    missing_counts: Counter[str],
) -> list[str]:
    platform = args.platform
    c_rows = rows_for(platform_rows, primary_label=PARKING2_C)
    e_rows = rows_for(platform_rows, primary_label=STRICT_E)
    drone_parking_c = rows_for(drone_parking_rows, primary_label=PARKING2_C)
    overall_penalty = mean([row["rotation_penalty"] for row in platform_rows])
    c_penalty = mean([row["rotation_penalty"] for row in c_rows])
    e_penalty = mean([row["rotation_penalty"] for row in e_rows])
    drone_parking_penalty = mean([row["rotation_penalty"] for row in drone_parking_rows])
    drone_parking_c_penalty = mean([row["rotation_penalty"] for row in drone_parking_c])

    lines = [
        "platform rotation mismatch findings",
        "",
        f"Input all_records_csv: {args.all_records_csv}",
        f"Platform: {platform}",
        f"Total platform records: {len(platform_rows)}",
        f"Scene counts: {scene_counts(platform_rows)}",
        f"Official IoU import_status: {OFFICIAL_IMPORT_STATUS}",
        f"Missing counts: {dict(missing_counts)}",
        "rotation_penalty_ratio is computed only when axis_aligned_iou_ignore_yaw > eps; otherwise it is NaN.",
        "",
        "Overall:",
        f"  {summary_line(platform, platform_rows)}",
        "",
        "Primary-label comparison:",
    ]
    for item in label_summary:
        lines.append(
            f"  {item['primary_label']}: n={item['num_records']}, "
            f"official={fmt(item['mean_top1_iou_recorded'])}, "
            f"ignore_yaw={fmt(item['mean_axis_aligned_iou_ignore_yaw'])}, "
            f"penalty={fmt(item['mean_rotation_penalty'])}, "
            f"penalty_ratio={fmt(item['mean_rotation_penalty_ratio'])}, "
            f"yaw_axis_deg={fmt(item['mean_yaw_to_axis_angle_deg'])}"
        )
    lines.append("")
    lines.append("Scene comparison:")
    for item in scene_summary:
        lines.append(
            f"  {item['scene']}: n={item['num_records']}, "
            f"official={fmt(item['mean_top1_iou_recorded'])}, "
            f"ignore_yaw={fmt(item['mean_axis_aligned_iou_ignore_yaw'])}, "
            f"penalty={fmt(item['mean_rotation_penalty'])}, "
            f"penalty_ratio={fmt(item['mean_rotation_penalty_ratio'])}"
        )

    lines.extend([
        "",
        "C/E comparison:",
        f"  {summary_line(f'{platform} C', c_rows)}",
        f"  {summary_line(f'{platform} E', e_rows)}",
        "",
        "Drone parking_2 reference:",
        f"  {summary_line('drone parking_2', drone_parking_rows)}",
        f"  {summary_line('drone parking_2 C', drone_parking_c)}",
        "",
        "Interpretation:",
    ])

    if math.isfinite(c_penalty) and c_penalty > 0.15:
        lines.append(f"  {platform} C 类也存在明显 rotation mismatch。")
    else:
        lines.append(f"  {platform} C 类 rotation mismatch 不算强，低 IoU 更可能来自 center/size/candidate/ranking。")

    if math.isfinite(c_penalty) and math.isfinite(e_penalty) and c_penalty > e_penalty + 0.05:
        lines.append("  C 类 rotation penalty 明显高于 E 类，rotation mismatch 对失败类别有区分作用。")
    elif math.isfinite(e_penalty):
        lines.append("  E 类并不只是因为 rotation penalty 更小才成功，仍可能有候选质量、场景清晰度或距离因素。")

    if math.isfinite(overall_penalty) and math.isfinite(drone_parking_penalty):
        if overall_penalty < drone_parking_penalty:
            lines.append("  该平台整体 rotation penalty 低于 Drone parking_2，Drone parking_2 的旋转框问题更严重。")
        else:
            lines.append("  该平台整体 rotation penalty 不低于 Drone parking_2；若指标仍较好，说明还有目标距离、相似目标数量、候选质量等因素在帮助该平台。")
    if math.isfinite(c_penalty) and math.isfinite(drone_parking_c_penalty):
        if c_penalty < drone_parking_c_penalty:
            lines.append("  该平台 C 类 rotation penalty 低于 Drone parking_2 C。")
        else:
            lines.append("  该平台 C 类 rotation penalty 接近或高于 Drone parking_2 C，需要结合场景和候选召回继续看。")
    return lines


def main() -> None:
    args = parse_args()
    all_rows = read_csv(repo_path(args.all_records_csv))
    platform_rows_raw = [row for row in all_rows if row.get("platform") == args.platform]
    if not platform_rows_raw:
        raise ValueError(f"No records for platform={args.platform!r} in {args.all_records_csv}")

    platform_rows = [analyze_record(row) for row in platform_rows_raw]
    drone_parking_raw = [row for row in all_rows if row.get("platform") == "drone" and row.get("scene") == DRONE_PARKING2]
    drone_parking_rows = [analyze_record(row) for row in drone_parking_raw]

    missing_counts: Counter[str] = Counter()
    for record in platform_rows:
        for reason in str(record.get("missing_reason", "")).split(";"):
            if reason:
                missing_counts[reason] += 1

    scene_summary = build_scene_summary(args.platform, platform_rows)
    scene_label_summary = build_scene_label_summary(args.platform, platform_rows)
    label_summary = build_label_summary(args.platform, platform_rows)
    high_penalty = sorted(platform_rows, key=lambda row: safe_float(row["rotation_penalty"], -999.0), reverse=True)[:100]
    low_penalty_low_iou = [
        row for row in platform_rows
        if safe_float(row["rotation_penalty"]) < 0.05 and safe_float(row["top1_iou_recorded"]) < 0.35
    ]

    out_dir = repo_path(args.out_dir)
    write_csv(out_dir / "rotation_mismatch_records.csv", platform_rows, RECORD_FIELDS)
    write_csv(out_dir / "rotation_summary_by_scene.csv", scene_summary, ["platform", "scene"] + SUMMARY_METRIC_FIELDS)
    write_csv(out_dir / "rotation_summary_by_scene_label.csv", scene_label_summary, ["platform", "scene", "primary_label"] + SUMMARY_METRIC_FIELDS)
    label_fields = [
        "platform", "primary_label", "num_records", "mean_top1_iou_recorded",
        "mean_axis_aligned_iou_ignore_yaw", "mean_rotation_penalty",
        "mean_rotation_penalty_ratio", "mean_yaw_to_axis_angle_deg",
        "ratio_rotation_penalty_gt_0_1", "ratio_rotation_penalty_gt_0_2", "ratio_rotation_penalty_gt_0_3",
    ]
    write_csv(out_dir / "rotation_summary_by_label.csv", label_summary, label_fields)
    case_fields = [
        "id", "scene", "primary_label", "top1_iou_recorded", "axis_aligned_iou_ignore_yaw",
        "rotation_penalty", "rotation_penalty_ratio", "yaw_to_axis_angle_deg", "utterance", "json_path",
    ]
    write_csv(out_dir / "high_rotation_penalty_cases.csv", high_penalty, case_fields)
    write_csv(out_dir / "low_rotation_penalty_low_iou_cases.csv", low_penalty_low_iou, case_fields)
    write_text(
        out_dir / "platform_rotation_findings.txt",
        build_findings(args, platform_rows, label_summary, scene_summary, drone_parking_rows, missing_counts),
    )

    print(f"Saved platform rotation mismatch analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
