"""Analyze why Outdoor_Day_penno_parking_2 fails in baseline Drone predictions.

This offline diagnostic compares Outdoor_Day_penno_parking_2 with
Outdoor_Day_penno_plaza using precomputed all_records.csv rows plus each
sample's prediction.json. It does not load models or modify evaluation output.

Example:
    python tools/analyze_scene_parking2_failure.py
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


DEFAULT_ALL_RECORDS_CSV = "outputs/diagnostic_baseline20_dq/all_records.csv"
DEFAULT_OUT_DIR = "outputs/scene_parking2_analysis"
PARKING2_SCENE = "Outdoor_Day_penno_parking_2"
PLAZA_SCENE = "Outdoor_Day_penno_plaza"
EPS = 1e-9

PRIMARY_LABELS = (
    "A_recall_fail_25",
    "B_ranking_fail_25",
    "C_coarse_success_precise_fail",
    "D_ranking_fail_50",
    "E_strict_success",
)

FLAG_BY_LABEL = {
    "A_recall_fail_25": "flag_A_recall_fail_25",
    "B_ranking_fail_25": "flag_B_ranking_fail_25",
    "C_coarse_success_precise_fail": "flag_C_coarse_success_precise_fail",
    "D_ranking_fail_50": "flag_D_ranking_fail_50",
    "E_strict_success": "flag_E_strict_success",
}

SCENE_COMPARE_COLUMNS = (
    "scene",
    "num_records",
    "mean_top1_iou",
    "median_top1_iou",
    "mean_max_top10_iou",
    "median_max_top10_iou",
    "acc25_top1",
    "acc50_top1",
    "acc25_top10",
    "acc50_top10",
    "A_count",
    "A_ratio",
    "B_count",
    "B_ratio",
    "C_count",
    "C_ratio",
    "D_count",
    "D_ratio",
    "E_count",
    "E_ratio",
)

LABEL_BY_SCENE_COLUMNS = (
    "scene",
    "primary_label",
    "count",
    "ratio",
)

GEOMETRY_COLUMNS = (
    "id",
    "scene",
    "frame_id",
    "utterance",
    "primary_label",
    "top1_iou",
    "max_top5_iou",
    "max_top10_iou",
    "acc25_top1",
    "acc50_top1",
    "acc50_top10",
    "json_path",
    "gt_center_x",
    "gt_center_y",
    "gt_center_z",
    "gt_size_x",
    "gt_size_y",
    "gt_size_z",
    "gt_volume",
    "gt_center_distance",
    "top1_center_x",
    "top1_center_y",
    "top1_center_z",
    "top1_size_x",
    "top1_size_y",
    "top1_size_z",
    "top1_volume",
    "top1_center_error",
    "top1_size_l1_error",
    "top1_volume_ratio",
    "best_top10_iou",
    "best_top10_rank",
    "best_top10_center_error",
    "best_top10_size_l1_error",
    "top1_to_best_top10_iou_gap",
    "failure_interpretation",
)

GEOMETRY_SUMMARY_COLUMNS = (
    "scene",
    "num_records",
    "mean_gt_volume",
    "median_gt_volume",
    "mean_gt_center_distance",
    "median_gt_center_distance",
    "mean_top1_center_error",
    "median_top1_center_error",
    "mean_top1_size_l1_error",
    "median_top1_size_l1_error",
    "mean_top1_volume_ratio",
    "median_top1_volume_ratio",
    "mean_best_top10_iou",
    "median_best_top10_iou",
    "mean_top1_to_best_top10_iou_gap",
    "median_top1_to_best_top10_iou_gap",
)

GEOMETRY_LABEL_SUMMARY_COLUMNS = (
    "scene",
    "primary_label",
    "num_records",
    "mean_gt_volume",
    "median_gt_volume",
    "mean_gt_center_distance",
    "median_gt_center_distance",
    "mean_top1_center_error",
    "mean_top1_size_l1_error",
    "mean_best_top10_iou",
    "mean_top1_to_best_top10_iou_gap",
)

REPRESENTATIVE_COLUMNS = (
    "id",
    "scene",
    "frame_id",
    "primary_label",
    "utterance",
    "top1_iou",
    "max_top10_iou",
    "best_top10_iou",
    "best_top10_rank",
    "top1_center_error",
    "top1_size_l1_error",
    "gt_volume",
    "gt_center_distance",
    "top1_to_best_top10_iou_gap",
    "failure_interpretation",
    "json_path",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline scene-level failure analysis for Drone parking_2 vs plaza."
    )
    parser.add_argument("--all_records_csv", default=DEFAULT_ALL_RECORDS_CSV)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--parking_scene", default=PARKING2_SCENE)
    parser.add_argument("--compare_scene", default=PLAZA_SCENE)
    return parser.parse_args()


def safe_float(value, default=math.nan):
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def safe_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def is_finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def finite_values(values):
    return [value for value in values if is_finite(value)]


def mean(values):
    values = finite_values(values)
    return sum(values) / len(values) if values else math.nan


def median(values):
    values = sorted(finite_values(values))
    if not values:
        return math.nan
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def ratio(count, denom):
    return float(count) / float(denom) if denom else math.nan


def safe_l2(values):
    if not all(is_finite(value) for value in values):
        return math.nan
    return math.sqrt(sum(value * value for value in values))


def safe_volume(size):
    if len(size) != 3 or not all(is_finite(value) for value in size):
        return math.nan
    return size[0] * size[1] * size[2]


def center(box):
    return box[:3] if box and len(box) >= 6 else [math.nan, math.nan, math.nan]


def size(box):
    return box[3:6] if box and len(box) >= 6 else [math.nan, math.nan, math.nan]


def center_error(box_a, box_b):
    ca = center(box_a)
    cb = center(box_b)
    if not all(is_finite(value) for value in ca + cb):
        return math.nan
    return safe_l2([ca[i] - cb[i] for i in range(3)])


def size_l1_error(box_a, box_b):
    sa = size(box_a)
    sb = size(box_b)
    if not all(is_finite(value) for value in sa + sb):
        return math.nan
    return sum(abs(sa[i] - sb[i]) for i in range(3))


def parse_box(value):
    if value is None:
        return [math.nan] * 6

    if isinstance(value, dict):
        if "center" in value and "size" in value:
            candidate = list(value.get("center", [])) + list(value.get("size", []))
            return parse_box(candidate)
        for key in ("box", "bbox", "pred_box", "gt_box", "value"):
            if key in value:
                box = parse_box(value[key])
                if valid_box(box):
                    return box
        return [math.nan] * 6

    if isinstance(value, (list, tuple)):
        numeric = [safe_float(item) for item in value[:6]]
        if len(value) >= 6 and all(is_finite(item) for item in numeric):
            return numeric
        for item in value:
            box = parse_box(item)
            if valid_box(box):
                return box

    return [math.nan] * 6


def valid_box(box):
    return isinstance(box, list) and len(box) == 6 and all(is_finite(value) for value in box)


def parse_box_list(value):
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("boxes", "pred_boxes", "top10_pred_boxes", "top10_boxes"):
            if key in value:
                return parse_box_list(value[key])
        return []
    if isinstance(value, (list, tuple)):
        if len(value) >= 6 and all(is_finite(safe_float(item)) for item in value[:6]):
            return [parse_box(value)]
        boxes = []
        for item in value:
            box = parse_box(item)
            if valid_box(box):
                boxes.append(box)
        return boxes
    return []


def first_present(record, keys):
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def read_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_top10_ious(prediction, row):
    raw = None
    if prediction:
        raw = first_present(prediction, ("top10_ious", "topk_ious", "ious"))
    values = []
    if isinstance(raw, (list, tuple)):
        for item in raw:
            number = safe_float(item)
            if is_finite(number):
                values.append(number)
    if values:
        return values
    fallback = safe_float(row.get("max_top10_iou"))
    return [fallback] if is_finite(fallback) else []


def extract_top10_boxes(prediction):
    if not prediction:
        return []
    for key in ("top10_pred_boxes", "top10_pred_bboxes", "top10_boxes", "top10_refined_pred_boxes"):
        boxes = parse_box_list(prediction.get(key))
        if boxes:
            return boxes
    for key in ("pred_boxes", "pred_bboxes", "boxes"):
        boxes = parse_box_list(prediction.get(key))
        if boxes:
            return boxes[:10]
    return []


def extract_gt_box(prediction):
    if not prediction:
        return [math.nan] * 6
    return parse_box(first_present(prediction, ("gt_box", "gt_bbox", "target_box", "target_bbox")))


def extract_top1_box(prediction, top10_boxes):
    if top10_boxes:
        return top10_boxes[0]
    if not prediction:
        return [math.nan] * 6
    for key in ("top1_pred_box", "top1_box", "pred_box", "pred_bbox", "top1_refined_box"):
        box = parse_box(prediction.get(key))
        if valid_box(box):
            return box
    for key in ("pred_boxes", "pred_bboxes", "boxes"):
        boxes = parse_box_list(prediction.get(key))
        if boxes:
            return boxes[0]
    return [math.nan] * 6


def best_top10(top10_ious, top10_boxes):
    values = [(index, value) for index, value in enumerate(top10_ious) if is_finite(value)]
    if not values:
        return math.nan, math.nan, [math.nan] * 6
    best_index, best_iou = max(values, key=lambda item: item[1])
    best_box = top10_boxes[best_index] if best_index < len(top10_boxes) else [math.nan] * 6
    return best_iou, best_index + 1, best_box


def interpret_failure(top1_iou, best_iou, gap, top1_center_err, top1_size_err, volume_ratio):
    parts = []
    if is_finite(best_iou) and best_iou < 0.25:
        parts.append("Top-10 最高 IoU 仍低于 0.25，主要是候选召回失败。")
    elif is_finite(top1_iou) and is_finite(best_iou) and top1_iou < 0.25 and best_iou >= 0.25:
        parts.append("Top-10 有较好候选但 Top-1 未达到 0.25，主要是排序失败。")
    elif is_finite(top1_iou) and 0.25 <= top1_iou < 0.5:
        parts.append("Top-1 已达到 0.25 但未达到 0.5，主要是粗定位成功但精定位不足。")
    elif is_finite(top1_iou) and top1_iou >= 0.5:
        parts.append("Top-1 已达到 0.5，属于严格成功样本。")

    if is_finite(gap) and gap >= 0.25:
        parts.append("Top-1 与 Top-10 最佳候选差距较大，排序问题明显。")
    if is_finite(top1_center_err) and top1_center_err >= 2.0:
        parts.append("Top-1 与 GT center error 大，可能是中心偏移问题。")
    if is_finite(top1_size_err) and top1_size_err >= 2.0:
        parts.append("Top-1 size error 大，可能是尺寸估计问题。")
    if is_finite(volume_ratio) and (volume_ratio < 0.5 or volume_ratio > 2.0):
        parts.append("Top-1 volume ratio 异常，可能是 box size 估计不稳定。")

    return " ".join(parts) if parts else "暂无明确规则解释，建议人工查看。"


def derive_geometry_row(row, prediction, scene):
    top1_iou = safe_float(row.get("top1_iou"))
    max_top5_iou = safe_float(row.get("max_top5_iou"))
    max_top10_iou = safe_float(row.get("max_top10_iou"))
    top10_boxes = extract_top10_boxes(prediction)
    top10_ious = extract_top10_ious(prediction, row)
    gt = extract_gt_box(prediction)
    top1 = extract_top1_box(prediction, top10_boxes)
    best_iou, best_rank, best_box = best_top10(top10_ious, top10_boxes)

    gt_ctr = center(gt)
    gt_sz = size(gt)
    top1_ctr = center(top1)
    top1_sz = size(top1)
    gt_vol = safe_volume(gt_sz)
    top1_vol = safe_volume(top1_sz)
    vol_ratio = top1_vol / gt_vol if is_finite(top1_vol) and is_finite(gt_vol) and abs(gt_vol) > EPS else math.nan
    top1_ctr_err = center_error(top1, gt)
    top1_sz_err = size_l1_error(top1, gt)
    best_ctr_err = center_error(best_box, gt)
    best_sz_err = size_l1_error(best_box, gt)
    gap = best_iou - top1_iou if is_finite(best_iou) and is_finite(top1_iou) else math.nan

    return {
        "id": row.get("id", ""),
        "scene": scene,
        "frame_id": row.get("frame_id", ""),
        "utterance": row.get("utterance", ""),
        "primary_label": row.get("primary_label", ""),
        "top1_iou": top1_iou,
        "max_top5_iou": max_top5_iou,
        "max_top10_iou": max_top10_iou,
        "acc25_top1": safe_float(row.get("acc25_top1")),
        "acc50_top1": safe_float(row.get("acc50_top1")),
        "acc50_top10": safe_float(row.get("acc50_top10")),
        "json_path": row.get("json_path", ""),
        "gt_center_x": gt_ctr[0],
        "gt_center_y": gt_ctr[1],
        "gt_center_z": gt_ctr[2],
        "gt_size_x": gt_sz[0],
        "gt_size_y": gt_sz[1],
        "gt_size_z": gt_sz[2],
        "gt_volume": gt_vol,
        "gt_center_distance": safe_l2(gt_ctr),
        "top1_center_x": top1_ctr[0],
        "top1_center_y": top1_ctr[1],
        "top1_center_z": top1_ctr[2],
        "top1_size_x": top1_sz[0],
        "top1_size_y": top1_sz[1],
        "top1_size_z": top1_sz[2],
        "top1_volume": top1_vol,
        "top1_center_error": top1_ctr_err,
        "top1_size_l1_error": top1_sz_err,
        "top1_volume_ratio": vol_ratio,
        "best_top10_iou": best_iou,
        "best_top10_rank": best_rank,
        "best_top10_center_error": best_ctr_err,
        "best_top10_size_l1_error": best_sz_err,
        "top1_to_best_top10_iou_gap": gap,
        "failure_interpretation": interpret_failure(top1_iou, best_iou, gap, top1_ctr_err, top1_sz_err, vol_ratio),
        "_missing_gt_box": not valid_box(gt),
        "_missing_top1_box": not valid_box(top1),
        "_missing_top10_boxes": not bool(top10_boxes),
        "_missing_top10_ious": not bool(top10_ious),
    }


def acc_value(row, field, fallback_iou_field=None, threshold=None):
    value = safe_float(row.get(field))
    if is_finite(value):
        return value
    if fallback_iou_field is not None and threshold is not None:
        iou = safe_float(row.get(fallback_iou_field))
        if is_finite(iou):
            return 1.0 if iou >= threshold else 0.0
    return math.nan


def summarize_scene(scene, rows):
    n = len(rows)
    summary = {
        "scene": scene,
        "num_records": n,
        "mean_top1_iou": mean(safe_float(row.get("top1_iou")) for row in rows),
        "median_top1_iou": median(safe_float(row.get("top1_iou")) for row in rows),
        "mean_max_top10_iou": mean(safe_float(row.get("max_top10_iou")) for row in rows),
        "median_max_top10_iou": median(safe_float(row.get("max_top10_iou")) for row in rows),
        "acc25_top1": mean(acc_value(row, "acc25_top1", "top1_iou", 0.25) for row in rows),
        "acc50_top1": mean(acc_value(row, "acc50_top1", "top1_iou", 0.5) for row in rows),
        "acc25_top10": mean(acc_value(row, "acc25_top10", "max_top10_iou", 0.25) for row in rows),
        "acc50_top10": mean(acc_value(row, "acc50_top10", "max_top10_iou", 0.5) for row in rows),
    }
    for prefix, label in zip(("A", "B", "C", "D", "E"), PRIMARY_LABELS):
        flag = FLAG_BY_LABEL[label]
        if rows and flag in rows[0]:
            count = sum(1 for row in rows if safe_bool(row.get(flag)))
        else:
            count = sum(1 for row in rows if row.get("primary_label") == label)
        summary[f"{prefix}_count"] = count
        summary[f"{prefix}_ratio"] = ratio(count, n)
    return summary


def label_by_scene(scene, rows):
    n = len(rows)
    counts = Counter(row.get("primary_label", "") for row in rows)
    return [
        {
            "scene": scene,
            "primary_label": label,
            "count": counts.get(label, 0),
            "ratio": ratio(counts.get(label, 0), n),
        }
        for label in PRIMARY_LABELS
    ]


def summarize_geometry(scene, rows):
    return {
        "scene": scene,
        "num_records": len(rows),
        "mean_gt_volume": mean(row["gt_volume"] for row in rows),
        "median_gt_volume": median(row["gt_volume"] for row in rows),
        "mean_gt_center_distance": mean(row["gt_center_distance"] for row in rows),
        "median_gt_center_distance": median(row["gt_center_distance"] for row in rows),
        "mean_top1_center_error": mean(row["top1_center_error"] for row in rows),
        "median_top1_center_error": median(row["top1_center_error"] for row in rows),
        "mean_top1_size_l1_error": mean(row["top1_size_l1_error"] for row in rows),
        "median_top1_size_l1_error": median(row["top1_size_l1_error"] for row in rows),
        "mean_top1_volume_ratio": mean(row["top1_volume_ratio"] for row in rows),
        "median_top1_volume_ratio": median(row["top1_volume_ratio"] for row in rows),
        "mean_best_top10_iou": mean(row["best_top10_iou"] for row in rows),
        "median_best_top10_iou": median(row["best_top10_iou"] for row in rows),
        "mean_top1_to_best_top10_iou_gap": mean(row["top1_to_best_top10_iou_gap"] for row in rows),
        "median_top1_to_best_top10_iou_gap": median(row["top1_to_best_top10_iou_gap"] for row in rows),
    }


def summarize_geometry_label(scene, label, rows):
    return {
        "scene": scene,
        "primary_label": label,
        "num_records": len(rows),
        "mean_gt_volume": mean(row["gt_volume"] for row in rows),
        "median_gt_volume": median(row["gt_volume"] for row in rows),
        "mean_gt_center_distance": mean(row["gt_center_distance"] for row in rows),
        "median_gt_center_distance": median(row["gt_center_distance"] for row in rows),
        "mean_top1_center_error": mean(row["top1_center_error"] for row in rows),
        "mean_top1_size_l1_error": mean(row["top1_size_l1_error"] for row in rows),
        "mean_best_top10_iou": mean(row["best_top10_iou"] for row in rows),
        "mean_top1_to_best_top10_iou_gap": mean(row["top1_to_best_top10_iou_gap"] for row in rows),
    }


def unique_extend(selected, candidates, limit):
    seen = {row["id"] for row in selected}
    for row in candidates:
        if row["id"] in seen:
            continue
        selected.append(row)
        seen.add(row["id"])
        if len(selected) >= limit:
            break
    return selected


def representative_for_label(rows, label, limit):
    rows = [row for row in rows if row.get("primary_label") == label]
    selected = []
    if label == "A_recall_fail_25":
        low = sorted(rows, key=lambda row: (safe_float(row["top1_iou"], 999.0), row["id"]))
        near = sorted(rows, key=lambda row: (abs(safe_float(row["max_top10_iou"], -999.0) - 0.25), row["id"]))
        unique_extend(selected, low, max(1, limit // 2))
        unique_extend(selected, near, limit)
    elif label == "C_coarse_success_precise_fail":
        near25 = sorted(rows, key=lambda row: (abs(safe_float(row["top1_iou"], 999.0) - 0.25), row["id"]))
        near50 = sorted(rows, key=lambda row: (abs(safe_float(row["top1_iou"], 999.0) - 0.5), row["id"]))
        unique_extend(selected, near25, max(1, limit // 2))
        unique_extend(selected, near50, limit)
    elif label == "D_ranking_fail_50":
        gap = sorted(rows, key=lambda row: (-safe_float(row["top1_to_best_top10_iou_gap"], -999.0), row["id"]))
        unique_extend(selected, gap, limit)
    elif label == "E_strict_success":
        high = sorted(rows, key=lambda row: (-safe_float(row["top1_iou"], -999.0), row["id"]))
        unique_extend(selected, high, limit)
    else:
        unique_extend(selected, rows, limit)
    unique_extend(selected, rows, limit)
    return selected[:limit]


def representative_cases(parking_rows):
    selected = []
    for label, limit in (
        ("A_recall_fail_25", 15),
        ("C_coarse_success_precise_fail", 15),
        ("D_ranking_fail_50", 10),
        ("E_strict_success", 10),
    ):
        selected.extend(representative_for_label(parking_rows, label, limit))
    return [{key: row.get(key, "") for key in REPRESENTATIVE_COLUMNS} for row in selected]


def diff_text(metric, parking_summary, plaza_summary):
    p = parking_summary.get(metric, math.nan)
    q = plaza_summary.get(metric, math.nan)
    if not is_finite(p) or not is_finite(q):
        return f"{metric}: parking_2={fmt(p)}, plaza={fmt(q)}, diff=nan"
    return f"{metric}: parking_2={fmt(p)}, plaza={fmt(q)}, diff={fmt(p - q)}"


def fmt(value):
    if not is_finite(value):
        return "nan"
    return f"{value:.4f}"


def missing_counts(rows):
    return {
        "missing_gt_box": sum(1 for row in rows if row.get("_missing_gt_box")),
        "missing_top1_box": sum(1 for row in rows if row.get("_missing_top1_box")),
        "missing_top10_boxes": sum(1 for row in rows if row.get("_missing_top10_boxes")),
        "missing_top10_ious": sum(1 for row in rows if row.get("_missing_top10_ious")),
    }


def build_findings(input_path, parking_scene, plaza_scene, scene_summary_rows, geometry_summary_rows, parking_rows, plaza_rows):
    scene_summary = {row["scene"]: row for row in scene_summary_rows}
    geom_summary = {row["scene"]: row for row in geometry_summary_rows}
    parking = scene_summary[parking_scene]
    plaza = scene_summary[plaza_scene]
    parking_geom = geom_summary[parking_scene]
    plaza_geom = geom_summary[plaza_scene]

    abc = [("A", parking["A_ratio"]), ("C", parking["C_ratio"]), ("D", parking["D_ratio"])]
    dominant = max(abc, key=lambda item: safe_float(item[1], -1.0))[0]
    suggestions = []
    if dominant == "C":
        suggestions.append("parking_2 在 A/C/D 中 C 类占比最高，说明大量样本卡在 0.25 到 0.5 的精定位区间。")
    if safe_float(parking["A_ratio"]) >= 0.25 and (
        safe_float(parking_geom["mean_best_top10_iou"]) < 0.3
        or safe_float(parking["acc50_top10"]) < safe_float(plaza["acc50_top10"]) * 0.5
    ):
        suggestions.append("Top-10 高阈值表现弱，候选召回 / candidate quality 是主要问题之一。")
    if safe_float(parking["C_ratio"]) >= 0.25 and (
        safe_float(parking["C_ratio"]) > safe_float(plaza["C_ratio"]) * 1.5
        or safe_float(parking_geom["median_top1_center_error"]) > safe_float(plaza_geom["median_top1_center_error"]) * 1.2
        or safe_float(parking_geom["median_top1_size_l1_error"]) > safe_float(plaza_geom["median_top1_size_l1_error"]) * 1.2
    ):
        suggestions.append("粗定位成功但 box precision 差，优先检查 localization / BoxRefine。")
    if safe_float(parking["D_ratio"]) >= 0.05 and safe_float(parking_geom["mean_top1_to_best_top10_iou_gap"]) > 0.2:
        suggestions.append("Top-10 内有更好候选但排序没选中，排序/reranking 也需要关注。")
    if safe_float(parking_geom["mean_gt_center_distance"]) > safe_float(plaza_geom["mean_gt_center_distance"]) * 1.2:
        suggestions.append("parking_2 目标距离明显更大，远距离/航拍视角可能是关键因素。")
    if safe_float(parking_geom["mean_gt_volume"]) < safe_float(plaza_geom["mean_gt_volume"]) * 0.8:
        suggestions.append("parking_2 目标体积更小，目标尺度可能是关键因素。")
    if not suggestions:
        suggestions.append("没有单一指标压倒性解释失败，建议结合代表样本做人工可视化复核。")

    parking_missing = missing_counts(parking_rows)
    plaza_missing = missing_counts(plaza_rows)

    lines = [
        "Outdoor_Day_penno_parking_2 failure analysis",
        "",
        f"Input CSV: {input_path}",
        f"parking_2 scene: {parking_scene}, n={parking['num_records']}",
        f"plaza scene: {plaza_scene}, n={plaza['num_records']}",
        "",
        "Accuracy comparison:",
        f"  Acc@0.25 Top1: parking_2={fmt(parking['acc25_top1'])}, plaza={fmt(plaza['acc25_top1'])}",
        f"  Acc@0.5 Top1:  parking_2={fmt(parking['acc50_top1'])}, plaza={fmt(plaza['acc50_top1'])}",
        f"  Acc@0.5 Top10: parking_2={fmt(parking['acc50_top10'])}, plaza={fmt(plaza['acc50_top10'])}",
        "",
        "A/B/C/D/E flag ratios:",
        f"  A recall fail@0.25: parking_2={fmt(parking['A_ratio'])}, plaza={fmt(plaza['A_ratio'])}",
        f"  B ranking fail@0.25: parking_2={fmt(parking['B_ratio'])}, plaza={fmt(plaza['B_ratio'])}",
        f"  C coarse-success precise-fail: parking_2={fmt(parking['C_ratio'])}, plaza={fmt(plaza['C_ratio'])}",
        f"  D ranking fail@0.5: parking_2={fmt(parking['D_ratio'])}, plaza={fmt(plaza['D_ratio'])}",
        f"  E strict success: parking_2={fmt(parking['E_ratio'])}, plaza={fmt(plaza['E_ratio'])}",
        f"  Dominant parking_2 failure among A/C/D: {dominant}",
        "",
        "Geometry / ranking gap comparison:",
        f"  {diff_text('mean_gt_volume', parking_geom, plaza_geom)}",
        f"  {diff_text('mean_gt_center_distance', parking_geom, plaza_geom)}",
        f"  {diff_text('mean_top1_center_error', parking_geom, plaza_geom)}",
        f"  {diff_text('mean_top1_size_l1_error', parking_geom, plaza_geom)}",
        f"  {diff_text('mean_best_top10_iou', parking_geom, plaza_geom)}",
        f"  {diff_text('mean_top1_to_best_top10_iou_gap', parking_geom, plaza_geom)}",
        "",
        "Missing prediction.json fields:",
        f"  parking_2: {parking_missing}",
        f"  plaza: {plaza_missing}",
        "",
        "Initial interpretation:",
    ]
    lines.extend(f"  - {item}" for item in suggestions)
    lines.extend([
        "",
        "Next-step recommendation:",
        "  先看 parking2_representative_cases.csv 中 A/C/D 样本的 prediction.json 和可视化。",
        "  如果 A 与 best_top10_iou 低占主导，优先分析 candidate generation/recall。",
        "  如果 C 与 center/size error 高占主导，优先分析 BoxRefine/localization precision。",
        "  如果 D 与 gap 高占主导，优先分析 ranking/reranking。",
    ])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    input_path = Path(args.all_records_csv)
    out_dir = Path(args.out_dir)
    target_scenes = (args.parking_scene, args.compare_scene)

    rows = read_csv_rows(input_path)
    rows_by_scene = {scene: [row for row in rows if row.get("scene") == scene] for scene in target_scenes}
    if not rows_by_scene[args.parking_scene]:
        raise ValueError(f"No rows found for scene {args.parking_scene!r} in {input_path}")
    if not rows_by_scene[args.compare_scene]:
        raise ValueError(f"No rows found for scene {args.compare_scene!r} in {input_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    scene_summary_rows = [summarize_scene(scene, rows_by_scene[scene]) for scene in target_scenes]
    write_csv(out_dir / "scene_compare_summary.csv", scene_summary_rows, SCENE_COMPARE_COLUMNS)

    label_rows = []
    for scene in target_scenes:
        label_rows.extend(label_by_scene(scene, rows_by_scene[scene]))
    write_csv(out_dir / "label_by_scene.csv", label_rows, LABEL_BY_SCENE_COLUMNS)

    geometry_by_scene = {}
    for scene in target_scenes:
        geometry_rows = []
        for row in rows_by_scene[scene]:
            prediction = read_json(row.get("json_path", ""))
            geometry_rows.append(derive_geometry_row(row, prediction, scene))
        geometry_by_scene[scene] = geometry_rows

    write_csv(out_dir / "parking2_error_geometry.csv", geometry_by_scene[args.parking_scene], GEOMETRY_COLUMNS)
    write_csv(out_dir / "plaza_error_geometry.csv", geometry_by_scene[args.compare_scene], GEOMETRY_COLUMNS)

    geometry_summary_rows = [summarize_geometry(scene, geometry_by_scene[scene]) for scene in target_scenes]
    write_csv(out_dir / "geometry_summary_by_scene.csv", geometry_summary_rows, GEOMETRY_SUMMARY_COLUMNS)

    label_geometry_rows = []
    for scene in target_scenes:
        rows_for_scene = geometry_by_scene[scene]
        for label in PRIMARY_LABELS:
            group = [row for row in rows_for_scene if row.get("primary_label") == label]
            label_geometry_rows.append(summarize_geometry_label(scene, label, group))
    write_csv(out_dir / "geometry_summary_by_scene_label.csv", label_geometry_rows, GEOMETRY_LABEL_SUMMARY_COLUMNS)

    representative_rows = representative_cases(geometry_by_scene[args.parking_scene])
    write_csv(out_dir / "parking2_representative_cases.csv", representative_rows, REPRESENTATIVE_COLUMNS)

    findings = build_findings(
        input_path.as_posix(),
        args.parking_scene,
        args.compare_scene,
        scene_summary_rows,
        geometry_summary_rows,
        geometry_by_scene[args.parking_scene],
        geometry_by_scene[args.compare_scene],
    )
    (out_dir / "parking2_findings.txt").write_text(findings, encoding="utf-8")

    files = [
        "scene_compare_summary.csv",
        "label_by_scene.csv",
        "parking2_error_geometry.csv",
        "plaza_error_geometry.csv",
        "geometry_summary_by_scene.csv",
        "geometry_summary_by_scene_label.csv",
        "parking2_representative_cases.csv",
        "parking2_findings.txt",
    ]
    print(f"Saved scene parking2 analysis to {out_dir.as_posix()}")
    print("Generated files:")
    for name in files:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
