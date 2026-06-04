"""Decompose parking_2 C-class IoU into axis overlap and oracle corrections.

This offline tool reads parking2_c_subtype.csv and recomputes axis-aligned
center+size interval IoU. It compares that IoU with the stored top1_iou,
computes per-axis overlap/boundary errors, and evaluates center/size oracle
corrections. It does not use GPU, checkpoints, or model code.

Example:
    python tools/analyze_parking2_c_iou_decomposition.py
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_PARKING2_C_CSV = "outputs/scene_parking2_c_detail_analysis/parking2_c_subtype.csv"
DEFAULT_OUT_DIR = "outputs/scene_parking2_c_iou_decomposition"
EPS = 1e-9
IOU_MISMATCH_THRESHOLD = 1e-3

REQUIRED_COLUMNS = (
    "id",
    "scene",
    "frame_id",
    "utterance",
    "top1_iou",
    "best_top10_iou",
    "best_top10_rank",
    "top1_to_best_top10_iou_gap",
    "c_subtype",
    "json_path",
    "gt_center_x",
    "gt_center_y",
    "gt_center_z",
    "gt_size_x",
    "gt_size_y",
    "gt_size_z",
    "gt_volume",
    "top1_center_x",
    "top1_center_y",
    "top1_center_z",
    "top1_size_x",
    "top1_size_y",
    "top1_size_z",
    "top1_volume",
)

DECOMP_COLUMNS = (
    "id",
    "scene",
    "frame_id",
    "utterance",
    "top1_iou",
    "best_top10_iou",
    "best_top10_rank",
    "top1_to_best_top10_iou_gap",
    "c_subtype",
    "json_path",
    "gt_center_x",
    "gt_center_y",
    "gt_center_z",
    "gt_size_x",
    "gt_size_y",
    "gt_size_z",
    "gt_volume",
    "top1_center_x",
    "top1_center_y",
    "top1_center_z",
    "top1_size_x",
    "top1_size_y",
    "top1_size_z",
    "top1_volume",
    "gt_x1",
    "gt_y1",
    "gt_z1",
    "gt_x2",
    "gt_y2",
    "gt_z2",
    "top1_x1",
    "top1_y1",
    "top1_z1",
    "top1_x2",
    "top1_y2",
    "top1_z2",
    "inter_x",
    "inter_y",
    "inter_z",
    "union_x",
    "union_y",
    "union_z",
    "overlap_ratio_x",
    "overlap_ratio_y",
    "overlap_ratio_z",
    "sym_overlap_x",
    "sym_overlap_y",
    "sym_overlap_z",
    "intersection_volume",
    "gt_volume_check",
    "top1_volume_check",
    "union_volume",
    "iou_recomputed",
    "iou_abs_diff",
    "iou_mismatch",
    "boundary_error_x1",
    "boundary_error_x2",
    "boundary_error_y1",
    "boundary_error_y2",
    "boundary_error_z1",
    "boundary_error_z2",
    "axis_boundary_error_x",
    "axis_boundary_error_y",
    "axis_boundary_error_z",
    "dominant_bad_axis",
    "dominant_boundary_axis",
    "oracle_center_iou",
    "oracle_size_iou",
    "oracle_center_size_iou",
    "oracle_center_pass_05",
    "oracle_size_pass_05",
    "oracle_center_size_pass_05",
    "oracle_center_gain",
    "oracle_size_gain",
    "oracle_center_size_gain",
    "oracle_best_type",
    "failure_decomposition",
)

SUMMARY_COLUMNS = (
    "num_samples",
    "mean_top1_iou",
    "median_top1_iou",
    "mean_iou_recomputed",
    "median_iou_recomputed",
    "num_iou_mismatch",
    "ratio_iou_mismatch",
    "mean_overlap_ratio_x",
    "median_overlap_ratio_x",
    "mean_overlap_ratio_y",
    "median_overlap_ratio_y",
    "mean_overlap_ratio_z",
    "median_overlap_ratio_z",
    "dominant_bad_axis_x_count",
    "dominant_bad_axis_y_count",
    "dominant_bad_axis_z_count",
    "dominant_bad_axis_x_ratio",
    "dominant_bad_axis_y_ratio",
    "dominant_bad_axis_z_ratio",
    "mean_axis_boundary_error_x",
    "median_axis_boundary_error_x",
    "mean_axis_boundary_error_y",
    "median_axis_boundary_error_y",
    "mean_axis_boundary_error_z",
    "median_axis_boundary_error_z",
    "mean_oracle_center_iou",
    "median_oracle_center_iou",
    "mean_oracle_size_iou",
    "median_oracle_size_iou",
    "mean_oracle_center_size_iou",
    "median_oracle_center_size_iou",
    "num_oracle_center_pass_05",
    "ratio_oracle_center_pass_05",
    "num_oracle_size_pass_05",
    "ratio_oracle_size_pass_05",
    "num_oracle_center_size_pass_05",
    "ratio_oracle_center_size_pass_05",
    "mean_oracle_center_gain",
    "median_oracle_center_gain",
    "mean_oracle_size_gain",
    "median_oracle_size_gain",
    "mean_oracle_center_size_gain",
    "median_oracle_center_size_gain",
)

BY_SUBTYPE_COLUMNS = (
    "c_subtype",
    "num_samples",
    "mean_top1_iou",
    "mean_overlap_ratio_x",
    "mean_overlap_ratio_y",
    "mean_overlap_ratio_z",
    "dominant_bad_axis_x_ratio",
    "dominant_bad_axis_y_ratio",
    "dominant_bad_axis_z_ratio",
    "mean_oracle_center_iou",
    "mean_oracle_size_iou",
    "mean_oracle_center_size_iou",
    "ratio_oracle_center_pass_05",
    "ratio_oracle_size_pass_05",
    "ratio_oracle_center_size_pass_05",
    "mean_oracle_center_gain",
    "mean_oracle_size_gain",
    "mean_oracle_center_size_gain",
)

DOMINANT_AXIS_CASE_COLUMNS = (
    "id",
    "top1_iou",
    "overlap_ratio_x",
    "overlap_ratio_y",
    "overlap_ratio_z",
    "dominant_bad_axis",
    "axis_boundary_error_x",
    "axis_boundary_error_y",
    "axis_boundary_error_z",
    "oracle_center_iou",
    "oracle_size_iou",
    "oracle_center_size_iou",
    "failure_decomposition",
    "utterance",
    "json_path",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Parking_2 C-class IoU decomposition and oracle analysis.")
    parser.add_argument("--parking2_c_csv", default=DEFAULT_PARKING2_C_CSV)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--iou_mismatch_threshold", type=float, default=IOU_MISMATCH_THRESHOLD)
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


def safe_div(num, den):
    if not is_finite(num) or not is_finite(den) or abs(den) <= EPS:
        return math.nan
    return num / den


def fmt(value):
    if not is_finite(value):
        return "nan"
    return f"{value:.4f}"


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {missing}. "
                "Regenerate parking2_c_subtype.csv with GT/Top1 box fields before running this script."
            )
        rows = []
        for row in reader:
            parsed = dict(row)
            for column in REQUIRED_COLUMNS:
                if column not in {"id", "scene", "frame_id", "utterance", "c_subtype", "json_path"}:
                    parsed[column] = safe_float(parsed.get(column))
            rows.append(parsed)
        return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def box_from_row(row, prefix):
    return {
        "cx": row[f"{prefix}_center_x"],
        "cy": row[f"{prefix}_center_y"],
        "cz": row[f"{prefix}_center_z"],
        "sx": row[f"{prefix}_size_x"],
        "sy": row[f"{prefix}_size_y"],
        "sz": row[f"{prefix}_size_z"],
    }


def make_box(cx, cy, cz, sx, sy, sz):
    return {"cx": cx, "cy": cy, "cz": cz, "sx": sx, "sy": sy, "sz": sz}


def box_to_interval(box):
    cx, cy, cz = box["cx"], box["cy"], box["cz"]
    sx, sy, sz = box["sx"], box["sy"], box["sz"]
    if not all(is_finite(value) for value in (cx, cy, cz, sx, sy, sz)):
        return {key: math.nan for key in ("x1", "y1", "z1", "x2", "y2", "z2")}
    return {
        "x1": cx - sx / 2.0,
        "y1": cy - sy / 2.0,
        "z1": cz - sz / 2.0,
        "x2": cx + sx / 2.0,
        "y2": cy + sy / 2.0,
        "z2": cz + sz / 2.0,
    }


def interval_length(interval, axis):
    a = interval[f"{axis}1"]
    b = interval[f"{axis}2"]
    if not is_finite(a) or not is_finite(b):
        return math.nan
    return max(0.0, b - a)


def volume_from_box(box):
    vals = (box["sx"], box["sy"], box["sz"])
    if not all(is_finite(value) for value in vals):
        return math.nan
    return vals[0] * vals[1] * vals[2]


def axis_intersection(gt_interval, pred_interval, axis):
    a1, a2 = gt_interval[f"{axis}1"], gt_interval[f"{axis}2"]
    b1, b2 = pred_interval[f"{axis}1"], pred_interval[f"{axis}2"]
    if not all(is_finite(value) for value in (a1, a2, b1, b2)):
        return math.nan
    return max(0.0, min(a2, b2) - max(a1, b1))


def axis_union_length(gt_interval, pred_interval, axis):
    a1, a2 = gt_interval[f"{axis}1"], gt_interval[f"{axis}2"]
    b1, b2 = pred_interval[f"{axis}1"], pred_interval[f"{axis}2"]
    if not all(is_finite(value) for value in (a1, a2, b1, b2)):
        return math.nan
    return max(a2, b2) - min(a1, b1)


def iou_axis_aligned(gt_box, pred_box):
    gt_i = box_to_interval(gt_box)
    pr_i = box_to_interval(pred_box)
    inter = {axis: axis_intersection(gt_i, pr_i, axis) for axis in ("x", "y", "z")}
    if not all(is_finite(value) for value in inter.values()):
        return math.nan
    inter_vol = inter["x"] * inter["y"] * inter["z"]
    gt_vol = volume_from_box(gt_box)
    pr_vol = volume_from_box(pred_box)
    union = gt_vol + pr_vol - inter_vol if is_finite(gt_vol) and is_finite(pr_vol) else math.nan
    return safe_div(inter_vol, union)


def dominant_min_axis(values):
    finite = {axis: value for axis, value in values.items() if is_finite(value)}
    if not finite:
        return "nan"
    return min(finite, key=finite.get)


def dominant_max_axis(values):
    finite = {axis: value for axis, value in values.items() if is_finite(value)}
    if not finite:
        return "nan"
    return max(finite, key=finite.get)


def best_oracle_type(gains):
    finite = {key: value for key, value in gains.items() if is_finite(value)}
    if not finite:
        return "nan"
    return max(finite, key=finite.get)


def explain(row):
    parts = []
    if row["iou_mismatch"]:
        parts.append("重算 IoU 与原始 IoU 不一致，优先检查 box 格式或 IoU 计算。")
    if row["oracle_center_pass_05"] and not row["oracle_size_pass_05"]:
        parts.append("中心修正即可过 0.5，主要是 center/boundary alignment 问题。")
    elif row["oracle_size_pass_05"] and not row["oracle_center_pass_05"]:
        parts.append("尺寸修正即可过 0.5，主要是 size/boundary scale 问题。")
    elif (not row["oracle_center_pass_05"]) and (not row["oracle_size_pass_05"]) and row["oracle_center_size_pass_05"]:
        parts.append("单独修 center 或 size 不够，需要联合修正 center 和 size。")
    elif not row["oracle_center_size_pass_05"]:
        parts.append("即使 center 和 size oracle 也不能过 0.5，可能存在 box 格式、方向、非 axis-aligned 表示或解析问题。")

    ox, oy, oz = row["overlap_ratio_x"], row["overlap_ratio_y"], row["overlap_ratio_z"]
    if is_finite(oz) and is_finite(ox) and is_finite(oy) and oz < min(ox, oy) - 0.05:
        parts.append("z 轴 overlap 是主要瓶颈，可能是高度/上下边界问题。")
    elif row["dominant_bad_axis"] in {"x", "y"}:
        parts.append("水平面 overlap 是主要瓶颈，可能是车辆相邻混淆或平面边界偏移。")
    return " ".join(parts) if parts else "暂无明确分解结论。"


def decompose_row(row, mismatch_threshold):
    gt_box = box_from_row(row, "gt")
    top_box = box_from_row(row, "top1")
    gt_i = box_to_interval(gt_box)
    top_i = box_to_interval(top_box)

    out = {key: row.get(key, "") for key in (
        "id", "scene", "frame_id", "utterance", "top1_iou", "best_top10_iou", "best_top10_rank",
        "top1_to_best_top10_iou_gap", "c_subtype", "json_path", "gt_center_x", "gt_center_y",
        "gt_center_z", "gt_size_x", "gt_size_y", "gt_size_z", "gt_volume", "top1_center_x",
        "top1_center_y", "top1_center_z", "top1_size_x", "top1_size_y", "top1_size_z", "top1_volume"
    )}

    for axis in ("x", "y", "z"):
        out[f"gt_{axis}1"] = gt_i[f"{axis}1"]
        out[f"gt_{axis}2"] = gt_i[f"{axis}2"]
        out[f"top1_{axis}1"] = top_i[f"{axis}1"]
        out[f"top1_{axis}2"] = top_i[f"{axis}2"]
        out[f"inter_{axis}"] = axis_intersection(gt_i, top_i, axis)
        out[f"union_{axis}"] = axis_union_length(gt_i, top_i, axis)
        gt_len = interval_length(gt_i, axis)
        top_len = interval_length(top_i, axis)
        out[f"overlap_ratio_{axis}"] = safe_div(out[f"inter_{axis}"], gt_len)
        out[f"sym_overlap_{axis}"] = safe_div(out[f"inter_{axis}"], max(gt_len, top_len) if is_finite(gt_len) and is_finite(top_len) else math.nan)

    out["intersection_volume"] = (
        out["inter_x"] * out["inter_y"] * out["inter_z"]
        if all(is_finite(out[f"inter_{axis}"]) for axis in ("x", "y", "z")) else math.nan
    )
    out["gt_volume_check"] = volume_from_box(gt_box)
    out["top1_volume_check"] = volume_from_box(top_box)
    out["union_volume"] = (
        out["gt_volume_check"] + out["top1_volume_check"] - out["intersection_volume"]
        if all(is_finite(out[key]) for key in ("gt_volume_check", "top1_volume_check", "intersection_volume")) else math.nan
    )
    out["iou_recomputed"] = safe_div(out["intersection_volume"], out["union_volume"])
    out["iou_abs_diff"] = abs(out["iou_recomputed"] - row["top1_iou"]) if is_finite(out["iou_recomputed"]) and is_finite(row["top1_iou"]) else math.nan
    out["iou_mismatch"] = is_finite(out["iou_abs_diff"]) and out["iou_abs_diff"] > mismatch_threshold

    for axis in ("x", "y", "z"):
        out[f"boundary_error_{axis}1"] = abs(top_i[f"{axis}1"] - gt_i[f"{axis}1"]) if is_finite(top_i[f"{axis}1"]) and is_finite(gt_i[f"{axis}1"]) else math.nan
        out[f"boundary_error_{axis}2"] = abs(top_i[f"{axis}2"] - gt_i[f"{axis}2"]) if is_finite(top_i[f"{axis}2"]) and is_finite(gt_i[f"{axis}2"]) else math.nan
        out[f"axis_boundary_error_{axis}"] = (
            out[f"boundary_error_{axis}1"] + out[f"boundary_error_{axis}2"]
            if is_finite(out[f"boundary_error_{axis}1"]) and is_finite(out[f"boundary_error_{axis}2"]) else math.nan
        )

    out["dominant_bad_axis"] = dominant_min_axis({axis: out[f"overlap_ratio_{axis}"] for axis in ("x", "y", "z")})
    out["dominant_boundary_axis"] = dominant_max_axis({axis: out[f"axis_boundary_error_{axis}"] for axis in ("x", "y", "z")})

    center_box = make_box(gt_box["cx"], gt_box["cy"], gt_box["cz"], top_box["sx"], top_box["sy"], top_box["sz"])
    size_box = make_box(top_box["cx"], top_box["cy"], top_box["cz"], gt_box["sx"], gt_box["sy"], gt_box["sz"])
    center_size_box = make_box(gt_box["cx"], gt_box["cy"], gt_box["cz"], gt_box["sx"], gt_box["sy"], gt_box["sz"])
    out["oracle_center_iou"] = iou_axis_aligned(gt_box, center_box)
    out["oracle_size_iou"] = iou_axis_aligned(gt_box, size_box)
    out["oracle_center_size_iou"] = iou_axis_aligned(gt_box, center_size_box)
    out["oracle_center_pass_05"] = is_finite(out["oracle_center_iou"]) and out["oracle_center_iou"] >= 0.5
    out["oracle_size_pass_05"] = is_finite(out["oracle_size_iou"]) and out["oracle_size_iou"] >= 0.5
    out["oracle_center_size_pass_05"] = is_finite(out["oracle_center_size_iou"]) and out["oracle_center_size_iou"] >= 0.5
    out["oracle_center_gain"] = out["oracle_center_iou"] - row["top1_iou"] if is_finite(out["oracle_center_iou"]) and is_finite(row["top1_iou"]) else math.nan
    out["oracle_size_gain"] = out["oracle_size_iou"] - row["top1_iou"] if is_finite(out["oracle_size_iou"]) and is_finite(row["top1_iou"]) else math.nan
    out["oracle_center_size_gain"] = out["oracle_center_size_iou"] - row["top1_iou"] if is_finite(out["oracle_center_size_iou"]) and is_finite(row["top1_iou"]) else math.nan
    out["oracle_best_type"] = best_oracle_type({
        "center": out["oracle_center_gain"],
        "size": out["oracle_size_gain"],
        "center_size": out["oracle_center_size_gain"],
    })
    out["failure_decomposition"] = explain(out)
    return out


def summarize(rows):
    n = len(rows)
    axis_counts = Counter(row["dominant_bad_axis"] for row in rows)
    return {
        "num_samples": n,
        "mean_top1_iou": mean(row["top1_iou"] for row in rows),
        "median_top1_iou": median(row["top1_iou"] for row in rows),
        "mean_iou_recomputed": mean(row["iou_recomputed"] for row in rows),
        "median_iou_recomputed": median(row["iou_recomputed"] for row in rows),
        "num_iou_mismatch": sum(bool(row["iou_mismatch"]) for row in rows),
        "ratio_iou_mismatch": ratio(sum(bool(row["iou_mismatch"]) for row in rows), n),
        "mean_overlap_ratio_x": mean(row["overlap_ratio_x"] for row in rows),
        "median_overlap_ratio_x": median(row["overlap_ratio_x"] for row in rows),
        "mean_overlap_ratio_y": mean(row["overlap_ratio_y"] for row in rows),
        "median_overlap_ratio_y": median(row["overlap_ratio_y"] for row in rows),
        "mean_overlap_ratio_z": mean(row["overlap_ratio_z"] for row in rows),
        "median_overlap_ratio_z": median(row["overlap_ratio_z"] for row in rows),
        "dominant_bad_axis_x_count": axis_counts.get("x", 0),
        "dominant_bad_axis_y_count": axis_counts.get("y", 0),
        "dominant_bad_axis_z_count": axis_counts.get("z", 0),
        "dominant_bad_axis_x_ratio": ratio(axis_counts.get("x", 0), n),
        "dominant_bad_axis_y_ratio": ratio(axis_counts.get("y", 0), n),
        "dominant_bad_axis_z_ratio": ratio(axis_counts.get("z", 0), n),
        "mean_axis_boundary_error_x": mean(row["axis_boundary_error_x"] for row in rows),
        "median_axis_boundary_error_x": median(row["axis_boundary_error_x"] for row in rows),
        "mean_axis_boundary_error_y": mean(row["axis_boundary_error_y"] for row in rows),
        "median_axis_boundary_error_y": median(row["axis_boundary_error_y"] for row in rows),
        "mean_axis_boundary_error_z": mean(row["axis_boundary_error_z"] for row in rows),
        "median_axis_boundary_error_z": median(row["axis_boundary_error_z"] for row in rows),
        "mean_oracle_center_iou": mean(row["oracle_center_iou"] for row in rows),
        "median_oracle_center_iou": median(row["oracle_center_iou"] for row in rows),
        "mean_oracle_size_iou": mean(row["oracle_size_iou"] for row in rows),
        "median_oracle_size_iou": median(row["oracle_size_iou"] for row in rows),
        "mean_oracle_center_size_iou": mean(row["oracle_center_size_iou"] for row in rows),
        "median_oracle_center_size_iou": median(row["oracle_center_size_iou"] for row in rows),
        "num_oracle_center_pass_05": sum(bool(row["oracle_center_pass_05"]) for row in rows),
        "ratio_oracle_center_pass_05": ratio(sum(bool(row["oracle_center_pass_05"]) for row in rows), n),
        "num_oracle_size_pass_05": sum(bool(row["oracle_size_pass_05"]) for row in rows),
        "ratio_oracle_size_pass_05": ratio(sum(bool(row["oracle_size_pass_05"]) for row in rows), n),
        "num_oracle_center_size_pass_05": sum(bool(row["oracle_center_size_pass_05"]) for row in rows),
        "ratio_oracle_center_size_pass_05": ratio(sum(bool(row["oracle_center_size_pass_05"]) for row in rows), n),
        "mean_oracle_center_gain": mean(row["oracle_center_gain"] for row in rows),
        "median_oracle_center_gain": median(row["oracle_center_gain"] for row in rows),
        "mean_oracle_size_gain": mean(row["oracle_size_gain"] for row in rows),
        "median_oracle_size_gain": median(row["oracle_size_gain"] for row in rows),
        "mean_oracle_center_size_gain": mean(row["oracle_center_size_gain"] for row in rows),
        "median_oracle_center_size_gain": median(row["oracle_center_size_gain"] for row in rows),
    }


def by_subtype_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["c_subtype"]].append(row)
    output = []
    for subtype, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        n = len(group)
        axis_counts = Counter(row["dominant_bad_axis"] for row in group)
        output.append({
            "c_subtype": subtype,
            "num_samples": n,
            "mean_top1_iou": mean(row["top1_iou"] for row in group),
            "mean_overlap_ratio_x": mean(row["overlap_ratio_x"] for row in group),
            "mean_overlap_ratio_y": mean(row["overlap_ratio_y"] for row in group),
            "mean_overlap_ratio_z": mean(row["overlap_ratio_z"] for row in group),
            "dominant_bad_axis_x_ratio": ratio(axis_counts.get("x", 0), n),
            "dominant_bad_axis_y_ratio": ratio(axis_counts.get("y", 0), n),
            "dominant_bad_axis_z_ratio": ratio(axis_counts.get("z", 0), n),
            "mean_oracle_center_iou": mean(row["oracle_center_iou"] for row in group),
            "mean_oracle_size_iou": mean(row["oracle_size_iou"] for row in group),
            "mean_oracle_center_size_iou": mean(row["oracle_center_size_iou"] for row in group),
            "ratio_oracle_center_pass_05": ratio(sum(bool(row["oracle_center_pass_05"]) for row in group), n),
            "ratio_oracle_size_pass_05": ratio(sum(bool(row["oracle_size_pass_05"]) for row in group), n),
            "ratio_oracle_center_size_pass_05": ratio(sum(bool(row["oracle_center_size_pass_05"]) for row in group), n),
            "mean_oracle_center_gain": mean(row["oracle_center_gain"] for row in group),
            "mean_oracle_size_gain": mean(row["oracle_size_gain"] for row in group),
            "mean_oracle_center_size_gain": mean(row["oracle_center_size_gain"] for row in group),
        })
    return output


def dominant_axis_cases(rows):
    cases = []
    seen = set()
    for axis in ("x", "y", "z"):
        candidates = [row for row in rows if row["dominant_bad_axis"] == axis]
        candidates = sorted(candidates, key=lambda row: (safe_float(row[f"overlap_ratio_{axis}"], 999.0), row["id"]))[:20]
        for row in candidates:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            cases.append({key: row.get(key, "") for key in DOMINANT_AXIS_CASE_COLUMNS})
    return cases


def findings_text(input_path, rows, summary):
    axis_counts = {axis: summary[f"dominant_bad_axis_{axis}_count"] for axis in ("x", "y", "z")}
    dominant_axis = max(axis_counts, key=axis_counts.get) if rows else "nan"
    boundary_means = {axis: summary[f"mean_axis_boundary_error_{axis}"] for axis in ("x", "y", "z")}
    boundary_axis = max(boundary_means, key=lambda axis: safe_float(boundary_means[axis], -1.0)) if rows else "nan"
    center_gain = summary["mean_oracle_center_gain"]
    size_gain = summary["mean_oracle_size_gain"]
    gain_winner = "center" if safe_float(center_gain, -999.0) >= safe_float(size_gain, -999.0) else "size"

    recommendations = []
    if summary["num_iou_mismatch"] > 0:
        recommendations.append("重算 axis-aligned IoU 与原始 top1_iou 大量不一致，先确认原评估是否使用 rotated GT 或不同 IoU 定义。")
    if summary["ratio_oracle_center_pass_05"] > summary["ratio_oracle_size_pass_05"]:
        recommendations.append("center-only oracle 过 0.5 的比例高于 size-only，BoxRefine 若继续推进应优先验证 center residual。")
    elif summary["ratio_oracle_size_pass_05"] > summary["ratio_oracle_center_pass_05"]:
        recommendations.append("size-only oracle 过 0.5 的比例高于 center-only，BoxRefine 应优先验证 size/scale residual。")
    if summary["ratio_oracle_center_size_pass_05"] > max(summary["ratio_oracle_center_pass_05"], summary["ratio_oracle_size_pass_05"]) + 0.1:
        recommendations.append("center+size 联合 oracle 明显更强，单独修 center 或 size 可能不够。")
    if summary["ratio_oracle_center_pass_05"] < 0.2 and summary["ratio_oracle_size_pass_05"] < 0.2:
        recommendations.append("单独 oracle pass 都偏少，简单 residual refinement 可能不足，需要候选、边界、方向或标注几何进一步分析。")

    lines = [
        "parking_2 C IoU decomposition findings",
        "",
        f"Input CSV: {input_path}",
        f"num_samples: {summary['num_samples']}",
        "",
        "IoU recomputation check:",
        f"  mean_top1_iou={fmt(summary['mean_top1_iou'])}, mean_iou_recomputed={fmt(summary['mean_iou_recomputed'])}",
        f"  num_iou_mismatch={summary['num_iou_mismatch']}, ratio_iou_mismatch={fmt(summary['ratio_iou_mismatch'])}",
        "",
        "Axis overlap ratios:",
        f"  x mean/median={fmt(summary['mean_overlap_ratio_x'])}/{fmt(summary['median_overlap_ratio_x'])}",
        f"  y mean/median={fmt(summary['mean_overlap_ratio_y'])}/{fmt(summary['median_overlap_ratio_y'])}",
        f"  z mean/median={fmt(summary['mean_overlap_ratio_z'])}/{fmt(summary['median_overlap_ratio_z'])}",
        f"  most common dominant_bad_axis={dominant_axis}",
        "",
        "Axis boundary errors:",
        f"  x mean/median={fmt(summary['mean_axis_boundary_error_x'])}/{fmt(summary['median_axis_boundary_error_x'])}",
        f"  y mean/median={fmt(summary['mean_axis_boundary_error_y'])}/{fmt(summary['median_axis_boundary_error_y'])}",
        f"  z mean/median={fmt(summary['mean_axis_boundary_error_z'])}/{fmt(summary['median_axis_boundary_error_z'])}",
        f"  largest mean boundary axis={boundary_axis}",
        "",
        "Oracle correction:",
        f"  center-only pass@0.5: {summary['num_oracle_center_pass_05']} / {summary['num_samples']} = {fmt(summary['ratio_oracle_center_pass_05'])}",
        f"  size-only pass@0.5: {summary['num_oracle_size_pass_05']} / {summary['num_samples']} = {fmt(summary['ratio_oracle_size_pass_05'])}",
        f"  center+size pass@0.5: {summary['num_oracle_center_size_pass_05']} / {summary['num_samples']} = {fmt(summary['ratio_oracle_center_size_pass_05'])}",
        f"  mean center gain={fmt(summary['mean_oracle_center_gain'])}, mean size gain={fmt(summary['mean_oracle_size_gain'])}, larger_gain={gain_winner}",
        "",
        "Initial conclusion:",
        f"  Dominant axis by overlap is {dominant_axis}; dominant boundary-error axis is {boundary_axis}.",
        "  Because center+size oracle uses the same center+size representation as the script, its IoU should be near 1; deviations would indicate parsing/calculation errors.",
        "",
        "Recommendations:",
    ]
    lines.extend(f"  - {item}" for item in recommendations)
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.parking2_c_csv)
    decomp = [decompose_row(row, args.iou_mismatch_threshold) for row in rows]

    write_csv(out_dir / "parking2_c_iou_decomposition.csv", decomp, DECOMP_COLUMNS)
    summary = summarize(decomp)
    write_csv(out_dir / "iou_decomposition_summary.csv", [summary], SUMMARY_COLUMNS)
    write_csv(out_dir / "iou_decomposition_by_subtype.csv", by_subtype_summary(decomp), BY_SUBTYPE_COLUMNS)
    write_csv(out_dir / "oracle_center_pass_cases.csv", [row for row in decomp if row["oracle_center_pass_05"]], DECOMP_COLUMNS)
    write_csv(out_dir / "oracle_size_pass_cases.csv", [row for row in decomp if row["oracle_size_pass_05"]], DECOMP_COLUMNS)
    write_csv(out_dir / "oracle_center_size_pass_cases.csv", [row for row in decomp if row["oracle_center_size_pass_05"]], DECOMP_COLUMNS)
    write_csv(out_dir / "iou_mismatch_cases.csv", [row for row in decomp if row["iou_mismatch"]], DECOMP_COLUMNS)
    write_csv(out_dir / "dominant_axis_cases.csv", dominant_axis_cases(decomp), DOMINANT_AXIS_CASE_COLUMNS)
    (out_dir / "parking2_c_iou_decomposition_findings.txt").write_text(
        findings_text(args.parking2_c_csv, decomp, summary), encoding="utf-8"
    )

    files = [
        "parking2_c_iou_decomposition.csv",
        "iou_decomposition_summary.csv",
        "iou_decomposition_by_subtype.csv",
        "oracle_center_pass_cases.csv",
        "oracle_size_pass_cases.csv",
        "oracle_center_size_pass_cases.csv",
        "iou_mismatch_cases.csv",
        "dominant_axis_cases.csv",
        "parking2_c_iou_decomposition_findings.txt",
    ]
    print(f"Saved parking2 C IoU decomposition analysis to {out_dir.as_posix()}")
    print("Generated files:")
    for name in files:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
