#!/usr/bin/env python3
"""Debug whether stored prediction IoUs can be reproduced offline.

This script reads existing prediction.json files through all_records.csv and
compares the recorded top1/top10 IoUs with several plausible box
interpretations. It does not load a model, checkpoint, or evaluation loader.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_ALL_RECORDS_CSV = "outputs/diagnostic_baseline20_dq/all_records.csv"
DEFAULT_SCENE = "Outdoor_Day_penno_parking_2"
DEFAULT_PRIMARY_LABEL = "C_coarse_success_precise_fail"
DEFAULT_OUTPUT_DIR = "outputs/iou_consistency_debug"
EPS = 1e-9


try:
    from utils.eval_det import iou3d_rotated_vs_aligned

    OFFICIAL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - reported in findings.
    iou3d_rotated_vs_aligned = None
    OFFICIAL_IMPORT_ERROR = repr(exc)

try:
    from utils.box_util import box3d_iou

    BOX3D_IOU_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - reported in findings.
    box3d_iou = None
    BOX3D_IOU_IMPORT_ERROR = repr(exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check consistency between recorded prediction IoUs and offline recomputation."
    )
    parser.add_argument("--all_records_csv", default=DEFAULT_ALL_RECORDS_CSV)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--primary_label", default=DEFAULT_PRIMARY_LABEL)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def repo_root() -> Path:
    return REPO_ROOT


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return repo_root() / path


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def fmt_float(value: Any, digits: int = 6) -> str:
    value = safe_float(value)
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def shape_of(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        arr = np.asarray(value, dtype=object)
    except Exception:
        return type(value).__name__
    if arr.shape:
        return "x".join(str(dim) for dim in arr.shape)
    return "scalar"


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


def first_box(value: Any) -> Any:
    arr = numeric_array(value)
    if arr is None:
        return None
    if arr.ndim == 1:
        return arr.tolist()
    if arr.ndim >= 2 and arr.shape[0] > 0:
        return arr[0].tolist()
    return None


def to_box6_first(value: Any) -> np.ndarray | None:
    arr = numeric_array(value)
    if arr is None:
        return None
    if arr.ndim == 1 and arr.shape[0] >= 6:
        return arr[:6].astype(np.float64)
    if arr.ndim >= 2 and arr.shape[-1] >= 6:
        flat = arr.reshape(-1, arr.shape[-1])
        return flat[0, :6].astype(np.float64)
    return None


def to_box6_list(value: Any) -> list[np.ndarray]:
    arr = numeric_array(value)
    if arr is None:
        return []
    if arr.ndim == 1 and arr.shape[0] >= 6:
        return [arr[:6].astype(np.float64)]
    if arr.ndim >= 2 and arr.shape[-1] >= 6:
        flat = arr.reshape(-1, arr.shape[-1])
        return [row[:6].astype(np.float64) for row in flat]
    return []


def to_corner8_first(value: Any) -> np.ndarray | None:
    arr = numeric_array(value)
    if arr is None:
        return None
    if arr.shape == (8, 3):
        return arr.astype(np.float64)
    if arr.ndim >= 3 and arr.shape[-2:] == (8, 3):
        flat = arr.reshape(-1, 8, 3)
        return flat[0].astype(np.float64)
    return None


def box_related_keys(record: dict[str, Any]) -> list[str]:
    needles = ("box", "bbox", "corner", "center", "size", "heading", "angle", "yaw", "rot")
    return sorted(key for key in record.keys() if any(token in key.lower() for token in needles))


def pick_pred_boxes(record: dict[str, Any]) -> tuple[str, Any]:
    for key in ("top10_pred_boxes", "top10_refined_pred_boxes", "pred_boxes", "boxes", "top10_boxes"):
        if key in record:
            return key, record.get(key)
    return "", None


def center_size_to_xyzxyz(box: np.ndarray) -> np.ndarray | None:
    if box.shape[0] < 6:
        return None
    size = box[3:6]
    if np.any(size <= 0):
        return None
    half = size / 2.0
    return np.concatenate([box[:3] - half, box[:3] + half])


def canonical_xyzxyz(box: np.ndarray) -> np.ndarray | None:
    if box.shape[0] < 6:
        return None
    mins = np.minimum(box[:3], box[3:6])
    maxs = np.maximum(box[:3], box[3:6])
    if np.any(maxs - mins <= 0):
        return None
    return np.concatenate([mins, maxs])


def iou_xyzxyz_arrays(a: np.ndarray | None, b: np.ndarray | None) -> float:
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


def iou_center_size(gt_value: Any, pred_value: Any) -> float:
    gt = to_box6_first(gt_value)
    pred = to_box6_first(pred_value)
    return iou_xyzxyz_arrays(center_size_to_xyzxyz(gt) if gt is not None else None,
                             center_size_to_xyzxyz(pred) if pred is not None else None)


def iou_xyzxyz(gt_value: Any, pred_value: Any) -> float:
    gt = to_box6_first(gt_value)
    pred = to_box6_first(pred_value)
    return iou_xyzxyz_arrays(canonical_xyzxyz(gt) if gt is not None else None,
                             canonical_xyzxyz(pred) if pred is not None else None)


def iou_official(gt_value: Any, pred_value: Any) -> float:
    if iou3d_rotated_vs_aligned is None:
        return math.nan
    gt = numeric_array(gt_value)
    pred = to_box6_first(pred_value)
    if gt is None or pred is None:
        return math.nan
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)
    elif gt.ndim >= 2:
        gt = gt.reshape(-1, gt.shape[-1])[:1]
    if gt.shape[-1] < 7:
        return math.nan
    try:
        gt_tensor = torch.as_tensor(gt, dtype=torch.float32)
        pred_tensor = torch.as_tensor(pred.reshape(1, 6), dtype=torch.float32)
        ious, _ = iou3d_rotated_vs_aligned(gt_tensor, pred_tensor)
        return safe_float(ious[0, 0].detach().cpu().item())
    except Exception:
        return math.nan


def iou_corner(gt_value: Any, pred_value: Any) -> float:
    if box3d_iou is None:
        return math.nan
    gt = to_corner8_first(gt_value)
    pred = to_corner8_first(pred_value)
    if gt is None or pred is None:
        return math.nan
    try:
        iou, _ = box3d_iou(gt, pred)
        return safe_float(iou)
    except Exception:
        return math.nan


def compute_top10(
    gt_value: Any,
    pred_boxes_value: Any,
    method: Callable[[Any, Any], float],
) -> list[float]:
    pred_boxes = to_box6_list(pred_boxes_value)
    return [method(gt_value, box.tolist()) for box in pred_boxes[:10]]


def clean_float_list(value: Any) -> list[float]:
    arr = numeric_array(value)
    if arr is None:
        return []
    return [safe_float(item) for item in arr.reshape(-1).tolist()]


def direct_diff_stats(recorded: list[float], recomputed: list[float]) -> tuple[float, float]:
    pairs = [
        (rec, rep)
        for rec, rep in zip(recorded, recomputed)
        if math.isfinite(rec) and math.isfinite(rep)
    ]
    if not pairs:
        return math.nan, math.nan
    diffs = [abs(rec - rep) for rec, rep in pairs]
    return min(diffs), sum(diffs) / len(diffs)


def sorted_match_mean_diff(recorded: list[float], recomputed: list[float]) -> float:
    rec = sorted([x for x in recorded if math.isfinite(x)])
    rep = sorted([x for x in recomputed if math.isfinite(x)])
    pairs = list(zip(rec, rep))
    if not pairs:
        return math.nan
    return sum(abs(a - b) for a, b in pairs) / len(pairs)


def choose_best_method(values: dict[str, float], recorded: float) -> tuple[str, float, float]:
    best_name = "none"
    best_iou = math.nan
    best_diff = math.nan
    for name, value in values.items():
        if not math.isfinite(value) or not math.isfinite(recorded):
            continue
        diff = abs(value - recorded)
        if not math.isfinite(best_diff) or diff < best_diff:
            best_name = name
            best_iou = value
            best_diff = diff
    return best_name, best_iou, best_diff


def infer_box_format(gt_shape: str, pred_shape: str, best_method: str) -> str:
    if best_method == "official":
        return "gt_rotated_7d9d__pred_center_size_6d"
    if best_method == "center_size":
        return "both_center_size_6d"
    if best_method == "xyzxyz":
        return "both_xyzxyz_6d"
    if best_method == "corner":
        return "both_corners_8x3"
    return f"unknown(gt={gt_shape},pred={pred_shape})"


def sample_rows(rows: list[dict[str, str]], scene: str, primary_label: str, num_samples: int) -> list[dict[str, str]]:
    selected = [
        row for row in rows
        if row.get("scene") == scene and row.get("primary_label") == primary_label
    ]
    selected.sort(key=lambda row: (row.get("frame_id", ""), row.get("id", "")))
    if num_samples > 0:
        return selected[:num_samples]
    return selected


def analyze_case(row: dict[str, str]) -> dict[str, Any]:
    json_path = resolve_path(row["json_path"])
    prediction = read_json(json_path)
    if isinstance(prediction, list):
        if len(prediction) != 1:
            raise ValueError(f"Expected one prediction record in {json_path}, got {len(prediction)}")
        prediction = prediction[0]
    if not isinstance(prediction, dict):
        raise ValueError(f"Expected dict prediction record in {json_path}, got {type(prediction).__name__}")

    gt_box = prediction.get("gt_box")
    pred_key, pred_boxes = pick_pred_boxes(prediction)
    pred0 = first_box(pred_boxes)
    pred_boxes_for_raw = pred_boxes
    top10_ious = clean_float_list(prediction.get("top10_ious"))
    top1_iou_recorded = safe_float(prediction.get("top1_iou", row.get("top1_iou")))
    top10_iou0_recorded = top10_ious[0] if top10_ious else math.nan

    iou_values = {
        "center_size": iou_center_size(gt_box, pred0),
        "xyzxyz": iou_xyzxyz(gt_box, pred0),
        "official": iou_official(gt_box, pred0),
        "corner": iou_corner(gt_box, pred0),
    }
    best_name, best_iou, best_diff = choose_best_method(iou_values, top1_iou_recorded)

    top10_recomputed_by_method = {
        name: compute_top10(gt_box, pred_boxes_for_raw, method)
        for name, method in (
            ("center_size", iou_center_size),
            ("xyzxyz", iou_xyzxyz),
            ("official", iou_official),
            ("corner", iou_corner),
        )
    }
    top10_method = best_name if best_name in top10_recomputed_by_method else "official"
    if top10_method == "corner" and not top10_recomputed_by_method[top10_method]:
        top10_method = "official"
    top10_recomputed = top10_recomputed_by_method.get(top10_method, [])
    top10_min_abs_diff, top10_mean_abs_diff = direct_diff_stats(top10_ious, top10_recomputed)
    top10_sorted_mean_abs_diff = sorted_match_mean_diff(top10_ious, top10_recomputed)

    gt_shape = shape_of(gt_box)
    pred_shape = shape_of(pred0)
    pred_boxes_shape = shape_of(pred_boxes)
    box_keys = box_related_keys(prediction)

    return {
        "id": prediction.get("id", row.get("id", "")),
        "primary_label": row.get("primary_label", ""),
        "top1_iou_recorded": top1_iou_recorded,
        "top10_iou0_recorded": top10_iou0_recorded,
        "top10_ious_recorded": json_dumps_compact(top10_ious),
        "box_format_guess": infer_box_format(gt_shape, pred_shape, best_name),
        "gt_box_shape": gt_shape,
        "top10_pred_boxes_shape": pred_boxes_shape,
        "pred_box_shape": pred_shape,
        "pred_boxes_key_used": pred_key,
        "box_related_keys": json_dumps_compact(box_keys),
        "gt_box_raw": json_dumps_compact(gt_box),
        "top10_pred_box0_raw": json_dumps_compact(pred0),
        "pred_box0_raw": json_dumps_compact(first_box(prediction.get("pred_boxes"))),
        "top1_pred_box_raw": json_dumps_compact(prediction.get("top1_pred_box")),
        "iou_center_size": iou_values["center_size"],
        "diff_center_size": abs(iou_values["center_size"] - top1_iou_recorded)
        if math.isfinite(iou_values["center_size"]) and math.isfinite(top1_iou_recorded) else math.nan,
        "iou_xyzxyz": iou_values["xyzxyz"],
        "diff_xyzxyz": abs(iou_values["xyzxyz"] - top1_iou_recorded)
        if math.isfinite(iou_values["xyzxyz"]) and math.isfinite(top1_iou_recorded) else math.nan,
        "iou_official": iou_values["official"],
        "diff_official": abs(iou_values["official"] - top1_iou_recorded)
        if math.isfinite(iou_values["official"]) and math.isfinite(top1_iou_recorded) else math.nan,
        "iou_corner": iou_values["corner"],
        "diff_corner": abs(iou_values["corner"] - top1_iou_recorded)
        if math.isfinite(iou_values["corner"]) and math.isfinite(top1_iou_recorded) else math.nan,
        "best_recompute_method": best_name,
        "best_recompute_iou": best_iou,
        "best_abs_diff": best_diff,
        "top10_match_method": top10_method,
        "top10_min_abs_diff": top10_min_abs_diff,
        "top10_mean_abs_diff": top10_mean_abs_diff,
        "top10_best_permutation_like_match": top10_sorted_mean_abs_diff,
        "json_path": row.get("json_path", ""),
    }


def finite_mean(values: list[float]) -> float:
    vals = [safe_float(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    if not vals:
        return math.nan
    return sum(vals) / len(vals)


def count_shapes(rows: list[dict[str, Any]], key: str) -> str:
    counts = Counter(str(row.get(key, "")) for row in rows)
    return ", ".join(f"{shape}:{count}" for shape, count in sorted(counts.items()))


def count_keys(rows: list[dict[str, Any]]) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        try:
            keys = json.loads(row.get("box_related_keys", "[]"))
        except json.JSONDecodeError:
            keys = []
        counts.update(keys)
    return ", ".join(f"{key}:{count}" for key, count in sorted(counts.items()))


def method_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("best_recompute_method", "none")) for row in rows)


def build_findings(
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict[str, Any]],
    all_records_count: int,
    selected_count_before_limit: int,
) -> list[str]:
    best_counts = method_counts(rows)
    center_mean = finite_mean([row["diff_center_size"] for row in rows])
    xyzxyz_mean = finite_mean([row["diff_xyzxyz"] for row in rows])
    official_mean = finite_mean([row["diff_official"] for row in rows])
    corner_mean = finite_mean([row["diff_corner"] for row in rows])
    top10_mean = finite_mean([row["top10_mean_abs_diff"] for row in rows])
    top10_perm_mean = finite_mean([row["top10_best_permutation_like_match"] for row in rows])
    best_method = best_counts.most_common(1)[0][0] if best_counts else "none"

    lines = [
        "IoU consistency debug findings",
        "",
        f"Input all_records_csv: {args.all_records_csv}",
        f"Resolved all_records rows: {all_records_count}",
        f"Scene filter: {args.scene}",
        f"Primary label filter: {args.primary_label}",
        f"Matching rows before sample limit: {selected_count_before_limit}",
        f"num_samples_analyzed: {len(rows)}",
        f"Output directory: {output_dir}",
        "",
        "Official IoU function:",
        "  expected export call: utils.eval_det.iou3d_rotated_vs_aligned(gt_box, top10_pred_boxes)",
        f"  import_status: {'ok' if iou3d_rotated_vs_aligned is not None else OFFICIAL_IMPORT_ERROR}",
        f"  corner_iou_import_status: {'ok' if box3d_iou is not None else BOX3D_IOU_IMPORT_ERROR}",
        "",
        "prediction.json box-related keys:",
        f"  {count_keys(rows)}",
        "",
        "Shape statistics:",
        f"  gt_box_shape: {count_shapes(rows, 'gt_box_shape')}",
        f"  top10_pred_boxes_shape: {count_shapes(rows, 'top10_pred_boxes_shape')}",
        f"  pred_box_shape(first top10 item): {count_shapes(rows, 'pred_box_shape')}",
        "",
        "Mean absolute diff to recorded top1_iou:",
        f"  center+size axis-aligned: {fmt_float(center_mean)}",
        f"  xyzxyz axis-aligned: {fmt_float(xyzxyz_mean)}",
        f"  official rotated-vs-aligned: {fmt_float(official_mean)}",
        f"  corner-based direct: {fmt_float(corner_mean)}",
        "",
        "Best recompute method counts:",
    ]
    if best_counts:
        for name, count in best_counts.most_common():
            lines.append(f"  {name}: {count}")
    else:
        lines.append("  none: 0")

    lines.extend([
        "",
        f"Most common best method: {best_method}",
        "",
        "Top10 correspondence:",
        f"  mean direct top10 abs diff ({best_method} or per-case best): {fmt_float(top10_mean)}",
        f"  mean sorted/permutation-like top10 abs diff: {fmt_float(top10_perm_mean)}",
    ])

    if rows and official_mean < 1e-4:
        lines.extend([
            "",
            "Conclusion:",
            "  official rotated-vs-aligned IoU reproduces recorded top1_iou.",
            "  prediction.json gt_box is rotated 7D/9D GT, while top10_pred_boxes are aligned 6D predictions.",
            "  The earlier center+size decomposition over GT first 6 dims is not equivalent to official IoU.",
        ])
    elif rows and min(center_mean, xyzxyz_mean, official_mean, corner_mean) < 1e-4:
        lines.extend([
            "",
            "Conclusion:",
            "  At least one recompute method closely reproduces recorded top1_iou; use the best method above for later analysis.",
        ])
    else:
        lines.extend([
            "",
            "Conclusion:",
            "  No recompute method closely reproduces recorded top1_iou.",
            "  Possible causes:",
            "    - top10_pred_boxes and top10_ious order are inconsistent;",
            "    - top1_iou comes from another head/ranking path;",
            "    - prediction.json exported boxes are not the boxes used by evaluation;",
            "    - coordinate transform or normalization differs between export and offline script.",
        ])

    return lines


def main() -> None:
    args = parse_args()
    all_records_path = resolve_path(args.all_records_csv)
    output_dir = resolve_path(args.output_dir)
    rows = read_csv(all_records_path)
    matching_before_limit = [
        row for row in rows
        if row.get("scene") == args.scene and row.get("primary_label") == args.primary_label
    ]
    selected_rows = sample_rows(rows, args.scene, args.primary_label, args.num_samples)
    if not selected_rows:
        raise ValueError(
            f"No rows matched scene={args.scene!r}, primary_label={args.primary_label!r} "
            f"in {all_records_path}"
        )

    case_rows = [analyze_case(row) for row in selected_rows]

    fieldnames = [
        "id",
        "primary_label",
        "top1_iou_recorded",
        "top10_iou0_recorded",
        "top10_ious_recorded",
        "box_format_guess",
        "gt_box_shape",
        "top10_pred_boxes_shape",
        "pred_box_shape",
        "pred_boxes_key_used",
        "box_related_keys",
        "gt_box_raw",
        "top10_pred_box0_raw",
        "pred_box0_raw",
        "top1_pred_box_raw",
        "iou_center_size",
        "diff_center_size",
        "iou_xyzxyz",
        "diff_xyzxyz",
        "iou_official",
        "diff_official",
        "iou_corner",
        "diff_corner",
        "best_recompute_method",
        "best_recompute_iou",
        "best_abs_diff",
        "top10_match_method",
        "top10_min_abs_diff",
        "top10_mean_abs_diff",
        "top10_best_permutation_like_match",
        "json_path",
    ]
    write_csv(output_dir / "iou_consistency_cases.csv", case_rows, fieldnames)
    findings = build_findings(
        args=args,
        output_dir=output_dir,
        rows=case_rows,
        all_records_count=len(rows),
        selected_count_before_limit=len(matching_before_limit),
    )
    write_text(output_dir / "iou_consistency_findings.txt", findings)
    print(f"Saved IoU consistency debug to {args.output_dir}")


if __name__ == "__main__":
    main()
