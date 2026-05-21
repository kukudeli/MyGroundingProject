"""Build diagnostic tables and small review subsets from prediction.json files.

This offline tool reads precomputed Drone and Quad evaluation predictions,
adds non-mutually-exclusive diagnostic flags plus one mutually-exclusive
primary label, and writes full records, summaries, and fixed-seed samples.

Example:
    python tools/build_diagnostic_subset.py \
      --drone_root logs/Train_drone_quad_Val_drone_quad/ablation_baseline_dq/eval/Val_drone/0520_1209/predictions \
      --quad_root logs/Train_drone_quad_Val_drone_quad/ablation_baseline_dq/eval/Val_quad/0520_1225/predictions \
      --out_dir outputs/diagnostic_baseline20_dq \
      --samples_per_group 20 \
      --samples_per_scene 5 \
      --seed 42
"""

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_DRONE_ROOT = (
    "logs/Train_drone_quad_Val_drone_quad/ablation_baseline_dq/eval/"
    "Val_drone/0520_1209/predictions"
)
DEFAULT_QUAD_ROOT = (
    "logs/Train_drone_quad_Val_drone_quad/ablation_baseline_dq/eval/"
    "Val_quad/0520_1225/predictions"
)
DEFAULT_OUT_DIR = "outputs/diagnostic_baseline20_dq"

REQUIRED_NUMERIC_FIELDS = ("top1_iou", "max_top5_iou", "max_top10_iou")

FLAG_COLUMNS = (
    "flag_A_recall_fail_25",
    "flag_B_ranking_fail_25",
    "flag_C_coarse_success_precise_fail",
    "flag_D_ranking_fail_50",
    "flag_E_strict_success",
)

PRIMARY_LABELS = (
    "A_recall_fail_25",
    "B_ranking_fail_25",
    "C_coarse_success_precise_fail",
    "D_ranking_fail_50",
    "E_strict_success",
)

CSV_COLUMNS = (
    "platform",
    "scene",
    "frame_id",
    "id",
    "utterance",
    "top1_iou",
    "max_top5_iou",
    "max_top10_iou",
    "acc25_top1",
    "acc50_top1",
    "acc25_top10",
    "acc50_top10",
    "primary_label",
    "flag_A_recall_fail_25",
    "flag_B_ranking_fail_25",
    "flag_C_coarse_success_precise_fail",
    "flag_D_ranking_fail_50",
    "flag_E_strict_success",
    "json_path",
)

SAMPLE_COLUMNS = (
    "id",
    "platform",
    "scene",
    "frame_id",
    "utterance",
    "top1_iou",
    "max_top5_iou",
    "max_top10_iou",
    "primary_label",
    "json_path",
)

MANUAL_REVIEW_COLUMNS = (
    "id",
    "platform",
    "scene",
    "frame_id",
    "primary_label",
    "utterance",
    "top1_iou",
    "max_top5_iou",
    "max_top10_iou",
    "json_path",
    "manual_target_size",
    "manual_distance",
    "manual_density",
    "manual_occlusion",
    "manual_similar_objects",
    "manual_edge_or_truncation",
    "manual_language_complexity",
    "manual_failure_reason",
    "manual_notes",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build diagnostic summaries and sample subsets from evaluation prediction.json files."
    )
    parser.add_argument("--drone_root", default=DEFAULT_DRONE_ROOT, help="Root directory containing Drone prediction.json files.")
    parser.add_argument("--quad_root", default=DEFAULT_QUAD_ROOT, help="Root directory containing Quad prediction.json files.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Directory to save diagnostic outputs.")
    parser.add_argument("--samples_per_group", type=int, default=20, help="Number of samples per platform and primary label.")
    parser.add_argument(
        "--samples_per_scene",
        type=int,
        default=5,
        help="Maximum number of samples per platform, scene, and primary label for scene-balanced outputs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for fixed sampling.")
    return parser.parse_args()


def display_path(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def read_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{display_path(path)}: invalid JSON ({exc})") from exc
    except OSError as exc:
        raise ValueError(f"{display_path(path)}: failed to read file ({exc})") from exc


def looks_like_single_record(data):
    return isinstance(data, dict) and any(field in data for field in REQUIRED_NUMERIC_FIELDS)


def coerce_records(data, json_path):
    if isinstance(data, list):
        records = []
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"{json_path}: list item {index} is not an object")
            records.append(dict(item))
        return records

    if looks_like_single_record(data):
        return [dict(data)]

    if isinstance(data, dict):
        records = []
        for key, value in data.items():
            if not isinstance(value, dict):
                raise ValueError(
                    f"{json_path}: dict value for key {key!r} is not an object; "
                    "expected a single record dict, a list of record dicts, or an id-to-record dict"
                )
            record = dict(value)
            record.setdefault("id", key)
            records.append(record)
        return records

    raise ValueError(f"{json_path}: expected a dict or list, got {type(data).__name__}")


def require_numeric(record, field, json_path):
    if field not in record:
        record_id = record.get("id", "<missing id>")
        raise ValueError(f"{json_path}: record {record_id!r} missing required field {field!r}")

    value = record[field]
    if isinstance(value, bool) or value is None:
        record_id = record.get("id", "<missing id>")
        raise ValueError(f"{json_path}: record {record_id!r} field {field!r} is not numeric: {value!r}")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        record_id = record.get("id", "<missing id>")
        raise ValueError(f"{json_path}: record {record_id!r} field {field!r} is not numeric: {value!r}") from exc

    if not math.isfinite(number):
        record_id = record.get("id", "<missing id>")
        raise ValueError(f"{json_path}: record {record_id!r} field {field!r} is not finite: {value!r}")

    return number


def compute_flags(top1_iou, max_top10_iou):
    flags = {
        "flag_A_recall_fail_25": max_top10_iou < 0.25,
        "flag_B_ranking_fail_25": max_top10_iou >= 0.25 and top1_iou < 0.25,
        "flag_C_coarse_success_precise_fail": top1_iou >= 0.25 and top1_iou < 0.5,
        "flag_D_ranking_fail_50": max_top10_iou >= 0.5 and top1_iou < 0.5,
        "flag_E_strict_success": top1_iou >= 0.5,
    }

    if flags["flag_E_strict_success"]:
        primary_label = "E_strict_success"
    elif flags["flag_D_ranking_fail_50"]:
        primary_label = "D_ranking_fail_50"
    elif flags["flag_C_coarse_success_precise_fail"]:
        primary_label = "C_coarse_success_precise_fail"
    elif flags["flag_B_ranking_fail_25"]:
        primary_label = "B_ranking_fail_25"
    elif flags["flag_A_recall_fail_25"]:
        primary_label = "A_recall_fail_25"
    else:
        primary_label = "unknown"

    return flags, primary_label


def parse_id_fields(record_id):
    if record_id is None:
        return "", "", ""

    parts = str(record_id).replace("\\", "/").split("/")
    platform_from_id = parts[0] if len(parts) >= 1 else ""
    scene = parts[1] if len(parts) >= 2 else ""
    frame_id = parts[2] if len(parts) >= 3 else ""
    return platform_from_id, scene, frame_id


def annotate_record(record, json_path, platform_hint):
    record = dict(record)
    platform_from_id, scene, frame_id = parse_id_fields(record.get("id"))
    if not record.get("platform"):
        record["platform"] = platform_from_id or platform_hint

    top1_iou = require_numeric(record, "top1_iou", json_path)
    max_top5_iou = require_numeric(record, "max_top5_iou", json_path)
    max_top10_iou = require_numeric(record, "max_top10_iou", json_path)

    flags, primary_label = compute_flags(top1_iou, max_top10_iou)

    record["top1_iou"] = top1_iou
    record["max_top5_iou"] = max_top5_iou
    record["max_top10_iou"] = max_top10_iou
    record["scene"] = scene
    record["frame_id"] = frame_id
    record["json_path"] = json_path
    record.update(flags)
    record["primary_label"] = primary_label
    return record


def load_root(root, platform_hint):
    root = Path(root)
    if not root.exists():
        raise ValueError(f"{platform_hint} input directory does not exist: {display_path(root)}")
    if not root.is_dir():
        raise ValueError(f"{platform_hint} input path is not a directory: {display_path(root)}")

    prediction_paths = sorted(root.rglob("prediction.json"))
    if not prediction_paths:
        raise ValueError(f"{platform_hint} input directory has no prediction.json files: {display_path(root)}")

    records = []
    for path in prediction_paths:
        json_path = display_path(path)
        data = read_json(path)
        for record in coerce_records(data, json_path):
            records.append(annotate_record(record, json_path, platform_hint))
    return records, prediction_paths


def bool_to_int(value):
    return 1 if bool(value) else 0


def metric_value(record, field):
    if field == "acc25_top1":
        return bool_to_int(record["top1_iou"] >= 0.25)
    if field == "acc50_top1":
        return bool_to_int(record["top1_iou"] >= 0.5)
    if field == "acc25_top10":
        return bool_to_int(record["max_top10_iou"] >= 0.25)
    if field == "acc50_top10":
        return bool_to_int(record["max_top10_iou"] >= 0.5)
    raise KeyError(field)


def csv_cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return ""
    return value


def row_for_columns(record, columns):
    row = {}
    for column in columns:
        if column in {"acc25_top1", "acc50_top1", "acc25_top10", "acc50_top10"}:
            row[column] = record.get(column, metric_value(record, column))
        else:
            row[column] = record.get(column, "")
        row[column] = csv_cell(row[column])
    return row


def write_jsonl(records, out_path):
    with Path(out_path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(records, out_path, columns):
    with Path(out_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(row_for_columns(record, columns))


def group_by_platform(records):
    grouped = defaultdict(list)
    for record in records:
        platform = str(record.get("platform") or "unknown")
        grouped[platform].append(record)
    return dict(sorted(grouped.items(), key=lambda item: item[0].lower()))


def group_by_platform_scene(records):
    grouped = defaultdict(list)
    for record in records:
        platform = str(record.get("platform") or "unknown")
        scene = str(record.get("scene") or "")
        grouped[(platform, scene)].append(record)
    return dict(sorted(grouped.items(), key=lambda item: (item[0][0].lower(), item[0][1].lower())))


def mean(records, field):
    return sum(float(record[field]) for record in records) / len(records)


def ratio(count, total):
    return count / total if total else 0.0


def build_platform_summary(records):
    summary_rows = []
    for platform, platform_records in group_by_platform(records).items():
        total = len(platform_records)
        row = {
            "platform": platform,
            "num_records": total,
            "mean_top1_iou": mean(platform_records, "top1_iou"),
            "mean_max_top5_iou": mean(platform_records, "max_top5_iou"),
            "mean_max_top10_iou": mean(platform_records, "max_top10_iou"),
            "acc25_top1": sum(metric_value(record, "acc25_top1") for record in platform_records) / total,
            "acc50_top1": sum(metric_value(record, "acc50_top1") for record in platform_records) / total,
            "acc25_top10": sum(metric_value(record, "acc25_top10") for record in platform_records) / total,
            "acc50_top10": sum(metric_value(record, "acc50_top10") for record in platform_records) / total,
        }
        for flag in FLAG_COLUMNS:
            count = sum(bool_to_int(record[flag]) for record in platform_records)
            row[f"{flag}_count"] = count
            row[f"{flag}_ratio"] = ratio(count, total)
        summary_rows.append(row)
    return summary_rows


def build_scene_summary(records):
    summary_rows = []
    for (platform, scene), scene_records in group_by_platform_scene(records).items():
        total = len(scene_records)
        row = {
            "platform": platform,
            "scene": scene,
            "num_records": total,
            "mean_top1_iou": mean(scene_records, "top1_iou"),
            "mean_max_top5_iou": mean(scene_records, "max_top5_iou"),
            "mean_max_top10_iou": mean(scene_records, "max_top10_iou"),
            "acc25_top1": sum(metric_value(record, "acc25_top1") for record in scene_records) / total,
            "acc50_top1": sum(metric_value(record, "acc50_top1") for record in scene_records) / total,
            "acc25_top10": sum(metric_value(record, "acc25_top10") for record in scene_records) / total,
            "acc50_top10": sum(metric_value(record, "acc50_top10") for record in scene_records) / total,
        }
        for flag in FLAG_COLUMNS:
            clean_name = clean_flag_name(flag)
            count = sum(bool_to_int(record[flag]) for record in scene_records)
            row[f"{clean_name}_count"] = count
            row[f"{clean_name}_ratio"] = ratio(count, total)
        summary_rows.append(row)
    return summary_rows


def write_summary_by_platform(records, out_path):
    summary_rows = build_platform_summary(records)
    columns = (
        "platform",
        "num_records",
        "mean_top1_iou",
        "mean_max_top5_iou",
        "mean_max_top10_iou",
        "acc25_top1",
        "acc50_top1",
        "acc25_top10",
        "acc50_top10",
    )
    flag_columns = []
    for flag in FLAG_COLUMNS:
        flag_columns.extend((f"{flag}_count", f"{flag}_ratio"))
    with Path(out_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + tuple(flag_columns))
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    return summary_rows


def write_scene_summary(records, out_path):
    summary_rows = build_scene_summary(records)
    columns = (
        "platform",
        "scene",
        "num_records",
        "mean_top1_iou",
        "mean_max_top5_iou",
        "mean_max_top10_iou",
        "acc25_top1",
        "acc50_top1",
        "acc25_top10",
        "acc50_top10",
    )
    flag_columns = []
    for flag in FLAG_COLUMNS:
        clean_name = clean_flag_name(flag)
        flag_columns.extend((f"{clean_name}_count", f"{clean_name}_ratio"))

    with Path(out_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + tuple(flag_columns))
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def clean_flag_name(flag_column):
    prefix = "flag_"
    if flag_column.startswith(prefix):
        return flag_column[len(prefix) :]
    return flag_column


def write_diagnostic_counts(records, out_path):
    rows = []
    for platform, platform_records in group_by_platform(records).items():
        total = len(platform_records)
        for flag in FLAG_COLUMNS:
            count = sum(bool_to_int(record[flag]) for record in platform_records)
            rows.append(
                {
                    "platform": platform,
                    "flag_name": clean_flag_name(flag),
                    "count": count,
                    "ratio": ratio(count, total),
                }
            )

    with Path(out_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("platform", "flag_name", "count", "ratio"))
        writer.writeheader()
        writer.writerows(rows)


def sanitize_filename_part(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_") or "unknown"


def stable_sample(records, limit, seed, seed_parts):
    if limit < 0:
        raise ValueError("sample limit must be non-negative")
    records = sorted(records, key=lambda record: (str(record.get("id", "")), record["json_path"]))
    if len(records) <= limit:
        return records

    seed_text = "|".join([str(seed)] + [str(part) for part in seed_parts])
    seed_value = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    sampled = random.Random(seed_value).sample(records, limit)
    return sorted(sampled, key=lambda record: (str(record.get("id", "")), record["json_path"]))


def scene_balanced_sample(records, platform, primary_label, samples_per_scene, seed):
    platform_key = str(platform).lower()
    scene_groups = defaultdict(list)
    for record in records:
        if str(record.get("platform") or "").lower() != platform_key:
            continue
        if record["primary_label"] != primary_label:
            continue
        scene_groups[str(record.get("scene") or "")].append(record)

    sampled_records = []
    for scene in sorted(scene_groups):
        sampled_records.extend(
            stable_sample(scene_groups[scene], samples_per_scene, seed, (platform_key, scene, primary_label))
        )
    return sorted(
        sampled_records,
        key=lambda record: (str(record.get("scene", "")), str(record.get("id", "")), record["json_path"]),
    )


def write_samples(records, samples_dir, samples_per_group, seed):
    if samples_per_group < 0:
        raise ValueError("--samples_per_group must be non-negative")

    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    grouped = group_by_platform(records)
    for platform, platform_records in grouped.items():
        for primary_label in PRIMARY_LABELS:
            group_records = [record for record in platform_records if record["primary_label"] == primary_label]
            group_records = sorted(group_records, key=lambda record: (str(record.get("id", "")), record["json_path"]))
            if len(group_records) > samples_per_group:
                sampled = rng.sample(group_records, samples_per_group)
                sampled = sorted(sampled, key=lambda record: (str(record.get("id", "")), record["json_path"]))
            else:
                sampled = group_records

            filename = f"{sanitize_filename_part(platform)}_{primary_label}.csv"
            write_csv(sampled, samples_dir / filename, SAMPLE_COLUMNS)


def write_scene_balanced_samples(records, samples_dir, samples_per_scene, seed):
    if samples_per_scene < 0:
        raise ValueError("--samples_per_scene must be non-negative")

    samples_dir = Path(samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)

    for platform in group_by_platform(records):
        for primary_label in PRIMARY_LABELS:
            sampled = scene_balanced_sample(records, platform, primary_label, samples_per_scene, seed)
            filename = f"{sanitize_filename_part(platform)}_{primary_label}_scene_balanced.csv"
            write_csv(sampled, samples_dir / filename, SAMPLE_COLUMNS)


def write_manual_review_template(records, out_path, samples_per_scene, seed):
    if samples_per_scene < 0:
        raise ValueError("--samples_per_scene must be non-negative")

    review_groups = (
        ("drone", "A_recall_fail_25"),
        ("drone", "C_coarse_success_precise_fail"),
        ("drone", "D_ranking_fail_50"),
        ("quad", "E_strict_success"),
    )

    sampled = []
    for platform, primary_label in review_groups:
        sampled.extend(scene_balanced_sample(records, platform, primary_label, samples_per_scene, seed))
    sampled = sorted(
        sampled,
        key=lambda record: (
            str(record.get("platform", "")),
            record["primary_label"],
            str(record.get("scene", "")),
            str(record.get("id", "")),
        ),
    )
    write_csv(sampled, out_path, MANUAL_REVIEW_COLUMNS)


def write_summary_txt(summary_rows, scene_summary_rows, out_path, drone_root, quad_root, out_dir):
    lines = [
        "Diagnostic baseline20 DQ summary",
        f"drone_root: {display_path(drone_root)}",
        f"quad_root: {display_path(quad_root)}",
        f"out_dir: {display_path(out_dir)}",
        "",
    ]

    scenes_by_platform = defaultdict(list)
    for scene_row in scene_summary_rows:
        scenes_by_platform[scene_row["platform"]].append(scene_row)

    for row in summary_rows:
        lines.extend(
            [
                f"[{row['platform']}]",
                f"num_records: {row['num_records']}",
                f"mean_top1_iou: {row['mean_top1_iou']:.4f}",
                f"mean_max_top5_iou: {row['mean_max_top5_iou']:.4f}",
                f"mean_max_top10_iou: {row['mean_max_top10_iou']:.4f}",
                f"acc25_top1: {row['acc25_top1']:.4f}",
                f"acc50_top1: {row['acc50_top1']:.4f}",
                f"acc25_top10: {row['acc25_top10']:.4f}",
                f"acc50_top10: {row['acc50_top10']:.4f}",
                "diagnostic_flags:",
            ]
        )
        for flag in FLAG_COLUMNS:
            clean_name = clean_flag_name(flag)
            lines.append(f"  {clean_name}: count={row[f'{flag}_count']} ratio={row[f'{flag}_ratio']:.4f}")
        lines.append("scene_summary:")
        for scene_row in scenes_by_platform.get(row["platform"], []):
            scene_name = scene_row["scene"] or "<missing_scene>"
            lines.extend(
                [
                    f"  {scene_name}:",
                    f"    num_records: {scene_row['num_records']}",
                    f"    acc25_top1: {scene_row['acc25_top1']:.4f}",
                    f"    acc50_top1: {scene_row['acc50_top1']:.4f}",
                    f"    acc25_top10: {scene_row['acc25_top10']:.4f}",
                    f"    acc50_top10: {scene_row['acc50_top10']:.4f}",
                    f"    A_recall_fail_25_ratio: {scene_row['A_recall_fail_25_ratio']:.4f}",
                    f"    C_coarse_success_precise_fail_ratio: {scene_row['C_coarse_success_precise_fail_ratio']:.4f}",
                    f"    E_strict_success_ratio: {scene_row['E_strict_success_ratio']:.4f}",
                ]
            )
        lines.append("")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    drone_records, drone_paths = load_root(args.drone_root, "drone")
    quad_records, quad_paths = load_root(args.quad_root, "quad")
    records = drone_records + quad_records
    if not records:
        raise ValueError("No records were loaded from prediction.json files.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(records, out_dir / "all_records.jsonl")
    write_csv(records, out_dir / "all_records.csv", CSV_COLUMNS)
    summary_rows = write_summary_by_platform(records, out_dir / "summary_by_platform.csv")
    scene_summary_rows = write_scene_summary(records, out_dir / "scene_summary.csv")
    write_diagnostic_counts(records, out_dir / "diagnostic_counts.csv")
    write_samples(records, out_dir / "samples", args.samples_per_group, args.seed)
    write_scene_balanced_samples(records, out_dir / "samples_scene_balanced", args.samples_per_scene, args.seed)
    write_manual_review_template(records, out_dir / "manual_review_template.csv", args.samples_per_scene, args.seed)
    write_summary_txt(summary_rows, scene_summary_rows, out_dir / "summary.txt", args.drone_root, args.quad_root, out_dir)

    print(f"Loaded {len(drone_records)} records from {len(drone_paths)} Drone prediction.json files.")
    print(f"Loaded {len(quad_records)} records from {len(quad_paths)} Quad prediction.json files.")
    print(f"Saved diagnostic outputs to {display_path(out_dir)}")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
