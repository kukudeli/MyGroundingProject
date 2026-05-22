"""Compare Baseline 20e and PCE+PER 20e predictions on manual review samples.

This offline script uses the sample ids in Baseline manual_review_filled.csv,
looks up the same ids in Baseline and PCE+PER all_records.csv files, then
writes sample-level changes, label transitions, grouped summaries, and a
human-readable report.

Example:
    python tools/compare_baseline_pceper_samples.py \
      --manual_csv outputs/diagnostic_baseline20_dq/manual_review_filled.csv \
      --baseline_csv outputs/diagnostic_baseline20_dq/all_records.csv \
      --pceper_csv outputs/diagnostic_pce_per20_dq/all_records.csv \
      --out_dir outputs/diagnostic_compare_baseline_pceper20
"""

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_MANUAL_CSV = "outputs/diagnostic_baseline20_dq/manual_review_filled.csv"
DEFAULT_BASELINE_CSV = "outputs/diagnostic_baseline20_dq/all_records.csv"
DEFAULT_PCEPER_CSV = "outputs/diagnostic_pce_per20_dq/all_records.csv"
DEFAULT_OUT_DIR = "outputs/diagnostic_compare_baseline_pceper20"

LABEL_ORDER = (
    "A_recall_fail_25",
    "B_ranking_fail_25",
    "C_coarse_success_precise_fail",
    "D_ranking_fail_50",
    "E_strict_success",
    "unknown",
)
LABEL_RANK = {label: index for index, label in enumerate(LABEL_ORDER)}

MANUAL_REQUIRED_COLUMNS = (
    "id",
    "platform",
    "scene",
    "frame_id",
    "primary_label",
    "utterance",
    "top1_iou",
    "max_top5_iou",
    "max_top10_iou",
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

PRED_REQUIRED_COLUMNS = (
    "id",
    "platform",
    "scene",
    "frame_id",
    "top1_iou",
    "max_top5_iou",
    "max_top10_iou",
    "acc25_top1",
    "acc50_top1",
    "acc50_top10",
    "primary_label",
    "flag_A_recall_fail_25",
    "flag_B_ranking_fail_25",
    "flag_C_coarse_success_precise_fail",
    "flag_D_ranking_fail_50",
    "flag_E_strict_success",
    "json_path",
)

SAMPLE_COMPARE_COLUMNS = (
    "id",
    "platform",
    "scene",
    "frame_id",
    "utterance",
    "manual_target_size",
    "manual_distance",
    "manual_density",
    "manual_occlusion",
    "manual_similar_objects",
    "manual_edge_or_truncation",
    "manual_language_complexity",
    "manual_failure_reason",
    "manual_notes",
    "baseline_primary_label",
    "pceper_primary_label",
    "baseline_top1_iou",
    "pceper_top1_iou",
    "delta_top1_iou",
    "baseline_max_top5_iou",
    "pceper_max_top5_iou",
    "delta_max_top5_iou",
    "baseline_max_top10_iou",
    "pceper_max_top10_iou",
    "delta_max_top10_iou",
    "baseline_acc25_top1",
    "pceper_acc25_top1",
    "baseline_acc50_top1",
    "pceper_acc50_top1",
    "baseline_acc50_top10",
    "pceper_acc50_top10",
    "baseline_json_path",
    "pceper_json_path",
    "change_type",
    "change_note",
)

TRANSITION_COLUMNS = (
    "baseline_primary_label",
    "pceper_primary_label",
    "count",
    "ratio",
)

GROUP_SUMMARY_COLUMNS = (
    "platform",
    "baseline_primary_label",
    "num_samples",
    "mean_baseline_top1_iou",
    "mean_pceper_top1_iou",
    "mean_delta_top1_iou",
    "mean_baseline_max_top10_iou",
    "mean_pceper_max_top10_iou",
    "mean_delta_max_top10_iou",
    "num_improved_top1",
    "ratio_improved_top1",
    "num_worsened_top1",
    "ratio_worsened_top1",
    "num_unchanged_top1",
    "ratio_unchanged_top1",
    "num_fixed_to_E",
    "ratio_fixed_to_E",
    "num_broken_from_E",
    "ratio_broken_from_E",
)

EPSILON_TOP1 = 0.02


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Baseline 20e and PCE+PER 20e on manual diagnostic samples."
    )
    parser.add_argument("--manual_csv", default=DEFAULT_MANUAL_CSV, help="Baseline manual review CSV with sample ids.")
    parser.add_argument("--baseline_csv", default=DEFAULT_BASELINE_CSV, help="Baseline all_records.csv.")
    parser.add_argument("--pceper_csv", default=DEFAULT_PCEPER_CSV, help="PCE+PER all_records.csv.")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Directory to save comparison outputs.")
    return parser.parse_args()


def display_path(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def check_csv_path(path, name):
    csv_path = Path(path)
    if not csv_path.exists():
        raise ValueError(f"{name} does not exist: {display_path(csv_path)}")
    if not csv_path.is_file():
        raise ValueError(f"{name} is not a file: {display_path(csv_path)}")
    if csv_path.stat().st_size == 0:
        raise ValueError(f"{name} is empty: {display_path(csv_path)}")
    return csv_path


def read_csv_rows(path, required_columns, name):
    csv_path = check_csv_path(path, name)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{name} has no header: {display_path(csv_path)}")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{name} missing required columns {missing}: {display_path(csv_path)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{name} has no data rows: {display_path(csv_path)}")
    return rows


def index_rows_by_id(rows, name):
    indexed = {}
    duplicate_ids = []
    for row in rows:
        sample_id = str(row.get("id", "")).strip()
        if not sample_id:
            raise ValueError(f"{name} contains a row with empty id")
        if sample_id in indexed:
            duplicate_ids.append(sample_id)
        indexed[sample_id] = row
    if duplicate_ids:
        duplicates = ", ".join(sorted(set(duplicate_ids))[:20])
        raise ValueError(f"{name} contains duplicate ids: {duplicates}")
    return indexed


def require_float(row, field, source_name):
    sample_id = row.get("id", "<missing id>")
    value = row.get(field, "")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source_name}: id {sample_id!r} field {field!r} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{source_name}: id {sample_id!r} field {field!r} is not finite: {value!r}")
    return number


def parse_acc_value(row, field, source_name):
    sample_id = row.get("id", "<missing id>")
    value = str(row.get(field, "")).strip().lower()
    if value in {"1", "1.0", "true", "t", "yes", "y"}:
        return 1
    if value in {"0", "0.0", "false", "f", "no", "n"}:
        return 0
    raise ValueError(f"{source_name}: id {sample_id!r} field {field!r} is not a boolean/0/1 value: {row.get(field)!r}")


def label_sort_key(label):
    return (LABEL_RANK.get(label, len(LABEL_ORDER)), label)


def group_sort_key(platform, label):
    return (str(platform).lower(),) + label_sort_key(label)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def ratio(count, total):
    return count / total if total else 0.0


def classify_change(baseline_label, pceper_label, delta_top1):
    if baseline_label == "E_strict_success" and pceper_label != "E_strict_success":
        return "broken_from_strict_success"
    if baseline_label == "C_coarse_success_precise_fail" and pceper_label == "E_strict_success":
        return "precise_improved"
    if baseline_label == "D_ranking_fail_50" and pceper_label == "E_strict_success":
        return "ranking_improved"
    if baseline_label != "E_strict_success" and pceper_label == "E_strict_success":
        return "fixed_to_strict_success"
    if baseline_label == "A_recall_fail_25" and pceper_label != "A_recall_fail_25":
        return "recall_improved"
    if baseline_label != "A_recall_fail_25" and pceper_label == "A_recall_fail_25":
        return "worsened_to_A"
    if baseline_label == pceper_label and delta_top1 > EPSILON_TOP1:
        return "label_unchanged_iou_improved"
    if baseline_label == pceper_label and delta_top1 < -EPSILON_TOP1:
        return "label_unchanged_iou_worsened"
    if baseline_label == pceper_label:
        return "label_unchanged"
    return "label_changed_other"


def make_change_note(baseline_label, pceper_label, change_type):
    if change_type == "fixed_to_strict_success":
        return f"Baseline {baseline_label} -> PCE+PER {pceper_label}，说明样本被修复为严格定位成功。"
    if change_type == "broken_from_strict_success":
        return f"Baseline {baseline_label} -> PCE+PER {pceper_label}，说明原本严格成功样本被破坏。"
    if change_type == "recall_improved":
        return f"Baseline {baseline_label} -> PCE+PER {pceper_label}，说明候选召回改善，但仍未达到严格定位成功。"
    if change_type == "precise_improved":
        return "Baseline C -> PCE+PER E，说明精定位改善，样本被修复。"
    if change_type == "ranking_improved":
        return "Baseline D -> PCE+PER E，说明排序改善，样本被修复。"
    if change_type == "worsened_to_A":
        return f"Baseline {baseline_label} -> PCE+PER A，说明 PCE+PER 后退化为候选召回失败。"
    if change_type == "label_unchanged_iou_improved":
        return f"Baseline {baseline_label} -> PCE+PER {pceper_label}，标签未变，但 Top-1 IoU 有提升。"
    if change_type == "label_unchanged_iou_worsened":
        return f"Baseline {baseline_label} -> PCE+PER {pceper_label}，标签未变，但 Top-1 IoU 下降。"
    if change_type == "label_unchanged":
        return f"Baseline {baseline_label} -> PCE+PER {pceper_label}，类别和 IoU 基本没变。"
    return f"Baseline {baseline_label} -> PCE+PER {pceper_label}，标签发生其他变化。"


def build_compare_rows(manual_rows, baseline_index, pceper_index):
    manual_ids = [str(row["id"]).strip() for row in manual_rows]
    missing_in_baseline = [sample_id for sample_id in manual_ids if sample_id not in baseline_index]
    missing_in_pceper = [sample_id for sample_id in manual_ids if sample_id not in pceper_index]
    if missing_in_baseline or missing_in_pceper:
        messages = []
        if missing_in_baseline:
            messages.append("missing in baseline_csv: " + ", ".join(missing_in_baseline))
        if missing_in_pceper:
            messages.append("missing in pceper_csv: " + ", ".join(missing_in_pceper))
        raise ValueError("; ".join(messages))

    compare_rows = []
    for manual_row in manual_rows:
        sample_id = str(manual_row["id"]).strip()
        baseline_row = baseline_index[sample_id]
        pceper_row = pceper_index[sample_id]

        baseline_top1 = require_float(baseline_row, "top1_iou", "baseline_csv")
        pceper_top1 = require_float(pceper_row, "top1_iou", "pceper_csv")
        baseline_top5 = require_float(baseline_row, "max_top5_iou", "baseline_csv")
        pceper_top5 = require_float(pceper_row, "max_top5_iou", "pceper_csv")
        baseline_top10 = require_float(baseline_row, "max_top10_iou", "baseline_csv")
        pceper_top10 = require_float(pceper_row, "max_top10_iou", "pceper_csv")

        delta_top1 = pceper_top1 - baseline_top1
        delta_top5 = pceper_top5 - baseline_top5
        delta_top10 = pceper_top10 - baseline_top10
        baseline_label = baseline_row["primary_label"]
        pceper_label = pceper_row["primary_label"]
        change_type = classify_change(baseline_label, pceper_label, delta_top1)

        compare_rows.append(
            {
                "id": sample_id,
                "platform": manual_row["platform"],
                "scene": manual_row["scene"],
                "frame_id": manual_row["frame_id"],
                "utterance": manual_row["utterance"],
                "manual_target_size": manual_row["manual_target_size"],
                "manual_distance": manual_row["manual_distance"],
                "manual_density": manual_row["manual_density"],
                "manual_occlusion": manual_row["manual_occlusion"],
                "manual_similar_objects": manual_row["manual_similar_objects"],
                "manual_edge_or_truncation": manual_row["manual_edge_or_truncation"],
                "manual_language_complexity": manual_row["manual_language_complexity"],
                "manual_failure_reason": manual_row["manual_failure_reason"],
                "manual_notes": manual_row["manual_notes"],
                "baseline_primary_label": baseline_label,
                "pceper_primary_label": pceper_label,
                "baseline_top1_iou": baseline_top1,
                "pceper_top1_iou": pceper_top1,
                "delta_top1_iou": delta_top1,
                "baseline_max_top5_iou": baseline_top5,
                "pceper_max_top5_iou": pceper_top5,
                "delta_max_top5_iou": delta_top5,
                "baseline_max_top10_iou": baseline_top10,
                "pceper_max_top10_iou": pceper_top10,
                "delta_max_top10_iou": delta_top10,
                "baseline_acc25_top1": parse_acc_value(baseline_row, "acc25_top1", "baseline_csv"),
                "pceper_acc25_top1": parse_acc_value(pceper_row, "acc25_top1", "pceper_csv"),
                "baseline_acc50_top1": parse_acc_value(baseline_row, "acc50_top1", "baseline_csv"),
                "pceper_acc50_top1": parse_acc_value(pceper_row, "acc50_top1", "pceper_csv"),
                "baseline_acc50_top10": parse_acc_value(baseline_row, "acc50_top10", "baseline_csv"),
                "pceper_acc50_top10": parse_acc_value(pceper_row, "acc50_top10", "pceper_csv"),
                "baseline_json_path": baseline_row["json_path"],
                "pceper_json_path": pceper_row["json_path"],
                "change_type": change_type,
                "change_note": make_change_note(baseline_label, pceper_label, change_type),
            }
        )
    return compare_rows


def build_transition_rows(compare_rows):
    total = len(compare_rows)
    counter = Counter((row["baseline_primary_label"], row["pceper_primary_label"]) for row in compare_rows)
    rows = []
    for (baseline_label, pceper_label), count in sorted(
        counter.items(), key=lambda item: label_sort_key(item[0][0]) + label_sort_key(item[0][1])
    ):
        rows.append(
            {
                "baseline_primary_label": baseline_label,
                "pceper_primary_label": pceper_label,
                "count": count,
                "ratio": ratio(count, total),
            }
        )
    return rows


def build_group_summary_rows(compare_rows):
    grouped = defaultdict(list)
    for row in compare_rows:
        grouped[(row["platform"], row["baseline_primary_label"])].append(row)

    summary_rows = []
    for (platform, baseline_label), rows in sorted(grouped.items(), key=lambda item: group_sort_key(*item[0])):
        total = len(rows)
        num_improved = sum(1 for row in rows if row["delta_top1_iou"] > EPSILON_TOP1)
        num_worsened = sum(1 for row in rows if row["delta_top1_iou"] < -EPSILON_TOP1)
        num_unchanged = total - num_improved - num_worsened
        num_fixed_to_e = sum(
            1
            for row in rows
            if row["baseline_primary_label"] != "E_strict_success" and row["pceper_primary_label"] == "E_strict_success"
        )
        num_broken_from_e = sum(
            1
            for row in rows
            if row["baseline_primary_label"] == "E_strict_success" and row["pceper_primary_label"] != "E_strict_success"
        )
        summary_rows.append(
            {
                "platform": platform,
                "baseline_primary_label": baseline_label,
                "num_samples": total,
                "mean_baseline_top1_iou": mean([row["baseline_top1_iou"] for row in rows]),
                "mean_pceper_top1_iou": mean([row["pceper_top1_iou"] for row in rows]),
                "mean_delta_top1_iou": mean([row["delta_top1_iou"] for row in rows]),
                "mean_baseline_max_top10_iou": mean([row["baseline_max_top10_iou"] for row in rows]),
                "mean_pceper_max_top10_iou": mean([row["pceper_max_top10_iou"] for row in rows]),
                "mean_delta_max_top10_iou": mean([row["delta_max_top10_iou"] for row in rows]),
                "num_improved_top1": num_improved,
                "ratio_improved_top1": ratio(num_improved, total),
                "num_worsened_top1": num_worsened,
                "ratio_worsened_top1": ratio(num_worsened, total),
                "num_unchanged_top1": num_unchanged,
                "ratio_unchanged_top1": ratio(num_unchanged, total),
                "num_fixed_to_E": num_fixed_to_e,
                "ratio_fixed_to_E": ratio(num_fixed_to_e, total),
                "num_broken_from_E": num_broken_from_e,
                "ratio_broken_from_E": ratio(num_broken_from_e, total),
            }
        )
    return summary_rows


def write_csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def format_row_brief(row):
    return (
        f"{row['id']} | {row['platform']} | {row['baseline_primary_label']} -> {row['pceper_primary_label']} | "
        f"delta_top1={row['delta_top1_iou']:.4f} | delta_top10={row['delta_max_top10_iou']:.4f}"
    )


def write_summary_txt(path, args, compare_rows, transition_rows, group_summary_rows):
    fixed_to_e = [
        row
        for row in compare_rows
        if row["baseline_primary_label"] != "E_strict_success" and row["pceper_primary_label"] == "E_strict_success"
    ]
    broken_from_e = [
        row
        for row in compare_rows
        if row["baseline_primary_label"] == "E_strict_success" and row["pceper_primary_label"] != "E_strict_success"
    ]
    top_improved = sorted(compare_rows, key=lambda row: row["delta_top1_iou"], reverse=True)[:10]
    top_worsened = sorted(compare_rows, key=lambda row: row["delta_top1_iou"])[:10]

    lines = [
        "Baseline 20e vs PCE+PER 20e manual sample comparison",
        f"manual_csv: {display_path(args.manual_csv)}",
        f"baseline_csv: {display_path(args.baseline_csv)}",
        f"pceper_csv: {display_path(args.pceper_csv)}",
        f"out_dir: {display_path(args.out_dir)}",
        "",
        f"num_samples: {len(compare_rows)}",
        f"overall_mean_delta_top1_iou: {mean([row['delta_top1_iou'] for row in compare_rows]):.4f}",
        f"overall_mean_delta_max_top10_iou: {mean([row['delta_max_top10_iou'] for row in compare_rows]):.4f}",
        "",
        "label_transition_table:",
    ]

    for row in transition_rows:
        lines.append(
            f"  {row['baseline_primary_label']} -> {row['pceper_primary_label']}: "
            f"count={row['count']} ratio={row['ratio']:.4f}"
        )

    lines.extend(["", "group_summary:"])
    for row in group_summary_rows:
        lines.extend(
            [
                f"  [{row['platform']} | {row['baseline_primary_label']}]",
                f"    num_samples: {row['num_samples']}",
                f"    mean_delta_top1_iou: {row['mean_delta_top1_iou']:.4f}",
                f"    mean_delta_max_top10_iou: {row['mean_delta_max_top10_iou']:.4f}",
                f"    improved/worsened/unchanged_top1: "
                f"{row['num_improved_top1']}/{row['num_worsened_top1']}/{row['num_unchanged_top1']}",
                f"    fixed_to_E: {row['num_fixed_to_E']} ({row['ratio_fixed_to_E']:.4f})",
                f"    broken_from_E: {row['num_broken_from_E']} ({row['ratio_broken_from_E']:.4f})",
            ]
        )

    lines.extend(["", "fixed_to_E_samples:"])
    if fixed_to_e:
        lines.extend(f"  {format_row_brief(row)}" for row in fixed_to_e)
    else:
        lines.append("  <none>")

    lines.extend(["", "broken_from_E_samples:"])
    if broken_from_e:
        lines.extend(f"  {format_row_brief(row)}" for row in broken_from_e)
    else:
        lines.append("  <none>")

    lines.extend(["", "top10_delta_top1_iou_improved:"])
    lines.extend(f"  {format_row_brief(row)}" for row in top_improved)

    lines.extend(["", "top10_delta_top1_iou_worsened:"])
    lines.extend(f"  {format_row_brief(row)}" for row in top_worsened)

    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    manual_rows = read_csv_rows(args.manual_csv, MANUAL_REQUIRED_COLUMNS, "manual_csv")
    baseline_rows = read_csv_rows(args.baseline_csv, PRED_REQUIRED_COLUMNS, "baseline_csv")
    pceper_rows = read_csv_rows(args.pceper_csv, PRED_REQUIRED_COLUMNS, "pceper_csv")

    index_rows_by_id(manual_rows, "manual_csv")
    baseline_index = index_rows_by_id(baseline_rows, "baseline_csv")
    pceper_index = index_rows_by_id(pceper_rows, "pceper_csv")

    compare_rows = build_compare_rows(manual_rows, baseline_index, pceper_index)
    transition_rows = build_transition_rows(compare_rows)
    group_summary_rows = build_group_summary_rows(compare_rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "sample_level_compare.csv", compare_rows, SAMPLE_COMPARE_COLUMNS)
    write_csv(out_dir / "label_transition_counts.csv", transition_rows, TRANSITION_COLUMNS)
    write_csv(out_dir / "group_summary.csv", group_summary_rows, GROUP_SUMMARY_COLUMNS)
    write_summary_txt(out_dir / "compare_summary.txt", args, compare_rows, transition_rows, group_summary_rows)

    print(f"Loaded {len(manual_rows)} manual samples.")
    print(f"Saved comparison outputs to {display_path(out_dir)}")
    if len(manual_rows) != 45:
        print(f"Warning: expected 45 manual samples, found {len(manual_rows)}.")


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
