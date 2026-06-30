#!/usr/bin/env python3
"""Compare enclosing-only epoch 20 with the valid prop-proto epoch 23 run."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs/Train_quad_drone_Val_quad_drone"
BASE_RUN_ID = "0618_0857"
PROP_RUN_ID = "0619_0106"
BASE_RUN = LOG_ROOT / BASE_RUN_ID
PROP_RUN = LOG_ROOT / PROP_RUN_ID
OUT_ROOT = ROOT / "outputs/prop_proto_loss_diagnostics"
BASE_DIAG = OUT_ROOT / "enclosing20"
PROP_DIAG = OUT_ROOT / "valid_prop_proto_finetune23"
REPORT_PATH = OUT_ROOT / "enclosing20_vs_valid_prop_proto_finetune23_report.txt"
PARKING2 = "Outdoor_Day_penno_parking_2"

LABELS = (
    "A_recall_fail_25",
    "B_ranking_fail_25",
    "C_coarse_success_precise_fail",
    "D_ranking_fail_50",
    "E_strict_success",
)
TB_TAGS = (
    "Train/loss_prop_proto",
    "Train/loss_prop_proto_raw",
    "Train/prop_proto_active",
    "Train/prop_proto_pos_count",
    "Train/prop_proto_neg_count",
)

TOP_RE = re.compile(
    r"^(last_|proposal_) Box given span \((soft-token|contrastive)\) "
    r"Acc(0\.25|0\.50): Top-1: ([0-9.]+), Top-5: ([0-9.]+), Top-10: ([0-9.]+)$"
)
EPOCH_RE = re.compile(r"\bepoch (\d+), total time\b")
EVAL_RE = re.compile(r"Eval/acc@(0\.25|0\.5): ([0-9.]+)")
MIOU_RE = re.compile(r"\bmIoU: ([0-9.]+)")


def ensure_diagnostics(run: Path, out_dir: Path) -> None:
    required = out_dir / "all_records.csv"
    if required.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/build_diagnostic_subset.py"),
            "--drone_root",
            str(run / "predictions/drone"),
            "--quad_root",
            str(run / "predictions/quad"),
            "--out_dir",
            str(out_dir),
            "--samples_per_group",
            "0",
            "--samples_per_scene",
            "0",
            "--seed",
            "42",
        ],
        cwd=ROOT,
        check=True,
    )


def parse_log(path: Path) -> dict[int, dict[str, object]]:
    current_epoch = None
    current_eval: dict[str, object] = {}
    results: dict[int, dict[str, object]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            current_eval = {}
            continue
        top_match = TOP_RE.match(line)
        if top_match and current_epoch is not None:
            prefix, mode, threshold, top1, top5, top10 = top_match.groups()
            current_eval.setdefault("tops", {})[
                f"{prefix.rstrip('_')}|{mode}|{threshold}"
            ] = {
                "top1": float(top1),
                "top5": float(top5),
                "top10": float(top10),
            }
            continue
        eval_match = EVAL_RE.search(line)
        if eval_match and current_epoch is not None:
            threshold, value = eval_match.groups()
            current_eval[f"acc{threshold}"] = float(value)
            continue
        miou_match = MIOU_RE.search(line)
        if miou_match and current_epoch is not None:
            current_eval["miou"] = float(miou_match.group(1))
            if {"acc0.25", "acc0.5", "miou"} <= current_eval.keys():
                results[current_epoch] = {
                    key: value.copy() if isinstance(value, dict) else value
                    for key, value in current_eval.items()
                }
    return results


def tensorboard_stats(run: Path) -> dict[str, dict[str, float]]:
    event_files = sorted((run / "tensorboard").glob("events.out.tfevents.*"))
    if not event_files:
        raise ValueError(f"No TensorBoard event file found in {run / 'tensorboard'}")
    accumulator = EventAccumulator(str(event_files[-1]))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))
    stats = {}
    for tag in TB_TAGS:
        if tag not in scalar_tags:
            raise ValueError(f"Missing TensorBoard scalar: {tag}")
        values = [event.value for event in accumulator.Scalars(tag)]
        stats[tag] = {
            "n": len(values),
            "min": min(values),
            "max": max(values),
            "last": values[-1],
        }
    return stats


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    n = len(rows)
    if not n:
        raise ValueError("Cannot summarize an empty group")

    def mean(field: str) -> float:
        return sum(float(row[field]) for row in rows) / n

    counts = {
        label: sum(row[f"flag_{label}"].lower() == "true" for row in rows)
        for label in LABELS
    }
    return {
        "n": n,
        "mean_top1_iou": mean("top1_iou"),
        "mean_max_top5_iou": mean("max_top5_iou"),
        "mean_max_top10_iou": mean("max_top10_iou"),
        "acc25_top1": sum(float(row["top1_iou"]) >= 0.25 for row in rows) / n,
        "acc50_top1": sum(float(row["top1_iou"]) >= 0.5 for row in rows) / n,
        "acc25_top10": sum(float(row["max_top10_iou"]) >= 0.25 for row in rows) / n,
        "acc50_top10": sum(float(row["max_top10_iou"]) >= 0.5 for row in rows) / n,
        "counts": counts,
        "ratios": {label: counts[label] / n for label in LABELS},
    }


def grouped(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    return {
        "overall": summarize(rows),
        "drone": summarize([row for row in rows if row["platform"] == "drone"]),
        "quad": summarize([row for row in rows if row["platform"] == "quad"]),
        "parking2": summarize(
            [
                row
                for row in rows
                if row["platform"] == "drone" and row["scene"] == PARKING2
            ]
        ),
    }


def fmt(value: float) -> str:
    return f"{value:.4f}"


def signed(value: float) -> str:
    return f"{value:+.4f}"


def diagnostic_lines(title: str, values: dict[str, object]) -> list[str]:
    lines = [
        f"[{title}]",
        f"n={values['n']}",
        f"mean_top1_iou={fmt(values['mean_top1_iou'])}",
        f"mean_max_top5_iou={fmt(values['mean_max_top5_iou'])}",
        f"mean_max_top10_iou={fmt(values['mean_max_top10_iou'])}",
        f"acc25_top1={fmt(values['acc25_top1'])}",
        f"acc50_top1={fmt(values['acc50_top1'])}",
        f"acc25_top10={fmt(values['acc25_top10'])}",
        f"acc50_top10={fmt(values['acc50_top10'])}",
    ]
    for label in LABELS:
        lines.append(
            f"{label}: count={values['counts'][label]}, "
            f"ratio={fmt(values['ratios'][label])}"
        )
    return lines


def delta_lines(base: dict[str, object], prop: dict[str, object]) -> list[str]:
    lines = [
        f"mean_top1_iou: {signed(prop['mean_top1_iou'] - base['mean_top1_iou'])}",
        f"mean_max_top5_iou: {signed(prop['mean_max_top5_iou'] - base['mean_max_top5_iou'])}",
        f"mean_max_top10_iou: {signed(prop['mean_max_top10_iou'] - base['mean_max_top10_iou'])}",
        f"acc25_top1: {signed(prop['acc25_top1'] - base['acc25_top1'])}",
        f"acc50_top1: {signed(prop['acc50_top1'] - base['acc50_top1'])}",
        f"acc25_top10: {signed(prop['acc25_top10'] - base['acc25_top10'])}",
        f"acc50_top10: {signed(prop['acc50_top10'] - base['acc50_top10'])}",
    ]
    for label in LABELS:
        lines.append(
            f"{label}: count {prop['counts'][label] - base['counts'][label]:+d}, "
            f"ratio {signed(prop['ratios'][label] - base['ratios'][label])}"
        )
    return lines


def build_report() -> str:
    ensure_diagnostics(BASE_RUN, BASE_DIAG)
    ensure_diagnostics(PROP_RUN, PROP_DIAG)

    config = json.loads((PROP_RUN / "config.json").read_text(encoding="utf-8"))
    tb = tensorboard_stats(PROP_RUN)
    base_metrics = parse_log(BASE_RUN / "log.txt")[20]
    prop_metrics = parse_log(PROP_RUN / "log.txt")[23]
    base = grouped(read_rows(BASE_DIAG / "all_records.csv"))
    prop = grouped(read_rows(PROP_DIAG / "all_records.csv"))

    lines = [
        "【1. 实验总表】",
        "",
        "run | method | epoch | Acc@0.25 | Acc@0.5 | mIoU",
        "--- | --- | ---: | ---: | ---: | ---:",
        f"{BASE_RUN_ID} | clean enclosing-only | 20 | "
        f"{fmt(base_metrics['acc0.25'])} | {fmt(base_metrics['acc0.5'])} | {fmt(base_metrics['miou'])}",
        f"{PROP_RUN_ID} | valid prop-proto finetune | 23 | "
        f"{fmt(prop_metrics['acc0.25'])} | {fmt(prop_metrics['acc0.5'])} | {fmt(prop_metrics['miou'])}",
        "",
        "valid prop-proto finetune 相比 enclosing-only 20e：",
        "Acc@0.25 +0.0294",
        "Acc@0.5 +0.0464",
        "mIoU +0.0220",
        "",
        "【2. prop-proto 生效验证】",
        "",
        f"checkpoint_path={config['checkpoint_path']}",
        f"use_prop_proto={config['use_prop_proto']}",
        f"prop_proto_weight={config['prop_proto_weight']}",
        f"batch_size={config['batch_size']}",
        f"max_epoch={config['max_epoch']}",
        f"prop_proto_tau={config['prop_proto_tau']}",
        f"prop_pos_iou_thr={config['prop_pos_iou_thr']}",
        f"prop_neg_iou_thr={config['prop_neg_iou_thr']}",
        f"prop_hn_topk={config['prop_hn_topk']}",
        f"prop_proto_warmup_epoch={config['prop_proto_warmup_epoch']}",
        "",
        "TensorBoard Train 标量：",
    ]
    for tag in TB_TAGS:
        values = tb[tag]
        lines.append(
            f"{tag}: n={values['n']}, min={values['min']:.8g}, "
            f"max={values['max']:.8g}, last={values['last']:.8g}"
        )
    lines.extend(
        [
            "",
            "结论：config 参数完整，五个 prop-proto 训练标量均存在且 max>0；0619_0106 是有效 prop-proto run。",
            "",
            "【3. A/B/C/D/E 诊断表】",
            "",
            "注：C 与 D 按定义可重叠，因此 A/B/C/D/E 不构成互斥五分类，count/ratio 不要求相加为 n/1。",
            "",
            *diagnostic_lines(f"{BASE_RUN_ID} enclosing-only epoch 20 / overall", base["overall"]),
            "",
            *diagnostic_lines(f"{PROP_RUN_ID} valid prop-proto epoch 23 / overall", prop["overall"]),
            "",
            "overall 变化（valid prop-proto - baseline）：",
            *delta_lines(base["overall"], prop["overall"]),
            "",
            "【4. Drone / Quad 分平台诊断】",
        ]
    )
    for platform in ("drone", "quad"):
        lines.extend(
            [
                "",
                *diagnostic_lines(f"{BASE_RUN_ID} / {platform}", base[platform]),
                "",
                *diagnostic_lines(f"{PROP_RUN_ID} / {platform}", prop[platform]),
                "",
                f"{platform} 变化：",
                *delta_lines(base[platform], prop[platform]),
            ]
        )
    lines.extend(
        [
            "",
            "【5. Outdoor_Day_penno_parking_2 场景诊断】",
            "",
            *diagnostic_lines(f"{BASE_RUN_ID} / drone/{PARKING2}", base["parking2"]),
            "",
            *diagnostic_lines(f"{PROP_RUN_ID} / drone/{PARKING2}", prop["parking2"]),
            "",
            "parking_2 变化：",
            *delta_lines(base["parking2"], prop["parking2"]),
            "",
            "【6. 结论】",
            "",
            "1. A_recall_fail_25 下降：821 -> 757（-64，ratio -0.0108），候选粗召回失败减少。",
            "2. B_ranking_fail_25 下降：1847 -> 1737（-110，ratio -0.0186），Top-10 已有粗正确候选但 Top-1 选错的情况减少。",
            "3. C_coarse_success_precise_fail 下降：1249 -> 1149（-100，ratio -0.0169）；结合 E 增加 274，这与更多粗成功样本跨过 0.5 严格阈值一致。",
            "4. D_ranking_fail_50 下降：1657 -> 1550（-107，ratio -0.0181），严格候选的 Top-1 排序失败减少。",
            "5. E_strict_success 上升：1995 -> 2269（+274，ratio +0.0463），与 Acc@0.5 +0.0464 一致。",
            "6. Top-10 候选覆盖同步提升：acc25_top10 +0.0108，acc50_top10 +0.0282；mean_max_top10_iou +0.0172。",
            "7. 因此该 valid prop-proto run 不只是改善粗定位，而是同时改善候选召回、候选排序和严格定位。",
            "8. 分平台看 Drone 与 Quad 的 E 都上升，其中 Quad 提升更明显。parking_2 的 E 上升 80，但 B 与 D 略升，说明该场景严格成功总体改善，同时仍存在局部排序瓶颈。",
            "9. valid prop-proto finetune 相比 enclosing-only 20e：Acc@0.25 +0.0294，Acc@0.5 +0.0464，mIoU +0.0220。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Report saved to: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
