#!/usr/bin/env python3
"""Generate the final three-way enclosing/proposal-prototype comparison."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = ROOT / "logs/Train_quad_drone_Val_quad_drone"
OUT_ROOT = ROOT / "outputs/prop_proto_loss_diagnostics"
REPORT = OUT_ROOT / "final_three_way_enclosing_prop_proto_report.txt"
PARKING2 = "Outdoor_Day_penno_parking_2"

RUNS = {
    "baseline": {
        "id": "0618_0857",
        "method": "enclosing-only 20e",
        "epoch": 20,
        "diag": "enclosing20",
    },
    "continued": {
        "id": "0619_1028",
        "method": "continued enclosing-only 20→23",
        "epoch": 23,
        "diag": "continued_enclosing_only23",
    },
    "prop": {
        "id": "0619_0106",
        "method": "valid prop-proto finetune 20→23",
        "epoch": 23,
        "diag": "valid_prop_proto_finetune23",
    },
}

LABELS = (
    "A_recall_fail_25",
    "B_ranking_fail_25",
    "C_coarse_success_precise_fail",
    "D_ranking_fail_50",
    "E_strict_success",
)

EPOCH_RE = re.compile(r"\bepoch (\d+), total time\b")
EVAL_RE = re.compile(r"Eval/acc@(0\.25|0\.5): ([0-9.]+)")
MIOU_RE = re.compile(r"\bmIoU: ([0-9.]+)")


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    epoch = None
    current: dict[str, float] = {}
    results: dict[int, dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = EPOCH_RE.search(line)
        if match:
            epoch = int(match.group(1))
            current = {}
            continue
        match = EVAL_RE.search(line)
        if match and epoch is not None:
            threshold, value = match.groups()
            current[f"acc{threshold}"] = float(value)
            continue
        match = MIOU_RE.search(line)
        if match and epoch is not None:
            current["miou"] = float(match.group(1))
            if {"acc0.25", "acc0.5", "miou"} <= current.keys():
                results[epoch] = current.copy()
    return results


def read_rows(diag_name: str) -> list[dict[str, str]]:
    path = OUT_ROOT / diag_name / "all_records.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    n = len(rows)

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


def groups(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
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


def table(title: str, values: dict[str, object]) -> list[str]:
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


def deltas(left: dict[str, object], right: dict[str, object]) -> list[str]:
    lines = []
    for field in (
        "mean_top1_iou",
        "mean_max_top5_iou",
        "mean_max_top10_iou",
        "acc25_top1",
        "acc50_top1",
        "acc25_top10",
        "acc50_top10",
    ):
        lines.append(f"{field}: {signed(right[field] - left[field])}")
    for label in LABELS:
        lines.append(
            f"{label}: count {right['counts'][label] - left['counts'][label]:+d}, "
            f"ratio {signed(right['ratios'][label] - left['ratios'][label])}"
        )
    return lines


def config_summary(run_id: str) -> dict[str, object]:
    return json.loads((LOG_ROOT / run_id / "config.json").read_text(encoding="utf-8"))


def build_report() -> str:
    metrics = {}
    diagnostics = {}
    configs = {}
    for key, run in RUNS.items():
        metrics[key] = parse_log(LOG_ROOT / run["id"] / "log.txt")[run["epoch"]]
        diagnostics[key] = groups(read_rows(run["diag"]))
        configs[key] = config_summary(run["id"])

    baseline = metrics["baseline"]
    continued = metrics["continued"]
    prop = metrics["prop"]

    lines = [
        "【1. 三组实验总表】",
        "",
        "run | method | start checkpoint | epoch | use_prop_proto | weight | Acc@0.25 | Acc@0.5 | mIoU",
        "--- | --- | --- | ---: | --- | ---: | ---: | ---: | ---:",
        f"0618_0857 | enclosing-only 20e | from scratch | 20 | False | 0.0 | "
        f"{fmt(baseline['acc0.25'])} | {fmt(baseline['acc0.5'])} | {fmt(baseline['miou'])}",
        f"0619_1028 | clean continued enclosing-only 20→23 | 0618_0857/ckpt_epoch_20.pth | 23 | False | 0.0 | "
        f"{fmt(continued['acc0.25'])} | {fmt(continued['acc0.5'])} | {fmt(continued['miou'])}",
        f"0619_0106 | valid prop-proto finetune 20→23 | 0618_0857/ckpt_epoch_20.pth | 23 | True | 0.001 | "
        f"{fmt(prop['acc0.25'])} | {fmt(prop['acc0.5'])} | {fmt(prop['miou'])}",
        "",
        "配置公平性核验：",
        f"- 0619_1028 checkpoint_path={configs['continued']['checkpoint_path']}",
        f"- 0619_0106 checkpoint_path={configs['prop']['checkpoint_path']}",
        f"- 两者 batch_size={configs['continued']['batch_size']}，max_epoch={configs['continued']['max_epoch']}。",
        f"- 0619_1028: use_prop_proto={configs['continued']['use_prop_proto']}, weight={configs['continued']['prop_proto_weight']}。",
        f"- 0619_0106: use_prop_proto={configs['prop']['use_prop_proto']}, weight={configs['prop']['prop_proto_weight']}。",
        "",
        "【2. 指标提升】",
        "",
        "prop-proto 20→23 相比 enclosing-only 20e：",
        f"- Acc@0.25 {signed(prop['acc0.25'] - baseline['acc0.25'])}",
        f"- Acc@0.5 {signed(prop['acc0.5'] - baseline['acc0.5'])}",
        f"- mIoU {signed(prop['miou'] - baseline['miou'])}",
        "",
        "prop-proto 20→23 相比 clean continued enclosing-only 20→23（公平主对比）：",
        f"- Acc@0.25 {signed(prop['acc0.25'] - continued['acc0.25'])}",
        f"- Acc@0.5 {signed(prop['acc0.5'] - continued['acc0.5'])}",
        f"- mIoU {signed(prop['miou'] - continued['miou'])}",
        "",
        "clean continued enclosing-only 20→23 相比 enclosing-only 20e：",
        f"- Acc@0.25 {signed(continued['acc0.25'] - baseline['acc0.25'])}",
        f"- Acc@0.5 {signed(continued['acc0.5'] - baseline['acc0.5'])}",
        f"- mIoU {signed(continued['miou'] - baseline['miou'])}",
        "",
        "【3. Overall A/B/C/D/E 公平诊断】",
        "",
        "注：C 与 D 可重叠，A/B/C/D/E 不构成互斥五分类。",
        "",
        *table("0618_0857 enclosing-only 20e / overall", diagnostics["baseline"]["overall"]),
        "",
        *table("0619_1028 continued enclosing-only / overall", diagnostics["continued"]["overall"]),
        "",
        *table("0619_0106 valid prop-proto / overall", diagnostics["prop"]["overall"]),
        "",
        "公平变化（prop-proto - continued-only）：",
        *deltas(diagnostics["continued"]["overall"], diagnostics["prop"]["overall"]),
        "",
        "参考：0618_0857 baseline vs valid prop-proto：",
        *deltas(diagnostics["baseline"]["overall"], diagnostics["prop"]["overall"]),
        "",
        "【4. Drone / Quad 公平诊断】",
    ]
    for platform in ("drone", "quad"):
        lines.extend(
            [
                "",
                *table(f"0619_1028 / {platform}", diagnostics["continued"][platform]),
                "",
                *table(f"0619_0106 / {platform}", diagnostics["prop"][platform]),
                "",
                f"{platform} 公平变化（prop-proto - continued-only）：",
                *deltas(diagnostics["continued"][platform], diagnostics["prop"][platform]),
            ]
        )
    lines.extend(
        [
            "",
            "【5. Drone Outdoor_Day_penno_parking_2 公平诊断】",
            "",
            *table("0619_1028 / drone/Outdoor_Day_penno_parking_2", diagnostics["continued"]["parking2"]),
            "",
            *table("0619_0106 / drone/Outdoor_Day_penno_parking_2", diagnostics["prop"]["parking2"]),
            "",
            "parking_2 公平变化（prop-proto - continued-only）：",
            *deltas(diagnostics["continued"]["parking2"], diagnostics["prop"]["parking2"]),
            "",
            "【6. 解释与最终结论】",
            "",
            "1. 单纯 continued enclosing-only 从 epoch 20 训练到 epoch 23 后三项指标均下降："
            "Acc@0.25 -0.0177、Acc@0.5 -0.0318、mIoU -0.0142。增加训练轮数本身没有带来收益。",
            "2. 在完全相同的起点、batch size 和训练终点下，valid prop-proto 相比 continued-only："
            "Acc@0.25 +0.0471、Acc@0.5 +0.0782、mIoU +0.0362。",
            "3. 公平 overall 诊断中，B 减少 305、C 减少 183、D 减少 106、E 增加 462。"
            "这是候选排序改善并转化为严格定位成功的直接证据。",
            "4. 严格候选覆盖也改善：acc50_top10 +0.0602、mean_max_top10_iou +0.0236。"
            "因此收益不仅来自 Top-1 重排，也包含更好的严格候选质量/覆盖。",
            "5. A 在公平对比中增加 26，acc25_top10 微降 0.0044。continued-only 虽产生了更多 IoU≥0.25 的粗候选，"
            "却伴随 B 增加和 E 大幅下降；prop-proto 的主要优势是把候选排对并推动样本跨过 0.5，而不是所有粗召回维度都单调改善。",
            "6. Quad 的公平收益最强：Acc@0.25 +0.0799、Acc@0.5 +0.1250，B/D 分别减少 290/125，E 增加 366。",
            "7. Drone 也获得稳定严格定位收益：Acc@0.5 +0.0322、E 增加 96。Drone 的 D 增加 19，"
            "但 acc50_top10 +0.0385 且 E 同时增加，说明更多样本获得严格候选，仍有一部分尚未排到 Top-1。",
            "8. parking_2 中 Acc@0.5 +0.0335、E 增加 94、A/B/C 均下降；D 增加 21，"
            "同样表明严格候选覆盖改善后仍保留局部精排空间。",
            "9. 综上，受控公平对照强力支持：最终提升不是由训练轮数增加造成，而是由 proposal-level prototype ranking finetuning 带来。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    REPORT.write_text(build_report(), encoding="utf-8")
    print(f"Report saved to: {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
