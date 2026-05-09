"""Plot platform imbalance probe curves.

Validation CSV is optional. The automatic export uses:

epoch,platform_id,platform_name,acc25,acc50,miou,count

If automatic per-platform validation export is not available, create a CSV
manually with this older compatible format:

epoch,platform,acc25,acc50,miou
10,drone,35.2,18.7,0.241
10,quad,41.5,23.1,0.269
20,drone,38.6,20.2,0.257
20,quad,42.0,23.4,0.271
"""

import argparse
import os
from pathlib import Path


def _import_deps():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("pandas is required for plotting. Please install pandas first.") from exc

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for plotting. Please install matplotlib first.") from exc

    return pd, plt


def _check_csv(path, name):
    if path is None:
        return None
    csv_path = Path(path)
    if not csv_path.exists():
        raise SystemExit(f"{name} does not exist: {csv_path}")
    if csv_path.stat().st_size == 0:
        raise SystemExit(f"{name} is empty: {csv_path}")
    return csv_path


def _read_csv(pd, path, name):
    csv_path = _check_csv(path, name)
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"{name} has no rows: {csv_path}")
    return df


def _smooth(series, smooth):
    if smooth <= 0:
        return series
    return series.ewm(alpha=1.0 - smooth, adjust=False).mean()


def _plot_grouped_line(df, x_col, y_col, group_col, out_path, ylabel, smooth, plt):
    fig, ax = plt.subplots()
    for group_name, group_df in df.groupby(group_col):
        group_df = group_df.sort_values(x_col)
        y = _smooth(group_df[y_col], smooth)
        ax.plot(group_df[x_col], y, label=str(group_name))
    ax.set_xlabel(x_col)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_train_csv(pd, plt, train_csv, out_dir, smooth):
    df = _read_csv(pd, train_csv, "train_csv")
    base_required = {"epoch", "batch_idx", "global_step", "platform_id", "platform_name", "count"}
    has_total_loss = "total_loss" in df.columns
    has_old_loss = "loss" in df.columns
    has_box_loss = "box_loss" in df.columns
    required = set(base_required)
    if has_total_loss:
        required.add("total_loss")
    elif has_old_loss:
        required.add("loss")
    else:
        raise SystemExit("train_csv must contain either total_loss or loss column.")
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"train_csv missing required columns: {sorted(missing)}")

    x_col = "global_step"
    if has_total_loss:
        _plot_grouped_line(
            df,
            x_col,
            "total_loss",
            "platform_name",
            os.path.join(out_dir, "platform_train_total_loss.png"),
            "total_loss",
            smooth,
            plt,
        )
    else:
        _plot_grouped_line(df, x_col, "loss", "platform_name", os.path.join(out_dir, "platform_train_loss.png"), "loss", smooth, plt)

    if has_box_loss:
        box_df = df.dropna(subset=["box_loss"])
        if not box_df.empty:
            _plot_grouped_line(
                box_df,
                x_col,
                "box_loss",
                "platform_name",
                os.path.join(out_dir, "platform_train_box_loss.png"),
                "box_loss",
                smooth,
                plt,
            )
    _plot_grouped_line(df, x_col, "count", "platform_name", os.path.join(out_dir, "platform_train_count.png"), "count", 0.0, plt)


def plot_val_csv(pd, plt, val_csv, out_dir, smooth):
    df = _read_csv(pd, val_csv, "val_csv")
    if "platform_name" in df.columns:
        platform_col = "platform_name"
    elif "platform" in df.columns:
        platform_col = "platform"
    else:
        raise SystemExit("val_csv must contain either platform_name or platform column.")

    required = {"epoch", platform_col, "acc25", "acc50", "miou"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"val_csv missing required columns: {sorted(missing)}")

    for metric in ["acc25", "acc50", "miou"]:
        _plot_grouped_line(
            df,
            "epoch",
            metric,
            platform_col,
            os.path.join(out_dir, f"platform_val_{metric}.png"),
            metric,
            smooth,
            plt,
        )

    pivot = df.pivot_table(index="epoch", columns=platform_col, values=["acc25", "acc50", "miou"], aggfunc="mean").sort_index()
    platforms = set(pivot.columns.get_level_values(1))
    if "drone" not in platforms or "quad" not in platforms:
        print("Warning: val_csv does not contain both drone and quad rows; skip platform_val_gap.png.")
        return

    gap_df = pd.DataFrame(index=pivot.index)
    gap_df["gap_acc25"] = (pivot[("acc25", "drone")] - pivot[("acc25", "quad")]).abs()
    gap_df["gap_acc50"] = (pivot[("acc50", "drone")] - pivot[("acc50", "quad")]).abs()
    gap_df["gap_miou"] = (pivot[("miou", "drone")] - pivot[("miou", "quad")]).abs()

    fig, ax = plt.subplots()
    for metric in gap_df.columns:
        ax.plot(gap_df.index, _smooth(gap_df[metric], smooth), label=metric)
    ax.set_xlabel("epoch")
    ax.set_ylabel("absolute gap")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "platform_val_gap.png"), dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot platform imbalance probe curves.")
    parser.add_argument("--train_csv", required=True, help="Path to platform_probe_train_loss.csv.")
    parser.add_argument("--val_csv", default=None, help="Optional validation metrics CSV: epoch,platform,acc25,acc50,miou.")
    parser.add_argument("--out_dir", required=True, help="Directory to save output plots.")
    parser.add_argument("--smooth", type=float, default=0.9, help="EMA smoothing factor. Use 0 to disable smoothing.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smooth < 0 or args.smooth >= 1:
        raise SystemExit("--smooth must be in [0, 1). Use 0 to disable smoothing.")

    pd, plt = _import_deps()
    os.makedirs(args.out_dir, exist_ok=True)
    plot_train_csv(pd, plt, args.train_csv, args.out_dir, args.smooth)
    if args.val_csv is not None:
        plot_val_csv(pd, plt, args.val_csv, args.out_dir, args.smooth)
    print(f"Saved plots to {args.out_dir}")


if __name__ == "__main__":
    main()
