#!/usr/bin/env python3
"""Plot entropy results from compute_entropy.py."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot entropy results for SFT checkpoints.")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=Path("results"),
        help=(
            "CSV file or directory containing per-run CSVs (default: results/). "
            "When a directory is provided, all *.csv files are loaded recursively."
        ),
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="(Deprecated) CSV produced by compute_entropy.py.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("entropy_plot.png"),
        help="Where to save the plot image.",
    )
    parser.add_argument(
        "--title",
        default="Average Token Entropy (probe set)",
        help="Plot title.",
    )
    parser.add_argument(
        "--group-by",
        choices=["size", "lr"],
        default="size",
        help="Group lines by size or learning rate.",
    )
    parser.add_argument(
        "--split-by-size",
        action="store_true",
        help="Save one plot per model size (uses --output-png as basename).",
    )
    return parser.parse_args()


def _normalize_lr(lr_value: str) -> str:
    match = re.search(r"LR(?P<lr>[0-9\.]+e[\-\+]?[0-9]+)", lr_value)
    if not match:
        return lr_value
    lr = match.group("lr")
    if "e-" in lr or "e+" in lr:
        return lr
    base, exp = lr.split("e", maxsplit=1)
    return f"{base}e-{exp}"


def _lr_to_float(lr_value: str) -> float | None:
    if lr_value is None:
        return None
    try:
        return float(lr_value)
    except (TypeError, ValueError):
        return None


def _derive_size(path: Path) -> str | None:
    for candidate in (path.parent.name, path.name):
        match = re.search(r"(?P<size>[0-9\.]+B)", candidate)
        if match:
            return match.group("size")
    return None


def _derive_lr(path: Path) -> str | None:
    match = re.search(r"LR(?P<lr>[0-9\.]+e[\-\+]?[0-9]+)", path.name)
    if match:
        lr = match.group("lr")
        return _normalize_lr(f"LR{lr}")
    match = re.search(r"LR(?P<lr>[0-9\.]+e[0-9]+)", path.name)
    if match:
        lr = match.group("lr")
        return _normalize_lr(f"LR{lr}")
    return None


def load_entropy_frames(input_path: Path) -> pd.DataFrame:
    if input_path.is_dir():
        csv_paths = sorted(input_path.rglob("*.csv"))
    elif input_path.is_file():
        csv_paths = [input_path]
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {input_path}")

    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df["source_file"] = str(csv_path)
        size = _derive_size(csv_path)
        lr = _derive_lr(csv_path)

        if "size" not in df.columns:
            df["size"] = size
        else:
            df["size"] = df["size"].replace("None", pd.NA).fillna(size)

        if "lr" not in df.columns:
            df["lr"] = lr
        else:
            df["lr"] = df["lr"].replace("None", pd.NA).fillna(lr)
            df["lr"] = df["lr"].astype(str).map(_normalize_lr)

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_path = args.input_csv if args.input_csv is not None else args.input_path
    if not input_path.is_absolute() and not input_path.exists():
        input_path = script_dir / input_path
    df = load_entropy_frames(input_path)

    if "checkpoint_step" in df.columns:
        df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce")
    df = df.sort_values(by=["size", "lr", "checkpoint_step"], na_position="last")

    output_png = args.output_png
    if not output_png.is_absolute():
        output_png = script_dir / output_png
    output_png.parent.mkdir(parents=True, exist_ok=True)

    def render_plot(plot_df: pd.DataFrame, title_suffix: str | None, target_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        if not args.split_by_size:
            summary = (
                plot_df.groupby(["size", "lr"], dropna=False)["avg_entropy"]
                .mean()
                .reset_index()
            )
            summary["lr_float"] = summary["lr"].astype(str).map(_lr_to_float)
            summary = summary.sort_values(by=["size", "lr_float"], na_position="last")
            for size, group in summary.groupby("size"):
                ax.plot(
                    group["lr_float"],
                    group["avg_entropy"],
                    marker="o",
                    label=f"{size}",
                )
            ax.set_xlabel("Learning rate")
            ax.set_xscale("log")
            ax.set_ylabel("Average token entropy")
        else:
            if args.group_by == "size":
                groups = plot_df.groupby("size")
                for size, group in groups:
                    pivot = group.pivot_table(
                        index="checkpoint_step",
                        columns="lr",
                        values="avg_entropy",
                        aggfunc="mean",
                    )
                    for lr in pivot.columns:
                        ax.plot(pivot.index, pivot[lr], marker="o", label=f"{size} | LR {lr}")
            else:
                groups = plot_df.groupby("lr")
                for lr, group in groups:
                    pivot = group.pivot_table(
                        index="checkpoint_step",
                        columns="size",
                        values="avg_entropy",
                        aggfunc="mean",
                    )
                    for size in pivot.columns:
                        ax.plot(pivot.index, pivot[size], marker="o", label=f"LR {lr} | {size}")

            ax.set_xlabel("Checkpoint step")
            ax.set_ylabel("Average token entropy")
        full_title = args.title
        if title_suffix:
            full_title = f"{full_title} ({title_suffix})"
        ax.set_title(full_title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(target_path, dpi=200)
        plt.close(fig)

    if args.split_by_size:
        for size, group in df.groupby("size"):
            suffix = str(size) if size not in {None, "nan"} else "unknown"
            target = output_png.with_name(f"{output_png.stem}_{suffix}{output_png.suffix}")
            render_plot(group, f"{size}" if size not in {None, "nan"} else "unknown", target)
    else:
        render_plot(df, None, output_png)


if __name__ == "__main__":
    main()
