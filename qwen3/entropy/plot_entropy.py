#!/usr/bin/env python3
"""Plot entropy results from compute_entropy.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot entropy results for SFT checkpoints.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("entropy_results.csv"),
        help="CSV produced by compute_entropy.py.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)

    if "checkpoint_step" in df.columns:
        df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce")
    df = df.sort_values(by=["size", "lr", "checkpoint_step"], na_position="last")

    fig, ax = plt.subplots(figsize=(10, 6))

    if args.group_by == "size":
        groups = df.groupby("size")
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
        groups = df.groupby("lr")
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
    ax.set_title(args.title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=8)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=200)


if __name__ == "__main__":
    main()
