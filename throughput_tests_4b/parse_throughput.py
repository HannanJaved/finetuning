#!/usr/bin/env python3
"""Parse throughput metrics from SLURM logs (HuggingFace Trainer format).

Extracts samples/s and runtime from the Trainer summary dict logged to stderr,
as well as per-step samples/s from intermediate step logs.  Derives GPU/node
counts from the filename pattern sft_tp_<N>g<M>n_*.

Usage:
    python parse_throughput.py                          # table to stdout
    python parse_throughput.py --output results.csv     # CSV
    python parse_throughput.py --plot scaling.png       # table + scaling plot
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any

# HF Trainer final summary: {'train_runtime': ..., 'train_samples_per_second': ..., ...}
# Accelerate prefixes lines with [hostname:rank]: so we match from { onwards.
_SUMMARY_RE = re.compile(r"\{[^{}]*'train_runtime'[^{}]*\}")

# Per-step log line dict, e.g. {'loss': 0.9, 'epoch': 0.01, ...}
# These do NOT contain samples/s so we skip them for throughput.

# Filename pattern: sft_tp_4g1n_3288955.out  →  gpus=4, nodes=1
_FNAME_RE = re.compile(r"_(\d+)g(\d+)n_")

# max_length from config yaml (used to convert samples/s → tokens/s estimate)
MAX_SEQ_LEN = 4096


def _parse_summary(text: str) -> dict[str, Any] | None:
    """Return the first train summary dict found in text, or None."""
    m = _SUMMARY_RE.search(text)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group())
        if isinstance(d, dict) and "train_runtime" in d:
            return d
    except Exception:
        pass
    return None


def _gpu_nodes_from_name(name: str) -> tuple[int | None, int | None]:
    m = _FNAME_RE.search(name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def parse_log(log_path: Path) -> dict[str, Any]:
    gpus, nodes = _gpu_nodes_from_name(log_path.name)

    # HF Trainer writes training output to stderr → .err file
    err_path = log_path.with_suffix(".err")

    summary: dict[str, Any] | None = None
    for path in (log_path, err_path):
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            summary = _parse_summary(text)
            if summary:
                break

    samples_per_s: float | None = summary.get("train_samples_per_second") if summary else None
    train_runtime: float | None = summary.get("train_runtime") if summary else None
    train_loss: float | None = summary.get("train_loss") if summary else None

    # Estimate tokens/s: samples/s × max_seq_len (upper bound — actual seqs are shorter)
    tokens_per_s: float | None = (
        samples_per_s * MAX_SEQ_LEN if samples_per_s is not None else None
    )

    # Scaling efficiency relative to 1-GPU baseline is computed after all rows are collected.
    return {
        "log_file": log_path.name,
        "gpus": gpus,
        "nodes": nodes,
        "samples_per_s": samples_per_s,
        "tokens_per_s_est": tokens_per_s,
        "train_runtime_s": train_runtime,
        "train_loss": train_loss,
    }


def _fmt(val: float | None, fmt: str = ".2f") -> str:
    return f"{val:{fmt}}" if val is not None else "n/a"


def print_table(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'log_file':42} | {'GPUs':>4} | {'samp/s':>8} | "
        f"{'tok/s (est)':>12} | {'runtime(s)':>10} | {'loss':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['log_file'][:42]:42} | "
            f"{row['gpus'] if row['gpus'] is not None else '?':>4} | "
            f"{_fmt(row['samples_per_s']):>8} | "
            f"{_fmt(row['tokens_per_s_est'], '.0f'):>12} | "
            f"{_fmt(row['train_runtime_s'], '.1f'):>10} | "
            f"{_fmt(row['train_loss']):>7}"
        )

    # Print scaling efficiency table if we have a 1-GPU baseline
    baseline = next((r for r in rows if r["gpus"] == 1 and r["samples_per_s"] is not None), None)
    if baseline and any(r["gpus"] != 1 and r["samples_per_s"] is not None for r in rows):
        print()
        print(f"{'log_file':42} | {'GPUs':>4} | {'scaling eff':>11}")
        print("-" * 62)
        base_sps = baseline["samples_per_s"]
        for row in rows:
            if row["samples_per_s"] is None or row["gpus"] is None:
                continue
            eff = row["samples_per_s"] / (base_sps * row["gpus"])
            print(
                f"{row['log_file'][:42]:42} | "
                f"{row['gpus']:>4} | "
                f"{eff:>10.1%}"
            )


def make_plot(rows: list[dict[str, Any]], plot_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot. Install with: pip install matplotlib")
        return

    valid = [r for r in rows if r["gpus"] is not None and r["samples_per_s"] is not None]
    if not valid:
        print("No valid data points for plotting.")
        return

    valid.sort(key=lambda r: r["gpus"])
    gpus = [r["gpus"] for r in valid]
    nodes = [r["nodes"] for r in valid]
    sps = [r["samples_per_s"] for r in valid]

    baseline_sps = sps[0] / gpus[0]  # samples/s per GPU at smallest config

    xlabels = [
        f"{g} GPU{'s' if g > 1 else ''}\n({n} node{'s' if n > 1 else ''})"
        for g, n in zip(gpus, nodes)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Left: throughput vs GPUs ---
    ax = axes[0]
    ax.plot(range(len(gpus)), sps, "o-", color="#2563eb", linewidth=2, markersize=7)
    ax.set_ylabel("Samples / second")
    ax.set_title("SFT-Capella Throughput Scaling\n(Qwen3-4B, SFT fine-tuning with TRL)")
    ax.set_xticks(range(len(gpus)))
    ax.set_xticklabels(xlabels)
    ax.grid(True, alpha=0.3)

    # --- Right: scaling efficiency ---
    ax = axes[1]
    eff = [s / (baseline_sps * g) * 100 for s, g in zip(sps, gpus)]
    bars = ax.bar(range(len(gpus)), eff, color="#2563eb", alpha=0.8)
    ax.set_ylabel("Scaling Efficiency (%)")
    ax.set_title("Parallel Scaling Efficiency on Capella for SFT on Qwen3-4B")
    ax.set_xticks(range(len(gpus)))
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 115)
    for bar, e in zip(bars, eff):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{e:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "SFT-Capella Throughput Sweep",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize HF Trainer throughput from SLURM logs.")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("/data/cat/ws/hama901h-Posttraining/.logs/Qwen3-4B/throughput/4B"),
        help="Directory containing SLURM .out/.err log files.",
    )
    parser.add_argument(
        "--glob",
        default="*.out",
        help="Glob pattern to select log files (default: *.out).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./throughput_4B.csv"),
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("./throughput_4B.png"),
        metavar="FILE.png",
        help="Save throughput + scaling efficiency plot to this path.",
    )

    args = parser.parse_args()
    logs_dir = args.logs_dir
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        return

    log_files = sorted(logs_dir.glob(args.glob))
    if not log_files:
        print(f"No logs matched {args.glob} in {logs_dir}")
        return

    rows = [parse_log(path) for path in log_files]

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote throughput summary to {args.output}")

    print_table(rows)

    if args.plot:
        make_plot(rows, args.plot)


if __name__ == "__main__":
    main()
