#!/usr/bin/env python3
"""Compare Qwen3 8B vs 32B runs without search-space grouping."""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Qwen3 8B vs 32B runs.")
    parser.add_argument("--overview-8b", required=True, help="Path to 8B overview.csv")
    parser.add_argument("--overview-32b", required=True, help="Path to 32B overview.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    return parser.parse_args()


def _read(path: str, label: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            row = dict(row)
            row["model_scale"] = label
            rows.append(row)
        return rows


def _key(row: Dict[str, str]) -> tuple:
    return (row["bucket"], row["search_type"])


def _join(rows8: List[Dict[str, str]], rows32: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_key: Dict[tuple, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows8:
        by_key[_key(row)]["8b"] = row
    for row in rows32:
        by_key[_key(row)]["32b"] = row

    combined = []
    for key, variants in sorted(by_key.items()):
        bucket, search_type = key
        row8 = variants.get("8b")
        row32 = variants.get("32b")
        combined.append(
            {
                "bucket": bucket,
                "search_type": search_type,
                "hidden_8b": row8.get("hidden") if row8 else "n/a",
                "hidden_32b": row32.get("hidden") if row32 else "n/a",
                "layers_8b": row8.get("layers") if row8 else "n/a",
                "layers_32b": row32.get("layers") if row32 else "n/a",
                "heads_mean_8b": row8.get("heads(mean)") if row8 else "n/a",
                "heads_mean_32b": row32.get("heads(mean)") if row32 else "n/a",
                "head_sz_mean_8b": row8.get("head_sz(mean)") if row8 else "n/a",
                "head_sz_mean_32b": row32.get("head_sz(mean)") if row32 else "n/a",
                "ffn_hid_mean_8b": row8.get("ffn_hid(mean)") if row8 else "n/a",
                "ffn_hid_mean_32b": row32.get("ffn_hid(mean)") if row32 else "n/a",
                "width/depth_8b": row8.get("width/depth") if row8 else "n/a",
                "width/depth_32b": row32.get("width/depth") if row32 else "n/a",
                "ffn/attn_params_8b": row8.get("ffn/attn_params") if row8 else "n/a",
                "ffn/attn_params_32b": row32.get("ffn/attn_params") if row32 else "n/a",
                "hidden/orig_8b": row8.get("hidden/orig") if row8 else "n/a",
                "hidden/orig_32b": row32.get("hidden/orig") if row32 else "n/a",
                "layers/orig_8b": row8.get("layers/orig") if row8 else "n/a",
                "layers/orig_32b": row32.get("layers/orig") if row32 else "n/a",
                "heads/orig_8b": row8.get("heads/orig") if row8 else "n/a",
                "heads/orig_32b": row32.get("heads/orig") if row32 else "n/a",
            }
        )
    return combined


def _write_csv(path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: str, rows: List[Dict[str, str]]) -> None:
    lines = [
        "# Qwen3 8B vs 32B comparison",
        "",
        "| bucket | search_type | width/depth 8B | width/depth 32B | ffn/attn 8B | ffn/attn 32B | hidden/orig 8B | hidden/orig 32B |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {bucket} | {search_type} | {wd8} | {wd32} | {ffn8} | {ffn32} | {h8} | {h32} |".format(
                bucket=row["bucket"],
                search_type=row["search_type"],
                wd8=row["width/depth_8b"],
                wd32=row["width/depth_32b"],
                ffn8=row["ffn/attn_params_8b"],
                ffn32=row["ffn/attn_params_32b"],
                h8=row["hidden/orig_8b"],
                h32=row["hidden/orig_32b"],
            )
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _plot_metric(rows: List[Dict[str, str]], key8: str, key32: str, title: str, ylabel: str, filename: str, output_dir: str) -> None:
    if plt is None:
        return

    labels = [f"b{row['bucket']}-{row['search_type']}" for row in rows]
    x = list(range(len(labels)))
    vals8 = [float(str(row[key8]).replace("*", "")) if row[key8] != "n/a" else float("nan") for row in rows]
    vals32 = [float(str(row[key32]).replace("*", "")) if row[key32] != "n/a" else float("nan") for row in rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.4
    ax.bar([xi - width / 2 for xi in x], vals8, width=width, label="8B", color="#4C78A8")
    ax.bar([xi + width / 2 for xi in x], vals32, width=width, label="32B", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close(fig)


def _plot_multi_panel(rows: List[Dict[str, str]], output_dir: str) -> None:
    if plt is None:
        return

    labels = [f"b{row['bucket']}-{row['search_type']}" for row in rows]
    x = list(range(len(labels)))
    width = 0.4
    panels = [
        ("width/depth", "width/depth_8b", "width/depth_32b"),
        ("ffn/attn params", "ffn/attn_params_8b", "ffn/attn_params_32b"),
        ("hidden/orig", "hidden/orig_8b", "hidden/orig_32b"),
        ("layers/orig", "layers/orig_8b", "layers/orig_32b"),
        ("heads/orig", "heads/orig_8b", "heads/orig_32b"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 14), sharex=True)
    for ax, (title, key8, key32) in zip(axes, panels):
        vals8 = [float(str(row[key8]).replace("*", "")) if row[key8] != "n/a" else float("nan") for row in rows]
        vals32 = [float(str(row[key32]).replace("*", "")) if row[key32] != "n/a" else float("nan") for row in rows]
        ax.bar([xi - width / 2 for xi in x], vals8, width=width, label="8B", color="#4C78A8")
        ax.bar([xi + width / 2 for xi in x], vals32, width=width, label="32B", color="#F58518")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Qwen3 8B vs 32B comparison (no search-space grouping)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "combined_multi_panel.png"), dpi=200)
    plt.close(fig)


def _plot_absolute_multi_panel(rows: List[Dict[str, str]], output_dir: str) -> None:
    if plt is None:
        return

    labels = [f"b{row['bucket']}-{row['search_type']}" for row in rows]
    x = list(range(len(labels)))
    width = 0.4
    panels = [
        ("hidden", "hidden_8b", "hidden_32b"),
        ("layers", "layers_8b", "layers_32b"),
        ("heads(mean)", "heads_mean_8b", "heads_mean_32b"),
        ("head_sz(mean)", "head_sz_mean_8b", "head_sz_mean_32b"),
        ("ffn_hid(mean)", "ffn_hid_mean_8b", "ffn_hid_mean_32b"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 14), sharex=True)
    for ax, (title, key8, key32) in zip(axes, panels):
        vals8 = [float(str(row[key8]).replace("*", "")) if row[key8] != "n/a" else float("nan") for row in rows]
        vals32 = [float(str(row[key32]).replace("*", "")) if row[key32] != "n/a" else float("nan") for row in rows]
        ax.bar([xi - width / 2 for xi in x], vals8, width=width, label="8B", color="#4C78A8")
        ax.bar([xi + width / 2 for xi in x], vals32, width=width, label="32B", color="#F58518")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Qwen3 absolute architecture metrics")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "absolute_metrics_multi_panel.png"), dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows8 = _read(args.overview_8b, "8b")
    rows32 = _read(args.overview_32b, "32b")

    os.makedirs(args.output_dir, exist_ok=True)
    combined = _join(rows8, rows32)

    _write_csv(os.path.join(args.output_dir, "combined_report.csv"), combined)
    _write_markdown(os.path.join(args.output_dir, "combined_report.md"), combined)

    _plot_metric(combined, "width/depth_8b", "width/depth_32b", "Width/Depth ratio", "width/depth", "width_depth_compare.png", args.output_dir)
    _plot_metric(combined, "ffn/attn_params_8b", "ffn/attn_params_32b", "FFN/Attn ratio", "ffn/attn params", "ffn_attn_compare.png", args.output_dir)
    _plot_metric(combined, "hidden/orig_8b", "hidden/orig_32b", "Hidden retention ratio", "hidden/orig", "hidden_retention_compare.png", args.output_dir)
    _plot_metric(combined, "layers/orig_8b", "layers/orig_32b", "Layer retention ratio", "layers/orig", "layers_retention_compare.png", args.output_dir)
    _plot_metric(combined, "heads/orig_8b", "heads/orig_32b", "Head retention ratio", "heads/orig", "heads_retention_compare.png", args.output_dir)

    _plot_multi_panel(combined, args.output_dir)
    _plot_absolute_multi_panel(combined, args.output_dir)

    print(f"Saved Qwen3 comparison report to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
