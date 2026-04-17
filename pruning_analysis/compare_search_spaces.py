#!/usr/bin/env python3
"""Generate a combined report comparing search-space analyses across model scales."""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - handled at runtime
    plt = None


WINNERS: Dict[str, List[Tuple[str, str]]] = {
    "6.9b": [
        ("finegrained", "0"),
        ("coarse_layerwise", "1"),
        ("coarse", "2"),
    ],
    "12b": [
        ("coarse", "0"),
        ("coarse_layerwise", "1"),
        ("coarse", "2"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare search-space pruning outputs between two model sizes.",
    )
    parser.add_argument(
        "--overview-6b",
        default=(
            "/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/"
            "hgf_ivw0083-Post-training/finetuning/pruning_analysis/output_search_spaces_pythia6.9b/"
            "search_space_overview.csv"
        ),
        help="Path to search_space_overview.csv for the 6.9b runs.",
    )
    parser.add_argument(
        "--overview-12b",
        default=(
            "/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/"
            "hgf_ivw0083-Post-training/finetuning/pruning_analysis/output_search_spaces_pythia12b/"
            "search_space_overview.csv"
        ),
        help="Path to search_space_overview.csv for the 12b runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/"
            "hgf_ivw0083-Post-training/finetuning/pruning_analysis/combined_report"
        ),
        help="Directory to write the combined report outputs.",
    )
    return parser.parse_args()


def _read_overview(path: str, label: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            row = dict(row)
            row["model_scale"] = label
            rows.append(row)
        return rows


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _make_key(row: Dict[str, str]) -> tuple:
    return (row["search_space"], row["bucket"])


def _is_winner(scale: str, search_space: str, bucket: str) -> bool:
    return (search_space, bucket) in WINNERS.get(scale, [])


def _join_rows(rows_6: List[Dict[str, str]], rows_12: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_key: Dict[tuple, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows_6:
        by_key[_make_key(row)]["6.9b"] = row
    for row in rows_12:
        by_key[_make_key(row)]["12b"] = row

    combined = []
    for key, variants in sorted(by_key.items()):
        search_space, bucket = key
        row6 = variants.get("6.9b")
        row12 = variants.get("12b")
        combined.append(
            {
                "search_space": search_space,
                "bucket": bucket,
                "winner_6.9b": "yes" if _is_winner("6.9b", search_space, bucket) else "no",
                "winner_12b": "yes" if _is_winner("12b", search_space, bucket) else "no",
                "hidden_6.9b": row6.get("hidden") if row6 else "n/a",
                "hidden_12b": row12.get("hidden") if row12 else "n/a",
                "layers_6.9b": row6.get("layers") if row6 else "n/a",
                "layers_12b": row12.get("layers") if row12 else "n/a",
                "heads_mean_6.9b": row6.get("heads(mean)") if row6 else "n/a",
                "heads_mean_12b": row12.get("heads(mean)") if row12 else "n/a",
                "head_sz_mean_6.9b": row6.get("head_sz(mean)") if row6 else "n/a",
                "head_sz_mean_12b": row12.get("head_sz(mean)") if row12 else "n/a",
                "ffn_hid_mean_6.9b": row6.get("ffn_hid(mean)") if row6 else "n/a",
                "ffn_hid_mean_12b": row12.get("ffn_hid(mean)") if row12 else "n/a",
                "q_groups_mean_6.9b": row6.get("q_groups(mean)") if row6 else "n/a",
                "q_groups_mean_12b": row12.get("q_groups(mean)") if row12 else "n/a",
                "params/orig_6.9b": row6.get("params/orig") if row6 else "n/a",
                "params/orig_12b": row12.get("params/orig") if row12 else "n/a",
                "attn_params/orig_6.9b": row6.get("attn_params/orig") if row6 else "n/a",
                "attn_params/orig_12b": row12.get("attn_params/orig") if row12 else "n/a",
                "ffn_params/orig_6.9b": row6.get("ffn_params/orig") if row6 else "n/a",
                "ffn_params/orig_12b": row12.get("ffn_params/orig") if row12 else "n/a",
                "heads/orig_6.9b": row6.get("heads/orig") if row6 else "n/a",
                "heads/orig_12b": row12.get("heads/orig") if row12 else "n/a",
                "layers/orig_6.9b": row6.get("layers/orig") if row6 else "n/a",
                "layers/orig_12b": row12.get("layers/orig") if row12 else "n/a",
                "hidden/orig_6.9b": row6.get("hidden/orig") if row6 else "n/a",
                "hidden/orig_12b": row12.get("hidden/orig") if row12 else "n/a",
                "width/depth_6.9b": row6.get("width/depth") if row6 else "n/a",
                "width/depth_12b": row12.get("width/depth") if row12 else "n/a",
                "ffn/attn_params_6.9b": row6.get("ffn/attn_params") if row6 else "n/a",
                "ffn/attn_params_12b": row12.get("ffn/attn_params") if row12 else "n/a",
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


def _write_txt(path: str, combined_rows: List[Dict[str, str]]) -> None:
    lines = [
        "=== Combined pruning report (6.9b vs 12b) ===",
        "",
        "Columns: params/orig, attn_params/orig, ffn_params/orig, heads/layers/hidden ratios.",
        "",
    ]
    header = (
        "search_space | bucket | params/orig 6.9b | params/orig 12b | "
        "attn/orig 6.9b | attn/orig 12b | ffn/orig 6.9b | ffn/orig 12b"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for row in combined_rows:
        lines.append(
            "{search_space} | {bucket} | {p6} | {p12} | {a6} | {a12} | {f6} | {f12}".format(
                search_space=row["search_space"],
                bucket=row["bucket"],
                p6=row["params/orig_6.9b"],
                p12=row["params/orig_12b"],
                a6=row["attn_params/orig_6.9b"],
                a12=row["attn_params/orig_12b"],
                f6=row["ffn_params/orig_6.9b"],
                f12=row["ffn_params/orig_12b"],
            )
        )
    lines.append("")
    lines.append("See combined_report.csv for full metrics.")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _write_markdown(path: str, combined_rows: List[Dict[str, str]]) -> None:
    lines = [
        "# Combined pruning report (6.9b vs 12b)",
        "",
        "Below is a summary of key ratios by search space and bucket.",
        "",
    "| search_space | bucket | winner (6.9b) | winner (12b) | params/orig (6.9b) | params/orig (12b) | "
    "attn/orig (6.9b) | attn/orig (12b) | ffn/orig (6.9b) | ffn/orig (12b) |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combined_rows:
        lines.append(
            "| {search_space} | {bucket} | {w6} | {w12} | {p6} | {p12} | {a6} | {a12} | {f6} | {f12} |".format(
                search_space=row["search_space"],
                bucket=row["bucket"],
                w6=row["winner_6.9b"],
                w12=row["winner_12b"],
                p6=row["params/orig_6.9b"],
                p12=row["params/orig_12b"],
                a6=row["attn_params/orig_6.9b"],
                a12=row["attn_params/orig_12b"],
                f6=row["ffn_params/orig_6.9b"],
                f12=row["ffn_params/orig_12b"],
            )
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def _plot_param_ratios(combined_rows: List[Dict[str, str]], output_dir: str) -> None:
    if plt is None:
        return

    labels = [f"{row['search_space']}-{row['bucket']}" for row in combined_rows]
    x = list(range(len(labels)))
    params_6 = [float(row["params/orig_6.9b"]) if row["params/orig_6.9b"] != "n/a" else float("nan")
                for row in combined_rows]
    params_12 = [float(row["params/orig_12b"]) if row["params/orig_12b"] != "n/a" else float("nan")
                 for row in combined_rows]

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.4
    ax.bar([xi - width / 2 for xi in x], params_6, width=width, label="6.9b", color="#4C78A8")
    ax.bar([xi + width / 2 for xi in x], params_12, width=width, label="12b", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("params/orig")
    ax.set_title("Pruned parameter retention: 6.9b vs 12b")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "params_orig_compare.png"), dpi=200)
    plt.close(fig)


def _plot_metric_side_by_side(
    combined_rows: List[Dict[str, str]],
    metric_6: str,
    metric_12: str,
    title: str,
    ylabel: str,
    filename: str,
    output_dir: str,
) -> None:
    if plt is None:
        return

    labels = [f"{row['search_space']}-{row['bucket']}" for row in combined_rows]
    x = list(range(len(labels)))
    values_6 = [
        float(row[metric_6]) if row[metric_6] != "n/a" else float("nan") for row in combined_rows
    ]
    values_12 = [
        float(row[metric_12]) if row[metric_12] != "n/a" else float("nan") for row in combined_rows
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.4
    bars_6 = ax.bar([xi - width / 2 for xi in x], values_6, width=width, label="6.9b", color="#4C78A8")
    bars_12 = ax.bar([xi + width / 2 for xi in x], values_12, width=width, label="12b", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    winners_6 = [row["winner_6.9b"] == "yes" for row in combined_rows]
    winners_12 = [row["winner_12b"] == "yes" for row in combined_rows]
    for bar, is_winner in zip(bars_6, winners_6):
        if is_winner:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                "★",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#4C78A8",
            )
    for bar, is_winner in zip(bars_12, winners_12):
        if is_winner:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                "★",
                ha="center",
                va="bottom",
                fontsize=10,
                color="#F58518",
            )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close(fig)


def _plot_retention_multi_panel(combined_rows: List[Dict[str, str]], output_dir: str) -> None:
    if plt is None:
        return

    labels = [f"{row['search_space']}-{row['bucket']}" for row in combined_rows]
    x = list(range(len(labels)))
    metrics = [
        ("hidden/orig", "hidden/orig_6.9b", "hidden/orig_12b"),
        ("layers/orig", "layers/orig_6.9b", "layers/orig_12b"),
        ("heads/orig", "heads/orig_6.9b", "heads/orig_12b"),
    ]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 9), sharex=True)
    width = 0.4
    winners_6 = [row["winner_6.9b"] == "yes" for row in combined_rows]
    winners_12 = [row["winner_12b"] == "yes" for row in combined_rows]
    for ax, (title, key6, key12) in zip(axes, metrics):
        vals_6 = [float(row[key6]) if row[key6] != "n/a" else float("nan") for row in combined_rows]
        vals_12 = [float(row[key12]) if row[key12] != "n/a" else float("nan") for row in combined_rows]
        bars_6 = ax.bar([xi - width / 2 for xi in x], vals_6, width=width, label="6.9b", color="#4C78A8")
        bars_12 = ax.bar([xi + width / 2 for xi in x], vals_12, width=width, label="12b", color="#F58518")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        for bar, is_winner in zip(bars_6, winners_6):
            if is_winner:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    "★",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#4C78A8",
                )
        for bar, is_winner in zip(bars_12, winners_12):
            if is_winner:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    "★",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#F58518",
                )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Retention ratios: 6.9b vs 12b")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "retention_ratios_compare.png"), dpi=200)
    plt.close(fig)


def _plot_full_multi_panel(combined_rows: List[Dict[str, str]], output_dir: str) -> None:
    if plt is None:
        return

    labels = [f"{row['search_space']}-{row['bucket']}" for row in combined_rows]
    x = list(range(len(labels)))
    width = 0.4

    panels = [
        ("width/depth", "width/depth_6.9b", "width/depth_12b"),
        ("ffn/attn params", "ffn/attn_params_6.9b", "ffn/attn_params_12b"),
        ("hidden/orig", "hidden/orig_6.9b", "hidden/orig_12b"),
        ("layers/orig", "layers/orig_6.9b", "layers/orig_12b"),
        ("heads/orig", "heads/orig_6.9b", "heads/orig_12b"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 14), sharex=True)
    winners_6 = [row["winner_6.9b"] == "yes" for row in combined_rows]
    winners_12 = [row["winner_12b"] == "yes" for row in combined_rows]
    for ax, (title, key6, key12) in zip(axes, panels):
        vals_6 = [float(row[key6]) if row[key6] != "n/a" else float("nan") for row in combined_rows]
        vals_12 = [float(row[key12]) if row[key12] != "n/a" else float("nan") for row in combined_rows]
        bars_6 = ax.bar([xi - width / 2 for xi in x], vals_6, width=width, label="6.9b", color="#4C78A8")
        bars_12 = ax.bar([xi + width / 2 for xi in x], vals_12, width=width, label="12b", color="#F58518")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        for bar, is_winner in zip(bars_6, winners_6):
            if is_winner:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    "★",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#4C78A8",
                )
        for bar, is_winner in zip(bars_12, winners_12):
            if is_winner:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    "★",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#F58518",
                )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Comparative ratios (side-by-side)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "combined_multi_panel.png"), dpi=200)
    plt.close(fig)


def _plot_absolute_multi_panel(combined_rows: List[Dict[str, str]], output_dir: str) -> None:
    if plt is None:
        return

    labels = [f"{row['search_space']}-{row['bucket']}" for row in combined_rows]
    x = list(range(len(labels)))
    width = 0.4

    panels = [
        ("hidden", "hidden_6.9b", "hidden_12b"),
        ("layers", "layers_6.9b", "layers_12b"),
        ("heads(mean)", "heads_mean_6.9b", "heads_mean_12b"),
        ("head_sz(mean)", "head_sz_mean_6.9b", "head_sz_mean_12b"),
        ("ffn_hid(mean)", "ffn_hid_mean_6.9b", "ffn_hid_mean_12b"),
        ("q_groups(mean)", "q_groups_mean_6.9b", "q_groups_mean_12b"),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 16), sharex=True)
    winners_6 = [row["winner_6.9b"] == "yes" for row in combined_rows]
    winners_12 = [row["winner_12b"] == "yes" for row in combined_rows]
    for ax, (title, key6, key12) in zip(axes, panels):
        vals_6 = [float(str(row[key6]).replace("*", "")) if row[key6] != "n/a" else float("nan")
                  for row in combined_rows]
        vals_12 = [float(str(row[key12]).replace("*", "")) if row[key12] != "n/a" else float("nan")
                   for row in combined_rows]
        bars_6 = ax.bar([xi - width / 2 for xi in x], vals_6, width=width, label="6.9b", color="#4C78A8")
        bars_12 = ax.bar([xi + width / 2 for xi in x], vals_12, width=width, label="12b", color="#F58518")
        ax.set_ylabel(title)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        for bar, is_winner in zip(bars_6, winners_6):
            if is_winner:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    "★",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#4C78A8",
                )
        for bar, is_winner in zip(bars_12, winners_12):
            if is_winner:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    "★",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#F58518",
                )

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels, rotation=20, ha="right")
    fig.suptitle("Absolute architecture metrics (side-by-side)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_dir, "absolute_metrics_multi_panel.png"), dpi=200)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    rows_6 = _read_overview(args.overview_6b, "6.9b")
    rows_12 = _read_overview(args.overview_12b, "12b")

    os.makedirs(args.output_dir, exist_ok=True)

    combined = _join_rows(rows_6, rows_12)
    _write_csv(os.path.join(args.output_dir, "combined_report.csv"), combined)
    _write_txt(os.path.join(args.output_dir, "combined_report.txt"), combined)
    _write_markdown(os.path.join(args.output_dir, "combined_report.md"), combined)
    _plot_metric_side_by_side(
        combined,
        "width/depth_6.9b",
        "width/depth_12b",
        "Width/Depth ratio comparison",
        "width/depth",
        "width_depth_compare.png",
        args.output_dir,
    )
    _plot_metric_side_by_side(
        combined,
        "ffn/attn_params_6.9b",
        "ffn/attn_params_12b",
        "FFN/Attn ratio comparison",
        "ffn/attn params",
        "ffn_attn_compare.png",
        args.output_dir,
    )
    _plot_metric_side_by_side(
        combined,
        "hidden/orig_6.9b",
        "hidden/orig_12b",
        "Hidden retention ratio",
        "hidden/orig",
        "hidden_retention_compare.png",
        args.output_dir,
    )
    _plot_metric_side_by_side(
        combined,
        "layers/orig_6.9b",
        "layers/orig_12b",
        "Layer retention ratio",
        "layers/orig",
        "layers_retention_compare.png",
        args.output_dir,
    )
    _plot_metric_side_by_side(
        combined,
        "heads/orig_6.9b",
        "heads/orig_12b",
        "Head retention ratio",
        "heads/orig",
        "heads_retention_compare.png",
        args.output_dir,
    )
    _plot_retention_multi_panel(combined, args.output_dir)
    _plot_full_multi_panel(combined, args.output_dir)
    _plot_absolute_multi_panel(combined, args.output_dir)

    print(f"Saved combined report to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
