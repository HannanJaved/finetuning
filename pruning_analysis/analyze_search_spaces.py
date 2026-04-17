#!/usr/bin/env python3
"""Batch analysis for pruned configs grouped by search space.

This script reuses pruning_analysis.py to compare each search space's buckets
against the same supernetwork/original config.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import pruning_analysis as pa  # noqa: E402

BUCKET_LABELS = {0: "410m", 1: "1b", 2: "2.8b"}
FILENAME_RE = re.compile(r"subnet_config_evolutionary_search_(.+)_([0-2])_100_epochs\.yaml$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pruning analysis for each search-space config set.",
    )
    parser.add_argument(
        "--config-dir",
        default=(
            "/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/"
            "hgf_ivw0083-checkpoints/pythia_all_configs/pythia_6.9b"
        ),
        help="Directory containing subnet_config_* YAML files.",
    )
    parser.add_argument(
        "--original",
        default=(
            "/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/"
            "hgf_ivw0083-checkpoints/checkpoints/EleutherAI/pythia-6.9b"
        ),
        help="Original/supernetwork config directory or file.",
    )
    parser.add_argument(
        "--output-root",
        default=(
            "/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/"
            "hgf_ivw0083-Post-training/finetuning/pruning_analysis/output_search_spaces"
        ),
        help="Root output directory for per-search-space analyses.",
    )
    return parser.parse_args()


def _load_original_stats(original_path: str) -> pa.ModelStats:
    resolved = pa._resolve_original_config(original_path)
    cfg = pa._load_yaml(resolved) if resolved.endswith((".yaml", ".yml")) else pa._load_json(resolved)
    return pa._extract_original(cfg, "original")


def _collect_configs(config_dir: str) -> Dict[str, Dict[int, str]]:
    grouped: Dict[str, Dict[int, str]] = {}
    for entry in Path(config_dir).glob("subnet_config_evolutionary_search_*_100_epochs.yaml"):
        match = FILENAME_RE.match(entry.name)
        if not match:
            continue
        search_space, bucket_str = match.groups()
        bucket = int(bucket_str)
        grouped.setdefault(search_space, {})[bucket] = str(entry)
    return grouped


def _write_summary_file(output_dir: str, name: str, headers: List[str], rows: List[List[str]],
                        summary_lines: List[str]) -> None:
    filename = f"summary_{name}_pruning.txt"
    path = os.path.join(output_dir, filename)
    content = [
        "=== Pruning analysis summary ===",
        "  (* = value is mean of heterogeneous per-layer list)",
        "",
        pa._make_table(headers, rows),
        "",
        "=== Per-model ratio breakdown ===",
        "",
        "\n".join(summary_lines).rstrip(),
        "",
        "Notes:",
        "  attn_params = sum_layers[ 2*hidden*(n_head*head_size) + 2*(n_head*head_size)*hidden ]",
        "  ffn_params  = sum_layers[ 2*hidden*intermediate_size ]",
        "  total_params includes embedding table (vocab_size * hidden_size)",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(content))


def main() -> int:
    args = parse_args()
    grouped = _collect_configs(args.config_dir)
    if not grouped:
        raise RuntimeError(f"No configs found in {args.config_dir}")

    original_stats = _load_original_stats(args.original)
    os.makedirs(args.output_root, exist_ok=True)

    overview_rows: List[List[str]] = []
    overview_headers = ["search_space", "bucket", *pa.HEADERS]

    for search_space, buckets in sorted(grouped.items()):
        output_dir = os.path.join(args.output_root, search_space)
        os.makedirs(output_dir, exist_ok=True)

        ordered = [buckets[b] for b in sorted(buckets.keys())]
        labels = [f"{search_space} ({BUCKET_LABELS.get(b, str(b))})" for b in sorted(buckets.keys())]

        pruned_stats = []
        for path, label in zip(ordered, labels):
            cfg = pa._load_yaml(path)
            pruned_stats.append(pa._extract_pruned(cfg, label))

        all_stats = [original_stats] + pruned_stats
        rows = pa._build_rows(all_stats, original_stats)

        pa._write_csv(os.path.join(output_dir, "pruning_summary.csv"), pa.HEADERS, rows)
        pa._write_json(
            os.path.join(output_dir, "pruning_summary.json"),
            {
                "original_config": pa._resolve_original_config(args.original),
                "pruned_configs": ordered,
                "summary_rows": [dict(zip(pa.HEADERS, row)) for row in rows],
            },
        )

        summary_lines = pa._summary_lines(pruned_stats, original_stats)
        _write_summary_file(output_dir, search_space, pa.HEADERS, rows, summary_lines)

        if pa.plt is not None:
            pa._plot_ratios_vs_original(pruned_stats, original_stats, output_dir)
            pa._plot_param_breakdown(all_stats, output_dir)
            pa._plot_params_est(all_stats, output_dir)
            pa._plot_width_depth(all_stats, output_dir)
            pa._plot_ffn_attn_ratios(all_stats, output_dir)
            pa._plot_heads_layers_ratios(all_stats, original_stats, output_dir)
            pa._plot_ffn_attn_ratio_vs_original(all_stats, original_stats, output_dir)

            heterogeneous = [
                s
                for s in pruned_stats
                if not (s.n_head.is_uniform and s.head_size.is_uniform and s.intermediate_size.is_uniform)
            ]
            pa._plot_layerwise_profiles(heterogeneous, original_stats, output_dir)
            pa._plot_layerwise_ratios(heterogeneous, original_stats, output_dir)
            pa._plot_ffn_attn_ratio_per_layer(heterogeneous, output_dir)

        for bucket_id, row in zip(sorted(buckets.keys()), rows[1:]):
            overview_rows.append([search_space, str(bucket_id), *row])

    overview_path = os.path.join(args.output_root, "search_space_overview.csv")
    with open(overview_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(overview_headers)
        writer.writerows(overview_rows)

    print(f"Saved search-space analyses to: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
