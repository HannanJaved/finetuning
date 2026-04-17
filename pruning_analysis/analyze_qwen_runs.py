#!/usr/bin/env python3
"""Analyze a fixed list of Qwen3 pruned configs without grouping by search space."""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import pruning_analysis as pa  # noqa: E402

BUCKET_RE = re.compile(r"_([0-2])_100_epochs\.yaml$")
SEARCH_RE = re.compile(r"_(evo|random)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a fixed set of Qwen3 pruned configs.",
    )
    parser.add_argument("--original", required=True, help="Original model directory or config.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--label", required=True, help="Label prefix for outputs.")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="List of pruned config YAML files.",
    )
    return parser.parse_args()


def _parse_meta(path: str) -> Tuple[str, str]:
    name = os.path.basename(path)
    bucket_match = BUCKET_RE.search(name)
    bucket = bucket_match.group(1) if bucket_match else "?"
    search_match = SEARCH_RE.search(name)
    search_type = search_match.group(1) if search_match else "unknown"
    return bucket, search_type


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    original_stats = pa._extract_original(
        pa._load_yaml(pa._resolve_original_config(args.original)), "original"
    )

    pruned_stats = []
    overview_rows: List[List[str]] = []
    config_meta: List[Tuple[str, str, str]] = []
    for path in args.configs:
        bucket, search_type = _parse_meta(path)
        label = f"{args.label}_b{bucket}_{search_type}"
        cfg = pa._load_yaml(path)
        pruned_stats.append(pa._extract_pruned(cfg, label))
        config_meta.append((bucket, search_type, path))

    all_stats = [original_stats] + pruned_stats
    rows = pa._build_rows(all_stats, original_stats)

    pa._write_csv(os.path.join(args.output_dir, "pruning_summary.csv"), pa.HEADERS, rows)
    pa._write_json(
        os.path.join(args.output_dir, "pruning_summary.json"),
        {
            "original_config": pa._resolve_original_config(args.original),
            "pruned_configs": args.configs,
            "summary_rows": [dict(zip(pa.HEADERS, row)) for row in rows],
        },
    )

    summary_lines = pa._summary_lines(pruned_stats, original_stats)
    summary_name = f"summary_{args.label}_pruning.txt"
    summary_path = os.path.join(args.output_dir, summary_name)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "=== Pruning analysis summary ===",
                    "  (* = value is mean of heterogeneous per-layer list)",
                    "",
                    pa._make_table(pa.HEADERS, rows),
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
            )
        )

    if pa.plt is not None:
        pa._plot_ratios_vs_original(pruned_stats, original_stats, args.output_dir)
        pa._plot_param_breakdown(all_stats, args.output_dir)
        pa._plot_params_est(all_stats, args.output_dir)
        pa._plot_width_depth(all_stats, args.output_dir)
        pa._plot_ffn_attn_ratios(all_stats, args.output_dir)
        pa._plot_heads_layers_ratios(all_stats, original_stats, args.output_dir)
        pa._plot_ffn_attn_ratio_vs_original(all_stats, original_stats, args.output_dir)

        heterogeneous = [
            s
            for s in pruned_stats
            if not (s.n_head.is_uniform and s.head_size.is_uniform and s.intermediate_size.is_uniform)
        ]
        pa._plot_layerwise_profiles(heterogeneous, original_stats, args.output_dir)
        pa._plot_layerwise_ratios(heterogeneous, original_stats, args.output_dir)
        pa._plot_ffn_attn_ratio_per_layer(heterogeneous, args.output_dir)

    for (bucket, search_type, _path), row in zip(config_meta, rows[1:]):
        overview_rows.append([bucket, search_type, *row])

    overview_path = os.path.join(args.output_dir, "overview.csv")
    with open(overview_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bucket", "search_type", *pa.HEADERS])
        writer.writerows(overview_rows)

    print(f"Saved Qwen3 analysis to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
