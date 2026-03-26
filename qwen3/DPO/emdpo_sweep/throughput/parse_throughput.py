#!/usr/bin/env python3
"""Parse throughput metrics from SLURM logs.

Looks for tokens/sec and samples/sec in log lines and summarizes per log file.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import mean
from typing import Iterable

TOKEN_PATTERNS = [
    re.compile(
        r"(?P<tokens>\d+(?:\.\d+)?)\s*(?:tok/s|tokens/s|tokens/sec|tokens\s*/\s*s)",
        re.IGNORECASE,
    ),
]
SAMPLE_PATTERNS = [
    re.compile(
        r"(?P<samples>\d+(?:\.\d+)?)\s*(?:samples/s|samples/sec|samples\s*/\s*s|samp/s)",
        re.IGNORECASE,
    ),
]


def _extract_values(patterns: Iterable[re.Pattern[str]], text: str, key: str) -> list[float]:
    values: list[float] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            raw = match.group(key)
            try:
                values.append(float(raw))
            except ValueError:
                continue
    return values


def _summarize(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return mean(values), max(values)


def parse_log(log_path: Path) -> dict[str, float | int | str | None]:
    tokens_values: list[float] = []
    samples_values: list[float] = []
    matched_lines = 0

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            tokens_values.extend(_extract_values(TOKEN_PATTERNS, line, "tokens"))
            samples_values.extend(_extract_values(SAMPLE_PATTERNS, line, "samples"))
            if ("tokens" in line.lower() and "/s" in line.lower()) or ("samples" in line.lower() and "/s" in line.lower()):
                matched_lines += 1

    tokens_mean, tokens_max = _summarize(tokens_values)
    samples_mean, samples_max = _summarize(samples_values)

    return {
        "log_file": log_path.name,
        "tokens_per_s_mean": tokens_mean,
        "tokens_per_s_max": tokens_max,
        "samples_per_s_mean": samples_mean,
        "samples_per_s_max": samples_max,
        "matched_lines": matched_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize throughput metrics from SLURM logs.")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("/data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/throughput"),
        help="Directory containing SLURM throughput logs.",
    )
    parser.add_argument(
        "--glob",
        default="*.out",
        help="Glob pattern to select log files (default: *.out).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path. If omitted, prints a table to stdout.",
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
        return

    header = (
        f"{'log_file':40} | {'tok/s mean':>10} | {'tok/s max':>10} | "
        f"{'samp/s mean':>11} | {'samp/s max':>11} | lines"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['log_file'][:40]:40} | "
            f"{row['tokens_per_s_mean'] if row['tokens_per_s_mean'] is not None else 'n/a':>10} | "
            f"{row['tokens_per_s_max'] if row['tokens_per_s_max'] is not None else 'n/a':>10} | "
            f"{row['samples_per_s_mean'] if row['samples_per_s_mean'] is not None else 'n/a':>11} | "
            f"{row['samples_per_s_max'] if row['samples_per_s_max'] is not None else 'n/a':>11} | "
            f"{row['matched_lines']}"
        )


if __name__ == "__main__":
    main()
