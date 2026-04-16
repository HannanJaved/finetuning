#!/usr/bin/env python3
"""Analyze pruning characteristics for GPT-NeoX/Pythia-style models.

This script only reads configuration files; it does not load weights.

Per-layer heterogeneous fields (n_head, head_size, intermediate_size,
n_query_groups, rope_n_elem) are stored as full lists when present so that
parameter estimates and layerwise plots are accurate rather than mean-based.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:  # pragma: no cover - handled at runtime
    plt = None
    mticker = None

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class LayerwiseField:
    """Holds a per-layer value that may be uniform (scalar) or heterogeneous (list)."""
    values: List[float]  # always a list, length == num_layers

    @property
    def is_uniform(self) -> bool:
        return len(set(self.values)) <= 1

    @property
    def mean(self) -> float:
        return sum(self.values) / len(self.values) if self.values else float("nan")

    @property
    def std(self) -> float:
        m = self.mean
        return math.sqrt(sum((v - m) ** 2 for v in self.values) / len(self.values))

    @property
    def minimum(self) -> float:
        return min(self.values) if self.values else float("nan")

    @property
    def maximum(self) -> float:
        return max(self.values) if self.values else float("nan")

    def summary(self) -> Dict[str, float]:
        return {"mean": self.mean, "std": self.std, "min": self.minimum, "max": self.maximum}

    def ratio_vs(self, baseline_scalar: float) -> "LayerwiseField":
        return LayerwiseField([v / baseline_scalar if baseline_scalar else float("nan") for v in self.values])


@dataclass
class ModelStats:
    name: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    # Per-layer fields (always stored as LayerwiseField for uniform treatment)
    n_head: LayerwiseField
    head_size: LayerwiseField
    intermediate_size: LayerwiseField
    n_query_groups: LayerwiseField
    rope_n_elem: Optional[LayerwiseField]
    # Derived scalars (computed from per-layer data)
    width_depth_ratio: float
    params_embed: float
    params_attn_total: float   # sum over layers of actual attn proj params
    params_ffn_total: float    # sum over layers of FFN params
    params_total: float


def _to_layerwise(raw, num_layers: int, fallback: Optional[float] = None) -> LayerwiseField:
    """Convert a scalar or list config value into a LayerwiseField of length num_layers."""
    if raw is None:
        val = fallback if fallback is not None else float("nan")
        return LayerwiseField([val] * num_layers)
    if isinstance(raw, list):
        return LayerwiseField([float(v) for v in raw])
    return LayerwiseField([float(raw)] * num_layers)


# ---------------------------------------------------------------------------
# Parameter estimation helpers
# ---------------------------------------------------------------------------

def _attn_params_per_layer(hidden: int, n_head: float, head_size: float) -> float:
    """
    Attention projection params for one layer.

    For GptNeoxMLP-style models the four projections are:
      Q: hidden -> n_head * head_size
      K: hidden -> n_head * head_size  (same for GQA groups, but here n_query_groups == n_head)
      V: hidden -> n_head * head_size
      O: n_head * head_size -> hidden
    Total = 4 * hidden * (n_head * head_size)

    When n_head * head_size == hidden this reduces to 4 * hidden^2.
    For pruned models they often differ, so we use the actual projection size.
    """
    proj_size = n_head * head_size
    return 2.0 * hidden * proj_size + 2.0 * proj_size * hidden  # Q+K+V shares proj_size; O maps back


def _ffn_params_per_layer(hidden: int, intermediate: float) -> float:
    """FFN params for one GptNeoxMLP layer: two linear projections."""
    return 2.0 * hidden * intermediate


def _compute_params(
    hidden_size: int,
    vocab_size: int,
    n_head_lw: LayerwiseField,
    head_size_lw: LayerwiseField,
    intermediate_lw: LayerwiseField,
) -> tuple:
    params_embed = float(vocab_size * hidden_size)
    attn_per_layer = [
        _attn_params_per_layer(hidden_size, nh, hs)
        for nh, hs in zip(n_head_lw.values, head_size_lw.values)
    ]
    ffn_per_layer = [
        _ffn_params_per_layer(hidden_size, im)
        for im in intermediate_lw.values
    ]
    return params_embed, sum(attn_per_layer), sum(ffn_per_layer)


# ---------------------------------------------------------------------------
# Config loading / extraction
# ---------------------------------------------------------------------------

def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: str) -> Dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read model_config.yaml. Install with `pip install pyyaml`."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_original_config(path: str) -> str:
    if os.path.isdir(path):
        for candidate in ["config.json", "model_config.yaml"]:
            full = os.path.join(path, candidate)
            if os.path.exists(full):
                return full
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Original config not found at {path}")


def _resolve_pruned_config(path: str) -> str:
    if os.path.isdir(path):
        for candidate in [
            os.path.join(path, "final", "model_config.yaml"),
            os.path.join(path, "model_config.yaml"),
            os.path.join(path, "config.yaml"),
        ]:
            if os.path.exists(candidate):
                return candidate
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"Pruned config not found at {path}")


def _extract_original(cfg: Dict, name: str) -> ModelStats:
    """Extract stats from a HuggingFace config.json or NeoX model_config.yaml."""
    if "hidden_size" in cfg:
        # HuggingFace format
        hidden_size = int(cfg["hidden_size"])
        num_layers = int(cfg["num_hidden_layers"])
        n_head_scalar = int(cfg["num_attention_heads"])
        intermediate_scalar = int(cfg["intermediate_size"])
        vocab_size = int(cfg.get("vocab_size") or cfg.get("padded_vocab_size") or 0)
        head_size_scalar = hidden_size / n_head_scalar
        n_query_groups_scalar = float(cfg.get("num_key_value_heads") or n_head_scalar)
    else:
        # NeoX / LitGPT format
        hidden_size = int(cfg["n_embd"])
        num_layers = int(cfg["n_layer"])
        n_head_scalar = int(cfg["n_head"])
        intermediate_scalar = int(cfg["intermediate_size"])
        vocab_size = int(cfg.get("vocab_size") or cfg.get("padded_vocab_size") or 0)
        head_size_scalar = float(cfg.get("head_size") or hidden_size / n_head_scalar)
        n_query_groups_scalar = float(cfg.get("n_query_groups") or n_head_scalar)

    n_head_lw = _to_layerwise(n_head_scalar, num_layers)
    head_size_lw = _to_layerwise(head_size_scalar, num_layers)
    intermediate_lw = _to_layerwise(intermediate_scalar, num_layers)
    n_query_groups_lw = _to_layerwise(n_query_groups_scalar, num_layers)
    rope_n_elem_lw = None  # original configs typically don't have this pre-computed

    params_embed, params_attn, params_ffn = _compute_params(
        hidden_size, vocab_size, n_head_lw, head_size_lw, intermediate_lw
    )
    return ModelStats(
        name=name,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        n_head=n_head_lw,
        head_size=head_size_lw,
        intermediate_size=intermediate_lw,
        n_query_groups=n_query_groups_lw,
        rope_n_elem=rope_n_elem_lw,
        width_depth_ratio=hidden_size / num_layers if num_layers else float("nan"),
        params_embed=params_embed,
        params_attn_total=params_attn,
        params_ffn_total=params_ffn,
        params_total=params_embed + params_attn + params_ffn,
    )


def _extract_pruned(cfg: Dict, name: str) -> ModelStats:
    """Extract stats from a pruned model_config.yaml (NeoX/LitGPT format)."""
    hidden_size = int(cfg["n_embd"])
    num_layers = int(cfg["n_layer"])
    vocab_size = int(cfg.get("vocab_size") or cfg.get("padded_vocab_size") or 0)

    n_head_lw = _to_layerwise(cfg["n_head"], num_layers)
    intermediate_lw = _to_layerwise(cfg["intermediate_size"], num_layers)

    # head_size: fall back to hidden/n_head per layer if absent
    head_size_raw = cfg.get("head_size")
    if head_size_raw is None:
        head_size_lw = LayerwiseField([hidden_size / nh for nh in n_head_lw.values])
    else:
        head_size_lw = _to_layerwise(head_size_raw, num_layers)

    n_query_groups_lw = _to_layerwise(cfg.get("n_query_groups") or cfg["n_head"], num_layers)

    rope_n_elem_raw = cfg.get("rope_n_elem")
    rope_n_elem_lw = _to_layerwise(rope_n_elem_raw, num_layers) if rope_n_elem_raw is not None else None

    params_embed, params_attn, params_ffn = _compute_params(
        hidden_size, vocab_size, n_head_lw, head_size_lw, intermediate_lw
    )
    return ModelStats(
        name=name,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        n_head=n_head_lw,
        head_size=head_size_lw,
        intermediate_size=intermediate_lw,
        n_query_groups=n_query_groups_lw,
        rope_n_elem=rope_n_elem_lw,
        width_depth_ratio=hidden_size / num_layers if num_layers else float("nan"),
        params_embed=params_embed,
        params_attn_total=params_attn,
        params_ffn_total=params_ffn,
        params_total=params_embed + params_attn + params_ffn,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(value: float, digits: int = 2) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def _fmt_B(value: float, digits: int = 2) -> str:
    return "n/a" if math.isnan(value) else f"{value / 1e9:.{digits}f}B"


def _make_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt_row(values: Iterable[str]) -> str:
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(values))

    sep = "-+-".join("-" * w for w in widths)
    lines = [_fmt_row(headers), sep]
    lines.extend(_fmt_row(row) for row in rows)
    return "\n".join(lines)


def _write_csv(path: str, headers: List[str], rows: List[List[str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([headers] + rows)


def _write_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _summary_filename(original_path: str) -> str:
    base = os.path.basename(original_path)
    if base in {"model_config.yaml", "model_config.yml", "config.json"}:
        base = os.path.basename(os.path.dirname(original_path))
    safe = base.replace(" ", "_")
    return f"summary_{safe}_pruning.txt"


# ---------------------------------------------------------------------------
# Table / summary building
# ---------------------------------------------------------------------------

def _build_rows(stats: List[ModelStats], baseline: ModelStats) -> List[List[str]]:
    rows = []
    for s in stats:
        head_ratio = s.n_head.mean / baseline.n_head.mean if baseline.n_head.mean else float("nan")
        layer_ratio = s.num_layers / baseline.num_layers if baseline.num_layers else float("nan")
        hidden_ratio = s.hidden_size / baseline.hidden_size if baseline.hidden_size else float("nan")
        inter_ratio = s.intermediate_size.mean / baseline.intermediate_size.mean if baseline.intermediate_size.mean else float("nan")
        qg_ratio = s.n_query_groups.mean / baseline.n_query_groups.mean if baseline.n_query_groups.mean else float("nan")
        hs_ratio = s.head_size.mean / baseline.head_size.mean if baseline.head_size.mean else float("nan")
        param_ratio = s.params_total / baseline.params_total if baseline.params_total else float("nan")
        attn_ratio = s.params_attn_total / baseline.params_attn_total if baseline.params_attn_total else float("nan")
        ffn_ratio = s.params_ffn_total / baseline.params_ffn_total if baseline.params_ffn_total else float("nan")
        ffn_to_attn = s.params_ffn_total / s.params_attn_total if s.params_attn_total else float("nan")

        rows.append([
            s.name,
            str(s.hidden_size),
            str(s.num_layers),
            _fmt(s.n_head.mean, 1) + ("*" if not s.n_head.is_uniform else ""),
            _fmt(s.head_size.mean, 1) + ("*" if not s.head_size.is_uniform else ""),
            _fmt(s.intermediate_size.mean, 0) + ("*" if not s.intermediate_size.is_uniform else ""),
            _fmt(s.n_query_groups.mean, 1) + ("*" if not s.n_query_groups.is_uniform else ""),
            _fmt(hidden_ratio, 3),
            _fmt(layer_ratio, 3),
            _fmt(head_ratio, 3),
            _fmt(hs_ratio, 3),
            _fmt(inter_ratio, 3),
            _fmt(qg_ratio, 3),
            _fmt(s.width_depth_ratio, 1),
            _fmt(ffn_to_attn, 2),
            _fmt_B(s.params_attn_total, 2),
            _fmt_B(s.params_ffn_total, 2),
            _fmt_B(s.params_total, 2),
            _fmt(param_ratio, 3),
            _fmt(attn_ratio, 3),
            _fmt(ffn_ratio, 3),
        ])
    return rows


HEADERS = [
    "model",
    "hidden", "layers",
    "heads(mean)", "head_sz(mean)", "ffn_hid(mean)", "q_groups(mean)",
    "hidden/orig", "layers/orig", "heads/orig", "head_sz/orig", "ffn_hid/orig", "q_groups/orig",
    "width/depth",
    "ffn/attn_params",
    "attn_params", "ffn_params", "total_params",
    "params/orig", "attn_params/orig", "ffn_params/orig",
]


def _summary_lines(stats: List[ModelStats], baseline: ModelStats) -> List[str]:
    lines = []
    for s in stats:
        header = f"{s.name} vs original:"
        lines.append(header)
        lines.append("-" * len(header))

        def ratio_line(label, val, base):
            r = val / base if base else float("nan")
            return f"  {label:<28} {_fmt(val, 2):>10}  (ratio: {_fmt(r, 3)})"

        lines.append(ratio_line("hidden_size", s.hidden_size, baseline.hidden_size))
        lines.append(ratio_line("num_layers", s.num_layers, baseline.num_layers))
        lines.append(ratio_line("n_head (mean)", s.n_head.mean, baseline.n_head.mean))
        lines.append(ratio_line("head_size (mean)", s.head_size.mean, baseline.head_size.mean))
        lines.append(ratio_line("intermediate_size (mean)", s.intermediate_size.mean, baseline.intermediate_size.mean))
        lines.append(ratio_line("n_query_groups (mean)", s.n_query_groups.mean, baseline.n_query_groups.mean))
        lines.append(ratio_line("attn_params", s.params_attn_total, baseline.params_attn_total))
        lines.append(ratio_line("ffn_params", s.params_ffn_total, baseline.params_ffn_total))
        lines.append(ratio_line("total_params", s.params_total, baseline.params_total))

        # Layerwise variability for heterogeneous models
        het_fields = [
            ("n_head",            s.n_head),
            ("head_size",         s.head_size),
            ("intermediate_size", s.intermediate_size),
            ("n_query_groups",    s.n_query_groups),
        ]
        if s.rope_n_elem is not None:
            het_fields.append(("rope_n_elem", s.rope_n_elem))

        has_variability = any(not lw.is_uniform for _, lw in het_fields)
        if has_variability:
            lines.append("  layerwise variability (* = heterogeneous across layers):")
            for fname, lw in het_fields:
                marker = "*" if not lw.is_uniform else " "
                lines.append(
                    f"  {marker} {fname:<24} mean={_fmt(lw.mean, 2):>8}  "
                    f"std={_fmt(lw.std, 2):>7}  "
                    f"min={_fmt(lw.minimum, 2):>7}  "
                    f"max={_fmt(lw.maximum, 2):>7}"
                )

        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]


def _plot_ratios_vs_original(stats: List[ModelStats], baseline: ModelStats, output_dir: str) -> None:
    """Bar chart: how much of each original dimension each pruned model retains."""
    labels = [s.name for s in stats]
    x = list(range(len(labels)))
    metrics = {
        "hidden/orig":    [s.hidden_size / baseline.hidden_size for s in stats],
        "layers/orig":    [s.num_layers / baseline.num_layers for s in stats],
        "heads/orig":     [s.n_head.mean / baseline.n_head.mean for s in stats],
        "head_size/orig": [s.head_size.mean / baseline.head_size.mean for s in stats],
        "ffn_hid/orig":   [s.intermediate_size.mean / baseline.intermediate_size.mean for s in stats],
        "q_groups/orig":  [s.n_query_groups.mean / baseline.n_query_groups.mean for s in stats],
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    n_metrics = len(metrics)
    bar_w = 0.12
    offsets = [(i - n_metrics / 2 + 0.5) * bar_w for i in range(n_metrics)]
    for (label, vals), offset, color in zip(metrics.items(), offsets, COLORS * 2):
        positions = [xi + offset for xi in x]
        ax.bar(positions, vals, width=bar_w, label=label, color=color, alpha=0.85)

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("fraction of original")
    ax.set_title("Pruning retention ratios vs original")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "retention_ratios.png"), dpi=200)
    plt.close(fig)


def _plot_param_breakdown(stats: List[ModelStats], output_dir: str) -> None:
    """Stacked bar: embed / attn / FFN param split."""
    labels = [s.name for s in stats]
    x = list(range(len(labels)))
    embed = [s.params_embed / 1e9 for s in stats]
    attn = [s.params_attn_total / 1e9 for s in stats]
    ffn = [s.params_ffn_total / 1e9 for s in stats]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, embed, label="embed", color=COLORS[0])
    ax.bar(x, attn, bottom=embed, label="attn", color=COLORS[1])
    ax.bar(x, ffn, bottom=[e + a for e, a in zip(embed, attn)], label="ffn", color=COLORS[2])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("params (B)")
    ax.set_title("Parameter breakdown: embed / attn / FFN")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "param_breakdown.png"), dpi=200)
    plt.close(fig)


def _plot_params_est(stats: List[ModelStats], output_dir: str) -> None:
    labels = [s.name for s in stats]
    x = list(range(len(labels)))
    totals = [s.params_total / 1e9 for s in stats]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, totals, color=COLORS[0])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("params (B)")
    ax.set_title("Estimated total parameters")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "params_est.png"), dpi=200)
    plt.close(fig)


def _plot_width_depth(stats: List[ModelStats], output_dir: str) -> None:
    labels = [s.name for s in stats]
    x = list(range(len(labels)))
    width_depth = [s.width_depth_ratio for s in stats]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, width_depth, marker="o", color=COLORS[0])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("width / depth")
    ax.set_title("Width/Depth ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "width_depth_ratio.png"), dpi=200)
    plt.close(fig)


def _plot_ffn_attn_ratios(stats: List[ModelStats], output_dir: str) -> None:
    labels = [s.name for s in stats]
    x = list(range(len(labels)))
    ffn_attn_params = [
        s.params_ffn_total / s.params_attn_total if s.params_attn_total else float("nan")
        for s in stats
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, ffn_attn_params, marker="o", color=COLORS[2])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("FFN params / Attn params")
    ax.set_title("FFN vs Attention ratios")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ffn_attn_ratios.png"), dpi=200)
    plt.close(fig)


def _plot_heads_layers_ratios(stats: List[ModelStats], baseline: ModelStats, output_dir: str) -> None:
    labels = [s.name for s in stats]
    x = list(range(len(labels)))
    heads_ratio = [s.n_head.mean / baseline.n_head.mean for s in stats]
    layers_ratio = [s.num_layers / baseline.num_layers for s in stats]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, heads_ratio, marker="o", label="heads/orig", color=COLORS[1])
    ax.plot(x, layers_ratio, marker="o", label="layers/orig", color=COLORS[3])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("ratio vs original")
    ax.set_title("Heads and layers ratios")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "heads_layers_ratios.png"), dpi=200)
    plt.close(fig)


def _plot_ffn_attn_ratio_vs_original(stats: List[ModelStats], baseline: ModelStats, output_dir: str) -> None:
    """
    Two-panel plot comparing FFN/Attn param ratios vs the original model.

    Top panel:    absolute FFN/Attn ratio for each model (how lopsided the
                  FFN vs attention budget is within each model).
    Bottom panel: (FFN_params / Attn_params) / (FFN_orig / Attn_orig) —
                  how much the balance has shifted relative to the original.
                  >1 means FFN was pruned less aggressively than attention.
                  <1 means attention was pruned less aggressively than FFN.
    """
    labels = [s.name for s in stats]
    x = list(range(len(stats)))

    ffn_attn_abs = [
        s.params_ffn_total / s.params_attn_total if s.params_attn_total else float("nan")
        for s in stats
    ]
    orig_ratio = baseline.params_ffn_total / baseline.params_attn_total if baseline.params_attn_total else float("nan")
    ffn_attn_rel = [r / orig_ratio if orig_ratio else float("nan") for r in ffn_attn_abs]

    fig, (ax_abs, ax_rel) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # --- absolute ratio ---
    bars = ax_abs.bar(x, ffn_attn_abs, color=COLORS[3], alpha=0.85)
    ax_abs.axhline(orig_ratio, color="gray", linewidth=1.0, linestyle="--",
                   label=f"original ({orig_ratio:.2f})")
    for bar, val in zip(bars, ffn_attn_abs):
        ax_abs.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax_abs.set_ylabel("FFN params / Attn params")
    ax_abs.set_title("FFN / Attention param ratio — absolute")
    ax_abs.legend(fontsize=8)
    ax_abs.grid(axis="y", alpha=0.3)

    # --- ratio vs original ---
    bars2 = ax_rel.bar(x, ffn_attn_rel, color=COLORS[4], alpha=0.85)
    ax_rel.axhline(1.0, color="gray", linewidth=1.0, linestyle="--", label="parity with original")
    for bar, val in zip(bars2, ffn_attn_rel):
        ax_rel.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax_rel.set_ylabel("(FFN/Attn) / (FFN_orig/Attn_orig)")
    ax_rel.set_title("FFN / Attention param ratio — relative to original\n"
                     "(>1 = FFN pruned less; <1 = Attn pruned less)")
    ax_rel.legend(fontsize=8)
    ax_rel.grid(axis="y", alpha=0.3)

    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(labels, rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ffn_attn_ratio_vs_original.png"), dpi=200)
    plt.close(fig)


def _plot_layerwise_profiles(stats: List[ModelStats], baseline: ModelStats, output_dir: str) -> None:
    """
    For each pruned model that has heterogeneous per-layer fields,
    plot all four fields across layers in one figure.
    Also overlay the baseline (uniform) as a dashed horizontal line.
    """
    fields_cfg = [
        ("n_head",            "# attention heads",   baseline.n_head.mean),
        ("head_size",         "head size (dim)",     baseline.head_size.mean),
        ("intermediate_size", "FFN hidden size",     baseline.intermediate_size.mean),
        ("n_query_groups",    "# query groups",      baseline.n_query_groups.mean),
    ]

    for s in stats:
        lw_fields = {
            "n_head":            s.n_head,
            "head_size":         s.head_size,
            "intermediate_size": s.intermediate_size,
            "n_query_groups":    s.n_query_groups,
        }
        if s.rope_n_elem is not None:
            lw_fields["rope_n_elem"] = s.rope_n_elem

        if all(lw.is_uniform for lw in lw_fields.values()):
            continue  # nothing interesting to plot per-layer

        n_fields = len(fields_cfg) + (1 if s.rope_n_elem is not None else 0)
        fig, axes = plt.subplots(n_fields, 1, figsize=(11, 3 * n_fields), sharex=True)
        if n_fields == 1:
            axes = [axes]

        plot_cfgs = list(fields_cfg)
        if s.rope_n_elem is not None:
            plot_cfgs.append(("rope_n_elem", "rope_n_elem", float("nan")))

        layer_idx = list(range(s.num_layers))
        for ax, (fname, ylabel, baseline_val) in zip(axes, plot_cfgs):
            lw = lw_fields.get(fname)
            if lw is None:
                ax.set_visible(False)
                continue
            color = COLORS[plot_cfgs.index((fname, ylabel, baseline_val)) % len(COLORS)]
            ax.bar(layer_idx, lw.values, color=color, alpha=0.75, label=fname)
            if not math.isnan(baseline_val):
                ax.axhline(baseline_val, color="gray", linewidth=1.0, linestyle="--",
                           label=f"orig ({baseline_val:.0f})")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=7, loc="upper right")

        axes[-1].set_xlabel("layer index")
        fig.suptitle(f"{s.name} — per-layer architecture profile", fontsize=11)
        fig.tight_layout()
        safe_name = s.name.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(os.path.join(output_dir, f"layerwise_{safe_name}.png"), dpi=200)
        plt.close(fig)


def _plot_layerwise_ratios(stats: List[ModelStats], baseline: ModelStats, output_dir: str) -> None:
    """
    For each heterogeneous pruned model, plot per-layer retention ratios
    (pruned / original) for heads, head_size, and intermediate_size.
    """
    for s in stats:
        if s.n_head.is_uniform and s.head_size.is_uniform and s.intermediate_size.is_uniform:
            continue

        layer_idx = list(range(s.num_layers))
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

        pairs = [
            (axes[0], s.n_head,            baseline.n_head.mean,            "n_head / orig",            COLORS[0]),
            (axes[1], s.head_size,         baseline.head_size.mean,         "head_size / orig",         COLORS[1]),
            (axes[2], s.intermediate_size, baseline.intermediate_size.mean, "intermediate_size / orig", COLORS[2]),
        ]
        for ax, lw, base_val, ylabel, color in pairs:
            ratios = [v / base_val if base_val else float("nan") for v in lw.values]
            ax.bar(layer_idx, ratios, color=color, alpha=0.75)
            ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(axis="y", alpha=0.3)

        axes[-1].set_xlabel("layer index")
        fig.suptitle(f"{s.name} — per-layer retention ratios vs original", fontsize=11)
        fig.tight_layout()
        safe_name = s.name.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(os.path.join(output_dir, f"layerwise_ratios_{safe_name}.png"), dpi=200)
        plt.close(fig)


def _plot_ffn_attn_ratio_per_layer(stats: List[ModelStats], output_dir: str) -> None:
    """
    For heterogeneous models: FFN-params / Attn-params ratio per layer,
    showing how the FFN/attention balance shifts across depth.
    """
    for s in stats:
        if s.n_head.is_uniform and s.intermediate_size.is_uniform:
            continue

        layer_idx = list(range(s.num_layers))
        attn_per = [
            _attn_params_per_layer(s.hidden_size, nh, hs)
            for nh, hs in zip(s.n_head.values, s.head_size.values)
        ]
        ffn_per = [_ffn_params_per_layer(s.hidden_size, im) for im in s.intermediate_size.values]
        ratio_per = [f / a if a else float("nan") for f, a in zip(ffn_per, attn_per)]

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(layer_idx, ratio_per, color=COLORS[3], alpha=0.75)
        global_ratio = s.params_ffn_total / s.params_attn_total if s.params_attn_total else float("nan")
        ax.axhline(global_ratio, color="gray", linestyle="--", linewidth=0.8,
                   label=f"global mean ({global_ratio:.2f})")
        ax.set_xlabel("layer index")
        ax.set_ylabel("FFN params / Attn params")
        ax.set_title(f"{s.name} — FFN/Attn param ratio per layer")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        safe_name = s.name.replace(" ", "_").replace("(", "").replace(")", "")
        fig.savefig(os.path.join(output_dir, f"ffn_attn_per_layer_{safe_name}.png"), dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze pruning ratios for Pythia/GPT-NeoX models.",
    )
    parser.add_argument(
        "--original",
        default=(
            "/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-checkpoints/"
            "checkpoints/EleutherAI/pythia-6.9b/model_config.yaml"
        ),
        help="Path to the original model directory or config file.",
    )
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=[
            "/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-checkpoints/"
            "pretrain_from_supernet/bucket_0",
            "/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-checkpoints/"
            "pretrain_from_supernet/bucket_1",
            "/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-checkpoints/"
            "pretrain_from_supernet/bucket_2",
        ],
        help="Paths to pruned bucket directories (or model_config.yaml files).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=["bucket_0 (410m)", "bucket_1 (1b)", "bucket_2 (2.8b)"],
        help="Labels for the pruned models (must match number of buckets).",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/hk-project-p0024043/hgf_ivw0083/ws/hkfswork/hgf_ivw0083-Post-training/"
            "finetuning/pruning_analysis/output"
        ),
        help="Directory for CSV/JSON summaries and plots.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    original_path = _resolve_original_config(args.original)
    original_cfg = _load_yaml(original_path) if original_path.endswith((".yaml", ".yml")) else _load_json(original_path)
    original_stats = _extract_original(original_cfg, "original")

    if args.labels and len(args.labels) != len(args.buckets):
        raise ValueError("Number of labels must match number of buckets.")

    pruned_stats: List[ModelStats] = []
    for idx, bucket in enumerate(args.buckets):
        label = args.labels[idx] if args.labels else f"bucket_{idx}"
        pruned_path = _resolve_pruned_config(bucket)
        pruned_cfg = _load_yaml(pruned_path)
        pruned_stats.append(_extract_pruned(pruned_cfg, label))

    all_stats = [original_stats] + pruned_stats

    # --- Console output ---
    rows = _build_rows(all_stats, original_stats)
    print("\n=== Pruning analysis summary ===")
    print("  (* = value is mean of heterogeneous per-layer list)\n")
    print(_make_table(HEADERS, rows))

    print("\n=== Per-model ratio breakdown ===\n")
    print("\n".join(_summary_lines(pruned_stats, original_stats)))

    print("Notes:")
    print("  attn_params = sum_layers[ 2*hidden*(n_head*head_size) + 2*(n_head*head_size)*hidden ]")
    print("  ffn_params  = sum_layers[ 2*hidden*intermediate_size ]")
    print("  total_params includes embedding table (vocab_size * hidden_size)")

    # --- File output ---
    output_dir = _ensure_output_dir(args.output_dir)

    _write_csv(os.path.join(output_dir, "pruning_summary.csv"), HEADERS, rows)

    summary_path = os.path.join(output_dir, _summary_filename(original_path))
    summary_lines = [
        "=== Pruning analysis summary ===",
        "  (* = value is mean of heterogeneous per-layer list)",
        "",
        _make_table(HEADERS, rows),
        "",
        "=== Per-model ratio breakdown ===",
        "",
        "\n".join(_summary_lines(pruned_stats, original_stats)).rstrip(),
        "",
        "Notes:",
        "  attn_params = sum_layers[ 2*hidden*(n_head*head_size) + 2*(n_head*head_size)*hidden ]",
        "  ffn_params  = sum_layers[ 2*hidden*intermediate_size ]",
        "  total_params includes embedding table (vocab_size * hidden_size)",
        "",
    ]
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary_lines))

    json_payload = {
        "original_config": original_path,
        "pruned_configs": [_resolve_pruned_config(b) for b in args.buckets],
        "summary_rows": [dict(zip(HEADERS, row)) for row in rows],
        "layerwise_data": {
            s.name: {
                "n_head":            s.n_head.values,
                "head_size":         s.head_size.values,
                "intermediate_size": s.intermediate_size.values,
                "n_query_groups":    s.n_query_groups.values,
                "rope_n_elem":       s.rope_n_elem.values if s.rope_n_elem else None,
            }
            for s in all_stats
            if not (s.n_head.is_uniform and s.head_size.is_uniform and s.intermediate_size.is_uniform)
        },
        "layerwise_summaries": {
            s.name: {
                fname: lw.summary()
                for fname, lw in [
                    ("n_head",            s.n_head),
                    ("head_size",         s.head_size),
                    ("intermediate_size", s.intermediate_size),
                    ("n_query_groups",    s.n_query_groups),
                ]
                if not lw.is_uniform
            }
            for s in all_stats
            if not (s.n_head.is_uniform and s.head_size.is_uniform and s.intermediate_size.is_uniform)
        },
    }
    _write_json(os.path.join(output_dir, "pruning_summary.json"), json_payload)

    # --- Plots ---
    if plt is not None:
        _plot_ratios_vs_original(pruned_stats, original_stats, output_dir)
        _plot_param_breakdown(all_stats, output_dir)
        _plot_params_est(all_stats, output_dir)
        _plot_width_depth(all_stats, output_dir)
        _plot_ffn_attn_ratios(all_stats, output_dir)
        _plot_heads_layers_ratios(all_stats, original_stats, output_dir)
        _plot_ffn_attn_ratio_vs_original(all_stats, original_stats, output_dir)

        heterogeneous = [
            s
            for s in pruned_stats
            if not (s.n_head.is_uniform and s.head_size.is_uniform and s.intermediate_size.is_uniform)
        ]
        _plot_layerwise_profiles(heterogeneous, original_stats, output_dir)
        _plot_layerwise_ratios(heterogeneous, original_stats, output_dir)
        _plot_ffn_attn_ratio_per_layer(heterogeneous, output_dir)
    else:
        print("\nWarning: matplotlib not available — skipping plots.")

    print(f"\nSaved summaries and plots to: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - runtime guard
        print(f"Error: {exc}", file=sys.stderr)
        raise
