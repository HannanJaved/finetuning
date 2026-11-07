#!/usr/bin/env python
"""Pick the RLVR starting points and write starts.tsv.

5 Qwen3-4B DPO checkpoints evenly spaced in Z-Dynamic rank (best ... worst), using the same
18-config grid and Z-Dynamic definition as the ICLR figures (qwen3_iclr_common).

Usage: python select_starts.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/horse/ws/hama901h-BFTranslation/plot_scripts/ICLR")
import qwen3_iclr_common as c  # noqa: E402

N_DPO = 5

metrics = c.metrics_from_final_models("4b")
z = c.compute_zscore_aggregate(metrics, c.DYNAMIC_BENCHMARKS)
names = pd.concat([df[["lr", "beta", "model_name"]] for df in metrics.values() if not df.empty]).drop_duplicates()
z = z.merge(names, on=["lr", "beta"]).sort_values("score", ascending=False).reset_index(drop=True)

picks = np.round(np.linspace(0, len(z) - 1, N_DPO)).astype(int)
roles = ["best"] + [f"mid{i}" for i in range(1, N_DPO - 1)] + ["worst"]
rows = [(role, z.model_name[i], c.model_path(z.model_name[i]), f"{z.score[i]:.4f}", f"{i + 1}/{len(z)}")
        for role, i in zip(roles, picks)]

out = Path(__file__).with_name("starts.tsv")
out.write_text("role\talias\tpath\tz_dynamic\tz_dynamic_rank\n" + "".join("\t".join(r) + "\n" for r in rows))
print(z[["lr", "beta", "score", "model_name"]].rename(columns={"score": "Z-Dynamic"}).to_string())
print(f"\nwrote {out}:\n{out.read_text()}")
