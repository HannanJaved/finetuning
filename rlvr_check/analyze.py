#!/usr/bin/env python
"""Does the DPO-stage ranking survive RLVR?

For every start in starts.tsv, collects pre- and post-RL Arena-Hard, MT-Bench, GSM8K and MATH-500,
then reports over the DPO starts:
  - Spearman rho of DPO-stage Z-Dynamic vs post-RL Chat-2 (the main question), plus pre- vs
    post-RL Chat-2 and per-benchmark rhos, each with an exact permutation p-value
  - whether the best DPO model (by Z-Dynamic) is still best on Chat-2 after RL
Chat-2 = mean z-score of Arena-Hard and MT-Bench across the DPO starts, z computed within a stage.

Usage: python analyze.py        (also writes rlvr_scores.csv)
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/horse/ws/hama901h-BFTranslation/plot_scripts/ICLR")
import qwen3_iclr_common as c  # noqa: E402
from register_models import RL_ROOT, post_alias  # noqa: E402

HERE = Path(__file__).resolve().parent
AH_ROOT = Path("/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/results/RLVR")  # arena_hard_leaderboard.sbatch
BENCH = ["Arena-Hard", "MT-Bench", "GSM8K", "MATH-500"]


def lm_eval_score(task_dir, path, task, metric):
    """Latest vLLM-backend lm-eval result (the grid's GSM8K protocol) for checkpoint `path`, in %."""
    for f in sorted((c.LM_EVAL_DIR / task_dir / c.flatten_path(path)).glob("results_*.json"), reverse=True):
        data = json.loads(f.read_text())
        if data["config"]["model"] == "vllm":
            return 100 * data["results"][task][metric]
    return np.nan


def spearman(x, y):
    return pd.Series(list(x)).rank().corr(pd.Series(list(y)).rank())


def exact_p(x, y):
    """Two-sided p-value for Spearman rho, enumerating every permutation of y."""
    obs = abs(spearman(x, y))
    return float(np.mean([abs(spearman(x, p)) >= obs - 1e-12 for p in itertools.permutations(list(y))]))


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def main():
    starts = pd.read_csv(HERE / "starts.tsv", sep="\t")
    ah = c.build_arena_hard_lookup(AH_ROOT)
    mt = c.build_mtbench_lookup(c.MTBENCH_QWEN_DIR)

    rows = []
    for s in starts.itertuples():
        for stage, alias, path in [("pre", s.alias, s.path), ("post", post_alias(s.alias), str(RL_ROOT / s.alias))]:
            rows.append({"role": s.role, "start": s.alias, "z_dynamic": s.z_dynamic, "stage": stage,
                         "Arena-Hard": ah.get(alias, np.nan),
                         "MT-Bench": mt.get(path, np.nan),
                         "GSM8K": lm_eval_score("gsm8k", path, "gsm8k", "exact_match,strict-match"),
                         "MATH-500": lm_eval_score("minerva_math500", path, "minerva_math500", "math_verify,none")})
    df = pd.DataFrame(rows)
    for stage in ("pre", "post"):
        m = df.stage == stage
        df.loc[m, "Chat-2"] = (zscore(df.loc[m, "Arena-Hard"]) + zscore(df.loc[m, "MT-Bench"])) / 2
    df.to_csv(HERE / "rlvr_scores.csv", index=False)

    wide = df.set_index(["role", "z_dynamic", "stage"])[BENCH + ["Chat-2"]].unstack("stage")
    wide.columns = [f"{b} {st}" for b, st in wide.columns]
    pd.set_option("display.width", 250)
    print(wide.round(2).to_string(), "\n")

    d = wide.reset_index()
    pairs = {
        "Z-Dynamic (DPO) vs Chat-2 post-RL": (d["z_dynamic"], d["Chat-2 post"]),
        "Chat-2 pre vs post-RL": (d["Chat-2 pre"], d["Chat-2 post"]),
        "Arena-Hard pre vs post-RL": (d["Arena-Hard pre"], d["Arena-Hard post"]),
        "MT-Bench pre vs post-RL": (d["MT-Bench pre"], d["MT-Bench post"]),
        "Z-Dynamic (DPO) vs GSM8K post-RL": (d["z_dynamic"], d["GSM8K post"]),
        "Z-Dynamic (DPO) vs MATH-500 post-RL": (d["z_dynamic"], d["MATH-500 post"]),
    }
    print(f"Spearman over n={len(d)} DPO starts (exact two-sided permutation p):")
    for name, (x, y) in pairs.items():
        if x.isna().any() or y.isna().any():
            print(f"  {name:38s} missing scores")
        else:
            print(f"  {name:38s} rho={spearman(x, y):+.3f}  p={exact_p(x, y):.3f}")

    if not d["Chat-2 post"].isna().any():
        best = d.loc[d["z_dynamic"].idxmax()]
        rank = int(d["Chat-2 post"].rank(ascending=False)[best.name])
        print(f"\nBest DPO model ({best['role']}, Z-Dynamic {best['z_dynamic']:.2f}): post-RL Chat-2 rank "
              f"{rank}/{len(d)} -> {'still best' if rank == 1 else 'NOT best'}")


if __name__ == "__main__":
    main()
