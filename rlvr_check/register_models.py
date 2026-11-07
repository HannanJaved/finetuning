#!/usr/bin/env python
"""Register the post-RL checkpoints with the existing eval pipeline.

Writes (next to this script)
  models_pre.txt / models_post.txt / models_all.txt   aliases, --models-file of the eval scripts
  model_paths.tsv                                     alias <TAB> checkpoint path (pre + post)
  api_config_rlvr.yaml                                the new api_config entries
With --apply the entries are appended to arena-hard-auto/config/api_config.yaml
(timestamped backup first; aliases already present are skipped).

Usage: python register_models.py [--apply]
"""
import argparse
import hashlib
import shutil
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
API_CONFIG = Path("/data/horse/ws/hama901h-BFTranslation/arena-hard-auto/config/api_config.yaml")
RL_ROOT = Path("/data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-4B-Base/RLVR-GSM8K")


def post_alias(alias):
    return f"rlvr-gsm8k-{alias}"


def entry(path, alias):
    # same port scheme as dpo_merging/05_register_merges.py
    port = 8000 + int(hashlib.md5(alias.encode()).hexdigest(), 16) % 700
    return {"model": str(path), "endpoints": [{"api_base": f"http://localhost:{port}/v1", "api_key": "-"}],
            "api_type": "openai", "parallel": 32, "max_tokens": 4096, "temperature": 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="append the entries to api_config.yaml")
    args = ap.parse_args()

    rows = [line.split("\t") for line in (HERE / "starts.tsv").read_text().splitlines()[1:]]
    pre = [(r[1], r[2]) for r in rows]
    post = [(post_alias(a), RL_ROOT / a) for a, _ in pre]

    (HERE / "models_pre.txt").write_text("".join(f"{a}\n" for a, _ in pre))
    (HERE / "models_post.txt").write_text("".join(f"{a}\n" for a, _ in post))
    (HERE / "models_all.txt").write_text("".join(f"{a}\n" for a, _ in pre + post))
    (HERE / "model_paths.tsv").write_text("".join(f"{a}\t{p}\n" for a, p in pre + post))

    new = {a: entry(p, a) for a, p in post}
    (HERE / "api_config_rlvr.yaml").write_text(yaml.safe_dump(new, sort_keys=False))
    for a, p in post:
        print(f"{a}: {p}  {'ready' if (p / 'config.json').exists() else 'NOT TRAINED YET'}")

    existing = yaml.safe_load(API_CONFIG.read_text()) or {}
    todo = {a: e for a, e in new.items() if a not in existing}
    if not args.apply:
        print(f"\n{len(todo)} entries to add (dry run; --apply appends them to {API_CONFIG})")
        return
    if todo:
        shutil.copy(API_CONFIG, API_CONFIG.with_suffix(f".yaml.bak.{int(time.time())}"))
        with API_CONFIG.open("a") as fh:
            fh.write("\n" + yaml.safe_dump(todo, sort_keys=False))
    print(f"\nappended {len(todo)} entries to {API_CONFIG}")


if __name__ == "__main__":
    main()
