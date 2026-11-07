#!/bin/bash
# Run at ZIH (needs internet): bundle what MN5 needs to rebuild venv-vllm offline.
#   python/   standalone CPython 3.11.13 (relocatable; MN5 nodes cannot download one)
#   wheels/   every package pinned in venv-vllm, as wheels
#   requirements.txt
# The two local installs (alignment-handbook, lm_eval) are left out; GRPO training does not use them.
set -euo pipefail
B=/data/horse/ws/hama901h-BFTranslation
OUT=$B/mn5_bundle
mkdir -p "$OUT"
export PIP_CACHE_DIR=$B/.cache/pip UV_CACHE_DIR=$B/.cache/uv

$B/venv-vllm/bin/python -m pip freeze | grep -vE '^-e | @ ' > "$OUT/requirements.txt"
$B/venv-vllm/bin/python -m pip wheel --no-deps -r "$OUT/requirements.txt" -w "$OUT/wheels"
uv python install 3.11.13 --install-dir "$OUT/python"
ls "$OUT"/python/*/bin/python3.11
du -sh "$OUT"/*
