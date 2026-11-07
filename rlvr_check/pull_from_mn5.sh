#!/bin/bash
# Run at ZIH after the MN5 runs finish: copy the RL checkpoints to where the ZIH eval pipeline
# (register_models.py, submit_evals.sh) expects them.
# Usage: bash pull_from_mn5.sh alias [alias ...]   (load the key first: ssh-add ~/.ssh/id_ed25519_mn5)
# Writes an "mn5" .claim first, so a capella/alpha duplicate that starts later exits instead of
# overwriting; skips an alias that a ZIH job has already claimed.
set -euo pipefail
SRC=eit449746@transfer1.bsc.es:/gpfs/projects/ehpc533/users/hannan/rlvr/RLVR-GSM8K
RL_ROOT=/data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-4B-Base/RLVR-GSM8K
mkdir -p "$RL_ROOT"

for alias in "$@"; do
    claim=$RL_ROOT/$alias.claim
    [[ -e $claim ]] || echo mn5 > "$claim"
    if [[ $(cat "$claim") != mn5 ]]; then
        echo "skip $alias: already claimed by ZIH job $(cat "$claim")" >&2
        continue
    fi
    rsync -avP -e "ssh -i $HOME/.ssh/id_ed25519_mn5" "$SRC/$alias/" "$RL_ROOT/$alias/"
done
