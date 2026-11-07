#!/bin/bash
# Run at ZIH, after build_mn5_bundle.sh: copy everything the MN5 RLVR runs need.
# Only the top-level files of each starting checkpoint go (model + tokenizer, ~8 GB each);
# --exclude='*/' skips the intermediate checkpoint-*/, runs/ and validation*/ folders.
# The key has a passphrase, so load it once first:
#   eval "$(ssh-agent)"; ssh-add ~/.ssh/id_ed25519_mn5
set -euo pipefail
HERE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/rlvr_check
B=/data/horse/ws/hama901h-BFTranslation
DEST=eit449746@transfer1.bsc.es:/gpfs/projects/ehpc533/users/hannan/rlvr
RSYNC=(rsync -avP -e "ssh -i $HOME/.ssh/id_ed25519_mn5")

ssh -i ~/.ssh/id_ed25519_mn5 eit449746@transfer1.bsc.es \
    mkdir -p /gpfs/projects/ehpc533/users/hannan/rlvr/{dpo-starts,hf_cache/hub,code,logs}

tail -n +2 "$HERE/starts.tsv" | cut -f3 | while read -r d; do
    "${RSYNC[@]}" --exclude='*/' "$d/" "$DEST/dpo-starts/$(basename "$d")/"
done
"${RSYNC[@]}" $B/.cache/openai___gsm8k "$DEST/hf_cache/"
"${RSYNC[@]}" $B/.cache/hub/datasets--openai--gsm8k "$DEST/hf_cache/hub/"
"${RSYNC[@]}" $B/mn5_bundle/ "$DEST/"
"${RSYNC[@]}" "$HERE"/{grpo_gsm8k.py,grpo_gsm8k.yaml,accelerate_fsdp.yaml,starts.tsv,setup_env_mn5.sh,train_grpo_mn5.sbatch,submit_train_mn5.sh} "$DEST/code/"
