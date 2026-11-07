#!/bin/bash
# Submit all post-RL evals (run register_models.py --apply first).
# Usage: bash submit_evals.sh            (dry run)
#        bash submit_evals.sh --submit
set -euo pipefail
HERE=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/rlvr_check
WS=/data/horse/ws/hama901h-BFTranslation
S=$WS/arena-hard-auto/scripts
BASELINE=qwen3-4b-sft-lr3e-5-AO   # the pre-RL SFT: same pairwise baseline as the DPO grid

if [[ "${1:-}" == "--submit" ]]; then EVALS_FLAG=--submit; DRY=(); else EVALS_FLAG=--dry-run; DRY=(--dry-run); fi
cd "$WS"

# Chat-2 = Arena-Hard + MT-Bench. Pre-RL starts are included so a missing score gets filled;
# --skip-completed skips everything already judged.
python "$S/submit_evals.py" --models-file "$HERE/models_all.txt" --baseline "$BASELINE" \
    --tasks arena-hard mtbench --skip-completed "$EVALS_FLAG"

# GSM8K: grid protocol (vLLM, 5-shot, no template). Pre-RL starts already have it.
"$WS/venv-lm-eval/bin/python" "$S/submit_gsm8k_from_list.py" --models-file "$HERE/models_post.txt" "${DRY[@]}"

# MATH-500: new benchmark, so pre- and post-RL.
while IFS=$'\t' read -r alias path; do
    cmd=(sbatch --job-name="math500_$alias" "$HERE/eval_math.sbatch" "$path")
    if [[ ${#DRY[@]} -eq 0 ]]; then "${cmd[@]}"; else echo "${cmd[@]}"; fi
done < "$HERE/model_paths.tsv"
