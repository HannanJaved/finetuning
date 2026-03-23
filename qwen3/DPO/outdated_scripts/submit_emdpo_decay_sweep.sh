#!/bin/bash
# Submit a sweep of EM-DPO with epsilon decay runs.
#
# Each run gets a unique config derived from the base config by patching
# the decay-specific fields. Config YAMLs are written alongside this script
# and persist on disk so SLURM jobs can read them after submission.
#
# Usage:
#   bash submit_emdpo_decay_sweep.sh [--dry-run]
#
#   --dry-run  Print the sbatch commands without submitting.
set -euo pipefail

ROOT=/data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO
SBATCH_SCRIPT="$ROOT/emdpo_with_decay_LR5e7_Beta1.sh"
BASE_CONFIG="$ROOT/config_emdpo_decay_LR5e7_Beta1.yaml"
BASE_OUTPUT=/data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-1.7B-Base/SFT/Lr5e5/EMDPODecay

DRY_RUN=0
for arg in "$@"; do
  [[ "$arg" == "--dry-run" ]] && DRY_RUN=1
done

# ---------------------------------------------------------------------------
# Sweep grid
# Each entry: "eps_max eps_min"
# Rationale:
#   0.30->0.10  baseline decay (conservative start, aggressive end)
#   0.20->0.10  shallower decay (less conservative start)
#   0.30->0.05  steeper end (very aggressive late)
#   0.30->0.20  mild decay    (stay conservative throughout)
#   0.10->0.10  flat eps=0.10 (ablation: no decay, matches best static result)
# ---------------------------------------------------------------------------
RUNS=(
  "0.30 0.10"
  "0.20 0.10"
  "0.30 0.05"
  "0.30 0.20"
  "0.10 0.10"
)

for run in "${RUNS[@]}"; do
  eps_max=$(echo "$run" | awk '{print $1}')
  eps_min=$(echo "$run" | awk '{print $2}')

  # Format for filenames: strip the dot, e.g. 0.30 -> 030
  fmt_max=$(echo "$eps_max" | tr -d '.')
  fmt_min=$(echo "$eps_min" | tr -d '.')
  tag="emdpo_decay_LR5e7_Beta1_epsmax${fmt_max}_epsmin${fmt_min}"

  cfg="$ROOT/config_${tag}.yaml"
  outdir="${BASE_OUTPUT}/epsmax${fmt_max}_epsmin${fmt_min}/"

  # Patch the base config: override the three decay fields + output_dir
  sed \
    -e "s|^emdpo_decay_eps_max:.*|emdpo_decay_eps_max: ${eps_max}|" \
    -e "s|^emdpo_decay_eps_min:.*|emdpo_decay_eps_min: ${eps_min}|" \
    -e "s|^output_dir:.*|output_dir: ${outdir}|" \
    "$BASE_CONFIG" > "$cfg"

  echo "Submitting: $tag  (eps_max=$eps_max  eps_min=$eps_min)"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "  [dry-run] sbatch --export=ALL,CONFIG_FILE=\"$cfg\" --job-name=\"$tag\" \"$SBATCH_SCRIPT\""
  else
    sbatch --export=ALL,CONFIG_FILE="$cfg" --job-name="$tag" "$SBATCH_SCRIPT"
  fi
done
