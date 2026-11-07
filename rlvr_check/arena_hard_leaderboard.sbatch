#!/usr/bin/env bash
#SBATCH --job-name=show-result-rlvr-4b
#SBATCH --account=p_neurasearch
#SBATCH --partition=barnard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/horse/ws/hama901h-BFTranslation/logs/plot_regen/show_result_rlvr_4b_%j.out
#SBATCH --error=/data/horse/ws/hama901h-BFTranslation/logs/plot_regen/show_result_rlvr_4b_%j.err
#
# Same as plot_scripts/ICLR/show_result_4b.sbatch, but writes to results/RLVR/ instead of
# results/ICLR27/: qwen3_iclr_common globs ICLR27/*/, so a new leaderboard there would silently
# override the paper's 4B Arena-Hard numbers.
set -euo pipefail
mkdir -p /data/horse/ws/hama901h-BFTranslation/logs/plot_regen
cd /data/horse/ws/hama901h-BFTranslation/arena-hard-auto
./venv/bin/python show_result.py \
  --category hard_prompt creative_writing \
  --output-csv \
  --output-dir results/RLVR/4b-gsm8k \
  --subdir ICLR/compared_with_qwen3-4b-sft-lr3e-5-AO
echo DONE
