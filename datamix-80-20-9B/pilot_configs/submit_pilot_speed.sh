#!/bin/bash
set -euo pipefail

DIR=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/datamix-80-20-9B

for script in \
  pilot_zero2_liger.sh \
  pilot_zero2_bs2.sh \
  pilot_zero2_liger_bs2.sh \
  pilot_zero2_nogc.sh \
  pilot_zero2_bs2_nogc.sh
do
  echo "Submitting $script"
  sbatch "$DIR/$script"
done

echo
echo "Baseline (production): ZeRO-2 ~8.5 s/step"
echo "Logs: /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/datamix-80-20-9B/pilot/"
echo
echo "Compare after jobs finish:"
echo "  grep -hE '[0-9]+/30.*s/it' $DIR/../.logs/datamix-80-20-9B/pilot/*.err | tail -30"
