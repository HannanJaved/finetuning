#!/bin/bash
set -euo pipefail

DIR=/data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/datamix-80-20-9B

for script in pilot_fsdp2.sh; do
  echo "Submitting $script"
  sbatch "$DIR/$script"
done

echo
echo "Logs: /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/datamix-80-20-9B/pilot/"
echo "Compare step time after jobs finish:"
echo "  grep -h 's/it' /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/.logs/datamix-80-20-9B/pilot/*.err | tail -20"
