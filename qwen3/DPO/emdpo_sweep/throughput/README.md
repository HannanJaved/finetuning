# EM-DPO throughput scaling sweep

This folder contains short EM-DPO runs (5% of one epoch) for throughput scaling on the same dataset/model as the existing `emdpo_sweep` configs.

## What’s included
- Configs per scale with `num_train_epochs: 0.05` and `save_strategy: "no"`.
- SLURM scripts for each scale:
  - 1 GPU / 1 node
  - 2 GPUs / 1 node
  - 4 GPUs / 1 node
  - 8 GPUs / 2 nodes
  - 12 GPUs / 3 nodes
  - 16 GPUs / 4 nodes

## Usage
Submit all runs at once:

```bash
sbatch /data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/emdpo_sweep/throughput/submit_throughput_sweep.sh
```

Or submit one script directly, e.g.:

```bash
sbatch /data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/emdpo_sweep/throughput/throughput_4gpu_1node.sh
```

## Notes
- Logs land in `/data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/throughput/`.
- Output directories are separated by scale under `.../EMDPO/throughput/`.
- If you want a different fraction of training, adjust `num_train_epochs` in the configs.

## Parse throughput
Use the helper to extract tokens/sec and samples/sec from logs:

```bash
python /data/cat/ws/hama901h-Posttraining/finetuning/qwen3/DPO/emdpo_sweep/throughput/parse_throughput.py \
  --logs-dir /data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/throughput \
  --glob "*.out" \
  --output /data/cat/ws/hama901h-Posttraining/.logs/Qwen3/DPO_N/throughput/throughput_summary.csv
```
