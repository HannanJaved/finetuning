# Qwen3 SFT Entropy Probe

This folder contains small scripts to compute **average token-level Shannon entropy** on a fixed probe set of prompts, then plot the results.

## Dataset choice

Defaults are set to **Nemotron SFT** splits (`code,math`) so every run uses the same prompt pool:

```
dataset="nvidia/Nemotron-Post-Training-Dataset-v2"
dataset_config="SFT"
dataset_split="code,math"
```

You can still swap datasets using `--dataset`, `--dataset-config`, and `--dataset-split` if you want a different probe set.

## What the scripts do

- **`compute_entropy.py`**
  - Samples (and caches) a fixed set of prompts from the chosen SFT dataset.
  - Computes average token-level entropy $H = -\sum_v p_\theta(v) \log p_\theta(v)$ per checkpoint.
  - Writes a CSV summary for plotting.

- **`plot_entropy.py`**
  - Reads the CSV and creates a line plot by checkpoint step.
- **`compute_entropy_first_token.py`**
  - Computes entropy at **only the first generated token** per prompt.
  - Writes a CSV with the same schema as `compute_entropy.py` for easy plotting.

## Usage

1. **Compute entropy**

```bash
python compute_entropy.py \
  --checkpoint-root /data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-X.XB-Base/SFT-sweep \
  --checkpoint-glob "Qwen3-*-SFT-LR*" \
  --checkpoint-subglob "checkpoint-*" \
  --dataset "nvidia/Nemotron-Post-Training-Dataset-v2" \
  --dataset-config "SFT" \
  --dataset-split "code,math" \
  --num-prompts 128 \
  --output-csv entropy_results.csv
```

2. **Plot**

```bash
python plot_entropy.py \
  --input-csv entropy_results.csv \
  --output-png entropy_plot.png
```

3. **First-token entropy (optional)**

```bash
python compute_entropy_first_token.py \
  --checkpoint-root /data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-X.XB-Base/SFT-sweep \
  --checkpoint-glob "Qwen3-*-SFT-LR*" \
  --checkpoint-subglob "checkpoint-*" \
  --dataset "nvidia/Nemotron-Post-Training-Dataset-v2" \
  --dataset-config "SFT" \
  --dataset-split "code,math" \
  --num-prompts 128 \
  --output-csv entropy_first_token_results.csv
```

## Notes

- The probe set is cached in `probe_prompts.jsonl` so entropy numbers are comparable across runs.
- If the run folders already contain the model weights (no `checkpoint-*`), use `--checkpoint-subglob none`.
- For very large checkpoints, start with small `--num-prompts` and `--batch-size`.
