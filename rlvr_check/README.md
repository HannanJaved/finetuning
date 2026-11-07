# RLVR check: does the DPO-stage ranking survive RL?

Tülu-3-style pipeline: DPO, then a short RL stage with verifiable rewards (GRPO on GSM8K train,
exact-match reward, no reward model). Every starting point gets identical RL settings. Then we
check whether the DPO-stage ranking (Z-Dynamic) still predicts Chat-2 (Arena-Hard + MT-Bench).

## Starting points (`starts.tsv`, written by `select_starts.py`)

Five Qwen3-4B DPO checkpoints at evenly spaced Z-Dynamic ranks of the 18-config grid (same
Z-Dynamic code as the ICLR figures):

| role  | config         | Z-Dynamic | rank  |
|-------|----------------|-----------|-------|
| best  | lr 4e-7, β 0.01 | +1.47     | 1/18  |
| mid1  | lr 2e-6, β 0.01 | +0.81     | 5/18  |
| mid2  | lr 1e-6, β 0.08 | −0.04     | 9/18  |
| mid3  | lr 2e-6, β 0.08 | −0.38     | 14/18 |
| worst | lr 8e-6, β 0.01 | −1.56     | 18/18 |

Why five DPO models and not four: with n=4, even a perfect ρ=1 has exact p = 2/24 ≈ 0.083, so the
test can never reject. With n=5 the minimum is 2/120 ≈ 0.017. To use a different set, change
`N_DPO` in `select_starts.py` and rerun it.

## RL settings (`grpo_gsm8k.yaml`, the same for every run)

- GRPO (TRL 0.29 in `venv-vllm`), vLLM colocate on one capella node (4×H100), FSDP full shard.
  FSDP is used because `venv-vllm` has no DeepSpeed.
- fp32 master weights with bf16 compute, the same regime as the ≥4B DPO runs. The final model is
  exported in bf16.
- 250 steps × 32 prompts × 8 generations (256 completions per step, fully on-policy), which is about 1.07 epochs.
- lr 1e-6 linear (3% warmup), KL β = 0.05 (Tülu 3's value), temperature 1.0, max completion 1024.
- Prompt: chat template, GSM8K question + "end with `#### <number>`".
- Reward: 1 if the **last number** in the completion equals the gold answer (open-instruct's GSM8K
  verifier), else 0. Text after the answer that contains a number makes the reward 0.

## Evaluation

| benchmark | protocol | pre-RL scores |
|---|---|---|
| Arena-Hard, MT-Bench (Chat-2) | existing pipeline (`submit_evals.py`), pairwise vs the **pre-RL SFT** (`qwen3-4b-sft-lr3e-5-AO`), the same baseline as the DPO grid | exist |
| GSM8K | grid protocol: vLLM, 5-shot, no template, strict-match | exist |
| MATH-500 | `minerva_math500`, vLLM, 4-shot, no template, `math_verify` | run here (new) |

The Arena-Hard leaderboard goes to `arena-hard-auto/results/RLVR/`, **not** `results/ICLR27/`.
`qwen3_iclr_common` globs `ICLR27/*/`, so a new leaderboard there would silently override the
paper's 4B numbers.

`analyze.py` reports over the 5 DPO starts:
ρ(Z-Dynamic, post-RL Chat-2) as the main result, ρ(pre-RL Chat-2, post-RL Chat-2), and per-benchmark ρ, each
with an exact permutation p-value. It also reports whether the best DPO model is still ranked first on Chat-2.
Chat-2 = mean of Arena-Hard and MT-Bench z-scores across the DPO starts, with z computed within each stage.

## Prerequisites (one-time, login node)

```bash
B=/data/horse/ws/hama901h-BFTranslation
# MATH-500 scoring needs math_verify + antlr4 4.11 (lm-eval's [math] extra); neither venv has them
$B/venv-vllm/bin/pip install "antlr4-python3-runtime==4.11" "math-verify[antlr4_11_0]"
# MATH-500 data into the cache the eval jobs use
HF_HOME=$B/.cache $B/venv-vllm/bin/python -c "from datasets import load_dataset; load_dataset('HuggingFaceH4/MATH-500', split='test')"
```

## Run order

```bash
cd /data/horse/ws/hama901h-Post-training/hama901h-Posttraining/finetuning/rlvr_check
# 1. smoke test (checks memory, vLLM weight sync and saving); then delete the output dir
sbatch --time=00:45:00 train_grpo.sbatch dpo-qwen3-4b-sft-lr3e5-ao-lr4e7-beta0.01 --max_steps 3 --output_dir <scratch dir>
# 2. the five RL runs
bash submit_train.sh            # dry run; add --submit to submit
# 3. once trained: register the RL checkpoints in api_config.yaml (backup written first)
python register_models.py --apply
# 4. Chat-2 + GSM8K + MATH-500
bash submit_evals.sh            # dry run; add --submit to submit
# 5. after Arena-Hard judgments finish
sbatch arena_hard_leaderboard.sbatch
# 6.
python analyze.py               # prints the table + Spearman; writes rlvr_scores.csv
```

### Alpha (8× A100-40GB)

From an alpha login node: `bash submit_train_alpha.sh --submit --account=<project>`.
`train_grpo_alpha.sbatch` keeps the same RL run (256 completions per step, one generation round
per step) and only changes memory settings: 2 per device × 16 accumulation × 8 GPUs, vLLM share 0.35
(4 per device ran out of memory on 40 GB in the smoke test).
Capella and alpha copies write to the same output dir. `grpo_gsm8k.py` atomically claims
`<output_dir>.claim` on start, so only the first job to start trains, on whichever cluster or
account, and later duplicates exit within a few minutes. If the claiming run fails, delete its
`.claim` file before resubmitting.

If step 1 runs out of memory, set `per_device_train_batch_size: 4` and
`gradient_accumulation_steps: 16`. The effective batch stays the same.

## Not verified yet

Checked on the login node: the config parses under TRL 0.29, the dataset loads offline, the reward
gives 1.0 on all 7,473 gold solutions, and `analyze.py` reproduces the paper's pre-RL scores.
Nothing GPU-side has run yet (FSDP + vLLM colocate, memory, step time), so run the smoke test first.
