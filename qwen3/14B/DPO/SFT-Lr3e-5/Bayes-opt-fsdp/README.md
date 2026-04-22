# Bayesian Optimization (DPO, FSDP) – SLURM runner

This folder launches a Bayesian optimization loop that submits one SLURM job per trial.
The controller itself runs as a long SLURM job so you don’t need to keep a login
session open.

## What it does

- Runs `bayes_opt.py` for 20 trials (Optuna TPE sampler).
- Each trial writes its own config/job script and submits with `sbatch`.
- Uses **0.5% validation split** from the training dataset (`dataset_test_split_size: 0.005`).
- Uses **FSDP** via `recipes/accelerate_configs/fsdp.yaml`.

## Files

- `bayes_opt.py` – Bayesian optimization controller.
- `run_bayes_opt.slurm` – SLURM job to run the controller for days if needed.
- `requirements.txt` – Python deps for the controller.

## Usage

Submit the controller job:

```bash
sbatch run_bayes_opt.slurm
```

## Notes

- Adjust SLURM resources in `run_bayes_opt.slurm` if your cluster requires
  different defaults. The controller is light; GPU is requested only if your
  policy needs at least one GPU allocated for job launchers.
- Ensure `optuna` and `pyyaml` are installed in the activated environment.
