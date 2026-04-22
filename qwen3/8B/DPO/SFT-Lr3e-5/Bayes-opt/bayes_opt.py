#!/usr/bin/env python3
import json
import math
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import optuna
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
JOB_TEMPLATE = SCRIPT_DIR / "LR1e7_Beta0.1.sh"
CONFIG_TEMPLATE = SCRIPT_DIR / "dpo_beta0.1_LR.yaml"

N_TRIALS = 20
POLL_SECONDS = 30
TRAINER_STATE_TIMEOUT_SECONDS = 60 * 30
TRAINER_STATE_POLL_SECONDS = 20
STALE_TRIAL_MAX_AGE_SECONDS = 60 * 30

# IMPORTANT: choose what to optimize
METRIC_KEY = "eval_loss"   # fallback to "loss" if eval not available
DIRECTION = "minimize"     # "maximize" for reward metrics


def write_config(trial_dir: Path, lr: float, beta: float) -> Path:
    with CONFIG_TEMPLATE.open() as f:
        cfg = yaml.safe_load(f)

    run_name = f"Qwen3-8B-DPO-BO-trial{trial_dir.name}-b{beta:.4f}-lr{lr:.2e}"
    out_dir = f"/data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-8B-Base/SFT-sweep/{run_name}/"

    cfg["learning_rate"] = float(lr)
    cfg["beta"] = float(beta)
    cfg["output_dir"] = out_dir

    # Keep global batch size 128 with per_device_train_batch_size=1.
    # global_batch = per_device_train_batch_size * gradient_accumulation_steps * world_size
    gpus_per_node = int(os.environ.get("SLURM_GPUS_ON_NODE", "1"))
    nodes = int(os.environ.get("SLURM_NNODES", "1"))
    world_size = max(1, gpus_per_node * nodes)
    per_device = int(cfg.get("per_device_train_batch_size", 1))
    target_global_batch = 128
    cfg["gradient_accumulation_steps"] = max(1, math.ceil(target_global_batch / (per_device * world_size)))

    # Use 1% of the dataset as validation by creating a test split.
    # The remaining 99% stays in train.
    cfg["dataset_test_split_size"] = 0.005
    cfg["dataset_test_split_seed"] = 42

    # Ensure we train and evaluate so BO can read eval metrics.
    cfg["do_train"] = True
    cfg["do_eval"] = True
    cfg["eval_strategy"] = "steps"
    cfg.setdefault("eval_steps", 100)
    cfg["save_strategy"] = "steps"
    cfg.setdefault("save_steps", cfg["eval_steps"])
    cfg.setdefault("save_total_limit", 3)

    config_path = trial_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return config_path


def write_job_script(trial_dir: Path, config_path: Path, run_name: str) -> Path:
    txt = JOB_TEMPLATE.read_text()

    txt = re.sub(r"^#SBATCH --job-name=.*$", f"#SBATCH --job-name={run_name}", txt, flags=re.M)
    txt = re.sub(r"^export WANDB_NAME=.*$", f"export WANDB_NAME={run_name}", txt, flags=re.M)
    txt = re.sub(r"^CONFIG_FILE=.*$", f"CONFIG_FILE={config_path}", txt, flags=re.M)

    job_path = trial_dir / "run.sh"
    job_path.write_text(txt)
    job_path.chmod(0o755)
    return job_path


def run_job_in_allocation(job_script: Path, trial_dir: Path) -> str:
    """
    Run the trial job directly inside the current Slurm allocation.
    This avoids sbatch queue latency and keeps the 8-GPU allocation busy.
    """
    stdout_path = trial_dir / "run.out"
    stderr_path = trial_dir / "run.err"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        result = subprocess.run(["bash", str(job_script)], stdout=stdout, stderr=stderr)
    return "COMPLETED" if result.returncode == 0 else "FAILED"


def wait_for_trainer_state(output_dir: Path) -> Path | None:
    deadline = time.time() + TRAINER_STATE_TIMEOUT_SECONDS
    while time.time() < deadline:
        candidates = list(output_dir.rglob("trainer_state.json"))
        if candidates:
            return max(candidates, key=lambda p: p.stat().st_mtime)
        time.sleep(TRAINER_STATE_POLL_SECONDS)
    return None


def extract_metric_from_trainer_state(output_dir: Path, metric_key: str) -> float:
    latest = wait_for_trainer_state(output_dir)

    hist = []
    if latest is not None:
        data = json.loads(latest.read_text())
        hist = data.get("log_history", [])

    keys_to_try = [metric_key]
    if metric_key != "eval_loss":
        keys_to_try.append("eval_loss")
    if metric_key != "loss":
        keys_to_try.append("loss")

    for key in keys_to_try:
        vals = [x[key] for x in hist if key in x]
        if vals:
            return float(vals[-1])

    # Fallback: check eval results files if trainer_state.json wasn't written.
    result_files = ["all_results.json", "eval_results.json", "metrics.json"]
    for name in result_files:
        for path in output_dir.rglob(name):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            for key in keys_to_try + [f"eval_{k}" for k in keys_to_try]:
                if key in data:
                    return float(data[key])

    raise RuntimeError(f"Metric '{metric_key}' not found under {output_dir}")


def cleanup_stale_trials(study: optuna.Study) -> int:
    """Mark stale RUNNING trials as FAIL so a restart can continue cleanly."""
    now = datetime.now()
    stale_count = 0
    for trial in study.get_trials(states=[optuna.trial.TrialState.RUNNING]):
        if trial.datetime_start is None:
            continue
        age_seconds = (now - trial.datetime_start).total_seconds()
        if age_seconds >= STALE_TRIAL_MAX_AGE_SECONDS:
            study._storage.set_trial_state_values(
                trial._trial_id,
                optuna.trial.TrialState.FAIL,
                values=None,
            )
            stale_count += 1
    return stale_count


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    beta = trial.suggest_float("beta", 0.01, 1.0, log=True)

    trial_dir = SCRIPT_DIR / "bo_trials" / f"{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"Qwen3-8B-DPO-BO-t{trial.number:03d}"

    config_path = write_config(trial_dir, lr, beta)
    job_script = write_job_script(trial_dir, config_path, run_name)
    job_id = str(trial.number)
    trial.set_user_attr("job_id", job_id)
    trial.set_user_attr("config_path", str(config_path))
    trial.set_user_attr("job_script", str(job_script))

    trial.set_user_attr("stdout_path", str(trial_dir / "run.out"))
    trial.set_user_attr("stderr_path", str(trial_dir / "run.err"))

    state = run_job_in_allocation(job_script, trial_dir)
    if state != "COMPLETED":
        trial.set_user_attr("terminal_state", state)
        raise optuna.exceptions.TrialPruned()

    # output_dir from config
    cfg = yaml.safe_load(config_path.read_text())
    try:
        metric = extract_metric_from_trainer_state(Path(cfg["output_dir"]), METRIC_KEY)
        return metric
    except Exception as exc:
        trial.set_user_attr("metric_error", str(exc))
        raise optuna.exceptions.TrialPruned()


def main():
    storage = f"sqlite:///{SCRIPT_DIR / 'optuna_qwen_dpo.db'}"
    sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=8)
    study = optuna.create_study(
        study_name="qwen3_8B_dpo_bo",
        direction=DIRECTION,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    stale = cleanup_stale_trials(study)
    if stale:
        print(f"Marked {stale} stale RUNNING trial(s) as FAIL.")

    completed_states = [
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.FAIL,
    ]
    completed = len(study.get_trials(states=completed_states))
    remaining = max(0, N_TRIALS - completed)

    if remaining == 0:
        print("All requested trials are already finished.")
        return

    study.optimize(objective, n_trials=remaining)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()