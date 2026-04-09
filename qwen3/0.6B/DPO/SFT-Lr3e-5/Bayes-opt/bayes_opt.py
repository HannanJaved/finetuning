#!/usr/bin/env python3
import json
import re
import subprocess
import time
from pathlib import Path

import optuna
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
JOB_TEMPLATE = SCRIPT_DIR / "LR1e7_Beta0.1.sh"
CONFIG_TEMPLATE = SCRIPT_DIR / "dpo_beta0.1_LR.yaml"

N_TRIALS = 25
POLL_SECONDS = 30

# IMPORTANT: choose what to optimize
METRIC_KEY = "eval_loss"   # fallback to "loss" if eval not available
DIRECTION = "minimize"     # "maximize" for reward metrics


def write_config(trial_dir: Path, lr: float, beta: float) -> Path:
    with CONFIG_TEMPLATE.open() as f:
        cfg = yaml.safe_load(f)

    run_name = f"Qwen3-0.6B-DPO-BO-trial{trial_dir.name}-b{beta:.4f}-lr{lr:.2e}"
    out_dir = f"/data/horse/ws/hama901h-BFTranslation/checkpoints/Qwen/Qwen3-0.6B-Base/SFT-sweep/{run_name}/"

    cfg["learning_rate"] = float(lr)
    cfg["beta"] = float(beta)
    cfg["output_dir"] = out_dir

    # Use 1% of the dataset as validation by creating a test split.
    # The remaining 99% stays in train.
    cfg["dataset_test_split_size"] = 0.01
    cfg["dataset_test_split_seed"] = 42

    # Enable evaluation so BO can read eval metrics.
    cfg["do_eval"] = True
    cfg["eval_strategy"] = "steps"

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


def submit_job(job_script: Path) -> str:
    out = subprocess.check_output(["sbatch", str(job_script)], text=True).strip()
    # e.g. "Submitted batch job 123456"
    job_id = out.split()[-1]
    return job_id


def wait_for_job(job_id: str):
    # Wait until sacct reports terminal state
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED"}
    while True:
        try:
            out = subprocess.check_output(
                ["sacct", "-j", job_id, "--format=State", "--noheader"],
                text=True
            )
            states = [s.strip().split()[0] for s in out.splitlines() if s.strip()]
            if states and any(s in terminal for s in states):
                # choose first terminal state encountered
                for s in states:
                    if s in terminal:
                        return s
        except subprocess.CalledProcessError:
            pass
        time.sleep(POLL_SECONDS)


def extract_metric_from_trainer_state(output_dir: Path, metric_key: str) -> float:
    # Finds most recent trainer_state.json under output_dir
    candidates = list(output_dir.rglob("trainer_state.json"))
    if not candidates:
        raise RuntimeError(f"No trainer_state.json found under {output_dir}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)

    data = json.loads(latest.read_text())
    hist = data.get("log_history", [])
    vals = [x[metric_key] for x in hist if metric_key in x]
    if not vals:
        raise RuntimeError(f"Metric '{metric_key}' not found in {latest}")
    return float(vals[-1])


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("learning_rate", 1e-7, 5e-5, log=True)
    beta = trial.suggest_float("beta", 0.05, 1.0)

    trial_dir = SCRIPT_DIR / "bo_trials" / f"{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"Qwen3-0.6B-DPO-BO-t{trial.number:03d}"

    config_path = write_config(trial_dir, lr, beta)
    job_script = write_job_script(trial_dir, config_path, run_name)
    job_id = submit_job(job_script)

    trial.set_user_attr("job_id", job_id)
    trial.set_user_attr("config_path", str(config_path))
    trial.set_user_attr("job_script", str(job_script))

    state = wait_for_job(job_id)
    if state != "COMPLETED":
        raise RuntimeError(f"Trial job {job_id} ended with state {state}")

    # output_dir from config
    cfg = yaml.safe_load(config_path.read_text())
    metric = extract_metric_from_trainer_state(Path(cfg["output_dir"]), METRIC_KEY)
    return metric


def main():
    storage = f"sqlite:///{SCRIPT_DIR / 'optuna_qwen_dpo.db'}"
    sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=8)
    study = optuna.create_study(
        study_name="qwen3_06b_dpo_bo",
        direction=DIRECTION,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)

    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()