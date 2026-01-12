#!/usr/bin/env python3
"""
Collect and summarize validation results from all experiments.

Usage:
    python collect_validation_results.py [RESULTS_DIR]

Default RESULTS_DIR: /data/cat/ws/hama901h-RL/evaluation_results/validation
"""

import json
import sys
from pathlib import Path
import csv


def collect_results(results_dir: Path) -> dict:
    """Collect all validation results from subdirectories."""
    all_results = {
        "experiments": [],
        "summary": {
            "total_experiments": 0,
            "sft_completed": 0,
            "dpo_completed": 0,
        }
    }
    
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue
        
        exp_result = {
            "name": exp_dir.name,
            "sft": None,
            "dpo": None
        }
        
        # Check for SFT results
        sft_results_file = exp_dir / "sft" / "validation_results.json"
        sft_eval_file = exp_dir / "sft" / "eval_results.json"
        
        if sft_results_file.exists():
            with open(sft_results_file) as f:
                exp_result["sft"] = json.load(f)
            all_results["summary"]["sft_completed"] += 1
        elif sft_eval_file.exists():
            with open(sft_eval_file) as f:
                exp_result["sft"] = {"metrics": json.load(f)}
            all_results["summary"]["sft_completed"] += 1
        
        # Check for DPO results
        dpo_results_file = exp_dir / "dpo" / "validation_results.json"
        dpo_eval_file = exp_dir / "dpo" / "eval_results.json"
        
        if dpo_results_file.exists():
            with open(dpo_results_file) as f:
                exp_result["dpo"] = json.load(f)
            all_results["summary"]["dpo_completed"] += 1
        elif dpo_eval_file.exists():
            with open(dpo_eval_file) as f:
                exp_result["dpo"] = {"metrics": json.load(f)}
            all_results["summary"]["dpo_completed"] += 1
        
        if exp_result["sft"] or exp_result["dpo"]:
            all_results["experiments"].append(exp_result)
            all_results["summary"]["total_experiments"] += 1
    
    return all_results


def save_summary_csv(results: dict, output_file: Path):
    """Save a CSV summary of the results."""
    rows = []
    
    for exp in results["experiments"]:
        row = {
            "experiment": exp["name"],
            "sft_eval_loss": None,
            "dpo_eval_loss": None,
            "dpo_rewards_chosen": None,
            "dpo_rewards_rejected": None,
            "dpo_rewards_margin": None,
            "dpo_logps_chosen": None,
            "dpo_logps_rejected": None,
        }
        
        if exp["sft"] and "metrics" in exp["sft"]:
            metrics = exp["sft"]["metrics"]
            row["sft_eval_loss"] = metrics.get("eval_loss")
        
        if exp["dpo"] and "metrics" in exp["dpo"]:
            metrics = exp["dpo"]["metrics"]
            row["dpo_eval_loss"] = metrics.get("eval_loss")
            row["dpo_rewards_chosen"] = metrics.get("eval_rewards/chosen")
            row["dpo_rewards_rejected"] = metrics.get("eval_rewards/rejected")
            row["dpo_rewards_margin"] = metrics.get("eval_rewards/margins")
            row["dpo_logps_chosen"] = metrics.get("eval_logps/chosen")
            row["dpo_logps_rejected"] = metrics.get("eval_logps/rejected")
        
        rows.append(row)
    
    if rows:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/cat/ws/hama901h-RL/evaluation_results/validation")
    
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"Collecting results from: {results_dir}")
    
    # Collect results
    results = collect_results(results_dir)
    
    # Save combined JSON
    json_output = results_dir / "all_validation_results.json"
    with open(json_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON results saved to: {json_output}")
    
    # Save CSV summary
    csv_output = results_dir / "validation_summary.csv"
    save_summary_csv(results, csv_output)
    print(f"CSV summary saved to: {csv_output}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {results['summary']['total_experiments']}")
    print(f"SFT validations completed: {results['summary']['sft_completed']}")
    print(f"DPO validations completed: {results['summary']['dpo_completed']}")
    print()
    
    # Print table
    print(f"{'Experiment':<40} {'SFT Loss':<12} {'DPO Loss':<12}")
    print("-" * 64)
    
    for exp in results["experiments"]:
        sft_loss = "N/A"
        dpo_loss = "N/A"
        
        if exp["sft"] and "metrics" in exp["sft"]:
            loss = exp["sft"]["metrics"].get("eval_loss")
            if loss is not None:
                sft_loss = f"{loss:.4f}"
        
        if exp["dpo"] and "metrics" in exp["dpo"]:
            loss = exp["dpo"]["metrics"].get("eval_loss")
            if loss is not None:
                dpo_loss = f"{loss:.4f}"
        
        print(f"{exp['name']:<40} {sft_loss:<12} {dpo_loss:<12}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
