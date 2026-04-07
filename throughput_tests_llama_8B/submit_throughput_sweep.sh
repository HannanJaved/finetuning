#!/bin/bash
set -euo pipefail

SCRIPT_DIR="/data/cat/ws/hama901h-Posttraining/finetuning/throughput_tests_llama_8B"

sbatch "$SCRIPT_DIR/throughput_1gpu_1node.sh"
sbatch "$SCRIPT_DIR/throughput_4gpu_1node.sh"
sbatch "$SCRIPT_DIR/throughput_8gpu_2node.sh"
sbatch "$SCRIPT_DIR/throughput_12gpu_3node.sh"
sbatch "$SCRIPT_DIR/throughput_16gpu_4node.sh"
