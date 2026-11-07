#!/bin/bash
# Run once on an MN5 login node: rebuild venv-vllm offline from the pushed bundle.
set -euo pipefail
R=/gpfs/projects/ehpc533/users/hannan/rlvr
$R/python/cpython-3.11.13-linux-x86_64-gnu/bin/python3.11 -m venv $R/venv
$R/venv/bin/pip install --no-index --find-links $R/wheels --no-deps -r $R/requirements.txt
$R/venv/bin/python -c "import torch, vllm, trl, transformers; print(torch.__version__, vllm.__version__, trl.__version__, transformers.__version__)"
