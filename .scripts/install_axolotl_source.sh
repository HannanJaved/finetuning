#!/bin/bash
#SBATCH --job-name=install_axolotl2
#SBATCH --output=.logs/misc/%x.out
#SBATCH --error=.logs/misc/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=4        
#SBATCH --mem=32G                
#SBATCH --time=2:00:00          
#SBATCH --partition=capella
#SBATCH --licenses=cat

source /data/cat/ws/hama901h-RL/.axolotl_venv/bin/activate

cd /data/cat/ws/hama901h-RL/

module load CUDA

export PYTHONPYCACHEPREFIX="$VIRTUAL_ENV/../.cache/pycache"
export HUGGINGFACE_HOME="$VIRTUAL_ENV/../.cache/huggingface"
export HF_HOME="$VIRTUAL_ENV/../.cache/huggingface"
export TORCH_HOME="$VIRTUAL_ENV/../.cache/torch"
export TRANSFORMERS_CACHE="$VIRTUAL_ENV/../.cache/transformers"
export XDG_CACHE_HOME="$VIRTUAL_ENV/../.cache"
export PIP_CACHE_DIR="//data/cat/ws/hama901h-RL/.cache/pip"

cd axolotl
pip3 install -e .