#!/bin/bash
#SBATCH --job-name=install_axolotl
#SBATCH --output=.logs/misc/%x.out
#SBATCH --error=.logs/misc/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=14        
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

export CMAKE_BUILD_PARALLEL_LEVEL=16
export MAKEFLAGS="-j16"
export TORCH_CUDA_ARCH_LIST="9.0"
export TORCH_EXTENSIONS_DIR="$VIRTUAL_ENV/../.cache/torch_extensions"

pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip3 install --no-build-isolation axolotl[deepspeed]