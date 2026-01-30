#!/bin/bash
#SBATCH --job-name=env_setup 
#SBATCH --partition=dev-g  
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1     
#SBATCH --gpus-per-node=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=7
#SBATCH --time=00:10:00
#SBATCH --account=project_462000963
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file

# mkdir -p logs

# Load modules
module load LUMI #Loads correct compilers for the accelerators, propably not needed
module use /appl/local/csc/modulefiles/ #Add the module path needed for csc modules in Lumi
module load pytorch/2.7


#Create venv
python -m venv .handbook_venv --system-site-packages

#Activate
source .handbook_venv/bin/activate

# Install pip packages
pip install git+https://github.com/huggingface/transformers.git
pip install trl==0.18.0




