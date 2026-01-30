#!/bin/bash
#SBATCH --job-name=env_setup 
#SBATCH --partition=dev-g  
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1     
#SBATCH --gpus-per-node=1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=7
#SBATCH --time=00:10:00
#SBATCH --account=project_462000353
#SBATCH -o %x.out
#SBATCH -e %x.err

# mkdir -p logs

# Load modules
module load LUMI #Loads correct compilers for the accelerators, propably not needed
module use /appl/local/csc/modulefiles/ #Add the module path needed for csc modules in Lumi
module load pytorch/2.4


#Create venv
python -m venv .trl_dpo_venv --system-site-packages

#Activate
source .trl_dpo_venv/bin/activate

# Install pip packages
# pip install --upgrade transformers 
# pip install --upgrade huggingface_hub
# pip install --upgrade accelerate
# pip install --upgrade deepspeed
pip install transformers==4.46.0
pip install trl==0.12.0



