#!/bin/bash
#SBATCH --job-name=ppo_forward
#SBATCH --output=result_redteam_forward.txt
#SBATCH --partition=job
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1  # Request 1 GPU

# Load the Conda environment

source /home/rce5022/miniconda3/bin/activate mlenv

# Run a Python script using Conda environment
cd ~/rl/curiosity_redteam
python -m experiments.imdb_toxicity_response.run_ppo --mode local --gpus 1