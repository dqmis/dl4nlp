#!/bin/bash
#SBATCH --job-name=train_en_ee_bt          # Job name
#SBATCH --output=out/train_model.out        # Standard output file
#SBATCH --error=out/train_model.err         # Standard error file
#SBATCH --partition=gpu                 # Partition (queue) name
#SBATCH --gpus=1                        # Number of GPUs
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=18              # Number of CPU cores per task
#SBATCH --time=04:00:00                 # Max runtime

# Load necessary modules or software environments if needed

# Activate the conda environment
source ~/.bashrc  # Ensures conda is available
conda activate venv

# Run the Python training script with the configuration file
python scripts/train.py train_bt_ee.yaml
