#!/bin/bash
#SBATCH --job-name=llm_ag_ee         # Job name
#SBATCH --output=out/en_ee/llm_augment.out        # Standard output file
#SBATCH --error=out/en_ee/llm_augment.err         # Standard error file
#SBATCH --partition=gpu                 # Partition (queue) name
#SBATCH --gpus=1                        # Number of GPUs
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=18              # Number of CPU cores per task
#SBATCH --time=04:00:00                 # Max runtime

# Activate the conda environment
source ~/.bashrc  # Ensures conda is available
conda activate venv

# Run the Python training script with the configuration file
python scripts/augment_data_llm.py --dataset_path=data/opus.nllb.en-ee/en-ee.txt/NLLB.en-ee.ee --output_path=out/augmented-llm-ee --lang_from=Estonian
