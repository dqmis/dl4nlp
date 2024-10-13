#!/bin/bash
#SBATCH --job-name=llm_af_augment         # Job name
#SBATCH --output=out/en_af/llm_augment.out        # Standard output file
#SBATCH --error=out/en_af/llm_augment.err         # Standard error file
#SBATCH --partition=gpu                 # Partition (queue) name
#SBATCH --gpus=1                        # Number of GPUs
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=18              # Number of CPU cores per task
#SBATCH --time=04:00:00                 # Max runtime

# Activate the conda environment
source ~/.bashrc  # Ensures conda is available
conda activate venv

# Run the Python training script with the configuration file
python scripts/augment_data_llm.py --dataset_path=data/opus.nllb.en-af/en-af.txt/NLLB.af-en.af --output_path=out/llm-af --lang_from=Afrikaans
