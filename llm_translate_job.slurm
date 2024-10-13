#!/bin/bash
#SBATCH --job-name=llm_tr_ee         # Job name
#SBATCH --output=out/en_ee/llm_translate.out        # Standard output file
#SBATCH --error=out/en_ee/llm_translate.err         # Standard error file
#SBATCH --partition=gpu                 # Partition (queue) name
#SBATCH --gpus=1                        # Number of GPUs
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=18              # Number of CPU cores per task
#SBATCH --time=05:00:00                 # Max runtime

# Activate the conda environment
source ~/.bashrc  # Ensures conda is available
conda activate venv

# Run the Python training script with the configuration file
python scripts/translate_data_llm.py --dataset_path=data/opus.nllb.en-ee/en-ee.txt/NLLB.en-ee.en --output_path=out/translated-llm-ee --lang_to=Estonian
