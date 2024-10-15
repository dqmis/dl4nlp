#!/bin/bash
#SBATCH --job-name=bt_en_af          # Job name
#SBATCH --output=out/en_af/backtranslate.out        # Standard output file
#SBATCH --error=out/en_af/backtranslate.err         # Standard error file
#SBATCH --partition=gpu                 # Partition (queue) name
#SBATCH --gpus=1                        # Number of GPUs
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=18              # Number of CPU cores per task
#SBATCH --time=05:00:00                 # Max runtime

# Activate the conda environment
source ~/.bashrc  # Ensures conda is available
conda activate venv

# Run the Python training script with the configuration file
python scripts/augment_data_backtranslate.py --dataset_path=data/opus.nllb.en-af/en-af.txt/NLLB.af-en.en --output_path=out/backtranslated-af --lang_from=af --lang_to=en
