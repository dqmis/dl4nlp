# Multilingual Neural Machine Translation for Low-Resource Languages

---

## Project Setup and Requirements

- ### Requirements

The project requires `Python ^3.11` version. Other dependencies are listed in the `pyproject.toml` file.

- ### Installation

The project uses Poetry to manage dependencies. To install the dependencies, run the following command:

```bash
# snel specific
/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda init bash
# restart shell
conda create python=3.11 -n venv
conda activate venv

pip install poetry

## or use poetry to create a virtual environment
# poetry env use python3.11
# poetry shell

poetry install
poetry run pre-commit install
```

- ### Data Preparation

```bash
python scripts/download_nllb.py
```

- ### Training

```bash
# snel specific - ask for resources and shell
# don't forget to allocate enough time
srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18  --time=00:01:00 --pty bash -i

conda activate venv
python scripts/train.py
```