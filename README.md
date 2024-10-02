# Multilingual Neural Machine Translation for Low-Resource Languages

---

## Project Setup and Requirements

- ### Requirements

The project requires `Python ^3.11` version. Other dependencies are listed in the `pyproject.toml` file.

- ### Installation

The project uses Poetry to manage dependencies. To install the dependencies, run the following command:

```bash
poetry env use python3.11
poetry shell
poetry install
poetry run pre-commit install
```

- ### Data Preparation

```bash
python scripts/download_nllb.py
```
