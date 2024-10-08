from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from src.utils.preprocessors import (
    preprocess_flores_function,
    preprocess_helsinki_function,
    preprocess_nllb_function,
)


def load_flores_dataset(
    source_lang: str, target_lang: str, tokenizer: AutoTokenizer
) -> dict:
    dataset = load_dataset(
        "facebook/flores",
        f"{source_lang}_Latn-{target_lang}_Latn",
    )
    processed_dataset: dict = dataset.map(
        preprocess_flores_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "source_lang": source_lang,
            "target_lang": target_lang,
        },
    )
    return processed_dataset


def load_ntrex_dataset(
    source_lang: str, target_lang: str, tokenizer: AutoTokenizer
) -> dict:
    source_dataset = load_dataset("davidstap/NTREX", f"{source_lang}_Latn")["test"]
    target_dataset = load_dataset("davidstap/NTREX", f"{target_lang}_Latn")["test"]

    source_dataset = source_dataset.rename_column(
        "text", f"sentence_{source_lang}_Latn"
    )
    target_dataset = target_dataset.rename_column(
        "text", f"sentence_{target_lang}_Latn"
    )

    dataset = concatenate_datasets([source_dataset, target_dataset], axis=1)

    processed_dataset: dict = dataset.map(
        preprocess_flores_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "source_lang": source_lang,
            "target_lang": target_lang,
        },
    )
    return processed_dataset


def load_helsinki_dataset(
    dataset_name: str,
    source_lang: str,
    target_lang: str,
    prefix: str,
    tokenizer: AutoTokenizer,
) -> dict:
    dataset = load_dataset(dataset_name, f"{source_lang}-{target_lang}")

    dataset = dataset["train"].select(range(232984))
    dataset = dataset.train_test_split(test_size=0.2)

    # Apply the preprocess function to dataset
    tokenized_dataset: dict = dataset.map(
        preprocess_helsinki_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "prefix": prefix,
        },
    )

    return tokenized_dataset


def load_nllb_dataset(
    dataset_path: str,
    source_lang: str,
    target_lang: str,
    tokenizer: AutoTokenizer,
    sample: float = 100,
) -> dict:
    assert f"{source_lang}-{target_lang}" in dataset_path, "Dataset language mismatch"
    dataset = load_dataset(dataset_path, data_files=["*.parquet"])
    dataset = dataset["train"]
    dataset = dataset.train_test_split(test_size=0.2)

    # Apply the preprocess function to dataset
    tokenized_dataset: dict = dataset.map(
        preprocess_nllb_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "source_lang": source_lang,
            "target_lang": target_lang,
        },
    )

    return tokenized_dataset


DATASETS = {
    "flores": load_flores_dataset,
    "helsinki": load_helsinki_dataset,
    "ntrex": load_ntrex_dataset,
    "nllb": load_nllb_dataset,
}
