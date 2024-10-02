from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from src.utils.preprocessors import (
    preprocess_flores_function,
    preprocess_helsinki_function,
)

DATASETS = {
    "flores": lambda source_lang, target_lang, tokenizer: load_flores_dataset(
        source_lang, target_lang, tokenizer
    ),
    "ntrex": lambda source_lang, target_lang, tokenizer: load_ntrex_dataset(
        source_lang, target_lang, tokenizer
    ),
    "helsinki": lambda dataset_name, source_lang, target_lang, prefix, tokenizer: load_helsinki_dataset(  # noqa
        dataset_name, source_lang, target_lang, prefix, tokenizer
    ),
}


def load_flores_dataset(source_lang: str, target_lang: str, tokenizer: AutoTokenizer) -> dict:
    dataset = load_dataset(
        "facebook/flores",
        f"{source_lang}_Latn-{target_lang}_Latn",
    )
    processed_dataset: dict = dataset.map(
        preprocess_flores_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "source_lang": "eng",
            "target_lang": "lit",
        },
    )
    return processed_dataset


def load_ntrex_dataset(source_lang: str, target_lang: str, tokenizer: AutoTokenizer) -> dict:
    source_dataset = load_dataset("davidstap/NTREX", f"{source_lang}_Latn")["test"]
    target_dataset = load_dataset("davidstap/NTREX", f"{target_lang}_Latn")["test"]

    source_dataset = source_dataset.rename_column("text", f"sentence_{source_lang}_Latn")
    target_dataset = target_dataset.rename_column("text", f"sentence_{target_lang}_Latn")

    dataset = concatenate_datasets([source_dataset, target_dataset], axis=1)

    processed_dataset: dict = dataset.map(
        preprocess_flores_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "source_lang": "eng",
            "target_lang": "lit",
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
    dataset = dataset["train"].shuffle(seed=42)
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
