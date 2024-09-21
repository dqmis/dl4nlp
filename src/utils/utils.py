import yaml
from transformers import AutoTokenizer

__all__ = ["preprocess_function", "load_config"]


def preprocess_function(
    examples: dict, tokenizer: AutoTokenizer, source_lang: str, target_lang: str, prefix: str
) -> dict:
    """
    Preprocess the given examples with the given tokenizer, source language, target language, and prefix.

    Args:
        examples (dict): The examples to preprocess.
        tokenizer (AutoTokenizer): The tokenizer to use for preprocessing.
        source_lang (str): The source language.
        target_lang (str): The target language.
        prefix (str): The prefix to use for the source language.

    Returns:
        dict: The preprocessed examples.
    """
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs: dict = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs


def load_config(config_file: str) -> dict[str, str]:
    """
    Load the configuration file from the given path.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        dict: The configuration file as a dictionary.
    """
    with open(config_file, "r") as file:
        config: dict[str, str] = yaml.safe_load(file)
    return config
