from transformers import AutoTokenizer

def preprocess_ntrex_function(
    source_examples: dict, target_examples: dict, tokenizer: AutoTokenizer
) -> dict:
    """
    Preprocess the given examples with the given tokenizer, source language, target language, and prefix. # noqa

    Args:
        examples (dict): The examples to preprocess.
        tokenizer (AutoTokenizer): The tokenizer to use for preprocessing.
        source_lang (str): The source language.
        target_lang (str): The target language.
        prefix (str): The prefix to use for the source language.

    Returns:
        dict: The preprocessed examples.
    """

    inputs = [example["text"] for example in source_examples]
    targets = [example["text"] for example in target_examples]
    model_inputs: dict = tokenizer(
        inputs,
        text_target=targets,
        max_length=128,
        truncation=True,
    )
    return model_inputs

def preprocess_helsinki_function(
    examples: dict, tokenizer: AutoTokenizer, source_lang: str, target_lang: str, prefix: str
) -> dict:
    """
    Preprocess the given examples with the given tokenizer, source language, target language, and prefix. # noqa

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


def preprocess_flores_function(
    examples: dict, tokenizer: AutoTokenizer, source_lang: str, target_lang: str
) -> dict:
    """
    Preprocess the given examples with the given tokenizer, source language, target language, and prefix. # noqa

    Args:
        examples (dict): The examples to preprocess.
        tokenizer (AutoTokenizer): The tokenizer to use for preprocessing.
        source_lang (str): The source language.
        target_lang (str): The target language.
        prefix (str): The prefix to use for the source language.

    Returns:
        dict: The preprocessed examples.
    """
    source_key = f"sentence_{source_lang}_Latn"
    target_key = f"sentence_{target_lang}_Latn"

    inputs = [example for example in examples[source_key]]
    targets = [example for example in examples[target_key]]
    model_inputs: dict = tokenizer(
        inputs,
        text_target=targets,
        max_length=128,
        truncation=True,
    )
    return model_inputs
