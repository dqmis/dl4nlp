from transformers import AutoTokenizer


def preprocess_function(
    examples: dict, tokenizer: AutoTokenizer, source_lang: str, target_lang: str, prefix: str
) -> dict:
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs: dict = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs
