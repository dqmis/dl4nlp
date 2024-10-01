import yaml


def load_config(config_file: str) -> dict[str, str]:
    """
    Load the configuration file from the given path.

    Args:
        config_file (str): The path to the configuration file.

    Returns:
        dict: The configuration file as a dictionary.
    """
    with open(config_file) as file:
        config: dict[str, str] = yaml.safe_load(file)
    return config


def populate_training_args(training_args: dict) -> dict:
    training_args["save_strategy"] = "epoch"
    training_args["eval_strategy"] = "epoch"
    training_args["push_to_hub"] = False
    training_args["include_inputs_for_metrics"] = True
    training_args["load_best_model_at_end"] = True
    training_args["predict_with_generate"] = True

    return training_args
