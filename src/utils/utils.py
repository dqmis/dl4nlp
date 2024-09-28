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
