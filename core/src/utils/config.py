"""Configuration parser utilities for experiments."""

import ast
import configparser
from pathlib import Path
from typing import Any, Dict, Union


def parse_value(value: str) -> Any:
    """Parse a string value into its appropriate Python type."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_config(config_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """
    Load and parse a configuration file.

    Parameters:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing the parsed configuration
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    parsed_config = {}
    for section in config.sections():
        parsed_config[section] = {}
        for key, value in config[section].items():
            parsed_config[section][key] = parse_value(value)

    return parsed_config


def setup_output_dirs(config: Dict[str, Dict[str, Any]]) -> None:
    """
    Create output directories specified in the configuration.

    Parameters:
        config: Parsed configuration dictionary
    """
    if "output" in config:
        for key, value in config["output"].items():
            if isinstance(value, str) and ("dir" in key or "directory" in key):
                Path(value).mkdir(parents=True, exist_ok=True)
