import yaml
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigLoadError(Exception):
    """Raised when config.yaml exists but cannot be read or parsed."""


def load_yaml_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Returns a dictionary suitable for passing to PipelineSettings and GraphSchema.

    Raises:
        ConfigLoadError: If the file exists but cannot be read or parsed.
            Callers must not silently swallow this — a corrupt config file should
            not cause the pipeline to run on unintended defaults.
    """
    if not os.path.exists(config_path):
        logger.warning("Config file not found at %s. Using environment variables and defaults.", config_path)
        return {}

    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    except Exception as exc:
        raise ConfigLoadError(
            f"Failed to load config file '{config_path}': {exc}"
        ) from exc

    logger.info("Loaded configuration from %s", config_path)
    return config_dict or {}
