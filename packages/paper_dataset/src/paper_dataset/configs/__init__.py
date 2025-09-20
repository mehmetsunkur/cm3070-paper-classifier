"""Configuration files for paper processing."""

import os
from pathlib import Path

# Path to the configs directory
CONFIGS_DIR = Path(__file__).parent
LABEL_CONFIG_PATH = CONFIGS_DIR / "label_config.yaml"

__all__ = ["CONFIGS_DIR", "LABEL_CONFIG_PATH"]