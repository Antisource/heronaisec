"""
Utility functions shared across the project.

This module intentionally contains only infrastructure code.
No attack logic, training logic, or evaluation logic belongs here.
"""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import torch
import yaml
from accelerate.utils import set_seed
from datetime import datetime

# ============================================================================
# Configuration loader
# ============================================================================

def load_config(config_path: str | Path) -> dict:
    """
    Load an experiment configuration from a YAML file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
 
# ============================================================================
# Reproducibility
# ============================================================================
  
def seed_everything(seed: int) -> None:
    """
    Seed all supported random number generators.

    Reproducibility is essential for comparing experiments.
    """

    set_seed(seed)

    # Prefer deterministic algorithms when available.
    #torch.use_deterministic_algorithms(True, warn_only=True)

# ============================================================================    
# Device Selection
# ============================================================================

def get_device() -> torch.device:
    """
    Select the best available compute device.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

# ============================================================================
# Directory creation
# ============================================================================

def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.
    """

    path = Path(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


# ============================================================================
# Timestamp
# ============================================================================

def timestamp() -> str:
    """
    Return a filesystem-safe timestamp for experiment outputs.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ============================================================================
# Create a new directory for each result
# ============================================================================

def create_run_directory(config: dict) -> Path:
    """
    Create a unique output directory for a single experiment run.

    Example
    -------
    results/
        baseline/
            20260719_113522/
    """

    run_dir = (
        Path(config["experiment"]["output_dir"])
        / config["experiment"]["name"]
        / timestamp()
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)

    return run_dir