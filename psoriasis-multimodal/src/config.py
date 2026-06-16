"""
Configuration loader and environment setup utilities.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    """
    Load the YAML configuration file.

    Args:
        path: Path to the config YAML file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required top-level keys
    required_keys = ["data", "training", "optimizer", "models", "paths"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config key: '{key}'")

    return config


def get_device() -> torch.device:
    """
    Auto-detect the best available compute device.

    Returns:
        torch.device: CUDA if available, then MPS (Apple Silicon), else CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Using CUDA: {gpu_name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[Device] Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("[Device] Using CPU (training will be slow)")
    return device


def setup_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] All random seeds set to {seed}")


def setup_environment(config_path: str = "configs/config.yaml"):
    """
    Full environment setup: load config, set seed, detect device.

    Args:
        config_path: Path to config YAML.

    Returns:
        tuple: (config dict, torch.device)
    """
    config = load_config(config_path)
    seed = config.get("seed", 42)
    setup_seed(seed)
    device = get_device()

    # Ensure output directories exist
    for dir_key in ["checkpoints", "results", "figures", "logs"]:
        dir_path = Path(config["paths"].get(dir_key, dir_key))
        dir_path.mkdir(parents=True, exist_ok=True)

    return config, device
