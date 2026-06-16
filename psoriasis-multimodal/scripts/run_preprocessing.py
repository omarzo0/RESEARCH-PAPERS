#!/usr/bin/env python3
"""
Run the full preprocessing pipeline: image quality filtering, resizing, label mapping, and split creation.

Usage:
    python scripts/run_preprocessing.py --config configs/config.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_environment
from src.preprocessing import run_preprocessing_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run data preprocessing pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config, device = setup_environment(args.config)

    print(f"\n{'='*60}")
    print("Running Preprocessing Pipeline")
    print(f"{'='*60}\n")

    metadata = run_preprocessing_pipeline(config)

    print(f"\n{'='*60}")
    print("Preprocessing Complete!")
    print(f"  Processed images: {len(metadata)}")
    print(f"  Splits saved to:  {config['data']['splits_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
