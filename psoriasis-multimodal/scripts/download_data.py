#!/usr/bin/env python3
"""
Download the Kaggle Psoriasis Skin Dataset.

Usage:
    python scripts/download_data.py [--output-dir data/raw]

Requires:
    - Kaggle API key at ~/.kaggle/kaggle.json
    - kaggle package installed (pip install kaggle)
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path


DATASET_SLUG = "pallapurajkumar/psoriasis-skin-dataset"
DEFAULT_OUTPUT = "data/raw"


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    # Also check env vars
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def download_dataset(output_dir: str) -> None:
    """
    Download and extract the psoriasis dataset from Kaggle.

    Args:
        output_dir: Directory to extract the dataset into.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not check_kaggle_credentials():
        print("=" * 70)
        print("ERROR: Kaggle API credentials not found.")
        print()
        print("To configure Kaggle API access:")
        print("  1. Go to https://www.kaggle.com/settings")
        print('  2. Click "Create New Token" under the API section')
        print("  3. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print()
        print("Alternatively, set environment variables:")
        print("  export KAGGLE_USERNAME=your_username")
        print("  export KAGGLE_KEY=your_api_key")
        print()
        print("Or download manually from:")
        print(f"  https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print(f"  Extract contents to: {output_path.resolve()}")
        print("=" * 70)
        sys.exit(1)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        print(f"[Download] Downloading dataset: {DATASET_SLUG}")
        print(f"[Download] Output directory: {output_path.resolve()}")

        api.dataset_download_files(
            DATASET_SLUG,
            path=str(output_path),
            unzip=False,
        )

        # Find and extract the zip file
        zip_files = list(output_path.glob("*.zip"))
        if zip_files:
            for zf in zip_files:
                print(f"[Download] Extracting: {zf.name}")
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(output_path)
                zf.unlink()  # Remove zip after extraction
                print(f"[Download] Removed archive: {zf.name}")

        # Verify download
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_count = sum(
            1
            for f in output_path.rglob("*")
            if f.suffix.lower() in image_extensions
        )

        if image_count > 0:
            print(f"[Download] Success! Found {image_count} images.")
            # List subdirectories (class folders)
            subdirs = [
                d for d in output_path.iterdir() if d.is_dir()
            ]
            for sd in sorted(subdirs):
                n = sum(1 for f in sd.rglob("*") if f.suffix.lower() in image_extensions)
                print(f"  └── {sd.name}: {n} images")
        else:
            print("[Download] WARNING: No images found after extraction.")
            print("  The dataset structure may differ from expected.")
            print(f"  Please check: {output_path.resolve()}")

    except Exception as e:
        print(f"[Download] Error: {e}")
        print()
        print("Falling back to manual download instructions:")
        print(f"  1. Visit: https://www.kaggle.com/datasets/{DATASET_SLUG}")
        print('  2. Click "Download" button')
        print(f"  3. Extract to: {output_path.resolve()}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download Kaggle Psoriasis Skin Dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for dataset (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    download_dataset(args.output_dir)


if __name__ == "__main__":
    main()
