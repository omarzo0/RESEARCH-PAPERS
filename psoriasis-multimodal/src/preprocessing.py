"""
Data preprocessing: image quality filtering, label engineering, and split creation.
"""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# ── Label Mapping ───────────────────────────────────────────────────────────

SEVERITY_TO_RESPONSE = {
    "Mild": 0,       # Non-Responder (PASI < 10)
    "Moderate": 1,   # Responder     (PASI 10–20)
    "Severe": 1,     # Responder     (PASI > 20)
}

CLASS_NAMES = ["Non-Responder", "Responder"]


def map_severity_to_response(label: str) -> int:
    """
    Map a severity class label to binary biologic-response proxy.

    Based on PASI clinical thresholds:
        Mild (PASI < 10)  → 0 (Non-Responder)
        Moderate/Severe   → 1 (Responder)

    Args:
        label: One of 'Mild', 'Moderate', 'Severe'.

    Returns:
        int: 0 (Non-Responder) or 1 (Responder).
    """
    label_clean = label.strip().capitalize()
    if label_clean not in SEVERITY_TO_RESPONSE:
        raise ValueError(
            f"Unknown severity label: '{label}'. "
            f"Expected one of: {list(SEVERITY_TO_RESPONSE.keys())}"
        )
    return SEVERITY_TO_RESPONSE[label_clean]


# ── Image Quality Filters ──────────────────────────────────────────────────

def compute_blur_score(image_path: str) -> float:
    """
    Compute blur score using Laplacian variance.

    Higher values = sharper image. Typical threshold: 100.

    Args:
        image_path: Path to the image file.

    Returns:
        float: Laplacian variance (blur score).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return cv2.Laplacian(img, cv2.CV_64F).var()


def compute_image_hash(image_path: str, hash_size: int = 8) -> str:
    """
    Compute average hash (aHash) for near-duplicate detection.

    Args:
        image_path: Path to the image file.
        hash_size: Size of the hash grid (default 8 → 64-bit hash).

    Returns:
        str: Hex string hash of the image.
    """
    try:
        import imagehash
        img = Image.open(image_path)
        return str(imagehash.average_hash(img, hash_size=hash_size))
    except ImportError:
        # Fallback: simple perceptual hash using PIL only
        img = Image.open(image_path).convert("L").resize((hash_size, hash_size))
        pixels = np.array(img)
        avg = pixels.mean()
        bits = (pixels > avg).flatten()
        return "".join(str(int(b)) for b in bits)


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    try:
        import imagehash
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2
    except (ImportError, ValueError):
        # Fallback for binary string hashes
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def detect_near_duplicates(
    image_paths: list,
    threshold: int = 5,
) -> list:
    """
    Detect near-duplicate images using average hash + Hamming distance.

    Args:
        image_paths: List of image file paths.
        threshold: Hamming distance threshold — images with distance
                   below this are considered duplicates.

    Returns:
        list: Indices of duplicate images to remove (keeps first occurrence).
    """
    print("[Preprocessing] Computing image hashes for duplicate detection...")
    hashes = []
    for p in tqdm(image_paths, desc="Hashing images"):
        hashes.append(compute_image_hash(p))

    duplicates_to_remove = set()
    for i in range(len(hashes)):
        if i in duplicates_to_remove:
            continue
        for j in range(i + 1, len(hashes)):
            if j in duplicates_to_remove:
                continue
            if hamming_distance(hashes[i], hashes[j]) <= threshold:
                duplicates_to_remove.add(j)

    print(f"[Preprocessing] Found {len(duplicates_to_remove)} near-duplicates.")
    return sorted(duplicates_to_remove)


def filter_low_quality(
    image_paths: list,
    blur_threshold: float = 100.0,
) -> tuple:
    """
    Filter out blurry images based on Laplacian variance.

    Args:
        image_paths: List of image file paths.
        blur_threshold: Images with blur score below this are removed.

    Returns:
        tuple: (kept_paths, removed_paths, blur_scores)
    """
    print(f"[Preprocessing] Filtering blurry images (threshold={blur_threshold})...")
    kept = []
    removed = []
    scores = []

    for p in tqdm(image_paths, desc="Checking blur"):
        score = compute_blur_score(p)
        scores.append(score)
        if score >= blur_threshold:
            kept.append(p)
        else:
            removed.append(p)

    print(f"[Preprocessing] Kept {len(kept)}, removed {len(removed)} blurry images.")
    return kept, removed, scores


# ── Image Resizing ──────────────────────────────────────────────────────────

def resize_image(
    image_path: str,
    output_path: str,
    size: int = 224,
) -> None:
    """
    Resize an image to size x size and save.

    Args:
        image_path: Source image path.
        output_path: Destination path.
        size: Target dimension (square).
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95)


def process_images(
    image_paths: list,
    labels: list,
    raw_dir: str,
    processed_dir: str,
    size: int = 224,
) -> pd.DataFrame:
    """
    Resize all images and build a metadata DataFrame.

    Args:
        image_paths: List of source image paths.
        labels: Corresponding severity labels.
        raw_dir: Root of the raw data directory.
        processed_dir: Root of the processed output directory.
        size: Target image size.

    Returns:
        pd.DataFrame: Metadata with columns [image_path, original_path,
                       severity, response_label].
    """
    processed_dir = Path(processed_dir)
    raw_dir = Path(raw_dir)

    records = []
    print(f"[Preprocessing] Resizing {len(image_paths)} images to {size}x{size}...")

    for img_path, label in tqdm(
        zip(image_paths, labels), total=len(image_paths), desc="Resizing"
    ):
        img_path = Path(img_path)
        # Preserve relative path structure
        try:
            rel_path = img_path.relative_to(raw_dir)
        except ValueError:
            rel_path = Path(label) / img_path.name

        output_path = processed_dir / rel_path
        resize_image(str(img_path), str(output_path), size)

        records.append(
            {
                "image_path": str(output_path),
                "original_path": str(img_path),
                "severity": label,
                "response_label": map_severity_to_response(label),
            }
        )

    df = pd.DataFrame(records)
    print(f"[Preprocessing] Processed {len(df)} images.")
    print(f"  Label distribution:")
    for cls_name, cls_val in zip(CLASS_NAMES, [0, 1]):
        n = (df["response_label"] == cls_val).sum()
        print(f"    {cls_name} ({cls_val}): {n} ({100*n/len(df):.1f}%)")

    return df


# ── Splits ──────────────────────────────────────────────────────────────────

def create_splits(
    metadata_df: pd.DataFrame,
    output_dir: str = "data/splits",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict:
    """
    Create stratified train / val / test splits.

    Args:
        metadata_df: DataFrame with 'image_path' and 'response_label' columns.
        output_dir: Directory to save split CSV files.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        dict: {'train': DataFrame, 'val': DataFrame, 'test': DataFrame}
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_df, valtest_df = train_test_split(
        metadata_df,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=metadata_df["response_label"],
    )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        valtest_df,
        test_size=relative_test_ratio,
        random_state=seed,
        stratify=valtest_df["response_label"],
    )

    splits = {"train": train_df, "val": val_df, "test": test_df}

    for split_name, split_df in splits.items():
        path = output_dir / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        print(
            f"[Splits] {split_name}: {len(split_df)} images "
            f"(label 0: {(split_df['response_label']==0).sum()}, "
            f"label 1: {(split_df['response_label']==1).sum()}) "
            f"→ {path}"
        )

    return splits


# ── Full Pipeline ───────────────────────────────────────────────────────────

def discover_images(raw_dir: str) -> tuple:
    """
    Discover images in the raw dataset directory.

    Expects directory structure: raw_dir/<severity_class>/<image_files>

    Args:
        raw_dir: Path to the raw data directory.

    Returns:
        tuple: (list of image paths, list of corresponding severity labels)
    """
    raw_dir = Path(raw_dir)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    image_paths = []
    labels = []

    # Try to find class subdirectories
    subdirs = sorted([d for d in raw_dir.rglob("*") if d.is_dir()])

    # If raw_dir has direct class folders
    for subdir in raw_dir.iterdir():
        if subdir.is_dir():
            # Check if this looks like a class folder
            class_name = subdir.name.strip()
            # Handle various naming conventions
            class_mapping = {
                "mild": "Mild",
                "moderate": "Moderate",
                "severe": "Severe",
                "normal skin": "Mild",
                "normal_skin": "Mild",
                "normal": "Mild",
                "psoriasis": "Severe"
            }

            matched_class = class_mapping.get(class_name.lower())
            if not matched_class:
                # Try to find class folders recursively
                for inner_dir in subdir.rglob("*"):
                    if inner_dir.is_dir():
                        inner_name = inner_dir.name.strip()
                        matched = class_mapping.get(inner_name.lower())
                        if matched:
                            for f in sorted(inner_dir.iterdir()):
                                if f.suffix.lower() in image_extensions:
                                    image_paths.append(str(f))
                                    labels.append(matched)
                continue

            for f in sorted(subdir.iterdir()):
                if f.suffix.lower() in image_extensions:
                    image_paths.append(str(f))
                    labels.append(matched_class)

    if not image_paths:
        # Fallback: recursively search for any class-named parent directories
        for f in raw_dir.rglob("*"):
            if f.suffix.lower() in image_extensions:
                # Try to infer class from parent directory names
                for parent in f.parents:
                    parent_name = parent.name.strip().lower()
                    if parent_name in ["mild", "moderate", "severe", "normal skin", "normal_skin", "normal", "psoriasis"]:
                        parent_mapping = {
                            "mild": "Mild",
                            "moderate": "Moderate",
                            "severe": "Severe",
                            "normal skin": "Mild",
                            "normal_skin": "Mild",
                            "normal": "Mild",
                            "psoriasis": "Severe"
                        }
                        image_paths.append(str(f))
                        labels.append(parent_mapping[parent_name])
                        break

    print(f"[Discovery] Found {len(image_paths)} images in {raw_dir}")
    if image_paths:
        from collections import Counter
        dist = Counter(labels)
        for cls, cnt in sorted(dist.items()):
            print(f"  {cls}: {cnt}")

    return image_paths, labels


def run_preprocessing_pipeline(config: dict) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Steps:
        1. Discover images in raw directory
        2. Filter blurry images
        3. Remove near-duplicates
        4. Resize to target size
        5. Map severity labels to binary response proxy
        6. Create stratified splits

    Args:
        config: Project configuration dictionary.

    Returns:
        pd.DataFrame: Final metadata DataFrame.
    """
    data_cfg = config["data"]

    # 1. Discover images
    image_paths, labels = discover_images(data_cfg["raw_dir"])
    if not image_paths:
        raise RuntimeError(
            f"No images found in {data_cfg['raw_dir']}. "
            "Please download the dataset first: python scripts/download_data.py"
        )

    # 2. Filter blurry images
    kept_paths, removed_paths, blur_scores = filter_low_quality(
        image_paths, data_cfg.get("blur_threshold", 100)
    )
    kept_labels = [
        labels[i] for i in range(len(image_paths)) if image_paths[i] in set(kept_paths)
    ]

    # 3. Remove near-duplicates
    dup_threshold = data_cfg.get("duplicate_hash_threshold", 5)
    dup_indices = detect_near_duplicates(kept_paths, threshold=dup_threshold)
    dup_set = set(dup_indices)
    clean_paths = [p for i, p in enumerate(kept_paths) if i not in dup_set]
    clean_labels = [l for i, l in enumerate(kept_labels) if i not in dup_set]

    # 4 & 5. Resize and build metadata with label mapping
    metadata_df = process_images(
        clean_paths,
        clean_labels,
        raw_dir=data_cfg["raw_dir"],
        processed_dir=data_cfg["processed_dir"],
        size=data_cfg.get("image_size", 224),
    )

    # 6. Create splits
    seed = config.get("seed", 42)
    create_splits(
        metadata_df,
        output_dir=data_cfg["splits_dir"],
        train_ratio=data_cfg.get("train_ratio", 0.70),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        seed=seed,
    )

    # Save full metadata
    metadata_path = Path(data_cfg["processed_dir"]) / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"[Preprocessing] Full metadata saved to {metadata_path}")

    return metadata_df
