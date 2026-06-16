"""
PyTorch Dataset and DataLoader factory for psoriasis images.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.augmentation import get_train_transforms, get_eval_transforms, CutMixMixUpCollator


class PsoriasisDataset(Dataset):
    """
    PyTorch Dataset for psoriasis skin images.

    Reads a CSV split file with columns:
        - image_path: path to the processed image
        - response_label: binary label (0=Non-Responder, 1=Responder)
        - severity: original severity class (Mild/Moderate/Severe)

    Args:
        csv_path: Path to the split CSV file (train.csv, val.csv, or test.csv).
        transform: Albumentations transform pipeline (or None).
    """

    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

        # Validate required columns
        required_cols = {"image_path", "response_label"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        # Validate all image paths exist
        missing_files = [
            p for p in self.df["image_path"]
            if not Path(p).exists()
        ]
        if missing_files:
            print(
                f"[Dataset] WARNING: {len(missing_files)} image files not found. "
                f"First few: {missing_files[:3]}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = row["image_path"]
        label = int(row["response_label"])

        # Load image as RGB numpy array
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label, image_path

    @property
    def labels(self) -> np.ndarray:
        """Return all labels as a numpy array (for stratification)."""
        return self.df["response_label"].values

    @property
    def class_counts(self) -> dict:
        """Return count of each class."""
        counts = self.df["response_label"].value_counts().to_dict()
        return {int(k): v for k, v in counts.items()}

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for loss balancing.

        Returns:
            torch.Tensor: Weight for each class.
        """
        counts = self.class_counts
        total = sum(counts.values())
        num_classes = len(counts)
        weights = [total / (num_classes * counts.get(i, 1)) for i in range(num_classes)]
        return torch.tensor(weights, dtype=torch.float32)


def _default_collate(batch):
    """Default collate — just stacks images/labels and keeps paths as a list."""
    images, labels, paths = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels, list(paths)


def get_dataloaders(config: dict) -> dict:
    """
    Create train, val, and test DataLoaders.

    Args:
        config: Full project config dict.

    Returns:
        dict: {'train': DataLoader, 'val': DataLoader, 'test': DataLoader,
               'train_dataset': Dataset, 'val_dataset': Dataset, 'test_dataset': Dataset}
    """
    data_cfg = config["data"]
    train_cfg = config["training"]
    aug_cfg = config.get("augmentation", {})
    splits_dir = Path(data_cfg["splits_dir"])

    # Build transforms
    train_transform = get_train_transforms(config)
    eval_transform = get_eval_transforms(config)

    # Create datasets
    train_dataset = PsoriasisDataset(
        csv_path=str(splits_dir / "train.csv"),
        transform=train_transform,
    )
    val_dataset = PsoriasisDataset(
        csv_path=str(splits_dir / "val.csv"),
        transform=eval_transform,
    )
    test_dataset = PsoriasisDataset(
        csv_path=str(splits_dir / "test.csv"),
        transform=eval_transform,
    )

    # Print dataset info
    print(f"[DataLoader] Train: {len(train_dataset)} images")
    print(f"[DataLoader] Val:   {len(val_dataset)} images")
    print(f"[DataLoader] Test:  {len(test_dataset)} images")

    # CutMix/MixUp collator for training
    use_cutmix = aug_cfg.get("cutmix_alpha", 0) > 0 or aug_cfg.get("mixup_alpha", 0) > 0
    if use_cutmix:
        train_collate = CutMixMixUpCollator(
            cutmix_alpha=aug_cfg.get("cutmix_alpha", 0.4),
            mixup_alpha=aug_cfg.get("mixup_alpha", 0.4),
            num_classes=2,
            prob=0.5,
        )
    else:
        train_collate = _default_collate

    batch_size = train_cfg.get("batch_size", 32)
    num_workers = data_cfg.get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_default_collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_default_collate,
        pin_memory=True,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }
