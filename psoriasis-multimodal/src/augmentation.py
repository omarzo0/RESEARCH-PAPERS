"""
Augmentation pipelines using Albumentations.
"""

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(config: dict) -> A.Compose:
    """
    Build the training augmentation pipeline.

    Includes: RandomResizedCrop, flips, color jitter, rotation, normalize.

    Args:
        config: Full config dict.

    Returns:
        albumentations.Compose: Training transform pipeline.
    """
    aug_cfg = config.get("augmentation", {})
    data_cfg = config["data"]
    size = data_cfg.get("image_size", 224)
    mean = data_cfg.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("imagenet_std", [0.229, 0.224, 0.225])

    transforms = [
        A.RandomResizedCrop(
            size=(size, size),
            scale=(aug_cfg.get("crop_scale_min", 0.8), aug_cfg.get("crop_scale_max", 1.0)),
            ratio=(0.9, 1.1),
        ),
    ]

    if aug_cfg.get("horizontal_flip", True):
        transforms.append(A.HorizontalFlip(p=0.5))

    if aug_cfg.get("vertical_flip", True):
        transforms.append(A.VerticalFlip(p=0.5))

    transforms.extend([
        A.ColorJitter(
            brightness=aug_cfg.get("brightness", 0.2),
            contrast=aug_cfg.get("contrast", 0.2),
            saturation=aug_cfg.get("saturation", 0.1),
            hue=0.0,
            p=0.8,
        ),
        A.Rotate(
            limit=aug_cfg.get("rotation_limit", 15),
            border_mode=0,  # BORDER_CONSTANT
            p=0.5,
        ),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms)


def get_eval_transforms(config: dict) -> A.Compose:
    """
    Build the evaluation (val/test) transform pipeline.

    Only resize + normalize, no augmentation.

    Args:
        config: Full config dict.

    Returns:
        albumentations.Compose: Evaluation transform pipeline.
    """
    data_cfg = config["data"]
    size = data_cfg.get("image_size", 224)
    mean = data_cfg.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = data_cfg.get("imagenet_std", [0.229, 0.224, 0.225])

    return A.Compose([
        A.Resize(height=size, width=size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ── CutMix / MixUp Collator ────────────────────────────────────────────────

class CutMixMixUpCollator:
    """
    Custom collate function that applies CutMix or MixUp at the batch level.

    With 50% probability applies CutMix, with 50% applies MixUp.
    Only used during training.

    Args:
        cutmix_alpha: Alpha parameter for CutMix Beta distribution.
        mixup_alpha: Alpha parameter for MixUp Beta distribution.
        num_classes: Number of output classes (for one-hot encoding).
        prob: Probability of applying either augmentation.
    """

    def __init__(
        self,
        cutmix_alpha: float = 0.4,
        mixup_alpha: float = 0.4,
        num_classes: int = 2,
        prob: float = 0.5,
    ):
        self.cutmix_alpha = cutmix_alpha
        self.mixup_alpha = mixup_alpha
        self.num_classes = num_classes
        self.prob = prob

    def __call__(self, batch):
        images, labels, paths = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        paths = list(paths)

        if np.random.random() > self.prob:
            # No augmentation
            return images, labels, paths

        # Convert labels to one-hot for mixing
        one_hot = torch.zeros(len(labels), self.num_classes)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)

        # Randomly choose CutMix or MixUp
        if np.random.random() < 0.5:
            images, one_hot = self._cutmix(images, one_hot)
        else:
            images, one_hot = self._mixup(images, one_hot)

        return images, one_hot, paths

    def _mixup(self, images, labels):
        """Apply MixUp augmentation."""
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]
        return mixed_images, mixed_labels

    def _cutmix(self, images, labels):
        """Apply CutMix augmentation."""
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)

        _, _, h, w = images.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        # Random center
        cy = np.random.randint(h)
        cx = np.random.randint(w)

        # Bounding box
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)

        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Adjust lambda based on actual cut area
        actual_lam = 1 - (y2 - y1) * (x2 - x1) / (h * w)
        mixed_labels = actual_lam * labels + (1 - actual_lam) * labels[index]

        return images, mixed_labels
