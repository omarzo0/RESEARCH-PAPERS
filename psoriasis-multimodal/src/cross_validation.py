"""
5-fold stratified cross-validation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from src.augmentation import get_train_transforms, get_eval_transforms, CutMixMixUpCollator
from src.dataset import PsoriasisDataset, _default_collate
from src.evaluate import compute_metrics
from src.train import Trainer


def run_cross_validation(
    model_factory,
    config: dict,
    device: torch.device,
    model_name: str = "model",
) -> dict:
    """
    Run 5-fold stratified cross-validation.

    Args:
        model_factory: Callable that returns a fresh model instance.
        config: Full config dict.
        device: torch.device.
        model_name: Name for logging.

    Returns:
        dict: {
            'fold_results': list of per-fold metric dicts,
            'mean': dict of mean metrics,
            'std': dict of std metrics
        }
    """
    cv_cfg = config.get("cross_validation", {})
    n_folds = cv_cfg.get("n_folds", 5)
    data_cfg = config["data"]
    train_cfg = config["training"]
    aug_cfg = config.get("augmentation", {})

    # Load the combined train + val data for CV
    splits_dir = Path(data_cfg["splits_dir"])
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    # Save combined CSV temporarily
    combined_csv = splits_dir / "_cv_combined.csv"
    full_df.to_csv(combined_csv, index=False)

    labels = full_df["response_label"].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.get("seed", 42))

    fold_results = []
    all_y_true = []
    all_y_prob = []

    print(f"\n{'='*60}")
    print(f"Cross-Validation: {model_name} ({n_folds} folds)")
    print(f"{'='*60}\n")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        print(f"\n--- Fold {fold}/{n_folds} ---")

        # Create fold-specific datasets
        train_transform = get_train_transforms(config)
        eval_transform = get_eval_transforms(config)

        # Full dataset with train transforms for training subset
        train_dataset = PsoriasisDataset(str(combined_csv), transform=train_transform)
        val_dataset = PsoriasisDataset(str(combined_csv), transform=eval_transform)

        train_subset = Subset(train_dataset, train_idx.tolist())
        val_subset = Subset(val_dataset, val_idx.tolist())

        # CutMix/MixUp collator
        use_cutmix = aug_cfg.get("cutmix_alpha", 0) > 0
        train_collate = (
            CutMixMixUpCollator(
                cutmix_alpha=aug_cfg.get("cutmix_alpha", 0.4),
                mixup_alpha=aug_cfg.get("mixup_alpha", 0.4),
            )
            if use_cutmix
            else _default_collate
        )

        train_loader = DataLoader(
            train_subset,
            batch_size=train_cfg.get("batch_size", 32),
            shuffle=True,
            num_workers=data_cfg.get("num_workers", 4),
            collate_fn=train_collate,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=train_cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=data_cfg.get("num_workers", 4),
            collate_fn=_default_collate,
            pin_memory=True,
        )

        # Fresh model for each fold
        model = model_factory()
        trainer = Trainer(
            model, config, device,
            experiment_name=f"{model_name}_fold{fold}",
        )

        # Train
        trainer.fit(train_loader, val_loader)

        # Evaluate with best model
        trainer.load_best_model()
        val_metrics = trainer.validate(val_loader)
        metrics = compute_metrics(val_metrics["y_true"], val_metrics["y_pred"], val_metrics["y_prob"])
        metrics["fold"] = fold
        fold_results.append(metrics)

        all_y_true.extend(val_metrics["y_true"])
        all_y_prob.extend(val_metrics["y_prob"])

        print(
            f"  Fold {fold}: AUC={metrics['auc_roc']:.4f}, "
            f"F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}"
        )

    # Aggregate results
    metric_keys = ["auc_roc", "f1_macro", "sensitivity", "specificity", "accuracy", "auprc"]
    mean_metrics = {k: np.mean([r[k] for r in fold_results]) for k in metric_keys}
    std_metrics = {k: np.std([r[k] for r in fold_results]) for k in metric_keys}

    print(f"\n{'='*60}")
    print(f"Cross-Validation Results ({n_folds} folds)")
    print(f"{'='*60}")
    for k in metric_keys:
        print(f"  {k:15s}: {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}")
    print(f"{'='*60}\n")

    # Save results
    results = {
        "model_name": model_name,
        "n_folds": n_folds,
        "fold_results": fold_results,
        "mean": mean_metrics,
        "std": std_metrics,
    }
    save_path = Path(config["paths"]["results"]) / f"{model_name}_cv_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[CV] Results saved → {save_path}")

    # Clean up temp file
    if combined_csv.exists():
        combined_csv.unlink()

    return results
