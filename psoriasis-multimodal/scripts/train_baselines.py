#!/usr/bin/env python3
"""
Train baseline CNN models (ResNet-50, DenseNet-121, VGG-16).

Usage:
    python scripts/train_baselines.py --config configs/config.yaml --model resnet50
    python scripts/train_baselines.py --config configs/config.yaml --model densenet121
    python scripts/train_baselines.py --config configs/config.yaml --model vgg16
    python scripts/train_baselines.py --config configs/config.yaml --model all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_environment
from src.dataset import get_dataloaders
from src.evaluate import compute_metrics, plot_roc_curve, plot_confusion_matrix, save_metrics
from src.models import get_model
from src.train import Trainer
from src.utils import count_parameters


BASELINE_MODELS = ["resnet50", "densenet121", "vgg16"]


def train_single_baseline(model_name: str, config: dict, device):
    """Train a single baseline model."""
    print(f"\n{'='*60}")
    print(f"Training Baseline: {model_name}")
    print(f"{'='*60}\n")

    # Create model
    model = get_model(model_name, config)
    params = count_parameters(model)
    print(f"Model parameters: {params['total_M']} total, {params['trainable_M']} trainable")

    # Create data loaders
    dataloaders = get_dataloaders(config)

    # Train
    trainer = Trainer(model, config, device, experiment_name=model_name)
    history = trainer.fit(dataloaders["train"], dataloaders["val"])

    # Evaluate on test set with best model
    trainer.load_best_model()
    test_results = trainer.validate(dataloaders["test"])

    metrics = compute_metrics(
        test_results["y_true"],
        test_results["y_pred"],
        test_results["y_prob"],
    )

    # Save metrics
    results_dir = Path(config["paths"]["results"])
    save_metrics(metrics, str(results_dir / f"{model_name}_metrics.json"))

    # Plot ROC curve
    figures_dir = Path(config["paths"]["figures"])
    plot_roc_curve(
        test_results["y_true"],
        test_results["y_prob"],
        model_name,
        str(figures_dir / f"{model_name}_roc.png"),
    )

    # Plot confusion matrix
    import numpy as np
    cm = np.array(metrics["confusion_matrix"])
    plot_confusion_matrix(
        cm, model_name,
        str(figures_dir / f"{model_name}_confusion_matrix.png"),
    )

    print(f"\n[{model_name}] Test Results:")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train baseline CNN models")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=BASELINE_MODELS + ["all"],
        help="Which baseline model to train",
    )
    args = parser.parse_args()

    config, device = setup_environment(args.config)

    if args.model == "all":
        all_results = {}
        for model_name in BASELINE_MODELS:
            metrics = train_single_baseline(model_name, config, device)
            all_results[model_name] = metrics

        # Print summary comparison
        print(f"\n{'='*60}")
        print("Baseline Comparison Summary")
        print(f"{'='*60}")
        print(f"{'Model':>15s} | {'AUC':>7s} | {'F1':>7s} | {'Sens':>7s} | {'Spec':>7s} | {'Acc':>7s}")
        print("-" * 65)
        for name, m in all_results.items():
            print(
                f"{name:>15s} | {m['auc_roc']:>7.4f} | {m['f1_macro']:>7.4f} | "
                f"{m['sensitivity']:>7.4f} | {m['specificity']:>7.4f} | {m['accuracy']:>7.4f}"
            )
    else:
        train_single_baseline(args.model, config, device)


if __name__ == "__main__":
    main()
