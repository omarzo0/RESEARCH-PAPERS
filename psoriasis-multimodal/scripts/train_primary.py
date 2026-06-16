#!/usr/bin/env python3
"""
Train primary models (EfficientNet-B3 and ViT-B/16).

Supports: training, hyperparameter search, and cross-validation.

Usage:
    # Hyperparameter search
    python scripts/train_primary.py --config configs/config.yaml --model efficientnet_b3 --mode hyperopt

    # Full training with best/default params
    python scripts/train_primary.py --config configs/config.yaml --model efficientnet_b3 --mode train

    # Cross-validation
    python scripts/train_primary.py --config configs/config.yaml --model efficientnet_b3 --mode cv

    # ViT training
    python scripts/train_primary.py --config configs/config.yaml --model vit_b16 --mode train
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import setup_environment
from src.dataset import get_dataloaders
from src.evaluate import compute_metrics, plot_roc_curve, plot_confusion_matrix, save_metrics, plot_training_history
from src.models import get_model
from src.train import Trainer
from src.utils import count_parameters


def run_training(model_name: str, config: dict, device):
    """Full training run."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}\n")

    # Check for best hyperparams from Optuna
    best_params_path = Path(config["paths"]["results"]) / f"{model_name}_best_hyperparams.json"
    if best_params_path.exists():
        print(f"[Train] Loading best hyperparams from {best_params_path}")
        with open(best_params_path) as f:
            best = json.load(f)["best_params"]
        # Apply best params
        config["optimizer"]["backbone_lr"] = best.get("learning_rate", config["optimizer"]["backbone_lr"])
        config["optimizer"]["head_lr"] = best.get("learning_rate", config["optimizer"]["head_lr"]) * 10
        config["optimizer"]["weight_decay"] = best.get("weight_decay", config["optimizer"]["weight_decay"])
        config["training"]["batch_size"] = best.get("batch_size", config["training"]["batch_size"])
        if "efficientnet" in model_name:
            config["models"]["efficientnet"]["dropout"] = best.get("dropout", config["models"]["efficientnet"]["dropout"])
        elif "vit" in model_name:
            config["models"]["vit"]["dropout"] = best.get("dropout", config["models"]["vit"]["dropout"])
        print(f"[Train] Applied: {best}")

    # Create model
    model = get_model(model_name, config)
    params = count_parameters(model)
    print(f"Parameters: {params['total_M']} total, {params['trainable_M']} trainable")

    # Data loaders
    dataloaders = get_dataloaders(config)

    # Train
    trainer = Trainer(model, config, device, experiment_name=model_name)
    history = trainer.fit(dataloaders["train"], dataloaders["val"])

    # Evaluate on test set
    trainer.load_best_model()
    test_results = trainer.validate(dataloaders["test"])
    metrics = compute_metrics(
        test_results["y_true"], test_results["y_pred"], test_results["y_prob"]
    )

    # Save everything
    results_dir = Path(config["paths"]["results"])
    figures_dir = Path(config["paths"]["figures"])

    save_metrics(metrics, str(results_dir / f"{model_name}_metrics.json"))
    plot_roc_curve(
        test_results["y_true"], test_results["y_prob"], model_name,
        str(figures_dir / f"{model_name}_roc.png"),
    )

    import numpy as np
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]), model_name,
        str(figures_dir / f"{model_name}_confusion_matrix.png"),
    )
    plot_training_history(history, str(figures_dir / f"{model_name}_training_history.png"))

    print(f"\n[{model_name}] Test Results:")
    print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")

    return metrics


def run_hyperopt(model_name: str, config: dict, device):
    """Run Optuna hyperparameter search."""
    from src.hyperopt import run_hyperopt as _run_hyperopt
    best_params = _run_hyperopt(config, device, model_name)
    return best_params


def run_cv(model_name: str, config: dict, device):
    """Run 5-fold cross-validation."""
    from src.cross_validation import run_cross_validation

    def model_factory():
        return get_model(model_name, config)

    results = run_cross_validation(model_factory, config, device, model_name)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train primary models")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["efficientnet_b3", "vit_b16"],
        help="Which primary model to train",
    )
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "hyperopt", "cv"],
        help="Training mode: train, hyperopt (Optuna search), or cv (cross-validation)",
    )
    args = parser.parse_args()

    config, device = setup_environment(args.config)

    if args.mode == "hyperopt":
        run_hyperopt(args.model, config, device)
    elif args.mode == "cv":
        run_cv(args.model, config, device)
    else:
        run_training(args.model, config, device)


if __name__ == "__main__":
    main()
