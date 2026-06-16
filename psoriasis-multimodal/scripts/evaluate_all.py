#!/usr/bin/env python3
"""
Comprehensive evaluation: compare all models on the test set.

Usage:
    python scripts/evaluate_all.py --config configs/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.config import setup_environment
from src.dataset import get_dataloaders
from src.evaluate import (
    compute_metrics,
    compare_models,
    mcnemar_test,
    plot_confusion_matrix,
    plot_roc_curves_comparison,
    save_metrics,
)
from src.models import get_model


ALL_MODELS = ["resnet50", "densenet121", "vgg16", "efficientnet_b3", "vit_b16"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate all trained models")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Specific models to evaluate (default: all with checkpoints)",
    )
    args = parser.parse_args()

    config, device = setup_environment(args.config)
    dataloaders = get_dataloaders(config)
    test_loader = dataloaders["test"]

    models_to_eval = args.models or ALL_MODELS
    results_dir = Path(config["paths"]["results"])
    figures_dir = Path(config["paths"]["figures"])

    all_metrics = {}
    all_predictions = {}  # for ROC comparison and McNemar test

    for model_name in models_to_eval:
        ckpt_path = Path(config["paths"]["checkpoints"]) / model_name / "best_model.pt"

        if not ckpt_path.exists():
            print(f"[Eval] Skipping {model_name} — no checkpoint at {ckpt_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")

        # Load model
        model = get_model(model_name, config)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model = model.to(device)
        model.eval()

        # Run inference on test set
        all_probs = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels, paths in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels_np)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_prob)
        all_metrics[model_name] = metrics
        all_predictions[model_name] = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        # Save individual metrics
        save_metrics(metrics, str(results_dir / f"{model_name}_test_metrics.json"))

        # Plot confusion matrix
        cm = np.array(metrics["confusion_matrix"])
        plot_confusion_matrix(
            cm, model_name,
            str(figures_dir / f"{model_name}_test_confusion_matrix.png"),
        )

        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
        print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")

    if len(all_metrics) < 2:
        print("\n[Eval] Not enough models for comparison. Done.")
        return

    # ── Comparison figures ──────────────────────────────────────────────────

    # ROC curves comparison
    roc_data = {
        name: {"y_true": pred["y_true"], "y_prob": pred["y_prob"]}
        for name, pred in all_predictions.items()
    }
    plot_roc_curves_comparison(roc_data, str(figures_dir / "all_models_roc_comparison.png"))

    # Grouped bar chart
    compare_models(all_metrics, str(figures_dir / "all_models_comparison_bar.png"))

    # ── McNemar's test (EfficientNet vs each baseline) ──────────────────────

    primary_model = "efficientnet_b3"
    if primary_model in all_predictions:
        print(f"\n{'='*50}")
        print(f"McNemar's Test: {primary_model} vs Baselines")
        print(f"{'='*50}")

        mcnemar_results = {}
        for baseline in ["resnet50", "densenet121", "vgg16"]:
            if baseline in all_predictions:
                result = mcnemar_test(
                    all_predictions[primary_model]["y_pred"],
                    all_predictions[baseline]["y_pred"],
                    all_predictions[primary_model]["y_true"],
                )
                mcnemar_results[f"{primary_model}_vs_{baseline}"] = result
                sig = "✓ Significant" if result["significant"] else "✗ Not significant"
                print(
                    f"  {primary_model} vs {baseline}: "
                    f"χ²={result['statistic']:.3f}, p={result['p_value']:.4f} — {sig}"
                )

        save_metrics(mcnemar_results, str(results_dir / "mcnemar_test_results.json"))

    # ── Summary table ───────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("Final Model Comparison — Test Set")
    print(f"{'='*80}")
    print(f"{'Model':>15s} | {'AUC-ROC':>8s} | {'F1':>8s} | {'Sens':>8s} | {'Spec':>8s} | {'Acc':>8s} | {'AUPRC':>8s}")
    print("-" * 80)
    for name in ALL_MODELS:
        if name in all_metrics:
            m = all_metrics[name]
            print(
                f"{name:>15s} | {m['auc_roc']:>8.4f} | {m['f1_macro']:>8.4f} | "
                f"{m['sensitivity']:>8.4f} | {m['specificity']:>8.4f} | "
                f"{m['accuracy']:>8.4f} | {m['auprc']:>8.4f}"
            )
    print(f"{'='*80}\n")

    # Save comparison as CSV
    comparison_rows = []
    for name in ALL_MODELS:
        if name in all_metrics:
            m = all_metrics[name]
            comparison_rows.append({
                "Model": name,
                "AUC-ROC": m["auc_roc"],
                "F1 (Macro)": m["f1_macro"],
                "Sensitivity": m["sensitivity"],
                "Specificity": m["specificity"],
                "Accuracy": m["accuracy"],
                "AUPRC": m["auprc"],
            })

    comparison_df = pd.DataFrame(comparison_rows)
    csv_path = results_dir / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"[Eval] Comparison table saved → {csv_path}")

    # Save as LaTeX table
    latex_path = results_dir / "model_comparison.tex"
    comparison_df.to_latex(latex_path, index=False, float_format="%.4f")
    print(f"[Eval] LaTeX table saved → {latex_path}")


if __name__ == "__main__":
    main()
