#!/usr/bin/env python3
"""
Generate all publication-quality figures.

Usage:
    python scripts/generate_figures.py --config configs/config.yaml
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import load_config

# ── Publication styling ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "pdf",
})


def generate_class_distribution(config, figures_dir):
    """Generate class distribution bar chart (original + remapped)."""
    splits_dir = Path(config["data"]["splits_dir"])
    dfs = {}
    for split in ["train", "val", "test"]:
        path = splits_dir / f"{split}.csv"
        if path.exists():
            dfs[split] = pd.read_csv(path)

    if not dfs:
        print("[Figures] No split CSVs found. Skipping class distribution.")
        return

    full_df = pd.concat(dfs.values(), ignore_index=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Original severity distribution
    if "severity" in full_df.columns:
        sev_counts = full_df["severity"].value_counts()
        colors_sev = ["#4CAF50", "#FF9800", "#E91E63"]
        sev_counts.reindex(["Mild", "Moderate", "Severe"]).plot.bar(
            ax=ax1, color=colors_sev, edgecolor="white", linewidth=1.5
        )
        ax1.set_title("Original Severity Distribution")
        ax1.set_xlabel("Severity Class")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=0)

        # Add count labels
        for p in ax1.patches:
            ax1.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

    # Binary label distribution
    label_counts = full_df["response_label"].value_counts().sort_index()
    label_names = ["Non-Responder (0)", "Responder (1)"]
    colors_bin = ["#2196F3", "#E91E63"]
    ax2.bar(label_names, label_counts.values, color=colors_bin, edgecolor="white", linewidth=1.5)
    ax2.set_title("Binary Response Label Distribution")
    ax2.set_xlabel("Response Class")
    ax2.set_ylabel("Count")

    for i, v in enumerate(label_counts.values):
        ax2.text(i, v + 5, str(v), ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    save_path = figures_dir / "fig_class_distribution.pdf"
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)
    print(f"[Figures] Saved class distribution → {save_path}")


def generate_training_curves_comparison(config, figures_dir):
    """Generate overlaid training curves for all models."""
    logs_dir = Path(config["paths"]["logs"])
    colors = {"resnet50": "#2196F3", "densenet121": "#4CAF50", "vgg16": "#FF9800",
              "efficientnet_b3": "#E91E63", "vit_b16": "#9C27B0"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for model_name, color in colors.items():
        csv_path = logs_dir / f"{model_name}_metrics.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        if "val_loss" in df.columns:
            ax1.plot(df["epoch"], df["val_loss"], color=color, lw=2, label=model_name)

        if "val_auc" in df.columns:
            ax2.plot(df["epoch"], df["val_auc"], color=color, lw=2, label=model_name)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Validation Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation AUC-ROC")
    ax2.set_title("Validation AUC Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = figures_dir / "fig_training_curves.pdf"
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)
    print(f"[Figures] Saved training curves → {save_path}")


def generate_confusion_matrix_grid(config, figures_dir):
    """Generate 2x3 grid of confusion matrices for all models."""
    results_dir = Path(config["paths"]["results"])
    models = ["resnet50", "densenet121", "vgg16", "efficientnet_b3", "vit_b16"]
    class_names = ["Non-Resp", "Responder"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, model_name in enumerate(models):
        metrics_path = results_dir / f"{model_name}_test_metrics.json"
        if not metrics_path.exists():
            metrics_path = results_dir / f"{model_name}_metrics.json"
        if not metrics_path.exists():
            axes[i].text(0.5, 0.5, f"{model_name}\n(no data)", ha="center", va="center")
            axes[i].set_title(model_name)
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[i], square=True, linewidths=0.5,
        )
        auc = metrics.get("auc_roc", 0)
        axes[i].set_title(f"{model_name}\n(AUC: {auc:.3f})", fontweight="bold")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")

    # Hide unused subplot
    if len(models) < 6:
        axes[-1].axis("off")

    plt.suptitle("Confusion Matrices — All Models", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path = figures_dir / "fig_confusion_matrices.pdf"
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"))
    plt.close(fig)
    print(f"[Figures] Saved confusion matrix grid → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    figures_dir = Path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Generating Publication Figures")
    print(f"Output: {figures_dir}")
    print(f"{'='*60}\n")

    generate_class_distribution(config, figures_dir)
    generate_training_curves_comparison(config, figures_dir)
    generate_confusion_matrix_grid(config, figures_dir)

    print(f"\n[Figures] All figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
