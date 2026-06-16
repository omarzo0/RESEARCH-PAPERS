"""
Evaluation metrics, plotting, and statistical testing.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script usage
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Publication-quality styling ─────────────────────────────────────────────
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
})

CLASS_NAMES = ["Non-Responder", "Responder"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute full evaluation metrics.

    Args:
        y_true: Ground truth labels (N,).
        y_pred: Predicted labels (N,).
        y_prob: Predicted probabilities for the positive class (N,).

    Returns:
        dict: All metrics including AUC, F1, precision, recall, specificity,
              accuracy, confusion matrix, and classification report.
    """
    # Core metrics
    auc_roc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    f1_per_class = f1_score(y_true, y_pred, average=None).tolist()
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")  # sensitivity
    accuracy = accuracy_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Sensitivity (recall for positive class)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )

    return {
        "auc_roc": float(auc_roc),
        "auprc": float(auprc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": f1_per_class,
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": report,
    }


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: str,
) -> None:
    """
    Plot ROC curve with AUC annotation.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities for the positive class.
        model_name: Model name for the title.
        save_path: Path to save the figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#9E9E9E", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#2196F3")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, format=save_path.split(".")[-1])
    plt.close(fig)
    print(f"[Plot] Saved ROC curve → {save_path}")


def plot_roc_curves_comparison(
    results: dict,
    save_path: str,
) -> None:
    """
    Plot overlaid ROC curves for multiple models.

    Args:
        results: Dict of model_name → {y_true, y_prob}.
        save_path: Path to save the figure.
    """
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(8, 7))

    for i, (name, data) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#9E9E9E", lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot] Saved comparison ROC curves → {save_path}")


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    save_path: str,
    class_names: list = None,
) -> None:
    """
    Plot annotated confusion matrix heatmap.

    Args:
        cm: Confusion matrix (2×2 array).
        model_name: Model name for the title.
        save_path: Path to save the figure.
        class_names: Class names for axis labels.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot] Saved confusion matrix → {save_path}")


def plot_training_history(
    history: list,
    save_path: str,
) -> None:
    """
    Plot training loss and AUC curves over epochs.

    Args:
        history: List of dicts with keys 'epoch', 'train_loss', 'val_loss',
                 'train_auc', 'val_auc'.
        save_path: Path to save the figure.
    """
    epochs = [h["epoch"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    if "train_loss" in history[0]:
        ax1.plot(epochs, [h["train_loss"] for h in history],
                 color="#2196F3", label="Train Loss", lw=2)
    if "val_loss" in history[0]:
        ax1.plot(epochs, [h["val_loss"] for h in history],
                 color="#E91E63", label="Val Loss", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # AUC curves
    if "train_auc" in history[0]:
        ax2.plot(epochs, [h.get("train_auc", 0) for h in history],
                 color="#2196F3", label="Train AUC", lw=2)
    if "val_auc" in history[0]:
        ax2.plot(epochs, [h.get("val_auc", 0) for h in history],
                 color="#E91E63", label="Val AUC", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC-ROC")
    ax2.set_title("Training & Validation AUC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot] Saved training history → {save_path}")


def compare_models(
    results: dict,
    save_path: str,
) -> None:
    """
    Grouped bar chart comparing metrics across models.

    Args:
        results: Dict of model_name → metrics dict.
        save_path: Path to save the figure.
    """
    metrics_to_plot = ["auc_roc", "f1_macro", "sensitivity", "specificity", "accuracy"]
    metric_labels = ["AUC-ROC", "F1 (Macro)", "Sensitivity", "Specificity", "Accuracy"]
    model_names = list(results.keys())

    x = np.arange(len(metric_labels))
    width = 0.8 / len(model_names)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, name in enumerate(model_names):
        values = [results[name].get(m, 0) for m in metrics_to_plot]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name,
                      color=colors[i % len(colors)], alpha=0.85)
        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc="lower right")
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"[Plot] Saved model comparison → {save_path}")


def mcnemar_test(
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """
    McNemar's test for comparing two classifiers.

    Tests whether the two models have significantly different error rates.

    Args:
        y_pred_a: Predictions from model A.
        y_pred_b: Predictions from model B.
        y_true: Ground truth labels.

    Returns:
        dict: {statistic, p_value, significant (at alpha=0.05)}
    """
    # Build contingency table
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # b = A correct, B wrong; c = A wrong, B correct
    b = np.sum(correct_a & ~correct_b)  # A right, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B right

    # McNemar's test with continuity correction
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    statistic = (abs(b - c) - 1) ** 2 / (b + c)

    # Chi-squared distribution with 1 df
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "b_a_right_b_wrong": int(b),
        "c_a_wrong_b_right": int(c),
    }


def save_metrics(metrics: dict, save_path: str) -> None:
    """Save metrics dict as JSON."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[Metrics] Saved → {save_path}")
