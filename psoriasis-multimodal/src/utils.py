"""
Shared utility classes and helper functions.
"""

import csv
import json
from pathlib import Path

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping to halt training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        mode: 'max' for metrics like AUC (higher is better),
              'min' for metrics like loss (lower is better).
        min_delta: Minimum change to qualify as an improvement.
        checkpoint_path: Path to save the best model checkpoint.
    """

    def __init__(
        self,
        patience: int = 7,
        mode: str = "max",
        min_delta: float = 0.0,
        checkpoint_path: str = "checkpoints/best_model.pt",
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        """
        Check whether training should stop.

        Args:
            score: Current epoch's monitored metric value.
            model: The model to save if this is the best score.
            epoch: Current epoch number.

        Returns:
            bool: True if training should stop.
        """
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            print(
                f"[EarlyStopping] No improvement for {self.patience} epochs. "
                f"Best {self.mode}: {self.best_score:.4f} at epoch {self.best_epoch}."
            )
            return True
        return False


class MetricLogger:
    """
    Logs per-epoch metrics to CSV and optionally to Weights & Biases.

    Args:
        log_dir: Directory to save log files.
        experiment_name: Name of the experiment (used as filename prefix).
        wandb_enabled: Whether to log to W&B.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "experiment",
        wandb_enabled: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        self.wandb_enabled = wandb_enabled
        self.history = []
        self._header_written = False

    def log(self, epoch: int, metrics: dict) -> None:
        """
        Log metrics for one epoch.

        Args:
            epoch: Epoch number.
            metrics: Dict of metric name → value.
        """
        row = {"epoch": epoch, **metrics}
        self.history.append(row)

        # Write to CSV
        fieldnames = list(row.keys())
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

        # Log to W&B
        if self.wandb_enabled:
            try:
                import wandb

                wandb.log(row, step=epoch)
            except ImportError:
                pass

    def save_summary(self, path: str = None) -> None:
        """Save the full training history as JSON."""
        path = path or (self.log_dir / "training_summary.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable and total parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        dict: {'total': int, 'trainable': int, 'frozen': int}
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "total_M": f"{total / 1e6:.2f}M",
        "trainable_M": f"{trainable / 1e6:.2f}M",
    }


def save_checkpoint(
    model: nn.Module,
    optimizer,
    epoch: int,
    metrics: dict,
    path: str,
) -> None:
    """
    Save a full training checkpoint.

    Args:
        model: The model.
        optimizer: The optimizer.
        epoch: Current epoch.
        metrics: Current metrics dict.
        path: Save path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer=None, device="cpu"):
    """
    Load a training checkpoint.

    Args:
        path: Checkpoint file path.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        device: Device to map tensors to.

    Returns:
        dict: Checkpoint metadata (epoch, metrics).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }
