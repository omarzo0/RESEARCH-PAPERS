"""
Unified training loop for all model architectures.

Supports: mixed precision, differential LR, cosine annealing,
early stopping, W&B/CSV logging.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.utils import EarlyStopping, MetricLogger, count_parameters


class Trainer:
    """
    Unified model trainer.

    Args:
        model: PyTorch model.
        config: Full config dict.
        device: torch.device.
        experiment_name: Name for logging and checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        experiment_name: str = "experiment",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.experiment_name = experiment_name

        train_cfg = config["training"]
        opt_cfg = config["optimizer"]
        sched_cfg = config["scheduler"]

        # Print model info
        params = count_parameters(model)
        print(f"[Trainer] Model: {experiment_name}")
        print(f"[Trainer] Parameters: {params['total_M']} total, {params['trainable_M']} trainable")

        # Loss function with label smoothing
        self.label_smoothing = train_cfg.get("label_smoothing", 0.1)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        self.criterion_soft = nn.KLDivLoss(reduction="batchmean")

        # Optimizer with differential learning rates
        self.optimizer = setup_optimizer(model, config)

        # Scheduler
        self.scheduler = setup_scheduler(self.optimizer, config)

        # Mixed precision
        self.use_amp = train_cfg.get("mixed_precision", True)
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp and device.type == "cuda")
        self.amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

        # Gradient clipping
        self.grad_clip = train_cfg.get("gradient_clip_max_norm", 1.0)

        # Early stopping
        checkpoint_dir = Path(config["paths"]["checkpoints"]) / experiment_name
        self.early_stopping = EarlyStopping(
            patience=train_cfg.get("early_stopping_patience", 7),
            mode="max",
            checkpoint_path=str(checkpoint_dir / "best_model.pt"),
        )

        # Metric logger
        wandb_cfg = config.get("wandb", {})
        self.logger = MetricLogger(
            log_dir=config["paths"]["logs"],
            experiment_name=experiment_name,
            wandb_enabled=wandb_cfg.get("enabled", False),
        )

        # Training history
        self.history = []

    def train_one_epoch(self, train_loader) -> dict:
        """
        Train for one epoch.

        Args:
            train_loader: Training DataLoader.

        Returns:
            dict: {'loss': float, 'auc': float}
        """
        self.model.train()
        running_loss = 0.0
        all_probs = []
        all_labels = []
        num_batches = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch in pbar:
            images, labels, _ = batch
            images = images.to(self.device)

            # Handle both hard labels (LongTensor) and soft labels (FloatTensor from CutMix/MixUp)
            is_soft = labels.dim() == 2  # one-hot / mixed labels
            if is_soft:
                labels_soft = labels.to(self.device)
                labels_hard = labels_soft.argmax(dim=1)
            else:
                labels_hard = labels.to(self.device)
                labels_soft = None

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits = self.model(images)

                if is_soft and labels_soft is not None:
                    # Soft label loss (KL divergence)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                    loss = self.criterion_soft(log_probs, labels_soft)
                else:
                    loss = self.criterion(logits, labels_hard)

            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate metrics
            running_loss += loss.item()
            num_batches += 1

            probs = torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels_hard.cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Epoch metrics
        avg_loss = running_loss / max(num_batches, 1)
        try:
            train_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            train_auc = 0.0

        return {"loss": avg_loss, "auc": train_auc}

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        """
        Evaluate on validation/test set.

        Args:
            val_loader: Validation DataLoader.

        Returns:
            dict: {'loss': float, 'auc': float, 'y_true': array, 'y_pred': array, 'y_prob': array}
        """
        self.model.eval()
        running_loss = 0.0
        all_probs = []
        all_preds = []
        all_labels = []
        num_batches = 0

        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            running_loss += loss.item()
            num_batches += 1

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / max(num_batches, 1)
        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = np.array(all_preds)

        try:
            val_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            val_auc = 0.0

        return {
            "loss": avg_loss,
            "auc": val_auc,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = None,
    ) -> list:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Number of epochs (overrides config if provided).

        Returns:
            list: Training history (list of per-epoch metric dicts).
        """
        if epochs is None:
            epochs = self.config["training"].get("epochs", 50)

        print(f"\n{'='*60}")
        print(f"Training: {self.experiment_name}")
        print(f"Epochs: {epochs} | Device: {self.device}")
        print(f"{'='*60}\n")

        best_auc = 0.0
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_one_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Step scheduler
            self.scheduler.step()

            # Log
            epoch_log = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_auc": train_metrics["auc"],
                "val_loss": val_metrics["loss"],
                "val_auc": val_metrics["auc"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "time_s": time.time() - epoch_start,
            }
            self.history.append(epoch_log)
            self.logger.log(epoch, epoch_log)

            # Print epoch summary
            print(
                f"Epoch {epoch:3d}/{epochs} │ "
                f"Train Loss: {train_metrics['loss']:.4f} │ "
                f"Train AUC: {train_metrics['auc']:.4f} │ "
                f"Val Loss: {val_metrics['loss']:.4f} │ "
                f"Val AUC: {val_metrics['auc']:.4f} │ "
                f"LR: {epoch_log['lr']:.2e} │ "
                f"Time: {epoch_log['time_s']:.1f}s"
            )

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                print(f"  ★ New best AUC: {best_auc:.4f}")

            # Early stopping
            should_stop = self.early_stopping(
                val_metrics["auc"], self.model, epoch
            )
            if should_stop:
                print(f"\n[Early Stopping] Stopped at epoch {epoch}.")
                break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete in {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"Best Val AUC: {self.early_stopping.best_score:.4f} "
              f"(epoch {self.early_stopping.best_epoch})")
        print(f"Best model saved to: {self.early_stopping.checkpoint_path}")
        print(f"{'='*60}\n")

        # Save training history
        self.logger.save_summary()

        return self.history

    def load_best_model(self) -> None:
        """Load the best checkpoint saved during training."""
        self.model.load_state_dict(
            torch.load(
                self.early_stopping.checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        )
        print(f"[Trainer] Loaded best model from {self.early_stopping.checkpoint_path}")


def setup_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Create AdamW optimizer with differential learning rates.

    Backbone layers get a lower LR, classification head gets a higher LR.

    Args:
        model: The model.
        config: Full config dict.

    Returns:
        torch.optim.AdamW: Configured optimizer.
    """
    opt_cfg = config["optimizer"]
    backbone_lr = opt_cfg.get("backbone_lr", 1e-4)
    head_lr = opt_cfg.get("head_lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.01)

    # Split parameters into backbone and head groups
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name or "classifier" in name or "fc" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    print(
        f"[Optimizer] AdamW — backbone LR: {backbone_lr}, "
        f"head LR: {head_lr}, weight decay: {weight_decay}"
    )
    return optimizer


def setup_scheduler(optimizer, config: dict):
    """
    Create cosine annealing with warm restarts scheduler.

    Args:
        optimizer: The optimizer.
        config: Full config dict.

    Returns:
        torch.optim.lr_scheduler: Configured scheduler.
    """
    sched_cfg = config["scheduler"]
    T_0 = sched_cfg.get("T_0", 10)
    T_mult = sched_cfg.get("T_mult", 1)
    eta_min = sched_cfg.get("eta_min", 1e-6)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )
    print(f"[Scheduler] CosineAnnealingWarmRestarts — T_0={T_0}, T_mult={T_mult}")
    return scheduler
