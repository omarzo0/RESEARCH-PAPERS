"""
Optuna hyperparameter optimization.
"""

import json
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import torch
from src.dataset import get_dataloaders
from src.train import Trainer


def create_objective(config: dict, device: torch.device, model_name: str):
    """
    Create an Optuna objective function for a given model.

    Args:
        config: Full config dict.
        device: torch.device.
        model_name: Name of the model to optimize.

    Returns:
        callable: Objective function for Optuna.
    """
    optuna_cfg = config.get("optuna", {})
    search_space = optuna_cfg.get("search_space", {})
    search_epochs = optuna_cfg.get("search_epochs", 15)

    def objective(trial):
        # Suggest hyperparameters
        lr_range = search_space.get("lr", [1e-5, 1e-3])
        lr = trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)

        dropout_range = search_space.get("dropout", [0.2, 0.5])
        dropout = trial.suggest_float("dropout", dropout_range[0], dropout_range[1])

        wd_range = search_space.get("weight_decay", [1e-4, 1e-1])
        weight_decay = trial.suggest_float("weight_decay", wd_range[0], wd_range[1], log=True)

        batch_choices = search_space.get("batch_size", [16, 32, 64])
        batch_size = trial.suggest_categorical("batch_size", batch_choices)

        # Update config with trial params
        trial_config = _deep_copy_config(config)
        trial_config["optimizer"]["backbone_lr"] = lr
        trial_config["optimizer"]["head_lr"] = lr * 10  # Head LR is 10x backbone
        trial_config["optimizer"]["weight_decay"] = weight_decay
        trial_config["training"]["batch_size"] = batch_size
        trial_config["training"]["epochs"] = search_epochs
        trial_config["training"]["early_stopping_patience"] = 5  # Shorter for search

        # Update model-specific dropout
        if model_name in ["efficientnet_b3", "efficientnet"]:
            trial_config["models"]["efficientnet"]["dropout"] = dropout
        elif model_name in ["vit_b16", "vit"]:
            trial_config["models"]["vit"]["dropout"] = dropout

        # Create data loaders with trial batch size
        dataloaders = get_dataloaders(trial_config)

        # Create model
        from src.models import get_model
        model = get_model(model_name, trial_config)

        # Train
        trainer = Trainer(
            model, trial_config, device,
            experiment_name=f"optuna_{model_name}_trial{trial.number}",
        )

        # Modified training loop with Optuna reporting
        train_loader = dataloaders["train"]
        val_loader = dataloaders["val"]

        best_auc = 0.0
        for epoch in range(1, search_epochs + 1):
            train_metrics = trainer.train_one_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)
            trainer.scheduler.step()

            val_auc = val_metrics["auc"]
            best_auc = max(best_auc, val_auc)

            # Report to Optuna for pruning
            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_auc

    return objective


def run_hyperopt(
    config: dict,
    device: torch.device,
    model_name: str,
) -> dict:
    """
    Run Optuna hyperparameter search.

    Args:
        config: Full config dict.
        device: torch.device.
        model_name: Name of the model to optimize.

    Returns:
        dict: Best hyperparameters found.
    """
    optuna_cfg = config.get("optuna", {})
    n_trials = optuna_cfg.get("n_trials", 50)

    print(f"\n{'='*60}")
    print(f"Hyperparameter Search: {model_name}")
    print(f"Trials: {n_trials} | Sampler: TPE | Pruner: Median")
    print(f"{'='*60}\n")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.get("seed", 42)),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name=f"{model_name}_hyperopt",
    )

    # Create objective
    objective = create_objective(config, device, model_name)

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
    )

    # Results
    best_params = study.best_params
    best_value = study.best_value

    print(f"\n{'='*60}")
    print(f"Best Trial: #{study.best_trial.number}")
    print(f"Best AUC: {best_value:.4f}")
    print(f"Best Parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    # Save results
    results_dir = Path(config["paths"]["results"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save best params
    best_path = results_dir / f"{model_name}_best_hyperparams.json"
    with open(best_path, "w") as f:
        json.dump(
            {"best_params": best_params, "best_auc": best_value, "n_trials": n_trials},
            f, indent=2,
        )
    print(f"[Optuna] Best params saved → {best_path}")

    # Save all trials
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": str(trial.state),
        })
    trials_path = results_dir / f"{model_name}_optuna_trials.json"
    with open(trials_path, "w") as f:
        json.dump(trials_data, f, indent=2)

    return best_params


def _deep_copy_config(config: dict) -> dict:
    """Deep copy a config dict without importing copy."""
    import json
    return json.loads(json.dumps(config))
