#!/usr/bin/env python3
"""
Generate XAI visualizations: Grad-CAM++, LIME, and Attention Rollout.

Usage:
    python scripts/run_xai.py --config configs/config.yaml --model efficientnet_b3 --method gradcam
    python scripts/run_xai.py --config configs/config.yaml --model efficientnet_b3 --method lime
    python scripts/run_xai.py --config configs/config.yaml --model vit_b16 --method attention_rollout
    python scripts/run_xai.py --config configs/config.yaml --model efficientnet_b3 --method all
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.config import setup_environment
from src.dataset import get_dataloaders
from src.evaluate import compute_metrics
from src.models import get_model
from src.explain import (
    GradCAMPlusPlusExplainer,
    LIMEExplainer,
    AttentionRolloutExplainer,
    generate_gradcam_grid,
    generate_lime_comparison,
)


def run_gradcam(model, dataloaders, config, device, model_name):
    """Generate Grad-CAM++ heatmaps."""
    figures_dir = Path(config["paths"]["figures"])

    # Get target layer for Grad-CAM
    if hasattr(model, "get_feature_layer"):
        target_layer = model.get_feature_layer()
    else:
        # Fallback: try to find the last conv layer
        target_layer = None
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                break
        if target_layer is None:
            print("[XAI] ERROR: Cannot determine target layer for Grad-CAM")
            return

    explainer = GradCAMPlusPlusExplainer(model, target_layer, device)

    # Generate heatmaps for test set
    results = explainer.batch_explain(
        dataloaders["test"],
        n_samples=30,
        save_dir=str(figures_dir / f"{model_name}_gradcam_individual"),
    )

    # Build severity label mapping for the grid
    test_csv = Path(config["data"]["splits_dir"]) / "test.csv"
    if test_csv.exists():
        test_df = pd.read_csv(test_csv)
        severity_labels = dict(zip(test_df["image_path"], test_df.get("severity", [])))
    else:
        severity_labels = {}

    # Generate 3x3 grid figure
    generate_gradcam_grid(
        results,
        str(figures_dir / f"{model_name}_gradcam_grid.png"),
        severity_labels=severity_labels,
    )

    print(f"[XAI] Grad-CAM++ complete. {len(results)} heatmaps generated.")


def run_lime(model, dataloaders, config, device, model_name):
    """Generate LIME explanations."""
    figures_dir = Path(config["paths"]["figures"])

    # Get test predictions to find correct and incorrect examples
    model.eval()
    correct_example = None
    incorrect_example = None

    for images, labels, paths in dataloaders["test"]:
        images_dev = images.to(device)
        with torch.no_grad():
            logits = model(images_dev)
            preds = logits.argmax(dim=1).cpu().numpy()

        labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels)

        for i in range(len(images)):
            import cv2
            orig = cv2.imread(paths[i])
            if orig is None:
                continue
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            orig = cv2.resize(orig, (224, 224))

            example = {
                "original": orig,
                "path": paths[i],
                "label": int(labels_np[i]),
                "prediction": int(preds[i]),
            }

            if preds[i] == labels_np[i] and correct_example is None:
                correct_example = example
            elif preds[i] != labels_np[i] and incorrect_example is None:
                incorrect_example = example

            if correct_example and incorrect_example:
                break
        if correct_example and incorrect_example:
            break

    # Generate comparison figure
    if correct_example or incorrect_example:
        generate_lime_comparison(
            model, device,
            correct_example, incorrect_example,
            str(figures_dir / f"{model_name}_lime_comparison.png"),
        )

    print("[XAI] LIME explanations complete.")


def run_attention_rollout(model, dataloaders, config, device, model_name):
    """Generate attention rollout maps for ViT."""
    figures_dir = Path(config["paths"]["figures"])

    if not hasattr(model, "get_attention_maps"):
        print("[XAI] ERROR: Model does not support attention rollout (not a ViT).")
        return

    explainer = AttentionRolloutExplainer(model, device)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Generate attention maps for a few test images
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    col_titles = ["Original", "Attention Rollout", "Overlay"]
    count = 0

    for images, labels, paths in dataloaders["test"]:
        for i in range(images.shape[0]):
            if count >= 3:
                break

            img_tensor = images[i]

            # Original image
            import cv2
            orig = cv2.imread(paths[i])
            if orig is None:
                continue
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            orig = cv2.resize(orig, (224, 224))

            # Attention rollout
            attention_map = explainer.rollout(img_tensor)
            overlay = explainer.visualize(attention_map, orig)

            axes[count, 0].imshow(orig)
            axes[count, 1].imshow(attention_map, cmap="jet")
            axes[count, 2].imshow(overlay)

            for col in range(3):
                axes[count, col].axis("off")
                if count == 0:
                    axes[count, col].set_title(col_titles[col], fontsize=13, fontweight="bold")

            count += 1
        if count >= 3:
            break

    plt.suptitle("ViT-B/16 Attention Rollout Maps", fontsize=15, fontweight="bold")
    plt.tight_layout()
    save_path = str(figures_dir / f"{model_name}_attention_rollout.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[XAI] Saved attention rollout → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate XAI visualizations")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["efficientnet_b3", "vit_b16", "resnet50", "densenet121", "vgg16"],
        help="Model to explain",
    )
    parser.add_argument(
        "--method", type=str, default="all",
        choices=["gradcam", "lime", "attention_rollout", "all"],
        help="XAI method to apply",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (default: auto-detect from checkpoints dir)",
    )
    args = parser.parse_args()

    config, device = setup_environment(args.config)

    # Load model
    model = get_model(args.model, config)

    # Load checkpoint
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = Path(config["paths"]["checkpoints"]) / args.model / "best_model.pt"

    if Path(ckpt_path).exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"[XAI] Loaded checkpoint: {ckpt_path}")
    else:
        print(f"[XAI] WARNING: No checkpoint found at {ckpt_path}. Using random weights.")

    model = model.to(device)
    model.eval()

    # Data loaders
    dataloaders = get_dataloaders(config)

    # Run requested method(s)
    if args.method in ("gradcam", "all"):
        if "vit" not in args.model:  # Grad-CAM is for CNNs
            run_gradcam(model, dataloaders, config, device, args.model)
        else:
            print("[XAI] Skipping Grad-CAM++ for ViT (use attention_rollout instead)")

    if args.method in ("lime", "all"):
        run_lime(model, dataloaders, config, device, args.model)

    if args.method in ("attention_rollout", "all"):
        if "vit" in args.model:
            run_attention_rollout(model, dataloaders, config, device, args.model)
        else:
            print("[XAI] Skipping attention rollout for CNN model")


if __name__ == "__main__":
    main()
