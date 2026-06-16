#!/usr/bin/env python3
"""
Generate a 4x4 publication-quality Grad-CAM++ visualization grid.
Replicates the template layout:
Columns: Original (Idx X) | Grad-CAM (Class 0) | Original (Idx Y) | Grad-CAM (Class 1)
Rows: 4 different representative pairs.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import setup_environment
from src.models import get_model
from src.explain import GradCAMPlusPlusExplainer


def main():
    config_path = "configs/config.yaml"
    config, device = setup_environment(config_path)

    # 1. Load the trained primary EfficientNet-B3 model
    model = get_model("efficientnet_b3", config)
    ckpt_path = Path(config["paths"]["checkpoints"]) / "efficientnet_b3" / "best_model.pt"
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Trained checkpoint not found at {ckpt_path}. Please run training first.")
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"[XAI Grid] Loaded trained model from {ckpt_path}")

    # 2. Get the target layer for Grad-CAM
    target_layer = model.get_feature_layer()
    explainer = GradCAMPlusPlusExplainer(model, target_layer, device)

    # 3. Load the test split and filter by classes
    test_csv = Path(config["data"]["splits_dir"]) / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split CSV not found at {test_csv}")
    
    test_df = pd.read_csv(test_csv)
    
    # Select 4 representative samples for Class 0 (Non-Responder) and 4 for Class 1 (Responder)
    # We will pick diverse samples by spacing out our selections
    class0_indices = test_df[test_df["response_label"] == 0].index.tolist()
    class1_indices = test_df[test_df["response_label"] == 1].index.tolist()
    
    # Pick 4 spread-out indices from each class to ensure diversity
    selected_c0 = [class0_indices[i] for i in np.linspace(0, len(class0_indices) - 1, 4, dtype=int)]
    selected_c1 = [class1_indices[i] for i in np.linspace(0, len(class1_indices) - 1, 4, dtype=int)]
    
    print(f"[XAI Grid] Selected Non-Responder (Class 0) indices: {selected_c0}")
    print(f"[XAI Grid] Selected Responder (Class 1) indices: {selected_c1}")

    # 4. Set up the plotting grid
    fig, axes = plt.subplots(4, 4, figsize=(12, 12), dpi=300)
    
    # Formatting parameters
    title_font = {"fontsize": 10, "fontweight": "normal", "fontfamily": "sans-serif"}
    
    # Process each row
    for row_idx in range(4):
        # Sample for left side (Class 0: Non-Responder)
        idx_c0 = selected_c0[row_idx]
        row_c0 = test_df.loc[idx_c0]
        img_path_c0 = row_c0["image_path"]
        
        # Sample for right side (Class 1: Responder)
        idx_c1 = selected_c1[row_idx]
        row_c1 = test_df.loc[idx_c1]
        img_path_c1 = row_c1["image_path"]
        
        # Helper to process and plot a sample pair
        def plot_pair(img_path, idx, class_label, col_orig, col_cam):
            # Load original image
            orig = cv2.imread(img_path)
            if orig is None:
                raise FileNotFoundError(f"Cannot read image: {img_path}")
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            
            # Preprocess tensor for model input (normalize as in validation/test)
            # Resize to (224, 224)
            orig_resized = cv2.resize(orig, (224, 224))
            
            # Normalize for network input (ImageNet norm)
            mean = np.array(config["data"]["imagenet_mean"])
            std = np.array(config["data"]["imagenet_std"])
            img_tensor = orig_resized.astype(np.float32) / 255.0
            img_tensor = (img_tensor - mean) / std
            img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).float()
            
            # Explain (Grad-CAM++)
            heatmap = explainer.explain(img_tensor, target_class=class_label)
            
            # Overlay heatmap on original resized image
            overlay = explainer.overlay(orig_resized, heatmap, alpha=0.4)
            
            # Plot Original
            ax_orig = axes[row_idx, col_orig]
            ax_orig.imshow(orig_resized)
            ax_orig.set_title(f"Original (Idx {idx})", **title_font)
            ax_orig.axis("off")
            
            # Plot Grad-CAM
            ax_cam = axes[row_idx, col_cam]
            ax_cam.imshow(overlay)
            ax_cam.set_title(f"Grad-CAM (Class {class_label})", **title_font)
            ax_cam.axis("off")

        # Plot Class 0 pair in columns 0 and 1
        plot_pair(img_path_c0, idx_c0, 0, col_orig=0, col_cam=1)
        
        # Plot Class 1 pair in columns 2 and 3
        plot_pair(img_path_c1, idx_c1, 1, col_orig=2, col_cam=3)

    # Tighten and save
    plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=1.5)
    
    figures_dir = Path(config["paths"]["figures"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_path = figures_dir / "efficientnet_b3_gradcam_grid.png"
    
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[XAI Grid] Saved premium 4x4 Grad-CAM grid to {save_path}")


if __name__ == "__main__":
    main()
