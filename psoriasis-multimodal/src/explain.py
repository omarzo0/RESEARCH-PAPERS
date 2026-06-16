"""
Explainability (XAI) pipeline: Grad-CAM++, LIME, and Attention Rollout.
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ── Grad-CAM++ ──────────────────────────────────────────────────────────────

class GradCAMPlusPlusExplainer:
    """
    Grad-CAM++ heatmap generator for CNN models.

    Uses the pytorch-grad-cam library for robust Grad-CAM++ computation.

    Args:
        model: Trained CNN model.
        target_layer: The convolutional layer to compute Grad-CAM++ on.
        device: torch.device.
    """

    def __init__(self, model, target_layer, device=None):
        self.model = model
        self.target_layer = target_layer
        self.device = device or next(model.parameters()).device

        # Use pytorch-grad-cam library
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        self.cam = GradCAMPlusPlus(
            model=self.model,
            target_layers=[self.target_layer],
        )
        self.ClassifierOutputTarget = ClassifierOutputTarget

    def explain(self, image_tensor, target_class=None):
        """
        Generate Grad-CAM++ heatmap for a single image.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, H, W) or (3, H, W).
            target_class: Target class index. If None, uses the predicted class.

        Returns:
            np.ndarray: Heatmap of shape (H, W) with values in [0, 1].
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        if target_class is not None:
            targets = [self.ClassifierOutputTarget(target_class)]
        else:
            targets = None  # Uses highest scoring class

        heatmap = self.cam(input_tensor=image_tensor, targets=targets)
        return heatmap[0]  # (H, W)

    def overlay(self, original_image, heatmap, alpha=0.4):
        """
        Overlay heatmap on original image.

        Args:
            original_image: RGB numpy array (H, W, 3) in [0, 255].
            heatmap: Heatmap array (H, W) in [0, 1].
            alpha: Blending factor.

        Returns:
            np.ndarray: Blended image (H, W, 3) in [0, 255].
        """
        from pytorch_grad_cam.utils.image import show_cam_on_image

        # Normalize original image to [0, 1]
        if original_image.max() > 1.0:
            original_image = original_image.astype(np.float32) / 255.0

        visualization = show_cam_on_image(
            original_image, heatmap, use_rgb=True, image_weight=1 - alpha
        )
        return visualization

    def batch_explain(self, dataloader, n_samples=50, save_dir=None):
        """
        Generate Grad-CAM++ heatmaps for N test images.

        Args:
            dataloader: Test DataLoader.
            n_samples: Number of samples to explain.
            save_dir: Directory to save individual heatmap images.

        Returns:
            list: List of (original, heatmap, overlay, label, pred) tuples.
        """
        results = []
        count = 0

        self.model.eval()
        for images, labels, paths in dataloader:
            for i in range(images.shape[0]):
                if count >= n_samples:
                    return results

                img_tensor = images[i]
                label = labels[i].item() if isinstance(labels[i], torch.Tensor) else labels[i]

                # Get prediction
                with torch.no_grad():
                    logits = self.model(img_tensor.unsqueeze(0).to(self.device))
                    pred = logits.argmax(dim=1).item()

                # Generate heatmap
                heatmap = self.explain(img_tensor)

                # Load original image for overlay
                orig = cv2.imread(paths[i])
                if orig is not None:
                    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                    orig_resized = cv2.resize(orig, (224, 224))
                    overlay = self.overlay(orig_resized, heatmap)
                else:
                    orig_resized = None
                    overlay = None

                results.append({
                    "original": orig_resized,
                    "heatmap": heatmap,
                    "overlay": overlay,
                    "label": label,
                    "prediction": pred,
                    "path": paths[i],
                })

                if save_dir:
                    save_path = Path(save_dir) / f"gradcam_{count:03d}.png"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    if overlay is not None:
                        Image.fromarray(overlay).save(save_path)

                count += 1

        return results


# ── LIME ────────────────────────────────────────────────────────────────────

class LIMEExplainer:
    """
    LIME (Local Interpretable Model-agnostic Explanations) for images.

    Uses the lime library to generate superpixel-based explanations.

    Args:
        model: Trained model.
        device: torch.device.
        preprocess_fn: Function to preprocess an image array for the model.
    """

    def __init__(self, model, device, preprocess_fn=None):
        self.model = model
        self.device = device
        self.model.eval()

        from lime import lime_image
        self.explainer = lime_image.LimeImageExplainer()

        # Default preprocessing: ImageNet normalize
        if preprocess_fn is None:
            self.preprocess_fn = self._default_preprocess
        else:
            self.preprocess_fn = preprocess_fn

    def _default_preprocess(self, images):
        """Default preprocessing for a batch of numpy images."""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        batch = []
        for img in images:
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = (img - mean) / std
            batch.append(img)
        return np.array(batch)

    def _predict_fn(self, images):
        """Prediction function for LIME."""
        processed = self.preprocess_fn(images)
        tensor = torch.from_numpy(processed).permute(0, 3, 1, 2).float().to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def explain(self, image_array, num_samples=1000, num_features=10):
        """
        Generate LIME explanation for a single image.

        Args:
            image_array: RGB numpy array (H, W, 3) in [0, 255].
            num_samples: Number of perturbation samples.
            num_features: Number of top features to return.

        Returns:
            lime explanation object.
        """
        explanation = self.explainer.explain_instance(
            image_array,
            self._predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=num_samples,
            num_features=num_features,
        )
        return explanation

    def visualize(self, explanation, label=None, positive_only=True):
        """
        Render LIME explanation as an image.

        Args:
            explanation: LIME explanation object.
            label: Target label to visualize. If None, uses top predicted.
            positive_only: Whether to show only positive contributing regions.

        Returns:
            tuple: (visualization_image, mask)
        """
        if label is None:
            label = explanation.top_labels[0]

        temp, mask = explanation.get_image_and_mask(
            label,
            positive_only=positive_only,
            num_features=5,
            hide_rest=False,
        )
        return temp, mask

    def get_top_features(self, explanation, label=None, k=5):
        """
        Get top-k contributing superpixels.

        Args:
            explanation: LIME explanation object.
            label: Target label.
            k: Number of top features.

        Returns:
            list: Top-k (feature_idx, weight) pairs.
        """
        if label is None:
            label = explanation.top_labels[0]

        local_exp = explanation.local_exp[label]
        return sorted(local_exp, key=lambda x: abs(x[1]), reverse=True)[:k]


# ── Attention Rollout (ViT) ─────────────────────────────────────────────────

class AttentionRolloutExplainer:
    """
    Attention rollout for Vision Transformers.

    Propagates attention weights across all transformer layers to produce
    a global attention map showing which image patches are most attended.

    Formula: A_rollout = Π(0.5 * I + 0.5 * A_i)

    Args:
        model: Trained ViT model with get_attention_maps() method.
        device: torch.device.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def rollout(self, image_tensor, head_fusion="mean", discard_ratio=0.1):
        """
        Compute attention rollout map.

        Args:
            image_tensor: Preprocessed image tensor (1, 3, H, W) or (3, H, W).
            head_fusion: How to fuse attention heads ('mean', 'max', 'min').
            discard_ratio: Fraction of lowest attention values to zero out.

        Returns:
            np.ndarray: Attention map of shape (H, W) in [0, 1].
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        # Get attention maps from all layers
        with torch.no_grad():
            attention_maps = self.model.get_attention_maps(image_tensor)

        # Process attention maps
        result = torch.eye(attention_maps[0].shape[-1]).to(self.device)

        for attention in attention_maps:
            # Fuse heads
            if head_fusion == "mean":
                attention_fused = attention.mean(dim=1)  # (B, seq, seq)
            elif head_fusion == "max":
                attention_fused = attention.max(dim=1).values
            elif head_fusion == "min":
                attention_fused = attention.min(dim=1).values
            else:
                raise ValueError(f"Unknown head fusion: {head_fusion}")

            attention_fused = attention_fused[0]  # Take first sample

            # Threshold low attention values
            flat = attention_fused.view(-1)
            threshold = flat.cpu().kthvalue(int(flat.size(0) * discard_ratio)).values.to(self.device)
            attention_fused = torch.where(
                attention_fused > threshold, attention_fused,
                torch.zeros_like(attention_fused)
            )

            # Re-normalize rows
            attention_fused = attention_fused / attention_fused.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            # Rollout: A = (0.5*I + 0.5*A) @ result
            I = torch.eye(attention_fused.shape[-1]).to(self.device)
            a = 0.5 * I + 0.5 * attention_fused
            result = a @ result

        # Extract CLS token attention (first row, skip CLS column)
        cls_attention = result[0, 1:]  # Skip CLS token

        # Reshape to spatial grid
        num_patches = cls_attention.shape[0]
        grid_size = int(num_patches ** 0.5)
        attention_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()

        # Normalize to [0, 1]
        attention_map = (attention_map - attention_map.min()) / (
            attention_map.max() - attention_map.min() + 1e-8
        )

        # Resize to image size
        attention_map = cv2.resize(attention_map, (224, 224))

        return attention_map

    def visualize(self, attention_map, original_image, alpha=0.4):
        """
        Overlay attention rollout map on original image.

        Args:
            attention_map: Attention map (H, W) in [0, 1].
            original_image: RGB numpy array (H, W, 3).
            alpha: Blending factor.

        Returns:
            np.ndarray: Blended visualization.
        """
        if original_image.max() > 1.0:
            original_image = original_image.astype(np.float32) / 255.0

        # Apply colormap
        heatmap = cv2.applyColorMap(
            (attention_map * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Blend
        blended = (1 - alpha) * original_image + alpha * heatmap
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

        return blended


# ── Publication Figure Generation ───────────────────────────────────────────

def generate_gradcam_grid(
    results: list,
    save_path: str,
    severity_labels: dict = None,
):
    """
    Generate a 3×3 Grad-CAM++ heatmap grid for publication.

    Rows: Mild / Moderate / Severe severity
    Columns: Original | Heatmap | Overlay

    Args:
        results: List of dicts from GradCAMPlusPlusExplainer.batch_explain().
        save_path: Path to save the figure.
        severity_labels: Dict mapping image paths to severity labels.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    severities = ["Mild", "Moderate", "Severe"]
    col_titles = ["Original", "Grad-CAM++", "Overlay"]

    for row, severity in enumerate(severities):
        # Find a result for this severity
        example = None
        for r in results:
            if severity_labels and severity_labels.get(r["path"]) == severity:
                example = r
                break
        if example is None and row < len(results):
            example = results[row]  # Fallback

        if example is not None and example["original"] is not None:
            # Original
            axes[row, 0].imshow(example["original"])
            # Heatmap
            axes[row, 1].imshow(example["heatmap"], cmap="jet")
            # Overlay
            if example["overlay"] is not None:
                axes[row, 2].imshow(example["overlay"])

        for col in range(3):
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=14, fontweight="bold")
        axes[row, 0].set_ylabel(severity, fontsize=13, rotation=90, labelpad=15)

    plt.suptitle("Grad-CAM++ Heatmaps by Severity Class", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[XAI] Saved Grad-CAM++ grid → {save_path}")


def generate_lime_comparison(
    model,
    device,
    correct_example,
    incorrect_example,
    save_path: str,
):
    """
    Generate LIME comparison figure: correct vs misclassified prediction.

    Args:
        model: Trained model.
        device: torch.device.
        correct_example: Dict with 'original' and 'path' for a correct prediction.
        incorrect_example: Dict with 'original' and 'path' for a misclassified example.
        save_path: Path to save the figure.
    """
    explainer = LIMEExplainer(model, device)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    titles = ["Original", "LIME (Positive)", "LIME (Negative)"]
    row_labels = ["Correct Prediction", "Misclassified"]

    for row, example in enumerate([correct_example, incorrect_example]):
        if example is None or example.get("original") is None:
            continue

        orig = example["original"]
        explanation = explainer.explain(orig)

        # Original
        axes[row, 0].imshow(orig)

        # Positive regions
        temp_pos, _ = explainer.visualize(explanation, positive_only=True)
        axes[row, 1].imshow(temp_pos)

        # Negative regions
        temp_neg, _ = explainer.visualize(explanation, positive_only=False)
        axes[row, 2].imshow(temp_neg)

        for col in range(3):
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(titles[col], fontsize=13, fontweight="bold")
        axes[row, 0].set_ylabel(row_labels[row], fontsize=12, rotation=90, labelpad=15)

    plt.suptitle("LIME Explanations: Correct vs Misclassified", fontsize=15, fontweight="bold")
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[XAI] Saved LIME comparison → {save_path}")
