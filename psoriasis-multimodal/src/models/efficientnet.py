"""
EfficientNet-B3 classifier — primary model.
"""

import torch
import torch.nn as nn
import timm


class EfficientNetB3Classifier(nn.Module):
    """
    EfficientNet-B3 fine-tuned for psoriasis biologic response prediction.

    Architecture:
        - EfficientNet-B3 backbone (ImageNet pretrained via timm)
        - Freeze all blocks except the last `unfreeze_blocks`
        - GAP → Dropout(0.4) → FC(1536→512) → ReLU → FC(512→256) → ReLU → FC(256→2)

    Args:
        config: Full config dict (efficientnet section is extracted).
        num_classes: Number of output classes.
    """

    def __init__(self, config: dict, num_classes: int = 2):
        super().__init__()
        model_cfg = config["models"]["efficientnet"]

        # Load pretrained backbone (no built-in classifier)
        self.backbone = timm.create_model(
            model_cfg.get("name", "efficientnet_b3"),
            pretrained=model_cfg.get("pretrained", True),
            num_classes=0,  # Remove original head; outputs pooled features
        )

        # Freeze strategy: unfreeze only the last N blocks
        unfreeze_blocks = model_cfg.get("unfreeze_blocks", 3)
        self._freeze_backbone(unfreeze_blocks)

        # Determine feature dimension from the backbone
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[1]

        # Classification head matching the plan:
        # GAP → Dropout → FC(1536→512) → ReLU → FC(512→256) → ReLU → FC(256→2)
        # (GAP is already applied by timm when num_classes=0)
        head_dims = model_cfg.get("head_dims", [feat_dim, 512, 256, num_classes])
        dropout = model_cfg.get("dropout", 0.4)

        layers = [nn.Dropout(dropout)]
        in_dim = head_dims[0] if head_dims[0] != num_classes else feat_dim

        # Build FC layers
        for i in range(1, len(head_dims)):
            layers.append(nn.Linear(in_dim, head_dims[i]))
            if i < len(head_dims) - 1:  # No activation after final layer
                layers.append(nn.ReLU(inplace=True))
            in_dim = head_dims[i]

        self.head = nn.Sequential(*layers)

        # Store target layer name for Grad-CAM
        self._target_layer = None

    def _freeze_backbone(self, unfreeze_blocks: int):
        """
        Freeze all backbone blocks except the last `unfreeze_blocks`.

        EfficientNet blocks are organized in self.backbone.blocks[0..N].
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N blocks
        if hasattr(self.backbone, "blocks"):
            total_blocks = len(self.backbone.blocks)
            unfreeze_from = max(0, total_blocks - unfreeze_blocks)
            for i in range(unfreeze_from, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        # Always unfreeze batch norm in unfrozen layers
        # (and the final norm/pooling layers)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Tensor: Logits of shape (B, num_classes).
        """
        features = self.backbone(x)  # (B, feat_dim) — GAP already applied
        logits = self.head(features)
        return logits

    def get_feature_layer(self):
        """
        Return the last convolutional layer for Grad-CAM.

        Returns:
            nn.Module: The target layer for Grad-CAM heatmap generation.
        """
        # For EfficientNet in timm, the last conv layer before GAP
        if hasattr(self.backbone, "conv_head"):
            return self.backbone.conv_head
        # Fallback: last block's last conv
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1]
        raise AttributeError("Cannot determine feature layer for Grad-CAM")
