"""
Baseline CNN models: ResNet-50, DenseNet-121, VGG-16.
"""

import torch
import torch.nn as nn
import timm


class ResNet50Classifier(nn.Module):
    """
    ResNet-50 baseline with configurable fine-tuning depth.

    Architecture:
        - ResNet-50 backbone (ImageNet pretrained)
        - Freeze all except last `fine_tune_layers` residual blocks
        - Custom head: GAP → Dropout → FC → ReLU → FC(num_classes)

    Args:
        config: Full config dict (resnet50 section is extracted).
        num_classes: Number of output classes.
    """

    def __init__(self, config: dict, num_classes: int = 2):
        super().__init__()
        model_cfg = config["models"]["resnet50"]

        # Load pretrained backbone
        self.backbone = timm.create_model(
            model_cfg.get("name", "resnet50"),
            pretrained=model_cfg.get("pretrained", True),
            num_classes=0,  # Remove original head
        )

        # Freeze layers
        fine_tune_layers = model_cfg.get("fine_tune_layers", 2)
        self._freeze_backbone(fine_tune_layers)

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[1]

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def _freeze_backbone(self, fine_tune_layers: int):
        """Freeze all layers except the last N residual blocks."""
        # ResNet has: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4
        layers = [
            self.backbone.conv1, self.backbone.bn1,
            self.backbone.layer1, self.backbone.layer2,
            self.backbone.layer3, self.backbone.layer4,
        ]
        # Freeze all except the last `fine_tune_layers`
        freeze_until = len(layers) - fine_tune_layers
        for i, layer in enumerate(layers):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 baseline with full fine-tuning.

    Architecture:
        - DenseNet-121 backbone (ImageNet pretrained)
        - Full fine-tuning (all layers unfrozen)
        - Custom head: GAP → Dropout → FC → ReLU → FC(num_classes)

    Args:
        config: Full config dict.
        num_classes: Number of output classes.
    """

    def __init__(self, config: dict, num_classes: int = 2):
        super().__init__()
        model_cfg = config["models"]["densenet121"]

        self.backbone = timm.create_model(
            model_cfg.get("name", "densenet121"),
            pretrained=model_cfg.get("pretrained", True),
            num_classes=0,
        )

        # Full fine-tune: all params unfrozen (default)
        if not model_cfg.get("full_fine_tune", True):
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class VGG16Classifier(nn.Module):
    """
    VGG-16 baseline with frozen convolutional layers, fine-tuned FC head.

    Architecture:
        - VGG-16 backbone (ImageNet pretrained)
        - Freeze all convolutional layers
        - Custom FC head: GAP → Dropout → FC → ReLU → Dropout → FC → ReLU → FC(num_classes)

    Args:
        config: Full config dict.
        num_classes: Number of output classes.
    """

    def __init__(self, config: dict, num_classes: int = 2):
        super().__init__()
        model_cfg = config["models"]["vgg16"]

        self.backbone = timm.create_model(
            model_cfg.get("name", "vgg16"),
            pretrained=model_cfg.get("pretrained", True),
            num_classes=0,
        )

        # Freeze all conv layers
        if model_cfg.get("fine_tune_fc_only", True):
            for name, param in self.backbone.named_parameters():
                if "features" in name or "pre_logits" in name:
                    param.requires_grad = False

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feat_dim = self.backbone(dummy).shape[1]

        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
