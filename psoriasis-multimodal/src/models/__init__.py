"""
Model registry — central factory for all architectures.
"""

from src.models.baselines import ResNet50Classifier, DenseNet121Classifier, VGG16Classifier
from src.models.efficientnet import EfficientNetB3Classifier
from src.models.vit import ViTB16Classifier


_MODEL_REGISTRY = {
    "resnet50": ResNet50Classifier,
    "densenet121": DenseNet121Classifier,
    "vgg16": VGG16Classifier,
    "efficientnet_b3": EfficientNetB3Classifier,
    "vit_b16": ViTB16Classifier,
}


def get_model(name: str, config: dict, num_classes: int = 2):
    """
    Instantiate a model by name.

    Args:
        name: One of 'resnet50', 'densenet121', 'vgg16', 'efficientnet_b3', 'vit_b16'.
        config: The full config dict (model-specific section is extracted internally).
        num_classes: Number of output classes (default 2 for binary).

    Returns:
        nn.Module: The instantiated model.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name](config=config, num_classes=num_classes)


__all__ = [
    "get_model",
    "ResNet50Classifier",
    "DenseNet121Classifier",
    "VGG16Classifier",
    "EfficientNetB3Classifier",
    "ViTB16Classifier",
]
