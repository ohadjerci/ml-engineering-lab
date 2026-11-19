"""Compatibility entry point for existing MNIST configurations and checkpoints."""

from src.models.classification_module import ClassificationLitModule


class MNISTLitModule(ClassificationLitModule):
    """Retain the original ten-class defaults and checkpoint parameter names."""
