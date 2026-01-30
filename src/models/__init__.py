"""Models module for federated learning."""

from .resnet import ResNet8, ResNet18, create_resnet8, create_resnet18
from .layer_mapping import (
    RESNET8_LAYER_MAPPING,
    RESNET18_LAYER_MAPPING,
    get_layer_mapping,
    get_num_layers,
    validate_layer_mapping
)

__all__ = [
    'ResNet8',
    'ResNet18',
    'create_resnet8',
    'create_resnet18',
    'RESNET8_LAYER_MAPPING',
    'RESNET18_LAYER_MAPPING',
    'get_layer_mapping',
    'get_num_layers',
    'validate_layer_mapping'
]
