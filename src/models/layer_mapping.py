"""
Layer-to-parameter mapping for layer-wise training.

Each entry maps a layer ID to the parameter names that belong to that layer.
Typically, each layer group contains a convolution and its associated batch normalization.
"""
from typing import Dict, List


# ResNet-8 layer mapping (10 layer groups)
RESNET8_LAYER_MAPPING: Dict[int, List[str]] = {
    0: ['pre_conv', 'pre_bn'],
    1: ['layers.0.0.conv1', 'layers.0.0.bn1'],
    2: ['layers.0.0.conv2', 'layers.0.0.bn2'],
    3: ['layers.1.0.conv1', 'layers.1.0.bn1'],
    4: ['layers.1.0.conv2', 'layers.1.0.bn2'],
    5: ['layers.1.0.downsample.0', 'layers.1.0.downsample.1'],
    6: ['layers.2.0.conv1', 'layers.2.0.bn1'],
    7: ['layers.2.0.conv2', 'layers.2.0.bn2'],
    8: ['layers.2.0.downsample.0', 'layers.2.0.downsample.1'],
    9: ['fc']
}


# ResNet-18 layer mapping (21 layer groups)
RESNET18_LAYER_MAPPING: Dict[int, List[str]] = {
    0: ['pre_conv', 'pre_bn'],
    # Layer group 1: 64 channels, 2 blocks
    1: ['layers.0.0.conv1', 'layers.0.0.bn1'],
    2: ['layers.0.0.conv2', 'layers.0.0.bn2'],
    3: ['layers.0.1.conv1', 'layers.0.1.bn1'],
    4: ['layers.0.1.conv2', 'layers.0.1.bn2'],
    # Layer group 2: 128 channels, 2 blocks (with downsample)
    5: ['layers.1.0.conv1', 'layers.1.0.bn1'],
    6: ['layers.1.0.conv2', 'layers.1.0.bn2'],
    7: ['layers.1.0.downsample.0', 'layers.1.0.downsample.1'],
    8: ['layers.1.1.conv1', 'layers.1.1.bn1'],
    9: ['layers.1.1.conv2', 'layers.1.1.bn2'],
    # Layer group 3: 256 channels, 2 blocks (with downsample)
    10: ['layers.2.0.conv1', 'layers.2.0.bn1'],
    11: ['layers.2.0.conv2', 'layers.2.0.bn2'],
    12: ['layers.2.0.downsample.0', 'layers.2.0.downsample.1'],
    13: ['layers.2.1.conv1', 'layers.2.1.bn1'],
    14: ['layers.2.1.conv2', 'layers.2.1.bn2'],
    # Layer group 4: 512 channels, 2 blocks (with downsample)
    15: ['layers.3.0.conv1', 'layers.3.0.bn1'],
    16: ['layers.3.0.conv2', 'layers.3.0.bn2'],
    17: ['layers.3.0.downsample.0', 'layers.3.0.downsample.1'],
    18: ['layers.3.1.conv1', 'layers.3.1.bn1'],
    19: ['layers.3.1.conv2', 'layers.3.1.bn2'],
    # Final classifier
    20: ['fc']
}


def get_layer_mapping(model_name: str) -> Dict[int, List[str]]:
    """
    Get the appropriate layer mapping for a model.
    
    Args:
        model_name: Name of the model ('resnet8' or 'resnet18')
        
    Returns:
        Dictionary mapping layer IDs to parameter name patterns
        
    Raises:
        ValueError: If model_name is not recognized
    """
    mappings = {
        'resnet8': RESNET8_LAYER_MAPPING,
        'resnet18': RESNET18_LAYER_MAPPING
    }
    
    if model_name.lower() not in mappings:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Available: {list(mappings.keys())}")
    
    return mappings[model_name.lower()]


def get_num_layers(model_name: str) -> int:
    """
    Get the number of layer groups for a model.
    
    Args:
        model_name: Name of the model ('resnet8' or 'resnet18')
        
    Returns:
        Number of layer groups
    """
    return len(get_layer_mapping(model_name))


def validate_layer_mapping(model, layer_mapping: Dict[int, List[str]]) -> bool:
    """
    Validate that a layer mapping matches the actual model parameters.
    
    Args:
        model: PyTorch model
        layer_mapping: Dictionary mapping layer IDs to parameter names
        
    Returns:
        True if mapping is valid, False otherwise
    """
    model_param_names = set(name for name, _ in model.named_parameters())
    
    for layer_id, param_patterns in layer_mapping.items():
        for pattern in param_patterns:
            # Check if pattern matches any parameter name
            matched = any(pattern in name for name in model_param_names)
            if not matched:
                print(f"Warning: Layer {layer_id} pattern '{pattern}' "
                      f"not found in model parameters")
                return False
    
    return True
