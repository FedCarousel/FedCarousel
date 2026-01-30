"""
Seed management for reproducibility across all experiments.
"""
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility across all random number generators.
    
    Args:
        seed: Integer seed value
        
    Note:
        This affects:
        - Python's random module
        - NumPy's random state
        - PyTorch's CPU and GPU random states
        - CuDNN deterministic behavior
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_generator(seed: int) -> torch.Generator:
    """
    Create a PyTorch Generator with fixed seed for reproducible data loading.
    
    Args:
        seed: Integer seed value
        
    Returns:
        torch.Generator with fixed seed
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
