"""Client implementations for federated learning."""

from .layer_wise_client import LayerWiseFlowerClient
from .fedavg_client import FedAvgClient
from .fedprox_client import FedProxClient

__all__ = [
    'LayerWiseFlowerClient',
    'FedAvgClient',
    'FedProxClient'
]