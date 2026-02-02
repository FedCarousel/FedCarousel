"""
FedProx client implementation.
Extends FedAvgClient with proximal term regularization.
"""
from typing import Dict, List, Tuple

import torch
from flwr.common import NDArrays, Scalar

from .fedavg_client import FedAvgClient
from ..utils.training import train


class FedProxClient(FedAvgClient):
    """
    FedProx client - extends FedAvgClient with proximal term.
    
    The proximal term penalizes local model deviation from global model,
    which helps with non-IID data distribution.
    
    Attributes:
        proximal_mu: Proximal term coefficient (higher = stronger regularization)
        global_params: Copy of global parameters for proximal computation
    """
    
    def __init__(
        self,
        trainloader,
        valloader,
        model,
        learning_rate: float = 0.01,
        local_epochs: int = 2,
        proximal_mu: float = 0.01,
        device: str = None
    ):
        """
        Args:
            trainloader: Training data loader
            valloader: Validation data loader
            model: PyTorch model (created externally)
            learning_rate: Learning rate for optimizer
            local_epochs: Number of local training epochs
            proximal_mu: FedProx proximal term coefficient
            device: Device to use (None for auto-detection)
        """
        super().__init__(
            trainloader=trainloader,
            valloader=valloader,
            model=model,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            device=device
        )
        self.proximal_mu = proximal_mu
        self.global_params: List[torch.Tensor] = None
    
    def set_parameters(self, parameters: List) -> None:
        """
        Load parameters and store a copy for proximal term computation.
        """
        super().set_parameters(parameters)
        # Store global parameters for FedProx proximal term
        self.global_params = [
            param.data.clone() for param in self.model.parameters()
        ]
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """
        Train with FedProx proximal term.
        
        Loss = CrossEntropy + (mu/2) * ||w - w_global||^2
        
        Args:
            parameters: Model parameters from server
            config: Configuration dictionary
            
        Returns:
            Tuple of (parameters, num_samples, metrics)
        """
        self.set_parameters(parameters)
        
        # Get config values or use defaults
        epochs = config.get("local_epochs", self.local_epochs)
        lr = config.get("learning_rate", self.learning_rate)
        mu = config.get("proximal_mu", self.proximal_mu)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=5e-4
        )
        
        # Use shared training function WITH proximal term
        train(
            self.model,
            self.trainloader,
            optimizer,
            self.device,
            epochs=epochs,
            proximal_mu=mu,
            global_params=self.global_params
        )
        
        num_samples = len(self.trainloader.dataset)
        
        return self.get_parameters({}), num_samples, {"proximal_mu": mu}