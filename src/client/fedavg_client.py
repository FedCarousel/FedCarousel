"""
FedAvg client implementation.
Follows the same structure as LayerWiseFlowerClient.
"""
from collections import OrderedDict
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import flwr as fl
from flwr.common import NDArrays, Scalar

from ..utils.training import train


class FedAvgClient(fl.client.NumPyClient):
    """
    Flower client for FedAvg algorithm.
    
    Attributes:
        trainloader: DataLoader for training data
        valloader: DataLoader for validation data
        model: PyTorch model
        device: Training device (CPU/GPU)
        learning_rate: Learning rate for optimizer
        local_epochs: Number of local training epochs
    """
    
    def __init__(
        self,
        trainloader,
        valloader,
        model: nn.Module,
        learning_rate: float = 0.001,
        local_epochs: int = 2,
        device: str = None
    ):
        """
        Args:
            trainloader: Training data loader
            valloader: Validation data loader
            model: PyTorch model (created externally)
            learning_rate: Learning rate for optimizer
            local_epochs: Number of local training epochs
            device: Device to use (None for auto-detection)
        """
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        
        self.device = torch.device(
            device if device else 
            ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
    
    def set_parameters(self, parameters: List) -> None:
        """
        Load parameters into model.
        Same implementation as LayerWiseFlowerClient.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({
            k: torch.tensor(v).to(self.device) 
            for k, v in params_dict
        })
        self.model.load_state_dict(state_dict, strict=True)
    
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Extract parameters from model.
        Same implementation as LayerWiseFlowerClient.
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """
        Train with standard FedAvg (no proximal term).
        
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
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=5e-4
        )
        
        train(
            self.model,
            self.trainloader,
            optimizer,
            self.device,
            epochs=epochs,
            proximal_mu=0.0,
            global_params=None
        )
        
        num_samples = len(self.trainloader.dataset)
        
        return self.get_parameters({}), num_samples, {}
    
    def evaluate(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on validation set.
        Same implementation as LayerWiseFlowerClient.
        """
        if self.valloader is None:
            return 0.0, 0, {}
        
        self.set_parameters(parameters)
        self.model.eval()
        
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for images, labels in self.valloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                num_samples += images.size(0)
        
        accuracy = correct / num_samples if num_samples > 0 else 0.0
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        
        return avg_loss, num_samples, {"accuracy": accuracy}