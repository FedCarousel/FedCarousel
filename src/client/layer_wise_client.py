"""
Layer-wise federated learning client.

Implements alternating training strategy:
- Global rounds: Train entire model
- Partial rounds: Train specific layers based on cluster assignment
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Tuple
import flwr as fl
from flwr.common import Scalar

from ..models.layer_mapping import get_layer_mapping


class LayerWiseFlowerClient(fl.client.NumPyClient):
    """
    Flower client with layer-wise training capability.
    
    Attributes:
        trainloader: DataLoader for training data
        valloader: DataLoader for validation data
        cluster_id: ID of the cluster this client belongs to
        client_id: Unique client ID
        model: PyTorch model
        device: Training device (CPU/GPU)
        layer_mapping: Dictionary mapping layer IDs to parameter names
    """
    
    def __init__(self, 
                 trainloader, 
                 valloader, 
                 model,
                 cluster_id: int,
                 client_id: int,
                 model_name: str,
                 device: str = None):
        """
        Args:
            trainloader: Training data loader
            valloader: Validation data loader
            model: PyTorch model
            cluster_id: Cluster assignment for this client
            client_id: Unique identifier for this client
            model_name: Name of model ('resnet8' or 'resnet18')
            device: Device to use (None for auto-detection)
        """
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.cluster_id = cluster_id
        self.client_id = client_id
        
        self.device = torch.device(
            device if device else 
            ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.layer_mapping = get_layer_mapping(model_name)

    def get_training_mode(self, round_num: int, cycle_length: int, 
                         global_rounds: int) -> Tuple[str, int]:
        """
        Determine training mode based on round number.
        
        Args:
            round_num: Current round number (1-indexed)
            cycle_length: Total rounds in one cycle
            global_rounds: Number of global rounds at cycle start
            
        Returns:
            Tuple of (mode, epochs) where mode is 'global' or 'partial'
        """
        position_in_cycle = (round_num - 1) % cycle_length
        
        if position_in_cycle < global_rounds:
            return "global", 2
        else:
            return "partial", 2

    def get_current_layer_to_train(self, round_num: int, cycle_length: int,
                                   global_rounds: int) -> int:
        """
        Determine which layer this client should train in partial mode.
        
        Uses formula: layer_id = (cluster_id + partial_round) % num_layers
        
        Args:
            round_num: Current round number (1-indexed)
            cycle_length: Total rounds in one cycle
            global_rounds: Number of global rounds at cycle start
            
        Returns:
            Layer ID to train, or -1 if in global mode
        """
        position_in_cycle = (round_num - 1) % cycle_length
        
        if position_in_cycle >= global_rounds:
            partial_round = position_in_cycle - global_rounds
            num_layers = len(self.layer_mapping)
            return (self.cluster_id + partial_round) % num_layers
        
        return -1

    def set_parameters(self, parameters):
        """Load parameters into model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({
            k: torch.tensor(v).to(self.device) 
            for k, v in params_dict
        })
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extract parameters from model."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def train_global(self, epochs: int, learning_rate: float = 0.01):
        """
        Train entire model (global mode).
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        self.model.train()
        
        # All parameters trainable
        for param in self.model.parameters():
            param.requires_grad = True
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=5e-4
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def train_specific_layer(self, layer_id: int, epochs: int, 
                            learning_rate: float = 0.01):
        """
        Train only a specific layer (partial mode).
        
        Strategy:
        - Keep model.train() so ALL BatchNorm layers use batch statistics
        - Freeze all parameters except those in target layer
        - Gradients only flow through target layer
        
        Args:
            layer_id: ID of layer to train
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        # CRITICAL: Keep model in train mode for BatchNorm
        self.model.train()
        
        layer_names = self.layer_mapping[layer_id]
        trainable_params = []
        
        # Freeze ALL parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze only target layer parameters
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                trainable_params.append(param)
        
        if len(trainable_params) == 0:
            print(f"⚠️  Client {self.client_id}: No parameters found for layer {layer_id}!")
            return
        
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=5e-4
        )
        
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Verify gradients exist (debugging)
                grad_norm = sum(
                    p.grad.norm().item() 
                    for p in trainable_params 
                    if p.grad is not None
                )
                if grad_norm == 0:
                    print(f"⚠️  Warning: Zero gradients for layer {layer_id}")
                
                optimizer.step()

    def get_layer_parameters(self, layer_id: int):
        """
        Extract parameters of a specific layer only.
        
        Used in partial mode to reduce communication cost.
        
        Args:
            layer_id: ID of layer to extract
            
        Returns:
            List of numpy arrays for this layer's parameters
        """
        layer_names = self.layer_mapping[layer_id]
        params = []
        param_names = []
        
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                params.append(param.detach().cpu().numpy())
                param_names.append(name)
        
        if len(params) == 0:
            raise RuntimeError(
                f"Client {self.client_id}: No parameters found for layer {layer_id}!"
            )
        
        print(f"   Client {self.client_id}: Sending {len(params)} params "
          f"for layer {layer_id}")
        return params

    def fit(self, parameters, config):
        """
        Train the model for one round.
        
        Args:
            parameters: Model parameters from server
            config: Configuration dictionary with:
                - round_num: Current round number
                - cycle_length: Rounds per cycle
                - global_rounds: Global rounds per cycle
                - learning_rate: Learning rate
                
        Returns:
            Tuple of (parameters, num_samples, metrics)
        """
        round_num = config.get("round_num", 1)
        cycle_length = config.get("cycle_length")
        global_rounds = config.get("global_rounds")
        learning_rate = config.get("learning_rate")
        
        # Load parameters from server
        self.set_parameters(parameters)
        
        # Determine training mode
        training_mode, epochs = self.get_training_mode(
            round_num, cycle_length, global_rounds
        )
        
        # Train accordingly
        if training_mode == "global":
            self.train_global(epochs, learning_rate)
            layer_trained = -1
            params_to_send = self.get_parameters(config)
            send_partial = False
        else:
            layer_to_train = self.get_current_layer_to_train(
                round_num, cycle_length, global_rounds
            )
            if layer_to_train != -1:
                self.train_specific_layer(layer_to_train, epochs, learning_rate)
                params_to_send = self.get_layer_parameters(layer_to_train)
                send_partial = True
            else:
                params_to_send = self.get_parameters(config)
                send_partial = False
            layer_trained = layer_to_train
                


        num_samples = len(self.trainloader.dataset)
        
        return params_to_send, num_samples, {
            "layer_trained": int(layer_trained),
            "training_mode": str(training_mode),
            "cluster_id": int(self.cluster_id),
            "send_partial": send_partial
        }

    def evaluate(self, parameters, config):
        """
        Evaluate model on validation set.
        
        Args:
            parameters: Model parameters to evaluate
            config: Configuration dictionary
            
        Returns:
            Tuple of (loss, num_samples, metrics)
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
