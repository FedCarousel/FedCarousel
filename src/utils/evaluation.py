"""
Centralized evaluation functions for federated learning.
"""
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, Callable
from flwr.common import parameters_to_ndarrays, Scalar


def get_evaluate_fn(testloader, model_factory: Callable, num_classes: int):
    """
    Create centralized evaluation function for Flower strategy.
    
    Args:
        testloader: DataLoader for test set
        model_factory: Function that creates the model (e.g., create_resnet8)
        num_classes: Number of output classes
        
    Returns:
        Evaluation function compatible with Flower strategy
    """
    def evaluate_fn(server_round: int, parameters, config: Dict[str, Scalar]):
        """
        Evaluate global model on centralized test set.
        
        Args:
            server_round: Current round number
            parameters: Model parameters to evaluate
            config: Configuration dictionary
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model_factory(num_classes=num_classes).to(device)
        
        # Handle both Parameters objects and raw arrays
        if hasattr(parameters, 'tensors'):
            params_arrays = parameters_to_ndarrays(parameters)
        else:
            params_arrays = parameters
        
        # Load parameters into model
        params_dict = zip(model.state_dict().keys(), params_arrays)
        state_dict = OrderedDict({
            k: torch.tensor(v).to(device) 
            for k, v in params_dict
        })
        model.load_state_dict(state_dict, strict=True)
        
        # Evaluate
        model.eval()
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        
        print(f"Round {server_round} - Test Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        
        return avg_loss, {"accuracy": accuracy}
    
    return evaluate_fn
