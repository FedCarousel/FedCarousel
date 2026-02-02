"""
Training utilities for federated learning.
Supports standard training and FedProx with proximal term.
"""
from typing import List, Optional

import torch
import torch.nn as nn


def train(
    model: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    proximal_mu: float = 0.0,
    global_params: Optional[List[torch.Tensor]] = None
) -> nn.Module:
    """
    Train model with optional FedProx proximal term.
    
    Args:
        model: PyTorch model to train
        trainloader: Training data loader
        optimizer: Optimizer instance
        device: Training device
        epochs: Number of local epochs
        proximal_mu: FedProx proximal term coefficient (0 = standard FedAvg)
        global_params: Global model parameters for FedProx
        
    Returns:
        Trained model
    """
    criterion = nn.CrossEntropyLoss()
    model.train()
    model.to(device)
    
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Add FedProx proximal term if enabled
            if proximal_mu > 0 and global_params is not None:
                proximal_term = 0.0
                for local_param, global_param in zip(model.parameters(), global_params):
                    proximal_term += (local_param - global_param).norm(2) ** 2
                loss += (proximal_mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
    
    return model


def test(
    model: nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device
) -> tuple:
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model to evaluate
        testloader: Test data loader
        device: Evaluation device
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(device)
    
    correct = 0
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            num_samples += images.size(0)
    
    accuracy = correct / num_samples if num_samples > 0 else 0.0
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
    
    return avg_loss, accuracy