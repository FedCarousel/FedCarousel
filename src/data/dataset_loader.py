"""
Unified dataset preparation.

Handles:
- CIFAR-10
- CIFAR-100
- Tiny ImageNet

With Dirichlet partitioning and client clustering.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import (ToTensor, Normalize, Compose, 
                                   RandomCrop, RandomHorizontalFlip)
from typing import Tuple, List, Dict
import gc

from .tinyimagenet import TinyImageNetDataset
from .clustering import (compute_client_signatures, kmeans_clustering, 
                        random_clustering, analyze_clustering_quality)
from ..utils.seed import set_seed


def get_transforms(dataset_name: str, augmentation: bool = True):
    """
    Get data transforms for a dataset.
    
    Args:
        dataset_name: 'cifar10', 'cifar100', or 'tinyimagenet'
        augmentation: Whether to apply data augmentation (for training)
        
    Returns:
        Tuple of (train_transform, test_transform)
    """
    if dataset_name == 'cifar10':
        normalize = Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2023, 0.1994, 0.2010))
        
        if augmentation:
            train_transform = Compose([
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize
            ])
        else:
            train_transform = Compose([ToTensor(), normalize])
        
        test_transform = Compose([ToTensor(), normalize])
    
    elif dataset_name == 'cifar100':
        normalize = Normalize((0.5071, 0.4867, 0.4408), 
                             (0.2675, 0.2565, 0.2761))
        
        if augmentation:
            train_transform = Compose([
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize
            ])
        else:
            train_transform = Compose([ToTensor(), normalize])
        
        test_transform = Compose([ToTensor(), normalize])
    
    elif dataset_name == 'tinyimagenet':
        normalize = Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        
        if augmentation:
            train_transform = Compose([
                RandomCrop(64, padding=8),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize
            ])
        else:
            train_transform = Compose([ToTensor(), normalize])
        
        test_transform = Compose([ToTensor(), normalize])
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_transform, test_transform


def load_centralized_dataset(dataset_name: str, 
                            data_root: str = './data',
                            augmentation: bool = True):
    """
    Load centralized dataset (before partitioning).
    
    Args:
        dataset_name: 'cifar10', 'cifar100', or 'tinyimagenet'
        data_root: Root directory for datasets
        augmentation: Whether to apply data augmentation
        
    Returns:
        Tuple of (trainset, testset, num_classes)
    """
    train_transform, test_transform = get_transforms(dataset_name, augmentation)
    
    if dataset_name == 'cifar10':
        trainset = CIFAR10(data_root, train=True, download=True, 
                          transform=train_transform)
        testset = CIFAR10(data_root, train=False, download=True, 
                         transform=test_transform)
        num_classes = 10
    
    elif dataset_name == 'cifar100':
        trainset = CIFAR100(f"{data_root}/CIFAR100", train=True, download=True, 
                           transform=train_transform)
        testset = CIFAR100(f"{data_root}/CIFAR100", train=False, download=True, 
                          transform=test_transform)
        num_classes = 100
    
    elif dataset_name == 'tinyimagenet':
        trainset = TinyImageNetDataset(
            root=f"{data_root}/tiny-imagenet-200",
            split='train',
            transform=train_transform
        )
        testset = TinyImageNetDataset(
            root=f"{data_root}/tiny-imagenet-200",
            split='val',
            transform=test_transform
        )
        num_classes = 200
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"📊 Dataset {dataset_name.upper()} loaded:")
    print(f"   Training samples: {len(trainset)}")
    print(f"   Test samples: {len(testset)}")
    print(f"   Number of classes: {num_classes}")
    
    return trainset, testset, num_classes


def partition_data_dirichlet(trainset, 
                             num_clients: int,
                             num_classes: int,
                             alpha: float,
                             seed: int = 42) -> List[List]:
    """
    Partition dataset using Dirichlet distribution (non-IID).
    
    Args:
        trainset: Training dataset
        num_clients: Number of clients
        num_classes: Number of classes in dataset
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        seed: Random seed
        
    Returns:
        List of datasets for each client
    """
    set_seed(seed)
    
    # Organize samples by class
    class_data = {i: [] for i in range(num_classes)}
    for idx, (image, label) in enumerate(trainset):
        class_data[label].append((image, label))
    
    print(f"\n👥 Partitioning data to {num_clients} clients (alpha={alpha})...")
    
    client_datasets = [[] for _ in range(num_clients)]
    
    # Distribute each class according to Dirichlet distribution
    for class_id in range(num_classes):
        class_samples = class_data[class_id]
        np.random.shuffle(class_samples)
        
        # Sample proportions from Dirichlet
        client_proportions = np.random.dirichlet([alpha] * num_clients)
        
        # Distribute samples according to proportions
        start_idx = 0
        for client_idx in range(num_clients):
            if client_idx == num_clients - 1:
                # Last client gets remaining samples
                client_samples = class_samples[start_idx:]
            else:
                num_samples = int(len(class_samples) * client_proportions[client_idx])
                client_samples = class_samples[start_idx:start_idx + num_samples]
                start_idx += num_samples
            
            client_datasets[client_idx].extend(client_samples)
    
    # Verify distribution
    total_distributed = sum(len(cd) for cd in client_datasets)
    print(f"   Original samples: {len(trainset)}")
    print(f"   Distributed samples: {total_distributed}")
    print(f"   Difference: {len(trainset) - total_distributed}")
    
    return client_datasets


def create_dataloaders(client_datasets: List[List],
                       batch_size: int,
                       val_split: float = 0.1,
                       num_workers: int = 2) -> Tuple[List, List]:
    """
    Create train and validation dataloaders for each client.
    
    Args:
        client_datasets: List of datasets for each client
        batch_size: Batch size for dataloaders
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        
    Returns:
        Tuple of (trainloaders, valloaders)
    """
    print(f"\n🔧 Creating dataloaders (batch_size={batch_size})...")
    
    trainloaders = []
    valloaders = []
    
    for client_idx, client_data in enumerate(client_datasets):
        if len(client_data) == 0:
            print(f"⚠️  Client {client_idx} has no data!")
            trainloaders.append(None)
            valloaders.append(None)
            continue
        
        # Shuffle client data
        np.random.shuffle(client_data)
        
        # Split into train and validation
        val_size = max(1, int(len(client_data) * val_split))
        train_data = client_data[val_size:]
        val_data = client_data[:val_size]
        
        # Ensure training set is not empty
        if len(train_data) == 0:
            train_data = client_data
            val_data = []
        
        # Create dataloaders
        trainloader = DataLoader(
            train_data,
            batch_size=min(batch_size, len(train_data)),
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True
        )
        
        valloader = DataLoader(
            val_data,
            batch_size=min(batch_size, len(val_data)) if val_data else 1,
            shuffle=False,
            num_workers=num_workers
        ) if val_data else None
        
        trainloaders.append(trainloader)
        valloaders.append(valloader)
        
        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"   Created {len(trainloaders)} client dataloaders")
    
    return trainloaders, valloaders


def prepare_federated_dataset(
    dataset_name: str,
    num_clients: int,
    num_clusters: int,
    alpha: float,
    batch_size: int,
    clustering_method: str = 'kmeans',
    data_root: str = './data',
    seed: int = 42
) -> Tuple:
    """
    Complete pipeline to prepare federated dataset.
    
    Args:
        dataset_name: 'cifar10', 'cifar100', or 'tinyimagenet'
        num_clients: Number of federated clients
        num_clusters: Number of clusters for grouping clients
        alpha: Dirichlet concentration parameter
        batch_size: Batch size for dataloaders
        clustering_method: 'kmeans' or 'random'
        data_root: Root directory for datasets
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (trainloaders, valloaders, testloader, 
                 cluster_assignments, client_to_cluster)
    """
    set_seed(seed)
    
    # Load centralized dataset
    trainset, testset, num_classes = load_centralized_dataset(
        dataset_name, data_root
    )
    
    # Create test dataloader
    testloader = DataLoader(testset, batch_size=batch_size, 
                           shuffle=False, num_workers=2)
    
    # Partition data to clients
    client_datasets = partition_data_dirichlet(
        trainset, num_clients, num_classes, alpha, seed
    )
    
    # Compute client signatures for clustering
    client_signatures = compute_client_signatures(client_datasets, num_classes)
    
    # Perform clustering
    if clustering_method == 'kmeans':
        cluster_assignments, client_to_cluster = kmeans_clustering(
            client_signatures, num_clusters, seed
        )
        # Analyze clustering quality
        analyze_clustering_quality(client_signatures, cluster_assignments, num_clusters)
    elif clustering_method == 'random':
        cluster_assignments, client_to_cluster = random_clustering(
            num_clients, num_clusters, seed
        )
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}")
    
    # Create dataloaders
    trainloaders, valloaders = create_dataloaders(
        client_datasets, batch_size
    )
    
    print(f"\n✅ Federated dataset preparation complete!")
    
    return trainloaders, valloaders, testloader, cluster_assignments, client_to_cluster

def prepare_federated_dataset_simple(
    dataset_name: str,
    num_clients: int,
    alpha: float,
    batch_size: int,
    data_root: str = './data',
    seed: int = 42
) -> Tuple[List, List, DataLoader]:
    """
    Simplified pipeline for FedAvg/FedProx (NO clustering).
    
    This function prepares federated data without computing client
    signatures or performing clustering, which is not needed for
    standard FedAvg and FedProx algorithms.
    
    Args:
        dataset_name: 'cifar10', 'cifar100', or 'tinyimagenet'
        num_clients: Number of federated clients
        alpha: Dirichlet concentration parameter (lower = more non-IID)
        batch_size: Batch size for dataloaders
        data_root: Root directory for datasets
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (trainloaders, valloaders, testloader)
    """
    set_seed(seed)
    
    # Load centralized dataset
    trainset, testset, num_classes = load_centralized_dataset(
        dataset_name, data_root
    )
    
    # Create test dataloader
    testloader = DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    # Partition data to clients using Dirichlet
    client_datasets = partition_data_dirichlet(
        trainset, num_clients, num_classes, alpha, seed
    )
    
    # Create dataloaders
    trainloaders, valloaders = create_dataloaders(
        client_datasets, batch_size
    )
    
    
    return trainloaders, valloaders, testloader