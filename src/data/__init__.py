"""Data module for federated learning datasets."""

from .dataset_loader import (
    prepare_federated_dataset,
    prepare_federated_dataset_simple,  # NEW: For FedAvg/FedProx
    load_centralized_dataset,
    partition_data_dirichlet,
    create_dataloaders,
    get_transforms
)
from .clustering import (
    kmeans_clustering,
    random_clustering,
    compute_client_signatures,
    analyze_clustering_quality
)
from .tinyimagenet import TinyImageNetDataset

__all__ = [
    # Dataset preparation
    'prepare_federated_dataset',
    'prepare_federated_dataset_simple',  
    'load_centralized_dataset',
    'partition_data_dirichlet',
    'create_dataloaders',
    'get_transforms',
    # Clustering (for layer-wise only)
    'kmeans_clustering',
    'random_clustering',
    'compute_client_signatures',
    'analyze_clustering_quality',
    # Custom datasets
    'TinyImageNetDataset'
]