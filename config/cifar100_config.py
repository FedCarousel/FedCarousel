"""
Configuration for CIFAR-100 experiments with ResNet-18.

Training cycle structure:
- Rounds 1-5: Global training (all clients, entire model)
- Rounds 6-26: Partial training (layer-wise per cluster)
- Cycle length: 26 rounds
- 21 clusters, 21 layers → perfect 1:1 rotation
"""

config = {
    # Dataset
    'dataset_name': 'cifar100',
    'num_classes': 100,
    'data_root': './data',
    
    # Model
    'model_name': 'resnet18',
    'use_maxpool': False,  # No maxpool for 32x32 images
    
    # Federated Learning
    'num_clients': 100,
    'num_clusters': 21,
    'num_rounds': 440,
    
    # Data Distribution
    'alpha_client': 0.1,  # Dirichlet concentration
    'clustering_method': 'kmeans',  # K-means clustering
    
    # Training
    'batch_size': 64,
    'learning_rate': 0.001,
    'global_epochs': 4,
    'partial_epochs': 4,
    
    # Cycle Configuration
    'cycle_length': 26,  # 5 global + 21 partial
    'global_rounds': 5,
    
    # Resources
    'client_resources': {
        'num_cpus': 2,
        'num_gpus': 0.2
    },
    'ray_init_args': {
        'num_gpus': 1,
        'num_cpus': 20,
        'object_store_memory': 10 * 1024 * 1024 * 1024,
        '_memory': 100 * 1024 * 1024 * 1024,  # 100GB
        'include_dashboard': False
    },
    
    # Reproducibility
    'seed': 15,
    
    # Evaluation
    'fraction_fit': 1.0,
    'fraction_evaluate': 0.0,
    
    # Output
    'results_dir': './Results',
    'save_model': False
}
