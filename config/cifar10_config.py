config = {
    # Dataset
    'dataset_name': 'cifar10',
    'num_classes': 10,
    'data_root': './data',
    
    # Model
    'model_name': 'resnet8',
    
    # Federated Learning
    'num_clients': 100,
    'num_clusters': 10,
    'num_rounds': 300,
    
    # Data Distribution
    'alpha_client': 0.1,  # Dirichlet concentration (lower = more non-IID)
    'clustering_method': 'random',  # 'random' or 'kmeans'
    
    # Training
    'batch_size': 32,
    'learning_rate': 0.01,
    'global_epochs': 2,
    'partial_epochs': 2,
    
    # Cycle Configuration
    'cycle_length': 13,  # 3 global + 10 partial
    'global_rounds': 3,  # Number of global rounds per cycle
    
    # Resources
    'client_resources': {
        'num_cpus': 2,
        'num_gpus': 0.12
    },
    'ray_init_args': {
        'num_gpus': 1,
        'num_cpus': 20,
        'object_store_memory': 10 * 1024 * 1024 * 1024,  # 10GB
        '_memory': 60 * 1024 * 1024 * 1024,  # 60GB
        'include_dashboard': False
    },
    
    # Reproducibility
    'seed': 15,
    
    # Evaluation
    'fraction_fit': 1.0,  # Fraction of clients to sample for training
    'fraction_evaluate': 0.1, 
    
    # Output
    'results_dir': './results',
    'save_model': False
}
