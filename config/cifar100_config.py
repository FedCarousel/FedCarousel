
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
    'clustering_method': 'random',  # 'random' or 'kmeans'
    
    # Training
    'batch_size': 64,
    'learning_rate': 0.001,
    'global_epochs': 2,
    'partial_epochs': 2,
    
    # Cycle Configuration
    'cycle_length': 24, 
    'global_rounds': 3,
    
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
    'fraction_evaluate': 0.1,
    
    # Output
    'results_dir': './Results',
    'save_model': False
}
