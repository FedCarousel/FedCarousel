config = {
    # Dataset
    'dataset_name': 'tinyimagenet',
    'num_classes': 200,
    'data_root': './data',
    
    # Model
    'model_name': 'resnet18',
    'use_maxpool': True,  # Use maxpool for 64x64 images
    
    # Federated Learning
    'num_clients': 100, 
    'num_clusters': 21,
    'num_rounds': 500,
    
    # Data Distribution
    'alpha_client': 0.1,  # Dirichlet concentration
    'clustering_method': 'random' ,  # 'random' or 'kmeans'
    
    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'global_epochs': 2,
    'partial_epochs': 2,
    
    # Cycle Configuration
    'cycle_length': 23,  
    'global_rounds': 3,
    
    # Resources
    'client_resources': {
        'num_cpus': 2,
        'num_gpus': 0.25
    },
    'ray_init_args': {
        'num_gpus': 1,
        'num_cpus': 20,
        'object_store_memory': 10 * 1024 * 1024 * 1024,
        '_memory': 90 * 1024 * 1024 * 1024,  # 90GB
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
