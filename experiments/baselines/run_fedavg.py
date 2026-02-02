"""
FedAvg baseline experiment for CIFAR-10.

This script runs standard FedAvg WITHOUT clustering,
using the simplified data preparation pipeline.
"""
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import flwr as fl
from flwr.common import Context
from flwr.server.strategy import FedAvg

from src.models import create_resnet8
from src.data import prepare_federated_dataset_simple  
from src.client import FedAvgClient
from src.utils import set_seed, get_evaluate_fn


# CONFIGURATION
CONFIG = {
    # Dataset
    'dataset_name': 'cifar10',
    'num_classes': 10,
    'data_root': './data',
    
    # Model
    'model_name': 'resnet8',
    
    # Federated Learning
    'num_clients': 100,
    'num_rounds': 100,
    'alpha': 0.1,  # Dirichlet concentration (lower = more non-IID)
    
    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'local_epochs': 2,
    
    # Reproducibility
    'seed': 15,
    
    # Resources 
    'client_resources': {'num_cpus': 2, 'num_gpus': 0.12},
    'ray_init_args': {
        'num_gpus': 1,
        'num_cpus': 20,
        'object_store_memory': 10 * 1024 * 1024 * 1024,
        '_memory': 60 * 1024 * 1024 * 1024,
        'include_dashboard': False
    },
    
    # Output
    'results_dir': './results/baselines'
}


def generate_client_fn(trainloaders, valloaders, cfg):
    """Generate client creation function for Flower simulation."""
    
    def client_fn(context: Context):
        client_id = int(context.node_config["partition-id"])
        
        # Create model 
        model = create_resnet8(num_classes=cfg['num_classes'])
        
        # Create FedAvg client 
        client = FedAvgClient(
            trainloader=trainloaders[client_id],
            valloader=valloaders[client_id],
            model=model,
            learning_rate=cfg['learning_rate'],
            local_epochs=cfg['local_epochs']
        )
        
        return client.to_client()
    
    return client_fn


def fit_config(server_round: int):
    """Generate configuration for each training round."""
    return {
        "round_num": server_round,
        "local_epochs": CONFIG['local_epochs'],
        "learning_rate": CONFIG['learning_rate']
    }


def main():
    """Main experiment execution."""
    print("="*70)
    print("FedAvg Baseline - CIFAR-10 with ResNet-8")
    print("="*70)
    print(f"Alpha (non-IID): {CONFIG['alpha']}")
    print(f"Clients: {CONFIG['num_clients']}")
    print(f"Rounds: {CONFIG['num_rounds']}")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(CONFIG['seed'])
    
    # Prepare dataset WITHOUT clustering
    trainloaders, valloaders, testloader = prepare_federated_dataset_simple(
        dataset_name=CONFIG['dataset_name'],
        num_clients=CONFIG['num_clients'],
        alpha=CONFIG['alpha'],
        batch_size=CONFIG['batch_size'],
        data_root=CONFIG['data_root'],
        seed=CONFIG['seed']
    )
    
    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.1,
        min_fit_clients=CONFIG['num_clients'],
        min_available_clients=CONFIG['num_clients'],
        evaluate_fn=get_evaluate_fn(
            testloader,
            create_resnet8,
            CONFIG['num_classes']
        ),
        on_fit_config_fn=fit_config
    )
    
    # Generate client function
    client_fn_callback = generate_client_fn(trainloaders, valloaders, CONFIG)
    
    # Run simulation
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn_callback,
        num_clients=CONFIG['num_clients'],
        config=fl.server.ServerConfig(num_rounds=CONFIG['num_rounds']),
        strategy=strategy,
        client_resources=CONFIG['client_resources'],
        ray_init_args=CONFIG['ray_init_args']
    )
    
    # Save results
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    results = {
        "accuracy": history.metrics_centralized,
        "loss_centralized": history.losses_centralized,
        "loss_distributed": history.losses_distributed,
        "config": CONFIG
    }
    
    output_file = os.path.join(
        CONFIG['results_dir'],
        f"fedavg_alpha{CONFIG['alpha']}_s{CONFIG['seed']}.json"
    )
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}")
    
    # Print final results
    if history.metrics_centralized:
        final_round, final_metrics = history.metrics_centralized[-1]
        print(f"\n🎉 Final Results:")
        print(f"   Round: {final_round}")
        print(f"   Test Accuracy: {final_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()