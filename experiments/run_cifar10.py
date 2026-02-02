"""
CIFAR-10 experiment with ResNet-8 and layer-wise federated learning.
"""
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import flwr as fl
from flwr.common import Context

from src.models import create_resnet8
from src.data import prepare_federated_dataset
from src.client.layer_wise_client import LayerWiseFlowerClient
from src.server.layer_wise_strategy import LayerWiseFedAvg
from src.utils.evaluation import get_evaluate_fn
from src.utils.seed import set_seed
from config.cifar10_config import config


def generate_client_fn(trainloaders, valloaders, client_to_cluster, cfg):
    """Generate client creation function for Flower simulation."""
    
    def client_fn(context: Context):
        client_id = int(context.node_config["partition-id"])
        cluster_id = client_to_cluster[client_id]
        
        # Create model
        model = create_resnet8(num_classes=cfg['num_classes'])
        
        # Create client
        client = LayerWiseFlowerClient(
            trainloader=trainloaders[client_id],
            valloader=valloaders[client_id],
            model=model,
            cluster_id=cluster_id,
            client_id=client_id,
            model_name=cfg['model_name']
        )
        
        return client.to_client()
    
    return client_fn


def fit_config(server_round: int, cfg):
    """Generate configuration for each training round."""
    return {
        "round_num": server_round,
        "cycle_length": cfg['cycle_length'],
        "global_rounds": cfg['global_rounds'],
        "learning_rate": cfg['learning_rate']
    }


def main():
    """Main experiment execution."""
    # Set seed for reproducibility
    set_seed(config['seed'])
    
    # Prepare dataset
    trainloaders, valloaders, testloader, cluster_assignments, client_to_cluster = \
        prepare_federated_dataset(
            dataset_name=config['dataset_name'],
            num_clients=config['num_clients'],
            num_clusters=config['num_clusters'],
            alpha=config['alpha_client'],
            batch_size=config['batch_size'],
            clustering_method=config['clustering_method'],
            data_root=config['data_root'],
            seed=config['seed']
        )
    
    # Create strategy
    print("\n🎯 Initializing federated strategy...")
    strategy = LayerWiseFedAvg(
        num_clusters=config['num_clusters'],
        model_name=config['model_name'],
        cycle_length=config['cycle_length'],
        global_rounds=config['global_rounds'],
        fraction_fit=config['fraction_fit'],
        fraction_evaluate=config['fraction_evaluate'],
        evaluate_fn=get_evaluate_fn(
            testloader, 
            create_resnet8, 
            config['num_classes']
        ),
        on_fit_config_fn=lambda round_num: fit_config(round_num, config)
    )
    
    # Generate client function
    client_fn_callback = generate_client_fn(
        trainloaders, valloaders, client_to_cluster, config
    )
    
    # Run simulation
    print("\n🚀 Starting federated learning simulation...")
    print(f"   Clients: {config['num_clients']}")
    print(f"   Clusters: {config['num_clusters']}")
    print(f"   Rounds: {config['num_rounds']}")
    print(f"   Cycle length: {config['cycle_length']}")
    print()
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn_callback,
        num_clients=config['num_clients'],
        config=fl.server.ServerConfig(num_rounds=config['num_rounds']),
        strategy=strategy,
        client_resources=config['client_resources'],
        ray_init_args=config['ray_init_args']
    )
    
    # Save results
    print("\n💾 Saving results...")
    os.makedirs(config['results_dir'], exist_ok=True)
    
    results = {
        "accuracy": history.metrics_centralized,
        "loss_centralised": history.losses_centralized,
        "loss_distributed": history.losses_distributed,
        "config": config
    }
    
    output_file = os.path.join(
        config['results_dir'], 
        f"cifar10_resnet8_{config['clustering_method']}.json"
    )
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_file}")
    
    # Print final results
    if history.metrics_centralized:
        final_round, final_acc = history.metrics_centralized[-1]
        print(f"\n🎉 Final Results:")
        print(f"   Round: {final_round}")
        print(f"   Test Accuracy: {final_acc['accuracy']:.4f}")


if __name__ == "__main__":
    main()
