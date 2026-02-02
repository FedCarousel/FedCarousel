"""
Layer-wise federated averaging strategy.

Implements:
- Global aggregation: Standard FedAvg across all parameters
- Partial aggregation: Layer-specific FedAvg per cluster
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import flwr as fl
from flwr.common import (
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    FitRes,
    Parameters,
    Scalar
)
from flwr.server.client_proxy import ClientProxy

from ..models.layer_mapping import get_layer_mapping


class LayerWiseFedAvg(fl.server.strategy.FedAvg):
    """
    Federated Averaging with layer-wise aggregation support.
    
    Features:
    - Alternates between global and partial aggregation modes
    - Maintains accumulated global parameters across rounds
    - Cluster-specific layer updates during partial rounds
    """
    
    def __init__(self, 
                 num_clusters: int,
                 model_name: str,
                 cycle_length: int,
                 global_rounds: int,
                 **kwargs):
        """
        Args:
            num_clusters: Number of client clusters
            model_name: Model architecture name ('resnet8' or 'resnet18')
            cycle_length: Number of rounds in one training cycle
            global_rounds: Number of global rounds at cycle start
            **kwargs: Arguments passed to FedAvg
        """
        super().__init__(**kwargs)
        self.num_clusters = num_clusters
        self.model_name = model_name
        self.cycle_length = cycle_length
        self.global_rounds = global_rounds
        
        # Store current global parameters
        self.current_global_params = None
        
        # Pre-compute layer-to-parameter mapping for efficiency
        self.layer_mapping = get_layer_mapping(model_name)
        self.layer_to_param_indices = self._build_layer_param_mapping()
        
    def _build_layer_param_mapping(self) -> Dict[int, List[int]]:
        """
        Build mapping from layer IDs to parameter indices.
        
        This is done once to avoid recreating the model at each aggregation.
        
        Returns:
            Dictionary mapping layer_id to list of parameter indices
        """
        from ..models.resnet import create_resnet8, create_resnet18
        
        # Create temporary model to get parameter names
        if self.model_name == 'resnet8':
            temp_model = create_resnet8()
        elif self.model_name == 'resnet18':
            # Dummy num_classes, only need structure
            temp_model = create_resnet18(num_classes=100)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        mapping = {}
        for layer_id, layer_names in self.layer_mapping.items():
            indices = []
            param_idx = 0
            
            for name, param in temp_model.named_parameters():
                if any(ln in name for ln in layer_names):
                    indices.append(param_idx)
                param_idx += 1
            
            mapping[layer_id] = indices
        
        del temp_model
        print(f"✅ Layer-to-parameter mapping built: {len(mapping)} layers")
        
        return mapping
        
    def get_training_mode(self, round_num: int) -> str:
        """
        Determine training mode for current round.
        
        Args:
            round_num: Current round number (1-indexed)
            
        Returns:
            'global' or 'partial'
        """
        position_in_cycle = (round_num - 1) % self.cycle_length
        return "global" if position_in_cycle < self.global_rounds else "partial"
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates from clients.
        
        Args:
            server_round: Current round number
            results: List of (client, fit_result) tuples
            failures: List of failed fits
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        training_mode = self.get_training_mode(server_round)
        
        if training_mode == "global":
            aggregated_params, metrics = self._aggregate_global(results)
        else:
            aggregated_params, metrics = self._aggregate_partial(
                server_round, results
            )
        
        # Update global parameters after EVERY round
        self.current_global_params = [
            param.copy() for param in parameters_to_ndarrays(aggregated_params)
        ]
        
        return aggregated_params, metrics
    
    def _aggregate_global(
        self, 
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        """
        Standard FedAvg aggregation across all parameters.
        
        Args:
            results: List of client results
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        base_params = parameters_to_ndarrays(results[0][1].parameters)
        aggregated_params = [
            np.zeros_like(p, dtype=np.float64) 
            for p in base_params
        ]
        
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        
        # Weighted average across all parameters
        for param_idx in range(len(aggregated_params)):
            for _, fit_res in results:
                client_params = parameters_to_ndarrays(fit_res.parameters)
                weight = fit_res.num_examples / total_examples
                aggregated_params[param_idx] += client_params[param_idx] * weight
        
        print(f"   ✅ Global aggregation complete ({len(results)} clients)")
        return ndarrays_to_parameters(aggregated_params), {}
    
    def _aggregate_partial(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> Tuple[Parameters, Dict[str, Scalar]]:
        """
        Layer-wise aggregation per cluster.
        
        Strategy:
        1. Start with current global parameters
        2. For each (cluster, layer) pair:
           - Aggregate updates from clients in that cluster
           - Update only the parameters of that layer
        3. Return updated global parameters
        
        Args:
            server_round: Current round number
            results: List of client results
            
        Returns:
            Tuple of (aggregated_parameters, metrics)
        """
        # Group results by (cluster_id, layer_id)
        cluster_layer_results: Dict[Tuple[int, int], List] = {}
        
        for client_proxy, fit_res in results:
            cluster_id = fit_res.metrics.get("cluster_id")
            layer_trained = fit_res.metrics.get("layer_trained")
            send_partial = fit_res.metrics.get("send_partial", False)
            
            if (layer_trained is not None and 
                layer_trained != -1 and 
                cluster_id is not None and
                send_partial):
                key = (cluster_id, layer_trained)
                if key not in cluster_layer_results:
                    cluster_layer_results[key] = []
                cluster_layer_results[key].append((client_proxy, fit_res))
        
        # Log active clusters
        print(f"   📊 Active clusters in round {server_round}:")
        for (cluster_id, layer_id), clients in cluster_layer_results.items():
            print(f"      Cluster {cluster_id} → Layer {layer_id}: "
                  f"{len(clients)} clients")
        
        # Start with current global parameters
        if self.current_global_params is None:
            base_params = parameters_to_ndarrays(results[0][1].parameters)
            aggregated_params = [p.copy() for p in base_params]
        else:
            aggregated_params = [p.copy() for p in self.current_global_params]
        
        # Update each layer with its cluster's aggregation
        for (cluster_id, layer_id), clients in cluster_layer_results.items():
            # Get parameter indices for this layer
            layer_param_indices = self.layer_to_param_indices[layer_id]
            
            # Aggregate across clients in this cluster for this layer
            total_examples = sum(fit_res.num_examples for _, fit_res in clients)
            
            try:
                for local_idx, global_idx in enumerate(layer_param_indices):
                    weighted_sum = np.zeros_like(
                        aggregated_params[global_idx], 
                        dtype=np.float64
                    )
                    
                    for _, fit_res in clients:
                        client_params = parameters_to_ndarrays(fit_res.parameters)
                        
                        # ✅ VÉRIFICATION CRITIQUE
                        if local_idx >= len(client_params):
                            raise RuntimeError(
                                f"Index mismatch: layer {layer_id} expects "
                                f"{len(layer_param_indices)} params, "
                                f"client sent {len(client_params)}"
                            )
                        
                        weight = fit_res.num_examples / total_examples
                        weighted_sum += client_params[local_idx] * weight
                    
                    aggregated_params[global_idx] = weighted_sum
                    
            except Exception as e:
                print(f"   ⚠️ ERROR in aggregation for cluster {cluster_id}, "
                    f"layer {layer_id}: {e}")
                print(f"   Skipping this layer update (using previous values)")
                continue
        
        
        return ndarrays_to_parameters(aggregated_params), {}
   
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        return super().aggregate_evaluate(server_round, results, failures)
