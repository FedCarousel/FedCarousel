"""
Client clustering strategies for federated learning.

Supports:
- K-means clustering based on data distribution
- Random clustering (baseline)
"""
import numpy as np
from typing import Dict, Tuple
from sklearn.cluster import KMeans


def compute_client_signatures(client_datasets: list, num_classes: int) -> np.ndarray:
    """
    Compute data distribution signature for each client.
    
    Args:
        client_datasets: List of datasets for each client
        num_classes: Total number of classes
        
    Returns:
        Array of shape (num_clients, num_classes) with normalized class distributions
    """
    num_clients = len(client_datasets)
    signatures = np.zeros((num_clients, num_classes))
    
    for client_idx, client_data in enumerate(client_datasets):
        for _, label in client_data:
            signatures[client_idx][label] += 1
    
    # Normalize signatures
    for i in range(num_clients):
        total = signatures[i].sum()
        if total > 0:
            signatures[i] /= total
    
    return signatures


def kmeans_clustering(client_signatures: np.ndarray, 
                     num_clusters: int, 
                     seed: int = 42) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Perform K-means clustering on client data distributions.
    
    Args:
        client_signatures: Array of shape (num_clients, num_classes)
        num_clusters: Number of clusters to create
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - cluster_assignments: Array mapping client_idx to cluster_id
        - client_to_cluster: Dict mapping client_id to cluster_id
    """
    print(f"\n🎯 Applying K-means clustering for {num_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init=20)
    cluster_assignments = kmeans.fit_predict(client_signatures)
    
    # Create mapping dictionary
    client_to_cluster = {
        client_id: int(cluster_assignments[client_id]) 
        for client_id in range(len(cluster_assignments))
    }
    
    # Print cluster statistics
    print(f"\n📍 Clustering Results:")
    for cluster_id in range(num_clusters):
        clients_in_cluster = np.where(cluster_assignments == cluster_id)[0]
        print(f"   Cluster {cluster_id}: {len(clients_in_cluster)} clients")
        print(f"      Client IDs: {clients_in_cluster.tolist()}")
        
        if len(clients_in_cluster) > 0:
            cluster_signature = client_signatures[clients_in_cluster].mean(axis=0)
            top_classes = np.argsort(cluster_signature)[-5:][::-1]
            print(f"      Top 5 classes: {top_classes.tolist()} "
                  f"({cluster_signature[top_classes]})")
    
    return cluster_assignments, client_to_cluster


def random_clustering(num_clients: int, 
                     num_clusters: int, 
                     seed: int = 42) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Perform random clustering (baseline).
    
    Args:
        num_clients: Number of clients
        num_clusters: Number of clusters to create
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of:
        - cluster_assignments: Array mapping client_idx to cluster_id
        - client_to_cluster: Dict mapping client_id to cluster_id
    """
    print(f"\n🎲 Random clustering into {num_clusters} clusters...")
    
    np.random.seed(seed)
    cluster_assignments = np.random.randint(0, num_clusters, size=num_clients)
    
    # Create mapping dictionary
    client_to_cluster = {
        client_id: int(cluster_assignments[client_id]) 
        for client_id in range(num_clients)
    }
    
    # Print cluster statistics
    print(f"\n📍 Clustering Results:")
    for cluster_id in range(num_clusters):
        clients_in_cluster = np.where(cluster_assignments == cluster_id)[0]
        print(f"   Cluster {cluster_id}: {len(clients_in_cluster)} clients")
        print(f"      Client IDs: {clients_in_cluster.tolist()}")
    
    return cluster_assignments, client_to_cluster


def analyze_clustering_quality(client_signatures: np.ndarray, 
                               cluster_assignments: np.ndarray,
                               num_clusters: int) -> dict:
    """
    Analyze the quality of clustering.
    
    Args:
        client_signatures: Array of shape (num_clients, num_classes)
        cluster_assignments: Array mapping client_idx to cluster_id
        num_clusters: Number of clusters
        
    Returns:
        Dictionary with clustering metrics
    """
    metrics = {
        'intra_cluster_variance': [],
        'inter_cluster_distance': []
    }
    
    # Compute cluster centroids
    centroids = []
    for cluster_id in range(num_clusters):
        clients_in_cluster = np.where(cluster_assignments == cluster_id)[0]
        if len(clients_in_cluster) > 0:
            centroid = client_signatures[clients_in_cluster].mean(axis=0)
            centroids.append(centroid)
            
            # Intra-cluster variance
            variance = np.var(client_signatures[clients_in_cluster], axis=0).mean()
            metrics['intra_cluster_variance'].append(variance)
    
    # Inter-cluster distances
    centroids = np.array(centroids)
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distance = np.linalg.norm(centroids[i] - centroids[j])
            metrics['inter_cluster_distance'].append(distance)
    
    metrics['avg_intra_cluster_variance'] = np.mean(metrics['intra_cluster_variance'])
    metrics['avg_inter_cluster_distance'] = np.mean(metrics['inter_cluster_distance'])
    
    print(f"\n📊 Clustering Quality Metrics:")
    print(f"   Average intra-cluster variance: {metrics['avg_intra_cluster_variance']:.4f}")
    print(f"   Average inter-cluster distance: {metrics['avg_inter_cluster_distance']:.4f}")
    
    return metrics
