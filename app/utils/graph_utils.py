"""Graph utilities for network topology management."""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, List
import pandas as pd


def generate_synthetic_network(num_nodes: int = 20, 
                               edge_probability: float = 0.3,
                               random_seed: int = 42) -> Tuple[np.ndarray, nx.Graph]:
    """
    Generate a synthetic network graph.
    
    Args:
        num_nodes: Number of nodes in the network
        edge_probability: Probability of edge creation (Erdos-Renyi model)
        random_seed: Random seed for reproducibility
        
    Returns:
        adjacency_matrix: NxN adjacency matrix
        graph: NetworkX graph object
    """
    np.random.seed(random_seed)
    
    # Create Erdos-Renyi random graph
    graph = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=random_seed)
    
    # Ensure graph is connected
    if not nx.is_connected(graph):
        # Add edges to make it connected
        components = list(nx.connected_components(graph))
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i + 1])[0]
            graph.add_edge(node1, node2)
    
    # Add edge weights (latency in ms)
    for u, v in graph.edges():
        graph[u][v]['latency'] = np.random.uniform(1, 50)  # 1-50ms
        graph[u][v]['bandwidth'] = np.random.uniform(1, 100)  # 1-100 Mbps
        graph[u][v]['congestion'] = np.random.uniform(0, 1)  # 0-1 (0=free, 1=full)
    
    # Add node features (capacity, processing time)
    for node in graph.nodes():
        graph.nodes[node]['capacity'] = np.random.uniform(50, 200)
        graph.nodes[node]['processing_time'] = np.random.uniform(0.1, 2.0)
    
    # Convert to adjacency matrix
    adjacency_matrix = nx.to_numpy_array(graph)
    
    return adjacency_matrix, graph


def load_network_from_csv(filepath: str) -> Tuple[np.ndarray, nx.Graph, Dict]:
    """
    Load network topology from CSV file.
    
    Expected CSV format:
    - edges.csv: source,target,latency,bandwidth,congestion
    - nodes.csv (optional): node_id,capacity,processing_time
    
    Args:
        filepath: Path to CSV file (or directory containing CSVs)
        
    Returns:
        adjacency_matrix: NxN adjacency matrix
        graph: NetworkX graph object
        metadata: Dictionary containing network metadata
    """
    graph = nx.DiGraph()
    metadata = {"source": filepath}
    
    try:
        # Load edges
        if filepath.endswith('.csv'):
            edges_df = pd.read_csv(filepath)
        else:
            edges_df = pd.read_csv(f"{filepath}/edges.csv")
        
        for _, row in edges_df.iterrows():
            source = int(row['source'])
            target = int(row['target'])
            
            edge_attrs = {
                'latency': float(row.get('latency', 10)),
                'bandwidth': float(row.get('bandwidth', 50)),
                'congestion': float(row.get('congestion', 0.5))
            }
            graph.add_edge(source, target, **edge_attrs)
        
        # Load nodes if available
        try:
            nodes_df = pd.read_csv(f"{filepath}/nodes.csv")
            for _, row in nodes_df.iterrows():
                node_id = int(row['node_id'])
                node_attrs = {
                    'capacity': float(row.get('capacity', 100)),
                    'processing_time': float(row.get('processing_time', 1.0))
                }
                graph.nodes[node_id].update(node_attrs)
        except FileNotFoundError:
            # Use default node attributes
            for node in graph.nodes():
                graph.nodes[node]['capacity'] = 100
                graph.nodes[node]['processing_time'] = 1.0
        
        metadata['num_nodes'] = graph.number_of_nodes()
        metadata['num_edges'] = graph.number_of_edges()
        
        # Convert to undirected for adjacency matrix
        undirected = graph.to_undirected()
        adjacency_matrix = nx.to_numpy_array(undirected, nodelist=sorted(graph.nodes()))
        
        return adjacency_matrix, graph, metadata
    
    except Exception as e:
        raise ValueError(f"Failed to load network from CSV: {str(e)}")


def get_edge_features(graph: nx.Graph, u: int, v: int) -> np.ndarray:
    """
    Extract edge features as a feature vector.
    
    Args:
        graph: NetworkX graph
        u: Source node
        v: Target node
        
    Returns:
        Feature vector: [latency, bandwidth, congestion]
    """
    edge_data = graph[u][v]
    return np.array([
        edge_data.get('latency', 10),
        edge_data.get('bandwidth', 50),
        edge_data.get('congestion', 0.5)
    ])


def compute_shortest_paths(graph: nx.Graph, source: int, target: int) -> Tuple[List[int], float]:
    """
    Compute shortest path using Dijkstra algorithm (baseline).
    
    Args:
        graph: NetworkX graph
        source: Source node
        target: Target node
        
    Returns:
        path: List of nodes in path
        total_cost: Total path cost
    """
    try:
        # Use latency as edge weight
        for u, v in graph.edges():
            graph[u][v]['weight'] = graph[u][v].get('latency', 10)
        
        path = nx.shortest_path(graph, source, target, weight='weight')
        total_cost = nx.shortest_path_length(graph, source, target, weight='weight')
        
        return path, total_cost
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [], float('inf')


def path_to_edges(path: List[int]) -> List[Tuple[int, int]]:
    """Convert a path (list of nodes) to a list of edges."""
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]
