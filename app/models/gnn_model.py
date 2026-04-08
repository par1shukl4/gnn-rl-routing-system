"""Graph Neural Network model for network routing."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import numpy as np
from typing import Tuple


class GraphNeuralNetwork(nn.Module):
    """
    Graph Convolutional Network for network routing.
    
    Encodes network topology and computes node embeddings.
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, output_dim: int = 32):
        """
        Initialize GNN.
        
        Args:
            input_dim: Input feature dimension per node
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
        """
        super(GraphNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Graph convolutional layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        # Activation and normalization
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge indices (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim) - optional
            
        Returns:
            Node embeddings (num_nodes, output_dim)
        """
        # First convolution
        x = self.conv1(x, edge_index)
        x = self.batch_norm1(x)
        x = self.relu(x)
        
        # Second convolution
        x = self.conv2(x, edge_index)
        x = self.batch_norm2(x)
        x = self.relu(x)
        
        # Third convolution
        x = self.conv3(x, edge_index)
        
        return x


class EdgeFeatureProcessor(nn.Module):
    """Process edge features for routing decisions."""
    
    def __init__(self, edge_feature_dim: int = 3, output_dim: int = 16):
        """
        Initialize edge feature processor.
        
        Args:
            edge_feature_dim: Dimension of edge features (latency, bandwidth, congestion)
            output_dim: Output dimension for processed features
        """
        super(EdgeFeatureProcessor, self).__init__()
        
        self.edge_feature_dim = edge_feature_dim
        self.output_dim = output_dim
        
        # MLP for edge feature processing
        self.mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Process edge features.
        
        Args:
            edge_features: Edge feature matrix (num_edges, edge_feature_dim)
            
        Returns:
            Processed edge features (num_edges, output_dim)
        """
        return self.mlp(edge_features)


def create_graph_data(adjacency_matrix: np.ndarray, 
                     edge_weights: np.ndarray = None) -> Data:
    """
    Convert adjacency matrix to PyTorch Geometric Data object.
    
    Args:
        adjacency_matrix: NxN adjacency matrix
        edge_weights: Optional edge weight matrix (latency, bandwidth, congestion)
        
    Returns:
        PyTorch Geometric Data object
    """
    # Create edge index from adjacency matrix
    edge_index = np.argwhere(adjacency_matrix > 0).T
    edge_index = torch.LongTensor(edge_index)
    
    # Node features (default: node degree)
    num_nodes = adjacency_matrix.shape[0]
    node_degree = adjacency_matrix.sum(axis=1)
    x = torch.FloatTensor(node_degree).unsqueeze(1)
    
    # Edge features
    edge_attr = None
    if edge_weights is not None:
        edge_attr = torch.FloatTensor(edge_weights)
    
    # Create graph data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data


def get_path_embedding(node_embeddings: torch.Tensor, path: list) -> torch.Tensor:
    """
    Get combined embedding for a path.
    
    Args:
        node_embeddings: Node embeddings from GNN (num_nodes, embedding_dim)
        path: List of node indices
        
    Returns:
        Path embedding (concatenated or aggregated node embeddings)
    """
    path_indices = torch.LongTensor(path)
    path_embeddings = node_embeddings[path_indices]
    
    # Aggregate embeddings (mean pooling)
    aggregated = path_embeddings.mean(dim=0)
    
    return aggregated
