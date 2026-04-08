"""Training pipeline for GNN + DQN routing system."""

import torch
import numpy as np
import logging
from typing import Dict, List, Tuple
import json
from pathlib import Path

from app.utils.graph_utils import generate_synthetic_network, load_network_from_csv, compute_shortest_paths
from app.models.gnn_model import GraphNeuralNetwork, create_graph_data
from app.rl.dqn_agent import DQNAgent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RoutingEnvironment:
    """Environment for RL-based routing."""
    
    def __init__(self, graph, gnn_model, num_nodes: int):
        """
        Initialize routing environment.
        
        Args:
            graph: NetworkX graph
            gnn_model: GNN model
            num_nodes: Number of nodes
        """
        self.graph = graph
        self.gnn_model = gnn_model
        self.num_nodes = num_nodes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def get_state(self, current_node: int, destination: int, path_history: List[int]) -> np.ndarray:
        """
        Get state representation for current routing situation.
        
        Args:
            current_node: Current position in network
            destination: Target node
            path_history: Nodes visited so far
            
        Returns:
            State vector
        """
        # State: [current_node, destination, path_length, visited_count, avg_congestion]
        path_length = len(path_history)
        visited_count = len(set(path_history))
        
        # Calculate average congestion on visited edges
        avg_congestion = 0.0
        if len(path_history) > 1:
            congestions = []
            for i in range(len(path_history) - 1):
                u, v = path_history[i], path_history[i+1]
                if self.graph.has_edge(u, v):
                    congestions.append(self.graph[u][v].get('congestion', 0.5))
            avg_congestion = np.mean(congestions) if congestions else 0.0
        
        # Get node embeddings
        adjacency = np.array(torch.tensor(
            [[1 if self.graph.has_edge(i, j) else 0 
              for j in range(self.num_nodes)] 
             for i in range(self.num_nodes)]
        ).float())
        
        graph_data = create_graph_data(adjacency)
        with torch.no_grad():
            node_embeddings = self.gnn_model(graph_data.x.to(self.device), 
                                            graph_data.edge_index.to(self.device))
        
        # Combine features
        current_embedding = node_embeddings[current_node].cpu().numpy()
        dest_embedding = node_embeddings[destination].cpu().numpy()
        
        state = np.concatenate([
            current_embedding,
            dest_embedding,
            np.array([path_length / 20, visited_count / self.num_nodes, avg_congestion])
        ])
        
        return state
    
    def get_next_actions(self, current_node: int, path_history: List[int]) -> List[int]:
        """Get available next nodes (neighbors)."""
        neighbors = list(self.graph.neighbors(current_node))
        # Avoid immediate backtracking
        if len(path_history) > 1:
            neighbors = [n for n in neighbors if n != path_history[-2]]
        return neighbors if neighbors else list(self.graph.neighbors(current_node))
    
    def compute_reward(self, current_node: int, next_node: int, destination: int, 
                      path_history: List[int], done: bool) -> float:
        """
        Compute reward for transition.
        
        Args:
            current_node: Current node
            next_node: Next node
            destination: Target node
            path_history: Path taken so far
            done: Whether episode is done
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        if done:
            if next_node == destination:
                # Success: reached destination
                reward = 100.0 - len(path_history) * 2  # Incentivize short paths
            else:
                # Failed: path too long
                reward = -50.0
        else:
            # Step reward
            if self.graph.has_edge(current_node, next_node):
                edge = self.graph[current_node][next_node]
                latency = edge.get('latency', 10)
                congestion = edge.get('congestion', 0.5)
                
                # Reward for low latency and congestion
                reward -= latency / 50  # Normalize to [-1, 0]
                reward -= congestion * 5  # Penalize congestion
                
                # Small reward for progress toward destination
                current_dist = len(list(self.graph.neighbors(current_node)))
                next_dist = len(list(self.graph.neighbors(next_node)))
                if next_dist > current_dist:
                    reward += 0.1
        
        return reward


def train_dqn_agent(graph, gnn_model, num_episodes: int = 100, 
                   num_training_pairs: int = 20, batch_size: int = 32) -> Dict:
    """
    Train DQN agent on network routing task.
    
    Args:
        graph: NetworkX graph
        gnn_model: GNN model
        num_episodes: Number of training episodes
        num_training_pairs: Number of source-destination pairs per episode
        batch_size: Training batch size
        
    Returns:
        Training metrics dictionary
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_nodes = graph.number_of_nodes()
    
    # Create environment
    env = RoutingEnvironment(graph, gnn_model, num_nodes)
    
    # Determine state and action dimensions
    # State: GNN embedding (32) + destination embedding (32) + metrics (3) = 67
    state_dim = 67
    # Action space: possible next hops (max = num_nodes)
    action_dim = num_nodes
    
    # Create DQN agent
    agent = DQNAgent(state_dim, action_dim, hidden_dim=128)
    
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'loss_history': [],
        'success_rate': []
    }
    
    logger.info(f"Starting DQN training on {num_nodes}-node network")
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    for episode in range(num_episodes):
        episode_reward = 0.0
        successful_routes = 0
        
        for _ in range(num_training_pairs):
            # Random source and destination
            source = np.random.randint(0, num_nodes)
            destination = np.random.randint(0, num_nodes)
            
            while destination == source:
                destination = np.random.randint(0, num_nodes)
            
            # Run episode
            current_node = source
            path_history = [current_node]
            done = False
            max_steps = num_nodes * 2
            
            while not done and len(path_history) < max_steps:
                # Get state
                state = env.get_state(current_node, destination, path_history)
                
                # Select action
                available_actions = env.get_next_actions(current_node, path_history)
                action = agent.select_action(state, training=True)
                
                # Ensure action is valid
                if action >= len(available_actions):
                    action = action % len(available_actions)
                
                next_node = available_actions[action]
                
                # Take step
                done = (next_node == destination) or (len(path_history) >= max_steps)
                reward = env.compute_reward(current_node, next_node, destination, 
                                           path_history, done)
                
                # Get next state
                next_state = env.get_state(next_node, destination, path_history + [next_node])
                
                # Store transition
                agent.store_transition(state, action, reward, next_state, done)
                
                # Train step
                loss = agent.train_step(batch_size)
                if loss is not None:
                    metrics['loss_history'].append(loss)
                
                episode_reward += reward
                current_node = next_node
                path_history.append(current_node)
            
            # Check success
            if current_node == destination:
                successful_routes += 1
        
        # Update target network
        agent.update_target_network()
        
        # Record metrics
        metrics['episode_rewards'].append(episode_reward / num_training_pairs)
        metrics['episode_lengths'].append(len(path_history))
        success_rate = successful_routes / num_training_pairs
        metrics['success_rate'].append(success_rate)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-10:])
            avg_success = np.mean(metrics['success_rate'][-10:])
            logger.info(f"Episode {episode+1}/{num_episodes} - "
                       f"Avg Reward: {avg_reward:.2f}, "
                       f"Success Rate: {avg_success:.2%}, "
                       f"Epsilon: {agent.epsilon:.3f}")
    
    logger.info("Training completed!")
    
    return agent, metrics


def save_training_state(agent, metrics, model_dir: str = "models"):
    """Save trained agent and metrics."""
    Path(model_dir).mkdir(exist_ok=True)
    
    # Save agent
    agent.save(f"{model_dir}/dqn_agent.pt")
    
    # Save metrics
    with open(f"{model_dir}/training_metrics.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = {
            k: v.tolist() if isinstance(v, np.ndarray) else v 
            for k, v in metrics.items()
        }
        json.dump(metrics_json, f, indent=2)
    
    logger.info(f"Model saved to {model_dir}/")
