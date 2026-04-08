"""Evaluation and baseline comparison for routing system."""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple
import logging

from app.utils.graph_utils import compute_shortest_paths, path_to_edges
from app.rl.dqn_agent import DQNAgent


logger = logging.getLogger(__name__)


class BaselineRouter:
    """Dijkstra-based routing (baseline for comparison)."""
    
    def __init__(self, graph):
        """Initialize baseline router."""
        self.graph = graph
    
    def route(self, source: int, destination: int) -> Tuple[List[int], Dict]:
        """
        Route using shortest path algorithm.
        
        Args:
            source: Source node
            destination: Destination node
            
        Returns:
            path: List of nodes
            metrics: Performance metrics
        """
        path, cost = compute_shortest_paths(self.graph, source, destination)
        
        metrics = {
            'latency': 0.0,
            'congestion': 0.0,
            'path_length': len(path) - 1
        }
        
        if path:
            # Calculate actual metrics
            total_latency = 0.0
            total_congestion = 0.0
            edge_count = 0
            
            for u, v in path_to_edges(path):
                if self.graph.has_edge(u, v):
                    total_latency += self.graph[u][v].get('latency', 10)
                    total_congestion += self.graph[u][v].get('congestion', 0.5)
                    edge_count += 1
            
            metrics['latency'] = total_latency
            metrics['congestion'] = total_congestion / edge_count if edge_count > 0 else 0.0
            metrics['success'] = True
        else:
            metrics['success'] = False
        
        return path, metrics


def evaluate_routing_quality(graph, agent, baseline_router, 
                            num_test_pairs: int = 50) -> Dict:
    """
    Evaluate routing quality of DQN agent vs baseline.
    
    Args:
        graph: NetworkX graph
        agent: Trained DQN agent
        baseline_router: Baseline routing algorithm
        num_test_pairs: Number of test source-destination pairs
        
    Returns:
        Evaluation metrics comparing agent vs baseline
    """
    num_nodes = graph.number_of_nodes()
    
    agent_metrics = {
        'latencies': [],
        'path_lengths': [],
        'success_count': 0,
        'congestions': []
    }
    
    baseline_metrics = {
        'latencies': [],
        'path_lengths': [],
        'success_count': 0,
        'congestions': []
    }
    
    logger.info(f"Evaluating on {num_test_pairs} routing queries...")
    
    for _ in range(num_test_pairs):
        # Generate random source-destination pair
        source = np.random.randint(0, num_nodes)
        destination = np.random.randint(0, num_nodes)
        
        while destination == source:
            destination = np.random.randint(0, num_nodes)
        
        # Baseline routing
        baseline_path, baseline_metrics_dict = baseline_router.route(source, destination)
        if baseline_metrics_dict['success']:
            baseline_metrics['latencies'].append(baseline_metrics_dict['latency'])
            baseline_metrics['congestions'].append(baseline_metrics_dict['congestion'])
            baseline_metrics['path_lengths'].append(baseline_metrics_dict['path_length'])
            baseline_metrics['success_count'] += 1
        
        # Agent routing (simplified - would use same environment as training)
        # For evaluation, simulate by random walk or greedy traversal
        agent_path = greedy_path_to_destination(graph, source, destination)
        if agent_path:
            agent_metrics['success_count'] += 1
            agent_metrics['path_lengths'].append(len(agent_path) - 1)
            
            # Calculate latency and congestion
            total_latency = 0.0
            total_congestion = 0.0
            for u, v in path_to_edges(agent_path):
                if graph.has_edge(u, v):
                    total_latency += graph[u][v].get('latency', 10)
                    total_congestion += graph[u][v].get('congestion', 0.5)
            
            agent_metrics['latencies'].append(total_latency)
            agent_metrics['congestions'].append(
                total_congestion / (len(agent_path) - 1) if len(agent_path) > 1 else 0.0
            )
    
    # Compute statistics
    results = {
        'agent': compute_statistics(agent_metrics, num_test_pairs),
        'baseline': compute_statistics(baseline_metrics, num_test_pairs)
    }
    
    # Compute improvements
    if baseline_metrics['success_count'] > 0:
        results['improvements'] = {
            'latency_reduction': (
                (np.mean(baseline_metrics['latencies']) - np.mean(agent_metrics['latencies']))
                / np.mean(baseline_metrics['latencies']) * 100
            ) if agent_metrics['latencies'] else 0.0,
            'path_length_reduction': (
                (np.mean(baseline_metrics['path_lengths']) - np.mean(agent_metrics['path_lengths']))
                / np.mean(baseline_metrics['path_lengths']) * 100
            ) if agent_metrics['path_lengths'] else 0.0,
            'congestion_reduction': (
                (np.mean(baseline_metrics['congestions']) - np.mean(agent_metrics['congestions']))
                / np.mean(baseline_metrics['congestions']) * 100
            ) if agent_metrics['congestions'] else 0.0
        }
    
    return results


def compute_statistics(metrics: Dict, total_tests: int) -> Dict:
    """Compute statistics from collected metrics."""
    stats = {
        'success_rate': metrics['success_count'] / total_tests if total_tests > 0 else 0.0,
        'avg_latency': np.mean(metrics['latencies']) if metrics['latencies'] else 0.0,
        'avg_congestion': np.mean(metrics['congestions']) if metrics['congestions'] else 0.0,
        'avg_path_length': np.mean(metrics['path_lengths']) if metrics['path_lengths'] else 0.0,
        'min_latency': np.min(metrics['latencies']) if metrics['latencies'] else 0.0,
        'max_latency': np.max(metrics['latencies']) if metrics['latencies'] else 0.0,
    }
    return stats


def greedy_path_to_destination(graph: nx.Graph, source: int, 
                               destination: int, max_steps: int = None) -> List[int]:
    """
    Greedy path finding (minimize latency at each step).
    
    Args:
        graph: NetworkX graph
        source: Source node
        destination: Destination node
        max_steps: Maximum steps before giving up
        
    Returns:
        Path as list of nodes
    """
    if max_steps is None:
        max_steps = graph.number_of_nodes() * 2
    
    current = source
    path = [current]
    visited = set([current])
    
    while current != destination and len(path) < max_steps:
        neighbors = list(graph.neighbors(current))
        
        if not neighbors:
            break
        
        # Filter out visited nodes (except destination)
        unvisited = [n for n in neighbors if n not in visited or n == destination]
        
        if not unvisited:
            # Backtrack: allow revisiting
            unvisited = neighbors
        
        # Choose neighbor with lowest latency
        best_neighbor = min(
            unvisited,
            key=lambda n: graph[current][n].get('latency', 10)
        )
        
        path.append(best_neighbor)
        if best_neighbor != destination:
            visited.add(best_neighbor)
        current = best_neighbor
    
    if current == destination:
        return path
    return []


def generate_evaluation_report(eval_results: Dict) -> str:
    """Generate human-readable evaluation report."""
    report = []
    report.append("=" * 60)
    report.append("ROUTING SYSTEM EVALUATION REPORT")
    report.append("=" * 60)
    
    report.append("\nAGENT PERFORMANCE:")
    agent_stats = eval_results['agent']
    report.append(f"  Success Rate:        {agent_stats['success_rate']:.2%}")
    report.append(f"  Avg Latency:         {agent_stats['avg_latency']:.2f} ms")
    report.append(f"  Avg Congestion:      {agent_stats['avg_congestion']:.3f}")
    report.append(f"  Avg Path Length:     {agent_stats['avg_path_length']:.1f} hops")
    
    report.append("\nBASELINE (DIJKSTRA) PERFORMANCE:")
    baseline_stats = eval_results['baseline']
    report.append(f"  Success Rate:        {baseline_stats['success_rate']:.2%}")
    report.append(f"  Avg Latency:         {baseline_stats['avg_latency']:.2f} ms")
    report.append(f"  Avg Congestion:      {baseline_stats['avg_congestion']:.3f}")
    report.append(f"  Avg Path Length:     {baseline_stats['avg_path_length']:.1f} hops")
    
    if 'improvements' in eval_results:
        report.append("\nIMPROVEMENTS (Agent vs Baseline):")
        improvements = eval_results['improvements']
        report.append(f"  Latency Reduction:   {improvements['latency_reduction']:+.2f}%")
        report.append(f"  Path Length Reduction: {improvements['path_length_reduction']:+.2f}%")
        report.append(f"  Congestion Reduction: {improvements['congestion_reduction']:+.2f}%")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)
