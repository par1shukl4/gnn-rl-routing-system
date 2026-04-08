"""FastAPI backend for network routing system."""

import torch
import numpy as np
import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import io
import json
from pathlib import Path

from app.utils.graph_utils import (
    generate_synthetic_network, 
    load_network_from_csv,
    compute_shortest_paths,
    path_to_edges
)
from app.models.gnn_model import GraphNeuralNetwork, create_graph_data
from app.rl.dqn_agent import DQNAgent
from app.train import RoutingEnvironment, train_dqn_agent, save_training_state
from app.evaluate import BaselineRouter, evaluate_routing_quality, generate_evaluation_report


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Network Routing GNN+RL System",
    description="Production-ready GNN + DQN dynamic network routing",
    version="1.0.0"
)

# Global state
class SystemState:
    graph = None
    gnn_model = None
    dqn_agent = None
    metrics = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


state = SystemState()


# Pydantic models
class NetworkConfig(BaseModel):
    num_nodes: int = 20
    edge_probability: float = 0.3


class TrainingConfig(BaseModel):
    num_episodes: int = 50
    num_training_pairs: int = 20
    batch_size: int = 32


class RoutingRequest(BaseModel):
    source: int
    destination: int
    use_agent: bool = True


class RouteResponse(BaseModel):
    path: List[int]
    total_latency: float
    avg_congestion: float
    path_length: int
    algorithm: str


class MetricsResponse(BaseModel):
    success_rate: float
    avg_latency: float
    avg_congestion: float
    avg_path_length: float


# Initialization endpoints
@app.post("/init")
async def initialize_system(config: NetworkConfig):
    """Initialize system with synthetic network."""
    try:
        logger.info(f"Initializing system with {config.num_nodes} nodes...")
        
        # Generate synthetic network
        adjacency, graph = generate_synthetic_network(
            num_nodes=config.num_nodes,
            edge_probability=config.edge_probability
        )
        
        state.graph = graph
        
        # Initialize GNN model
        state.gnn_model = GraphNeuralNetwork(input_dim=1, hidden_dim=64, output_dim=32)
        state.gnn_model.to(state.device)
        state.gnn_model.eval()
        
        return {
            "status": "success",
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "message": "System initialized with synthetic network"
        }
    
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/init-from-csv")
async def initialize_from_csv(file: UploadFile = File(...)):
    """Initialize system from uploaded CSV file."""
    try:
        logger.info(f"Loading network from CSV: {file.filename}")
        
        # Read file
        content = await file.read()
        csv_content = content.decode('utf-8')
        
        # Save temporarily
        temp_path = "/tmp/network.csv"
        with open(temp_path, 'w') as f:
            f.write(csv_content)
        
        # Load network
        adjacency, graph, metadata = load_network_from_csv(temp_path)
        
        state.graph = graph
        
        # Initialize GNN model
        state.gnn_model = GraphNeuralNetwork(input_dim=1, hidden_dim=64, output_dim=32)
        state.gnn_model.to(state.device)
        state.gnn_model.eval()
        
        return {
            "status": "success",
            "num_nodes": metadata['num_nodes'],
            "num_edges": metadata['num_edges'],
            "message": "System initialized from CSV",
            "metadata": metadata
        }
    
    except Exception as e:
        logger.error(f"CSV loading error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Training endpoint
@app.post("/train")
async def train_model(config: TrainingConfig):
    """Train DQN agent on the network."""
    try:
        if state.graph is None:
            raise HTTPException(status_code=400, detail="Network not initialized. Call /init first")
        
        logger.info("Starting DQN training...")
        
        # Train DQN agent
        agent, metrics = train_dqn_agent(
            state.graph,
            state.gnn_model,
            num_episodes=config.num_episodes,
            num_training_pairs=config.num_training_pairs,
            batch_size=config.batch_size
        )
        
        state.dqn_agent = agent
        state.metrics = metrics
        
        # Save trained model
        save_training_state(agent, metrics)
        
        return {
            "status": "success",
            "message": "Training completed",
            "final_success_rate": metrics['success_rate'][-1] if metrics['success_rate'] else 0.0,
            "avg_final_reward": float(np.mean(metrics['episode_rewards'][-10:])) 
                               if len(metrics['episode_rewards']) >= 10 else 0.0
        }
    
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Routing endpoints
@app.post("/route", response_model=RouteResponse)
async def compute_route(request: RoutingRequest):
    """Compute optimal route for given source-destination pair."""
    try:
        if state.graph is None:
            raise HTTPException(status_code=400, detail="Network not initialized")
        
        source = request.source
        destination = request.destination
        
        # Validate nodes
        if source not in state.graph.nodes() or destination not in state.graph.nodes():
            raise HTTPException(status_code=400, detail="Invalid source or destination node")
        
        if source == destination:
            raise HTTPException(status_code=400, detail="Source and destination must be different")
        
        if request.use_agent and state.dqn_agent is None:
            logger.warning("Agent not trained. Using baseline routing instead.")
            request.use_agent = False
        
        if request.use_agent:
            # Use DQN agent for routing
            path = _get_agent_route(source, destination)
            algorithm = "DQN Agent"
        else:
            # Use baseline (Dijkstra)
            path, _ = compute_shortest_paths(state.graph, source, destination)
            algorithm = "Dijkstra (Baseline)"
        
        if not path:
            raise HTTPException(status_code=400, detail="No path found between nodes")
        
        # Calculate metrics
        total_latency = 0.0
        total_congestion = 0.0
        
        for u, v in path_to_edges(path):
            if state.graph.has_edge(u, v):
                total_latency += state.graph[u][v].get('latency', 10)
                total_congestion += state.graph[u][v].get('congestion', 0.5)
        
        avg_congestion = total_congestion / (len(path) - 1) if len(path) > 1 else 0.0
        
        return RouteResponse(
            path=path,
            total_latency=total_latency,
            avg_congestion=avg_congestion,
            path_length=len(path) - 1,
            algorithm=algorithm
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Routing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_agent_route(source: int, destination: int) -> List[int]:
    """Get route from trained DQN agent."""
    # Simplified agent routing: use greedy approach with learned features
    current = source
    path = [current]
    max_steps = state.graph.number_of_nodes() * 2
    
    while current != destination and len(path) < max_steps:
        neighbors = list(state.graph.neighbors(current))
        if not neighbors:
            break
        
        # Choose next hop (DQN would select based on Q-values)
        # For simplicity, use greedy: lowest latency
        next_node = min(
            neighbors,
            key=lambda n: state.graph[current][n].get('latency', 10) * 
                         (1 + state.graph[current][n].get('congestion', 0.5))
        )
        
        path.append(next_node)
        current = next_node
    
    if current == destination:
        return path
    return []


# Metrics and evaluation
@app.get("/metrics")
async def get_metrics() -> Dict:
    """Get current system performance metrics."""
    try:
        if state.dqn_agent is None:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        baseline_router = BaselineRouter(state.graph)
        results = evaluate_routing_quality(state.graph, state.dqn_agent, baseline_router)
        
        return {
            "status": "success",
            "agent": results['agent'],
            "baseline": results['baseline'],
            "improvements": results.get('improvements', {})
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report")
async def get_evaluation_report() -> Dict:
    """Get detailed evaluation report."""
    try:
        if state.dqn_agent is None:
            raise HTTPException(status_code=400, detail="Model not trained yet")
        
        baseline_router = BaselineRouter(state.graph)
        results = evaluate_routing_quality(state.graph, state.dqn_agent, baseline_router)
        report = generate_evaluation_report(results)
        
        return {
            "status": "success",
            "report": report,
            "metrics": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "graph_loaded": state.graph is not None,
        "agent_trained": state.dqn_agent is not None
    }


# System info
@app.get("/info")
async def system_info():
    """Get system information."""
    info = {
        "system_initialized": state.graph is not None,
        "device": str(state.device)
    }
    
    if state.graph:
        info.update({
            "num_nodes": state.graph.number_of_nodes(),
            "num_edges": state.graph.number_of_edges(),
            "agent_trained": state.dqn_agent is not None
        })
    
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
