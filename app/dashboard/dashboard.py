"""Interactive web dashboard for network routing visualization."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import networkx as nx
import requests
import json
import pandas as pd
from typing import List, Dict, Tuple
import logging


logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
st.set_page_config(
    page_title="Network Routing Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-left: 5px solid #FF6B6B;
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success {
        color: #28a745;
        font-weight: bold;
    }
    .warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'last_route' not in st.session_state:
        st.session_state.last_route = None


def get_system_status() -> Dict:
    """Get current system status from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/info", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Connection Error: {str(e)}")
        return {"error": str(e)}


def initialize_synthetic_network(num_nodes: int, edge_prob: float):
    """Initialize synthetic network via API."""
    try:
        with st.spinner("Initializing network..."):
            response = requests.post(
                f"{API_BASE_URL}/init",
                json={"num_nodes": num_nodes, "edge_probability": edge_prob},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            st.success(f"Network initialized: {result['num_nodes']} nodes, {result['num_edges']} edges")
            return result
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        return None


def train_model(num_episodes: int, num_pairs: int, batch_size: int):
    """Train DQN model via API."""
    try:
        with st.spinner("Training DQN agent... This may take a few minutes..."):
            response = requests.post(
                f"{API_BASE_URL}/train",
                json={
                    "num_episodes": num_episodes,
                    "num_training_pairs": num_pairs,
                    "batch_size": batch_size
                },
                timeout=300  # 5 minute timeout for training
            )
            response.raise_for_status()
            result = response.json()
            st.session_state.trained = True
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Success Rate", f"{result['final_success_rate']:.2%}")
            with col2:
                st.metric("Avg Final Reward", f"{result['avg_final_reward']:.2f}")
            
            return result
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None


def compute_route(source: int, destination: int, use_agent: bool = True) -> Dict:
    """Compute route via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/route",
            json={
                "source": source,
                "destination": destination,
                "use_agent": use_agent
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        st.session_state.last_route = result
        return result
    except Exception as e:
        st.error(f"Routing failed: {str(e)}")
        return None


def get_evaluation_metrics() -> Dict:
    """Get evaluation metrics from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to fetch metrics: {str(e)}")
        return None


def visualize_network(num_nodes: int, highlighted_path: List[int] = None):
    """Create interactive network visualization."""
    # Generate a consistent network graph for visualization
    np.random.seed(42)
    G = nx.erdos_renyi_graph(num_nodes, 0.3, seed=42)
    
    # Ensure connected
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = list(components[i])[0]
            v = list(components[i + 1])[0]
            G.add_edge(u, v)
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_colors = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Color edges based on whether they're in the highlighted path
        if highlighted_path and edge in [(highlighted_path[i], highlighted_path[i+1]) 
                                         for i in range(len(highlighted_path)-1)]:
            edge_colors.extend(['#FF4136', '#FF4136', '#FF4136'])
        else:
            edge_colors.extend(['#999999', '#999999', '#999999'])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='#999999'),
        hoverinfo='none',
        showlegend=False
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_color = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Node {node}")
        
        # Color nodes based on path
        if highlighted_path:
            if node == highlighted_path[0]:
                node_color.append('green')  # Source
            elif node == highlighted_path[-1]:
                node_color.append('red')  # Destination
            elif node in highlighted_path:
                node_color.append('orange')  # On path
            else:
                node_color.append('lightblue')
        else:
            node_color.append('lightblue')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[f"{i}" for i in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=20,
            color=node_color,
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title="Network Topology",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    return fig


def visualize_metrics(metrics: Dict):
    """Create metrics comparison visualization."""
    if not metrics or 'agent' not in metrics:
        st.warning("No metrics available yet. Train the model first.")
        return
    
    agent_stats = metrics['agent']
    baseline_stats = metrics['baseline']
    
    # Create comparison dataframe
    comparison_data = {
        'Metric': ['Success Rate', 'Avg Latency (ms)', 'Avg Congestion', 'Avg Path Length'],
        'Agent': [
            agent_stats['success_rate'],
            agent_stats['avg_latency'],
            agent_stats['avg_congestion'],
            agent_stats['avg_path_length']
        ],
        'Baseline': [
            baseline_stats['success_rate'],
            baseline_stats['avg_latency'],
            baseline_stats['avg_congestion'],
            baseline_stats['avg_path_length']
        ]
    }
    
    df = pd.DataFrame(comparison_data)
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(name='Agent', x=df['Metric'], y=[
            df['Agent'][0] * 100,
            df['Agent'][1],
            df['Agent'][2] * 100,
            df['Agent'][3]
        ]),
        go.Bar(name='Baseline', x=df['Metric'], y=[
            df['Baseline'][0] * 100,
            df['Baseline'][1],
            df['Baseline'][2] * 100,
            df['Baseline'][3]
        ])
    ])
    
    fig.update_layout(
        barmode='group',
        title="Agent vs Baseline Comparison",
        yaxis_title="Value",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main dashboard application."""
    init_session_state()
    
    # Header
    st.title("🌐 Network Routing System")
    st.markdown("**GNN + DQN Dynamic Network Routing Dashboard**")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # System status
        with st.expander("System Status"):
            status = get_system_status()
            if "error" not in status:
                if status.get('system_initialized'):
                    st.success("✓ Network initialized")
                    st.metric("Nodes", status.get('num_nodes', 'N/A'))
                    st.metric("Edges", status.get('num_edges', 'N/A'))
                    st.metric("Agent Trained", "Yes" if status.get('agent_trained') else "No")
                else:
                    st.info("Initialize network first")
        
        st.divider()
        
        # Network initialization
        st.subheader("1. Initialize Network")
        init_type = st.radio("Select initialization type:", 
                            ["Synthetic", "CSV Upload"])
        
        if init_type == "Synthetic":
            col1, col2 = st.columns(2)
            with col1:
                num_nodes = st.slider("Number of nodes", 5, 50, 20)
            with col2:
                edge_prob = st.slider("Edge probability", 0.1, 0.8, 0.3)
            
            if st.button("Initialize Synthetic Network", use_container_width=True):
                initialize_synthetic_network(num_nodes, edge_prob)
        
        else:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file is not None:
                if st.button("Load from CSV", use_container_width=True):
                    with st.spinner("Loading network..."):
                        files = {"file": (uploaded_file.name, uploaded_file)}
                        try:
                            response = requests.post(
                                f"{API_BASE_URL}/init-from-csv",
                                files=files,
                                timeout=10
                            )
                            response.raise_for_status()
                            result = response.json()
                            st.success(f"Loaded: {result['num_nodes']} nodes, {result['num_edges']} edges")
                        except Exception as e:
                            st.error(f"Load failed: {str(e)}")
        
        st.divider()
        
        # Training
        st.subheader("2. Train Model")
        col1, col2, col3 = st.columns(3)
        with col1:
            episodes = st.number_input("Episodes", 10, 500, 50, step=10)
        with col2:
            pairs = st.number_input("Pairs/episode", 5, 100, 20, step=5)
        with col3:
            batch = st.number_input("Batch size", 8, 128, 32, step=8)
        
        if st.button("Train Model", use_container_width=True, type="primary"):
            train_model(episodes, pairs, batch)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Network Topology")
        # Get number of nodes from status
        status = get_system_status()
        num_nodes = status.get('num_nodes', 20) if "error" not in status else 20
        
        # Routing query
        st.markdown("---")
        st.subheader("3. Route Query")
        col_src, col_dst = st.columns(2)
        with col_src:
            source = st.number_input("Source", 0, num_nodes-1, 0)
        with col_dst:
            destination = st.number_input("Destination", 0, num_nodes-1, num_nodes-1)
        
        use_agent = st.checkbox("Use trained agent", value=True, disabled=not st.session_state.trained)
        
        if st.button("Compute Route", use_container_width=True):
            route_result = compute_route(source, destination, use_agent)
            if route_result:
                # Visualize the route
                fig = visualize_network(num_nodes, route_result['path'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Display route details
                with st.expander("Route Details", expanded=True):
                    st.write(f"**Path**: {' → '.join(map(str, route_result['path']))}")
                    st.write(f"**Algorithm**: {route_result['algorithm']}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Path Length", f"{route_result['path_length']} hops")
                    with col2:
                        st.metric("Total Latency", f"{route_result['total_latency']:.2f} ms")
                    with col3:
                        st.metric("Avg Congestion", f"{route_result['avg_congestion']:.3f}")
        else:
            # Show network without highlighted path
            fig = visualize_network(num_nodes)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Performance Metrics")
        
        tab1, tab2 = st.tabs(["Metrics", "Report"])
        
        with tab1:
            if st.button("Refresh Metrics", use_container_width=True):
                metrics = get_evaluation_metrics()
                if metrics and "agent" in metrics:
                    visualize_metrics(metrics)
                    
                    st.markdown("---")
                    st.subheader("Detailed Statistics")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Agent**")
                        for key, val in metrics['agent'].items():
                            if isinstance(val, float):
                                st.write(f"{key}: {val:.3f}")
                            else:
                                st.write(f"{key}: {val}")
                    
                    with col2:
                        st.markdown("**Baseline**")
                        for key, val in metrics['baseline'].items():
                            if isinstance(val, float):
                                st.write(f"{key}: {val:.3f}")
                            else:
                                st.write(f"{key}: {val}")
                    
                    if 'improvements' in metrics:
                        st.markdown("---")
                        st.markdown("**Improvements**")
                        for key, val in metrics['improvements'].items():
                            st.write(f"{key}: {val:+.2f}%")
        
        with tab2:
            if st.button("Generate Report", use_container_width=True):
                try:
                    response = requests.get(f"{API_BASE_URL}/report", timeout=30)
                    response.raise_for_status()
                    report = response.json()
                    st.code(report['report'], language='text')
                except Exception as e:
                    st.error(f"Failed to generate report: {str(e)}")


if __name__ == "__main__":
    main()
