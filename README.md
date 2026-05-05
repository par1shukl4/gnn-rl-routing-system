Developed by: Pari Shukla  
NSUT Engineering | pari.shukla.ug23@nsut.ac.in  

---

## Project Overview

The GNN-RL Routing System is a learning-based network optimization framework designed to address routing inefficiencies in dynamic and large-scale network topologies. By integrating Graph Neural Networks (GNNs) with Deep Reinforcement Learning (DRL), the system learns adaptive routing policies that outperform traditional heuristic-based protocols under variable traffic conditions.

This project models communication networks as graph structures and formulates routing as a sequential decision-making problem, enabling intelligent, congestion-aware, and generalizable path selection.

---

## Core Scientific Frameworks

### 1. Structural Learning: Graph Neural Networks

Traditional routing algorithms fail to capture complex relational dependencies across nodes. This system leverages Graph Neural Networks to encode topological and edge-level information.

Mechanism:  
Each node aggregates feature information from its neighbors through message passing, generating a latent embedding that captures both local and global network structure.

Scientific Rationale:  
GNNs provide permutation-invariant representations and enable inductive generalization across unseen graph topologies, making them well-suited for dynamic routing environments.

---

### 2. Sequential Optimization: Reinforcement Learning

Routing is modeled as a Markov Decision Process (MDP), where an agent learns optimal forwarding decisions through interaction with the network environment.

Components:  
State: Graph embeddings representing current network conditions  
Action: Selection of next-hop node  
Reward: Function of latency, congestion, and successful packet delivery  

Scientific Rationale:  
Reinforcement Learning enables adaptive policy learning without explicit supervision, allowing the system to optimize long-term network performance under uncertainty.

---

### 3. Policy Generalization and Scalability

The integration of GNN embeddings with RL policies allows the system to generalize learned behaviors across varying network sizes and configurations.

Outcome:  
The trained model demonstrates strong transferability to unseen topologies and maintains performance under dynamic traffic distributions.

---

## Technical Implementation

### Tech Stack

- Language: Python  
- Deep Learning: PyTorch  
- Graph Processing: PyTorch Geometric / DGL  
- Environment Modeling: OpenAI Gym (or custom simulator)  
- Graph Utilities: NetworkX  

---

### Key Features

- Adaptive routing under dynamic traffic conditions  
- End-to-end learning pipeline integrating GNN and RL  
- Congestion-aware decision-making  
- Scalable across diverse network structures  
- Generalization to unseen environments  

---

## System Architecture

1. Graph Construction  
Network represented as nodes and weighted edges (latency, bandwidth)

2. GNN Encoder  
Extracts structural embeddings from the graph

3. RL Agent  
Uses embeddings to learn optimal routing policies

4. Simulation Environment  
Generates traffic scenarios and provides reward feedback

---

## Evaluation Framework

Performance is evaluated across multiple metrics:

- Average Path Length  
- End-to-End Latency  
- Network Throughput  
- Packet Delivery Ratio  
- Convergence Speed  

---

## Results

The GNN-RL framework achieves improved routing efficiency compared to traditional shortest-path and heuristic-based approaches, particularly in environments with high variability and congestion.

The model demonstrates strong generalization capabilities and stable convergence behavior during training.

---

## Future Work

- Multi-agent reinforcement learning for decentralized routing  
- Integration with Software-Defined Networking (SDN) controllers  
- Real-world deployment on large-scale network datasets  
- Advanced reward shaping for improved convergence  

---

## Installation and Setup

Clone the repository:

