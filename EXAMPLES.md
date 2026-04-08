# API Examples and Usage Guide

## Setup

### Start the System

```bash
# Option 1: Direct Python
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# Option 2: Docker Compose
docker-compose up

# Option 3: Using run script
chmod +x run.sh
./run.sh
```

### Verify API is Running

```bash
curl http://localhost:8000/health
# Output: {"status": "healthy", "graph_loaded": false, "agent_trained": false}
```

---

## Example 1: Synthetic Network Training

Complete workflow for creating and training on a synthetic network.

### Step 1: Initialize Synthetic Network

```bash
curl -X POST "http://localhost:8000/init" \
  -H "Content-Type: application/json" \
  -d '{
    "num_nodes": 25,
    "edge_probability": 0.3
  }' | jq .
```

**Response**:
```json
{
  "status": "success",
  "num_nodes": 25,
  "num_edges": 87,
  "message": "System initialized with synthetic network"
}
```

### Step 2: Train DQN Agent

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "num_episodes": 100,
    "num_training_pairs": 20,
    "batch_size": 32
  }' | jq .
```

**Response**:
```json
{
  "status": "success",
  "message": "Training completed",
  "final_success_rate": 0.87,
  "avg_final_reward": 52.34
}
```

Training takes ~3-5 minutes depending on hardware.

### Step 3: Test Individual Routes

```bash
# Route from node 0 to node 20
curl -X POST "http://localhost:8000/route" \
  -H "Content-Type: application/json" \
  -d '{
    "source": 0,
    "destination": 20,
    "use_agent": true
  }' | jq .
```

**Response**:
```json
{
  "path": [0, 3, 7, 12, 18, 20],
  "total_latency": 78.5,
  "avg_congestion": 0.456,
  "path_length": 5,
  "algorithm": "DQN Agent"
}
```

### Step 4: Compare with Baseline

```bash
# Same route using Dijkstra
curl -X POST "http://localhost:8000/route" \
  -H "Content-Type: application/json" \
  -d '{
    "source": 0,
    "destination": 20,
    "use_agent": false
  }' | jq .
```

### Step 5: Get Evaluation Metrics

```bash
curl -X GET "http://localhost:8000/metrics" | jq .
```

**Response**:
```json
{
  "status": "success",
  "agent": {
    "success_rate": 0.92,
    "avg_latency": 48.3,
    "avg_congestion": 0.41,
    "avg_path_length": 3.2,
    "min_latency": 12.5,
    "max_latency": 95.2
  },
  "baseline": {
    "success_rate": 0.88,
    "avg_latency": 52.1,
    "avg_congestion": 0.45,
    "avg_path_length": 3.5,
    "min_latency": 11.8,
    "max_latency": 98.7
  },
  "improvements": {
    "latency_reduction": 7.29,
    "path_length_reduction": 8.57,
    "congestion_reduction": 8.89
  }
}
```

---

## Example 2: Loading from CSV

Use the included example network or your own data.

### Step 1: Prepare CSV Files

Required file: `edges.csv`
```csv
source,target,latency,bandwidth,congestion
0,1,10.5,75.0,0.3
0,3,15.2,50.0,0.5
1,2,8.3,100.0,0.2
...
```

Optional file: `nodes.csv`
```csv
node_id,capacity,processing_time
0,100,1.0
1,120,0.8
...
```

### Step 2: Load Network from CSV

```bash
curl -X POST "http://localhost:8000/init-from-csv" \
  -F "file=@example_network.csv" | jq .
```

**Response**:
```json
{
  "status": "success",
  "num_nodes": 20,
  "num_edges": 41,
  "message": "System initialized from CSV",
  "metadata": {
    "source": "/tmp/network.csv",
    "num_nodes": 20,
    "num_edges": 41
  }
}
```

### Step 3: Train and Evaluate

Same as synthetic workflow (Steps 2-5 above).

---

## Example 3: Batch Routing Evaluation

Evaluate performance on multiple routes.

### Python Script Example

```python
import requests
import json
from itertools import combinations
import random

API_URL = "http://localhost:8000"

# Initialize
requests.post(f"{API_URL}/init", json={
    "num_nodes": 20,
    "edge_probability": 0.35
})

# Train
requests.post(f"{API_URL}/train", json={
    "num_episodes": 50,
    "num_training_pairs": 15,
    "batch_size": 32
})

# Test multiple routes
results = {"agent": [], "baseline": []}

for i in range(20):
    source = random.randint(0, 19)
    dest = random.randint(0, 19)
    
    if source == dest:
        continue
    
    # Agent routing
    resp_agent = requests.post(f"{API_URL}/route", json={
        "source": source,
        "destination": dest,
        "use_agent": True
    })
    results["agent"].append(resp_agent.json())
    
    # Baseline routing
    resp_baseline = requests.post(f"{API_URL}/route", json={
        "source": source,
        "destination": dest,
        "use_agent": False
    })
    results["baseline"].append(resp_baseline.json())

# Print comparison
print("Agent routes (avg latency):", 
      sum(r["total_latency"] for r in results["agent"]) / len(results["agent"]))
print("Baseline routes (avg latency):", 
      sum(r["total_latency"] for r in results["baseline"]) / len(results["baseline"]))
```

---

## Example 4: Advanced Configuration

### Large Network with Custom Settings

```bash
# Initialize large network
curl -X POST "http://localhost:8000/init" \
  -H "Content-Type: application/json" \
  -d '{
    "num_nodes": 100,
    "edge_probability": 0.15
  }'

# Train with more episodes
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "num_episodes": 200,
    "num_training_pairs": 50,
    "batch_size": 64
  }'
```

### Quick Evaluation Mode

```bash
# Minimal training for quick testing
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "num_episodes": 20,
    "num_training_pairs": 10,
    "batch_size": 16
  }'
```

---

## Example 5: Analyzing Route Quality

### Get Detailed Report

```bash
curl -X GET "http://localhost:8000/report" | jq '.report'
```

**Sample Output**:
```
============================================================
ROUTING SYSTEM EVALUATION REPORT
============================================================

AGENT PERFORMANCE:
  Success Rate:        92.00%
  Avg Latency:         48.25 ms
  Avg Congestion:      0.410
  Avg Path Length:     3.2 hops

BASELINE (DIJKSTRA) PERFORMANCE:
  Success Rate:        88.00%
  Avg Latency:         52.10 ms
  Avg Congestion:      0.450
  Avg Path Length:     3.5 hops

IMPROVEMENTS (Agent vs Baseline):
  Latency Reduction:   +7.41%
  Path Length Reduction: +8.57%
  Congestion Reduction: +8.89%

============================================================
```

---

## Example 6: Error Handling

### Invalid Node IDs

```bash
curl -X POST "http://localhost:8000/route" \
  -H "Content-Type: application/json" \
  -d '{
    "source": 0,
    "destination": 999,
    "use_agent": true
  }'

# Response (400):
# {"detail": "Invalid source or destination node"}
```

### Not Initialized

```bash
curl -X POST "http://localhost:8000/train"

# Response (400):
# {"detail": "Network not initialized. Call /init first"}
```

### Not Trained

```bash
curl -X GET "http://localhost:8000/metrics"

# Response (400):
# {"detail": "Model not trained yet"}
```

---

## Example 7: Performance Monitoring

### Check System Status

```bash
curl -X GET "http://localhost:8000/info" | jq .
```

**Response**:
```json
{
  "system_initialized": true,
  "device": "cuda",
  "num_nodes": 25,
  "num_edges": 87,
  "agent_trained": true
}
```

### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

---

## Example 8: Docker Usage

### Start Full Stack

```bash
# Build and start
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
docker-compose logs -f dashboard

# Execute command in container
docker-compose exec api python -c "import torch; print(torch.cuda.is_available())"

# Stop services
docker-compose down
```

### Scale API (if needed)

```bash
docker-compose up -d --scale api=3
```

---

## Example 9: Integration with Monitoring

### Prometheus-style Metrics

```bash
# Create a simple monitoring endpoint
curl -X GET "http://localhost:8000/metrics" \
  -H "Accept: application/json" \
  -o metrics.json

# Parse and use in monitoring systems
jq '.agent.success_rate' metrics.json
```

### Log Output for Aggregation

```bash
# Capture structured logs
docker-compose logs api > api_logs.txt

# Useful for ELK stack or similar
```

---

## Example 10: Custom Network Analysis

### Analyze Specific Network Pattern

```python
import requests
import networkx as nx
import json

API_URL = "http://localhost:8000"

# Initialize network
requests.post(f"{API_URL}/init", json={"num_nodes": 30, "edge_probability": 0.25})

# Train model
train_resp = requests.post(f"{API_URL}/train", json={
    "num_episodes": 75,
    "num_training_pairs": 25,
    "batch_size": 32
})

# Analyze core paths
core_routes = []
metrics_dict = {}

for source in range(0, 10):  # Test specific nodes
    for dest in range(10, 20):
        route_resp = requests.post(f"{API_URL}/route", json={
            "source": source,
            "destination": dest,
            "use_agent": True
        })
        if route_resp.status_code == 200:
            route_data = route_resp.json()
            core_routes.append({
                "source": source,
                "dest": dest,
                "path": route_data["path"],
                "latency": route_data["total_latency"]
            })

# Analyze results
avg_latency = sum(r["latency"] for r in core_routes) / len(core_routes)
print(f"Average core path latency: {avg_latency:.2f} ms")
print(f"Analyzed {len(core_routes)} routes")
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill it
kill -9 <PID>

# Or use different port
python -m uvicorn app.api.main:app --port 8001
```

### Out of Memory During Training

```bash
# Reduce batch size and replay buffer
# Edit app/rl/dqn_agent.py:
# - Change batch_size from 32 to 16
# - Change replay_buffer_size from 10000 to 5000
```

### GPU Not Detected

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## Performance Tips

1. **Use GPU for Training**
   - 5-10x faster than CPU
   - Automatic detection in code

2. **Batch Multiple Routes**
   - Much faster than sequential API calls
   - Use Python script for bulk testing

3. **Cache Metrics**
   - Don't call `/metrics` in tight loops
   - Cache results for 1-5 minutes

4. **Tune Hyperparameters**
   - More episodes for better convergence
   - Smaller batch size if memory limited
   - Larger networks = longer training

---

## Next Steps

1. Experiment with different network sizes
2. Load your own network topology
3. Integrate with your existing systems
4. Monitor performance in production
5. Fine-tune hyperparameters for your use case

For more details, see main README.md and API documentation at `/docs`
