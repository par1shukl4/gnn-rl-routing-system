# Deployment Guide

Complete guide for deploying the Network Routing GNN+RL system to production.

## Table of Contents
1. [Local Development](#local-development)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Checklist](#production-checklist)
5. [Monitoring & Logging](#monitoring--logging)
6. [Troubleshooting](#troubleshooting)

---

## Local Development

### Quick Setup

```bash
# Clone repository
git clone <repo-url>
cd network-routing-gnn-rl

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Create necessary directories
mkdir -p models data logs
```

### Running Services Separately

**Terminal 1: API Server**
```bash
python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2: Dashboard**
```bash
streamlit run app/dashboard/dashboard.py
```

**Terminal 3: Optional - Monitor logs**
```bash
tail -f logs/api.log
```

---

## Docker Deployment

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 4GB RAM
- Optional: GPU support (nvidia-docker)

### Build and Deploy

#### Option 1: Docker Compose (Recommended)

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Verify services are running
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Option 2: Individual Docker Containers

```bash
# Build image
docker build -t routing-system:latest .

# Run API server
docker run -d \
  --name routing-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  routing-system:latest \
  python -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# Run dashboard
docker run -d \
  --name routing-dashboard \
  -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  --link routing-api:api \
  routing-system:latest \
  streamlit run app/dashboard/dashboard.py
```

### Access Services

- **API**: http://localhost:8000
  - API Docs: http://localhost:8000/docs
  - ReDoc: http://localhost:8000/redoc
- **Dashboard**: http://localhost:8501

### Database Persistence

Models and data are stored in volumes:
- `./models/` - Trained models
- `./data/` - Network data files

To backup:
```bash
docker-compose exec api tar czf /tmp/models.tar.gz /app/models
docker cp routing-api:/tmp/models.tar.gz ./backup/
```

---

## Cloud Deployment

### AWS Deployment

#### Using EC2 + Docker

1. **Launch EC2 Instance**
```bash
# t3.large recommended for CPU, g4dn.xlarge for GPU
# Ubuntu 22.04 LTS AMI
```

2. **Install Docker**
```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker ubuntu
```

3. **Deploy Application**
```bash
git clone <repo-url>
cd network-routing-gnn-rl
docker-compose up -d
```

4. **Configure Security Group**
   - Allow inbound on ports 8000 (API) and 8501 (Dashboard)
   - Restrict to your IP range in production

#### Using Elastic Container Service (ECS)

```yaml
# Create task definition
version: '3.8'
services:
  api:
    image: routing-system:latest
    container_port: 8000
    memory: 2048
    cpu: 512
    
  dashboard:
    image: routing-system:latest
    container_port: 8501
    memory: 1024
    cpu: 256
```

### Google Cloud Run Deployment

```bash
# Configure gcloud
gcloud auth configure-docker

# Build and push image
docker build -t gcr.io/<PROJECT_ID>/routing-system:latest .
docker push gcr.io/<PROJECT_ID>/routing-system:latest

# Deploy to Cloud Run
gcloud run deploy routing-api \
  --image gcr.io/<PROJECT_ID>/routing-system:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 1
```

### Azure Deployment

```bash
# Login
az login

# Create resource group
az group create --name routing-rg --location eastus

# Create container registry
az acr create --resource-group routing-rg \
  --name routingregistry --sku Basic

# Build and push
docker build -t routing-system:latest .
docker tag routing-system:latest routingregistry.azurecr.io/routing-system:latest
az acr build --registry routingregistry \
  --image routing-system:latest .

# Deploy to Container Instances
az container create --resource-group routing-rg \
  --name routing-api \
  --image routingregistry.azurecr.io/routing-system:latest \
  --cpu 2 --memory 2 \
  --ports 8000 8501 \
  --environment-variables API_HOST=0.0.0.0
```

### Heroku Deployment

```bash
# Create app
heroku create routing-system

# Set buildpack
heroku buildpacks:add heroku/python

# Deploy
git push heroku main

# View logs
heroku logs --tail

# Scale dynos (optional)
heroku ps:scale web=2
```

---

## Production Checklist

### Before Going Live

- [ ] Update `pyproject.toml` with pinned versions
- [ ] Set environment variables in `.env`
- [ ] Configure logging to files instead of stdout
- [ ] Enable HTTPS/TLS
- [ ] Set up database backup strategy
- [ ] Configure monitoring and alerts
- [ ] Load test the system
- [ ] Document deployment procedure
- [ ] Set up CI/CD pipeline
- [ ] Configure auto-scaling (if cloud)

### Environment Variables

Create `.env` file:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=False

# Database
MODELS_PATH=/app/models
DATA_PATH=/app/data

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/routing-api/api.log

# Performance
MAX_WORKERS=4
BATCH_SIZE=32

# Security
ALLOWED_ORIGINS=["https://yourdomain.com"]
API_KEY_REQUIRED=True

# GPU
CUDA_VISIBLE_DEVICES=0
```

### Docker Production Build

```dockerfile
# Multi-stage optimized build
FROM python:3.11-slim as base

FROM base as dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install --user --no-cache-dir -e .

FROM base as runtime
COPY --from=dependencies /root/.local /root/.local
COPY app /app/
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
WORKDIR /app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    restart: always
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
    volumes:
      - models:/app/models
      - data:/app/data
      - logs:/var/log/routing-api
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - routing

  dashboard:
    build: .
    restart: always
    ports:
      - "8501:8501"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - models:/app/models
      - data:/app/data
    depends_on:
      api:
        condition: service_healthy
    networks:
      - routing
    command: streamlit run app/dashboard/dashboard.py

  nginx:
    image: nginx:alpine
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    networks:
      - routing

volumes:
  models:
  data:
  logs:

networks:
  routing:
    driver: bridge
```

---

## Monitoring & Logging

### Structured Logging Setup

```python
# In app/api/main.py
import logging
import logging.handlers

# Configure JSON logging
formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)

file_handler = logging.handlers.RotatingFileHandler(
    '/var/log/routing-api/api.log',
    maxBytes=10485760,  # 10MB
    backupCount=10
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
```

### Prometheus Metrics

```python
# Add to app/api/main.py
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
route_requests = Counter('route_requests_total', 'Total routing requests')
route_latency = Histogram('route_latency_seconds', 'Route computation time')
training_duration = Histogram('training_seconds', 'Training time')

@app.get("/metrics")
async def prometheus_metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### ELK Stack Integration

```yaml
# docker-compose.yml addition
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
  environment:
    - discovery.type=single-node
  ports:
    - "9200:9200"

kibana:
  image: docker.elastic.co/kibana/kibana:8.0.0
  ports:
    - "5601:5601"

filebeat:
  image: docker.elastic.co/beats/filebeat:8.0.0
  volumes:
    - /var/log/routing-api:/var/log/routing-api:ro
```

### CloudWatch (AWS)

```python
# app/api/main.py
import watchtower
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(watchtower.CloudWatchLogHandler())
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs api

# Common issues:
# 1. Port already in use
lsof -i :8000
kill -9 <PID>

# 2. Insufficient memory
docker stats

# 3. GPU not found
docker run --gpus all <image>
```

### High Memory Usage

```bash
# Profile memory usage
python -m memory_profiler app/api/main.py

# Reduce batch size
# Edit app/rl/dqn_agent.py:
# replay_buffer_size = 5000  # Was 10000
```

### Slow Training

```bash
# Check if GPU is being used
nvidia-smi

# If CPU-only, consider:
# 1. Reduce network size
# 2. Reduce num_episodes
# 3. Use GPU instance type
# 4. Optimize batch size
```

### API Timeout

```bash
# Increase timeout in docker-compose.yml
healthcheck:
  timeout: 30s  # Increased from 10s
  
# Or increase nginx timeout
# In nginx.conf: proxy_read_timeout 300s;
```

---

## Scaling

### Horizontal Scaling (Multiple API instances)

```bash
# With Docker Compose
docker-compose up -d --scale api=3

# With Kubernetes
kubectl scale deployment routing-api --replicas=3
```

### Load Balancing

```nginx
# nginx.conf
upstream api_backend {
    server routing-api-1:8000;
    server routing-api-2:8000;
    server routing-api-3:8000;
}

server {
    listen 80;
    location /api {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
    }
}
```

### Kubernetes Deployment

```yaml
# routing-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: routing-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: routing-api
  template:
    metadata:
      labels:
        app: routing-api
    spec:
      containers:
      - name: api
        image: routing-system:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: routing-api-service
spec:
  selector:
    app: routing-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Backup & Recovery

### Backup Models and Data

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR=/backups
DATE=$(date +%Y%m%d_%H%M%S)

docker-compose exec -T api tar czf /tmp/models_${DATE}.tar.gz /app/models
docker cp routing-api:/tmp/models_${DATE}.tar.gz ${BACKUP_DIR}/

# Upload to S3
aws s3 cp ${BACKUP_DIR}/models_${DATE}.tar.gz s3://my-bucket/backups/
```

### Restore from Backup

```bash
# Restore models
aws s3 cp s3://my-bucket/backups/models_20240101_120000.tar.gz .
docker-compose exec -T api tar xzf /tmp/models.tar.gz -C /app/
```

---

## Security Best Practices

1. **Use HTTPS**
```nginx
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/certs/cert.pem;
    ssl_certificate_key /etc/nginx/certs/key.pem;
}
```

2. **API Authentication**
```python
# app/api/security.py
from fastapi.security import HTTPBearer, HTTPAuthCredentials

security = HTTPBearer()

@app.post("/train")
async def train_model(config: TrainingConfig, credentials: HTTPAuthCredentials = Depends(security)):
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=403, detail="Invalid credentials")
    # ... rest of function
```

3. **Rate Limiting**
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/route")
@limiter.limit("100/minute")
async def compute_route(request: RoutingRequest):
    # ...
```

4. **Input Validation**
```python
from pydantic import validator

class RoutingRequest(BaseModel):
    source: int
    destination: int
    
    @validator('source', 'destination')
    def validate_nodes(cls, v):
        if v < 0 or v >= 1000:
            raise ValueError('Invalid node ID')
        return v
```

---

## Continuous Integration/Deployment

### GitHub Actions Example

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Build Docker image
        run: docker build -t routing-system:latest .
      - name: Run tests
        run: docker run routing-system:latest python -m pytest
      - name: Push to Docker Hub
        run: docker push myusername/routing-system:latest
      - name: Deploy to production
        run: |
          ssh user@production-server
          docker pull myusername/routing-system:latest
          docker-compose down && docker-compose up -d
```

---

**For more information, see README.md and EXAMPLES.md**
