# Deployment Guide

## Overview

This guide covers deploying the Cogniverse system in various environments, from local development to production cloud deployments.

## Local Development

### Prerequisites

- Python 3.12+
- Docker
- 16GB+ RAM
- 20GB+ disk space
- CUDA-capable GPU (optional)

### Quick Setup

```bash
# Clone repository
git clone <repo-url>
cd cogniverse

# Install dependencies
pip install uv
uv sync

# Start Vespa
./scripts/start_vespa.sh

# Start Phoenix (optional)
python scripts/start_phoenix.py

# Verify services
curl http://localhost:8080/ApplicationStatus  # Vespa
curl http://localhost:6006/health            # Phoenix (if started)
```

### Service Ports

- **Vespa**: 8080 (HTTP), 19071 (Config)
- **Phoenix**: 6006
- **Ollama**: 11434
- **Application**: 8000

## Docker Deployment

### Running Services Individually

```bash
# Start Vespa with persistent storage
docker run -d --name vespa \
  -p 8080:8080 -p 19071:19071 \
  -v vespa-data:/opt/vespa/var \
  vespaengine/vespa:latest

# Start Phoenix for observability
docker run -d --name phoenix \
  -p 6006:6006 \
  -v phoenix-data:/data \
  -e PHOENIX_WORKING_DIR=/data \
  arizephoenix/phoenix:latest

# Start Ollama for LLM inference
docker run -d --name ollama \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama:latest

# Pull required models
docker exec ollama ollama pull llava:7b
docker exec ollama ollama pull deepseek-r1:7b
```

### Building Application Image

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir uv && \
    uv pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Cloud Deployment

### AWS Deployment

#### Infrastructure Setup

```terraform
# main.tf
resource "aws_ecs_cluster" "cogniverse" {
  name = "cogniverse-cluster"
}

resource "aws_ecs_service" "vespa" {
  name            = "vespa"
  cluster         = aws_ecs_cluster.cogniverse.id
  task_definition = aws_ecs_task_definition.vespa.arn
  desired_count   = 3

  deployment_configuration {
    maximum_percent         = 200
    minimum_healthy_percent = 100
  }
}

resource "aws_ecs_task_definition" "vespa" {
  family                   = "vespa"
  requires_compatibilities = ["FARGATE"]
  network_mode            = "awsvpc"
  cpu                     = "4096"
  memory                  = "16384"

  container_definitions = jsonencode([
    {
      name  = "vespa"
      image = "vespaengine/vespa:latest"
      portMappings = [
        {
          containerPort = 8080
          protocol      = "tcp"
        }
      ]
      mountPoints = [
        {
          sourceVolume  = "vespa-data"
          containerPath = "/opt/vespa/var"
        }
      ]
    }
  ])
}
```

#### Deployment Script

```bash
#!/bin/bash
# deploy-aws.sh

# Build and push Docker image
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin $ECR_REGISTRY

docker build -t cogniverse .
docker tag cogniverse:latest $ECR_REGISTRY/cogniverse:latest
docker push $ECR_REGISTRY/cogniverse:latest

# Update ECS service
aws ecs update-service \
  --cluster cogniverse-cluster \
  --service cogniverse-app \
  --force-new-deployment
```

### GCP Deployment

#### Cloud Run Setup

```yaml
# service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: cogniverse
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
    spec:
      containers:
      - image: gcr.io/PROJECT_ID/cogniverse
        resources:
          limits:
            cpu: "4"
            memory: "8Gi"
        env:
        - name: VESPA_HOST
          value: vespa-service
        - name: PHOENIX_HOST
          value: phoenix-service
```

```bash
# Deploy to Cloud Run
gcloud run deploy cogniverse \
  --image gcr.io/PROJECT_ID/cogniverse \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --allow-unauthenticated
```

### Modal Deployment (Serverless)

```python
# modal_app.py
import modal

app = modal.App("cogniverse")

# Define image with dependencies
image = modal.Image.debian_slim().pip_install_from_requirements("requirements.txt")

@app.function(
    image=image,
    gpu="T4",
    memory=8192,
    timeout=300,
    secrets=[modal.Secret.from_name("cogniverse-secrets")]
)
async def process_video(video_url: str, profile: str):
    from src.app.ingestion import process_single_video
    return await process_single_video(video_url, profile)

@app.web_endpoint(method="POST")
async def search(query: str, profile: str = "frame_based_colpali"):
    from src.app.search import perform_search
    return await perform_search(query, profile)
```

```bash
# Deploy to Modal
modal deploy modal_app.py
```

## Kubernetes Deployment

### Helm Chart

```yaml
# helm/cogniverse/values.yaml
replicaCount: 3

image:
  repository: cogniverse
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt
  hosts:
    - host: cogniverse.example.com
      paths:
        - path: /
          pathType: Prefix

resources:
  limits:
    cpu: 4
    memory: 8Gi
  requests:
    cpu: 2
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

vespa:
  enabled: true
  replicas: 3
  storage: 100Gi
  memory: 16Gi

phoenix:
  enabled: true
  storage: 50Gi
```

```bash
# Deploy with Helm
helm install cogniverse ./helm/cogniverse \
  --namespace cogniverse \
  --create-namespace
```

## Production Configuration

### Environment Variables

```bash
# Production settings
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export WORKERS=4

# Service endpoints
export VESPA_HOST=vespa.internal.example.com
export VESPA_PORT=8080
export PHOENIX_HOST=phoenix.internal.example.com
export PHOENIX_PORT=6006

# Model configuration
export MODEL_CACHE_DIR=/mnt/models
export REMOTE_INFERENCE_URL=https://inference.example.com
export REMOTE_INFERENCE_API_KEY=${SECRET_API_KEY}

# Performance tuning
export VESPA_FEED_MAX_WORKERS=8
export VESPA_FEED_MAX_QUEUE_SIZE=1000
export BATCH_SIZE=50
export MAX_CONCURRENT_VIDEOS=4
```

### Secrets Management

```bash
# Using AWS Secrets Manager
aws secretsmanager create-secret \
  --name cogniverse/production \
  --secret-string '{
    "openai_api_key": "sk-...",
    "modal_token": "...",
    "inference_api_key": "..."
  }'

# Using Kubernetes Secrets
kubectl create secret generic cogniverse-secrets \
  --from-literal=openai-api-key=$OPENAI_API_KEY \
  --from-literal=modal-token=$MODAL_TOKEN \
  -n cogniverse
```

## Monitoring & Observability

### Prometheus Metrics

```yaml
# prometheus-config.yaml
scrape_configs:
  - job_name: 'cogniverse'
    static_configs:
      - targets: ['cogniverse:8000']
    metrics_path: '/metrics'
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Cogniverse Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])"
          }
        ]
      },
      {
        "title": "Response Time",
        "targets": [
          {
            "expr": "http_request_duration_seconds"
          }
        ]
      }
    ]
  }
}
```

### Logging

```python
# logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/var/log/cogniverse/app.log",
            "maxBytes": 104857600,  # 100MB
            "backupCount": 10,
            "formatter": "json"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}
```

## Scaling Strategies

### Horizontal Scaling

```yaml
# HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cogniverse-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cogniverse
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Vertical Scaling

```bash
# Resize instances
kubectl set resources deployment cogniverse \
  --limits=cpu=8,memory=16Gi \
  --requests=cpu=4,memory=8Gi
```

## Health Checks

### Application Health

```python
# health.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "vespa": check_vespa_health(),
        "phoenix": check_phoenix_health(),
        "models": check_model_health()
    }

@app.get("/ready")
async def ready():
    if all([vespa_ready(), phoenix_ready(), models_loaded()]):
        return {"status": "ready"}
    return {"status": "not_ready"}, 503
```

## Backup & Recovery

### Data Backup

```bash
#!/bin/bash
# backup.sh

# Backup Vespa data
docker exec vespa vespa-backup \
  --outputdir /backup \
  --cluster mycluster

# Backup Phoenix data
docker exec phoenix \
  tar -czf /backup/phoenix-$(date +%Y%m%d).tar.gz /data

# Upload to S3
aws s3 cp /backup/ s3://cogniverse-backups/$(date +%Y%m%d)/ --recursive
```

### Disaster Recovery

```bash
#!/bin/bash
# restore.sh

# Restore from S3
aws s3 cp s3://cogniverse-backups/$BACKUP_DATE/ /restore/ --recursive

# Restore Vespa
docker exec vespa vespa-restore \
  --inputdir /restore/vespa

# Restore Phoenix
docker exec phoenix \
  tar -xzf /restore/phoenix-$BACKUP_DATE.tar.gz -C /
```

## Security Hardening

### Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cogniverse-network-policy
spec:
  podSelector:
    matchLabels:
      app: cogniverse
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
```

### TLS Configuration

```bash
# Generate certificates
certbot certonly \
  --standalone \
  -d cogniverse.example.com

# Configure nginx
server {
    listen 443 ssl http2;
    server_name cogniverse.example.com;
    
    ssl_certificate /etc/letsencrypt/live/cogniverse.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/cogniverse.example.com/privkey.pem;
    
    location / {
        proxy_pass http://cogniverse:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Troubleshooting

### Common Issues

**Vespa Connection Failed**
```bash
# Check Vespa status
curl http://vespa:8080/ApplicationStatus

# Check logs
docker logs vespa --tail 100

# Restart Vespa
docker restart vespa
```

**Out of Memory**
```bash
# Increase memory limits
docker update --memory 32g vespa
kubectl set resources deployment cogniverse --limits=memory=16Gi
```

**Slow Inference**
```bash
# Check GPU availability
nvidia-smi

# Scale inference workers
kubectl scale deployment inference --replicas=5
```

**Model Loading Issues**
```bash
# Pre-download models
python scripts/download_models.py --all

# Mount model cache
docker run -v /models:/app/models cogniverse
```