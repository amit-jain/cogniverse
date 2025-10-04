# Deployment Guide

## Overview

Cogniverse is a multi-tenant video search system with multi-agent orchestration, DSPy-powered optimization, and comprehensive observability. This guide covers deployment from local development to production cloud environments.

## Architecture Components

**Core Services:**
- **Application**: Multi-agent orchestration with routing, video search, and summarization agents
- **Vespa**: Multi-tenant vector database with schema-per-tenant isolation
- **Phoenix**: Distributed tracing and experiment tracking
- **Ollama**: Local LLM inference for memory management and query analysis

**Optional Services:**
- **Modal**: Serverless GPU inference for video processing
- **Prometheus/Grafana**: Metrics and monitoring
- **Loki**: Log aggregation

## Local Development

### Prerequisites

- Python 3.12+
- Docker
- 16GB+ RAM (32GB recommended)
- 20GB+ disk space
- CUDA-capable GPU (optional, for local model inference)

### Quick Setup

```bash
# Clone repository
git clone <repo-url>
cd cogniverse

# Install dependencies
pip install uv
uv sync

# Start Vespa
docker run -d --name vespa \
  -p 8080:8080 -p 19071:19071 \
  -v vespa-data:/opt/vespa/var \
  vespaengine/vespa:latest

# Start Phoenix
docker run -d --name phoenix \
  -p 6006:6006 \
  -v phoenix-data:/data \
  -e PHOENIX_WORKING_DIR=/data \
  arizephoenix/phoenix:latest

# Start Ollama
docker run -d --name ollama \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama:latest

# Pull required Ollama models
docker exec ollama ollama pull llama3.2
docker exec ollama ollama pull nomic-embed-text

# Verify services
curl http://localhost:8080/ApplicationStatus  # Vespa
curl http://localhost:6006/health            # Phoenix
curl http://localhost:11434/api/tags         # Ollama
```

### Service Ports

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Vespa HTTP | 8080 | HTTP | Document feed & search queries |
| Vespa Config | 19071 | HTTP | Schema deployment |
| Phoenix | 6006 | HTTP | Telemetry & experiments dashboard |
| Phoenix Collector | 4317 | gRPC | OTLP span collection |
| Ollama | 11434 | HTTP | LLM inference API |
| Application | 8000 | HTTP | Main API |

### Environment Configuration

```bash
# Create .env file
cat > .env <<EOF
# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# Telemetry
TELEMETRY_ENABLED=true
TELEMETRY_LEVEL=detailed
PHOENIX_ENABLED=true
PHOENIX_COLLECTOR_ENDPOINT=localhost:4317
TELEMETRY_SYNC_EXPORT=false  # Use async batch export

# Vespa
VESPA_HOST=localhost
VESPA_PORT=8080

# Ollama
OLLAMA_BASE_URL=http://localhost:11434/v1

# Multi-tenant
DEFAULT_TENANT_ID=default

# Performance
VESPA_FEED_MAX_WORKERS=8
BATCH_SIZE=50
EOF

# Source environment
source .env
```

## Docker Compose Deployment

### Complete Stack

```yaml
# docker-compose.yml
version: '3.8'

services:
  vespa:
    image: vespaengine/vespa:latest
    ports:
      - "8080:8080"
      - "19071:19071"
    volumes:
      - vespa-data:/opt/vespa/var
    environment:
      - VESPA_CONFIGSERVERS=localhost
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/ApplicationStatus"]
      interval: 30s
      timeout: 10s
      retries: 3

  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
      - "4317:4317"  # OTLP gRPC collector
    volumes:
      - phoenix-data:/data
    environment:
      - PHOENIX_WORKING_DIR=/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6006/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  cogniverse:
    build: .
    ports:
      - "8000:8000"
    environment:
      - VESPA_HOST=vespa
      - VESPA_PORT=8080
      - PHOENIX_COLLECTOR_ENDPOINT=phoenix:4317
      - OLLAMA_BASE_URL=http://ollama:11434/v1
      - ENVIRONMENT=production
      - TELEMETRY_ENABLED=true
    depends_on:
      - vespa
      - phoenix
      - ollama
    volumes:
      - model-cache:/app/models
      - ./configs:/app/configs:ro

volumes:
  vespa-data:
  phoenix-data:
  ollama-data:
  model-cache:
```

```bash
# Deploy stack
docker-compose up -d

# Pull Ollama models
docker-compose exec ollama ollama pull llama3.2
docker-compose exec ollama ollama pull nomic-embed-text

# View logs
docker-compose logs -f cogniverse

# Scale application
docker-compose up -d --scale cogniverse=3
```

## Cloud Deployment

### AWS ECS Deployment

#### Infrastructure (Terraform)

```terraform
# main.tf
provider "aws" {
  region = "us-west-2"
}

# ECS Cluster
resource "aws_ecs_cluster" "cogniverse" {
  name = "cogniverse-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

# Vespa Service
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
        },
        {
          containerPort = 19071
          protocol      = "tcp"
        }
      ]
      mountPoints = [
        {
          sourceVolume  = "vespa-data"
          containerPath = "/opt/vespa/var"
        }
      ]
      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8080/ApplicationStatus || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 60
      }
    }
  ])

  volume {
    name = "vespa-data"
    efs_volume_configuration {
      file_system_id = aws_efs_file_system.vespa.id
    }
  }
}

# Application Service
resource "aws_ecs_task_definition" "cogniverse" {
  family                   = "cogniverse"
  requires_compatibilities = ["FARGATE"]
  network_mode            = "awsvpc"
  cpu                     = "2048"
  memory                  = "8192"

  container_definitions = jsonencode([
    {
      name  = "cogniverse"
      image = "${aws_ecr_repository.cogniverse.repository_url}:latest"
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      environment = [
        {
          name  = "VESPA_HOST"
          value = "vespa.internal.cogniverse.local"
        },
        {
          name  = "PHOENIX_COLLECTOR_ENDPOINT"
          value = "phoenix.internal.cogniverse.local:4317"
        },
        {
          name  = "ENVIRONMENT"
          value = "production"
        }
      ]
      secrets = [
        {
          name      = "OPENAI_API_KEY"
          valueFrom = aws_secretsmanager_secret.cogniverse.arn
        }
      ]
    }
  ])
}

# ECS Services
resource "aws_ecs_service" "vespa" {
  name            = "vespa"
  cluster         = aws_ecs_cluster.cogniverse.id
  task_definition = aws_ecs_task_definition.vespa.arn
  desired_count   = 3
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.vespa.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.vespa.arn
    container_name   = "vespa"
    container_port   = 8080
  }
}

resource "aws_ecs_service" "cogniverse" {
  name            = "cogniverse"
  cluster         = aws_ecs_cluster.cogniverse.id
  task_definition = aws_ecs_task_definition.cogniverse.arn
  desired_count   = 5
  launch_type     = "FARGATE"

  network_configuration {
    subnets         = aws_subnet.private[*].id
    security_groups = [aws_security_group.cogniverse.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.cogniverse.arn
    container_name   = "cogniverse"
    container_port   = 8000
  }

  # Auto-scaling
  depends_on = [aws_lb_listener.cogniverse]
}

# Auto-scaling
resource "aws_appautoscaling_target" "cogniverse" {
  max_capacity       = 20
  min_capacity       = 2
  resource_id        = "service/${aws_ecs_cluster.cogniverse.name}/${aws_ecs_service.cogniverse.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cogniverse_cpu" {
  name               = "cogniverse-cpu-autoscaling"
  policy_type        = "TargetTrackingScaling"
  resource_id        = aws_appautoscaling_target.cogniverse.resource_id
  scalable_dimension = aws_appautoscaling_target.cogniverse.scalable_dimension
  service_namespace  = aws_appautoscaling_target.cogniverse.service_namespace

  target_tracking_scaling_policy_configuration {
    predefined_metric_specification {
      predefined_metric_type = "ECSServiceAverageCPUUtilization"
    }
    target_value = 70.0
  }
}
```

#### Deployment Script

```bash
#!/bin/bash
# deploy-aws.sh

set -e

# Variables
AWS_REGION="us-west-2"
ECR_REGISTRY="123456789.dkr.ecr.us-west-2.amazonaws.com"
IMAGE_NAME="cogniverse"
CLUSTER_NAME="cogniverse-cluster"
SERVICE_NAME="cogniverse"

# Build image
echo "Building Docker image..."
docker build -t $IMAGE_NAME .

# Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_REGISTRY

# Tag and push
echo "Pushing image to ECR..."
docker tag $IMAGE_NAME:latest $ECR_REGISTRY/$IMAGE_NAME:latest
docker tag $IMAGE_NAME:latest $ECR_REGISTRY/$IMAGE_NAME:$(git rev-parse --short HEAD)
docker push $ECR_REGISTRY/$IMAGE_NAME:latest
docker push $ECR_REGISTRY/$IMAGE_NAME:$(git rev-parse --short HEAD)

# Update ECS service
echo "Updating ECS service..."
aws ecs update-service \
  --cluster $CLUSTER_NAME \
  --service $SERVICE_NAME \
  --force-new-deployment \
  --region $AWS_REGION

echo "Deployment initiated. Monitor progress:"
echo "aws ecs wait services-stable --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION"
```

### GCP Cloud Run Deployment

```bash
#!/bin/bash
# deploy-gcp.sh

set -e

# Variables
PROJECT_ID="cogniverse-prod"
REGION="us-central1"
SERVICE_NAME="cogniverse"

# Build and push image
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME

# Deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
  --platform managed \
  --region $REGION \
  --memory 8Gi \
  --cpu 4 \
  --timeout 300 \
  --concurrency 100 \
  --min-instances 2 \
  --max-instances 20 \
  --set-env-vars "ENVIRONMENT=production,VESPA_HOST=vespa-service" \
  --set-secrets "OPENAI_API_KEY=cogniverse-secrets:latest" \
  --allow-unauthenticated

# Set up custom domain
gcloud run domain-mappings create \
  --service $SERVICE_NAME \
  --domain api.cogniverse.example.com \
  --region $REGION
```

### Modal Deployment (Serverless GPU)

```python
# modal_app.py
import modal

app = modal.App("cogniverse")

# GPU-optimized image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        "apt-get update && apt-get install -y ffmpeg git",
        "huggingface-cli download vidore/colsmol-500m",
    )
)

# Video processing function with GPU
@app.function(
    image=image,
    gpu="A10G",  # 24GB VRAM
    memory=32768,  # 32GB RAM
    timeout=1800,  # 30 minutes
    secrets=[modal.Secret.from_name("cogniverse-secrets")],
    volumes={"/models": modal.Volume.from_name("model-cache")}
)
async def process_video(
    video_url: str,
    profile: str = "video_colpali_smol500_mv_frame",
    tenant_id: str = "default"
):
    """Process video with ColPali/VideoPrism on GPU"""
    from src.app.ingestion import VideoIngestionPipeline

    pipeline = VideoIngestionPipeline(profile=profile, tenant_id=tenant_id)
    result = await pipeline.process_video_from_url(video_url)

    return {
        "video_id": result.video_id,
        "documents_created": len(result.documents),
        "processing_time_seconds": result.processing_time
    }

# Search endpoint (CPU-only, fast)
@app.function(
    image=image,
    memory=8192,
    timeout=30,
    secrets=[modal.Secret.from_name("cogniverse-secrets")]
)
async def search(
    query: str,
    profile: str = "video_colpali_smol500_mv_frame",
    ranking_strategy: str = "hybrid_float_bm25",
    top_k: int = 10,
    tenant_id: str = "default"
):
    """Execute video search"""
    from src.app.agents.enhanced_video_search_agent import EnhancedVideoSearchAgent

    agent = EnhancedVideoSearchAgent(profile=profile, tenant_id=tenant_id)
    results = await agent.search(
        query=query,
        ranking_strategy=ranking_strategy,
        top_k=top_k
    )

    return {
        "results": [r.to_dict() for r in results],
        "count": len(results)
    }

# FastAPI web endpoint
@app.function(image=image)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from src.api.routes import router

    app = FastAPI(title="Cogniverse API")
    app.include_router(router)

    return app
```

```bash
# Deploy to Modal
modal deploy modal_app.py

# Test endpoints
modal run modal_app.py::search --query "machine learning tutorial"
modal run modal_app.py::process_video --video-url "https://example.com/video.mp4"
```

## Kubernetes Deployment

### Helm Chart Structure

```
helm/cogniverse/
├── Chart.yaml
├── values.yaml
├── templates/
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── ingress.yaml
│   ├── configmap.yaml
│   ├── secrets.yaml
│   ├── hpa.yaml
│   ├── pdb.yaml
│   └── servicemonitor.yaml
└── values-production.yaml
```

### Values Configuration

```yaml
# helm/cogniverse/values.yaml
replicaCount: 3

image:
  repository: cogniverse
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8000
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: api.cogniverse.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: cogniverse-tls
      hosts:
        - api.cogniverse.example.com

resources:
  limits:
    cpu: 4000m
    memory: 8Gi
  requests:
    cpu: 2000m
    memory: 4Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

podDisruptionBudget:
  enabled: true
  minAvailable: 1

# Vespa configuration
vespa:
  enabled: true
  replicas: 3
  storage: 100Gi
  memory: 16Gi
  cpu: 4000m

# Phoenix configuration
phoenix:
  enabled: true
  storage: 50Gi
  memory: 4Gi

# Ollama configuration
ollama:
  enabled: true
  gpu:
    enabled: true
    count: 1
  models:
    - llama3.2
    - nomic-embed-text

# Environment variables
env:
  ENVIRONMENT: production
  TELEMETRY_ENABLED: "true"
  TELEMETRY_LEVEL: detailed
  PHOENIX_ENABLED: "true"
  LOG_LEVEL: INFO

# Secrets (from Kubernetes secrets)
secrets:
  enabled: true
  name: cogniverse-secrets
  keys:
    - OPENAI_API_KEY
    - MODAL_TOKEN
```

### Deploy with Helm

```bash
# Add Helm repo (if using packaged chart)
helm repo add cogniverse https://charts.cogniverse.example.com
helm repo update

# Install
helm install cogniverse cogniverse/cogniverse \
  --namespace cogniverse \
  --create-namespace \
  --values values-production.yaml \
  --wait

# Upgrade
helm upgrade cogniverse cogniverse/cogniverse \
  --namespace cogniverse \
  --values values-production.yaml \
  --wait

# Rollback
helm rollback cogniverse 1 --namespace cogniverse
```

## Production Configuration

### Multi-Tenant Schema Deployment

```python
# scripts/deploy_tenant_schemas.py
"""Deploy schemas for all tenants on production Vespa"""
import asyncio
from src.backends.vespa.schema_manager import VespaSchemaManager

async def deploy_all_tenant_schemas():
    manager = VespaSchemaManager(
        vespa_endpoint="https://vespa.prod.example.com",
        vespa_port=443
    )

    tenants = ["acme", "globex", "initech"]  # Production tenants
    base_schemas = [
        "video_colpali_smol500_mv_frame",
        "video_videoprism_base_mv_chunk_30s"
    ]

    for tenant in tenants:
        for schema in base_schemas:
            tenant_schema = f"{schema}_{tenant}"
            if not manager.schema_exists(tenant_schema):
                print(f"Deploying schema: {tenant_schema}")
                manager.deploy_tenant_schema(
                    base_schema=schema,
                    tenant_id=tenant
                )
            else:
                print(f"Schema exists: {tenant_schema}")

if __name__ == "__main__":
    asyncio.run(deploy_all_tenant_schemas())
```

```bash
# Deploy tenant schemas
uv run python scripts/deploy_tenant_schemas.py
```

### Environment-Specific Configuration

```bash
# Production environment variables
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export WORKERS=8

# Service endpoints (internal DNS)
export VESPA_HOST=vespa.internal.cogniverse.local
export VESPA_PORT=8080
export PHOENIX_COLLECTOR_ENDPOINT=phoenix.internal.cogniverse.local:4317
export OLLAMA_BASE_URL=http://ollama.internal.cogniverse.local:11434/v1

# Telemetry configuration
export TELEMETRY_ENABLED=true
export TELEMETRY_LEVEL=detailed
export PHOENIX_ENABLED=true
export TELEMETRY_SYNC_EXPORT=false  # Use batch async export

# Performance tuning
export VESPA_FEED_MAX_WORKERS=16
export VESPA_FEED_MAX_QUEUE_SIZE=2048
export BATCH_SIZE=100
export MAX_CONCURRENT_VIDEOS=8

# Model configuration
export MODEL_CACHE_DIR=/mnt/models
export REMOTE_INFERENCE_URL=https://api.modal.com/v1
export REMOTE_INFERENCE_API_KEY=${MODAL_TOKEN}

# Multi-tenant settings
export DEFAULT_TENANT_ID=default
export MAX_CACHED_TENANTS=200
```

## Monitoring & Observability

### Phoenix Dashboard Access

```bash
# Port-forward Phoenix in Kubernetes
kubectl port-forward -n cogniverse svc/phoenix 6006:6006

# Access dashboard
open http://localhost:6006

# View tenant-specific traces
# Navigate to project: cogniverse-{tenant_id}-video-search
```

### Prometheus Metrics

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'cogniverse'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - cogniverse
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
```

### Grafana Dashboards

**Cogniverse Overview Dashboard:**
- Request rate (queries/sec)
- P50/P95/P99 latency
- Error rate by tenant
- Vespa query performance
- Agent routing distribution
- Phoenix experiment success rate

**Import dashboard JSON:**
```bash
# Import pre-built dashboard
kubectl create configmap grafana-dashboard-cogniverse \
  --from-file=dashboards/cogniverse-overview.json \
  -n monitoring
```

## Backup & Disaster Recovery

### Automated Backup Script

```bash
#!/bin/bash
# scripts/backup-production.sh

set -e

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/$BACKUP_DATE"
S3_BUCKET="s3://cogniverse-backups-prod"

mkdir -p $BACKUP_DIR

# Backup Vespa data
echo "Backing up Vespa..."
kubectl exec -n cogniverse deployment/vespa -- \
  vespa-backup --cluster video --outputdir /backup

kubectl cp cogniverse/vespa:/backup $BACKUP_DIR/vespa/

# Backup Phoenix data
echo "Backing up Phoenix..."
kubectl exec -n cogniverse deployment/phoenix -- \
  tar -czf /tmp/phoenix-backup.tar.gz /data

kubectl cp cogniverse/phoenix:/tmp/phoenix-backup.tar.gz \
  $BACKUP_DIR/phoenix/phoenix-data.tar.gz

# Backup configurations
echo "Backing up configurations..."
kubectl get configmap -n cogniverse -o yaml > $BACKUP_DIR/configmaps.yaml
kubectl get secret -n cogniverse -o yaml > $BACKUP_DIR/secrets.yaml

# Upload to S3
echo "Uploading to S3..."
aws s3 sync $BACKUP_DIR $S3_BUCKET/$BACKUP_DATE/ \
  --storage-class GLACIER \
  --sse AES256

# Cleanup local backup
rm -rf $BACKUP_DIR

echo "Backup complete: $S3_BUCKET/$BACKUP_DATE/"
```

```bash
# Schedule daily backups (cron)
0 2 * * * /app/scripts/backup-production.sh >> /var/log/backups.log 2>&1
```

## Security

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: cogniverse-network-policy
  namespace: cogniverse
spec:
  podSelector:
    matchLabels:
      app: cogniverse
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow from ingress controller
    - from:
      - namespaceSelector:
          matchLabels:
            name: ingress-nginx
      ports:
      - protocol: TCP
        port: 8000
  egress:
    # Allow to Vespa
    - to:
      - podSelector:
          matchLabels:
            app: vespa
      ports:
      - protocol: TCP
        port: 8080
    # Allow to Phoenix
    - to:
      - podSelector:
          matchLabels:
            app: phoenix
      ports:
      - protocol: TCP
        port: 4317
    # Allow DNS
    - to:
      - namespaceSelector: {}
      ports:
      - protocol: UDP
        port: 53
```

### Pod Security Policy

```yaml
# pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: cogniverse-restricted
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - configMap
    - emptyDir
    - projected
    - secret
    - downwardAPI
    - persistentVolumeClaim
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  fsGroup:
    rule: RunAsAny
```

## Troubleshooting

### Common Issues

**1. Vespa Connection Timeout**
```bash
# Check Vespa pod status
kubectl get pods -n cogniverse -l app=vespa

# Check logs
kubectl logs -n cogniverse deployment/vespa --tail=100

# Test connectivity
kubectl exec -n cogniverse deployment/cogniverse -- \
  curl http://vespa:8080/ApplicationStatus

# Restart if needed
kubectl rollout restart deployment/vespa -n cogniverse
```

**2. Out of Memory (OOMKilled)**
```bash
# Check pod events
kubectl describe pod -n cogniverse <pod-name>

# Increase memory limits
kubectl set resources deployment/cogniverse \
  --limits=memory=16Gi \
  --requests=memory=8Gi \
  -n cogniverse
```

**3. Slow Phoenix Span Export**
```bash
# Check Phoenix collector is accessible
kubectl exec -n cogniverse deployment/cogniverse -- \
  curl http://phoenix:4317

# Check batch export configuration
kubectl set env deployment/cogniverse \
  TELEMETRY_SYNC_EXPORT=false \
  -n cogniverse
```

**4. Tenant Schema Not Found**
```bash
# List all schemas in Vespa
curl http://vespa:8080/document/v1/ | jq '.schemas'

# Deploy missing schema
kubectl exec -n cogniverse deployment/cogniverse -- \
  python scripts/deploy_tenant_schemas.py
```

**5. Model Loading Failures**
```bash
# Check model cache volume
kubectl get pvc -n cogniverse

# Pre-download models
kubectl exec -n cogniverse deployment/cogniverse -- \
  python scripts/download_models.py --all

# Verify model files
kubectl exec -n cogniverse deployment/cogniverse -- \
  ls -lh /models/
```

## Related Documentation

- [Architecture Overview](architecture.md) - System design
- [Multi-Tenant System](multi-tenant-system.md) - Tenant isolation
- [Phoenix Integration](phoenix-integration.md) - Observability
- [System Flows](system-flows.md) - Request traces

**Last Updated**: 2025-10-04
