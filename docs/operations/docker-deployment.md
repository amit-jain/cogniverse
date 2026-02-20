# Docker Deployment Guide

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Development Deployment](#development-deployment)
- [Production Deployment](#production-deployment)
- [Service Configuration](#service-configuration)
- [Multi-Tenant Setup](#multi-tenant-setup)
- [Operations](#operations)
- [Monitoring & Health Checks](#monitoring-health-checks)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

Cogniverse provides Docker Compose configurations for both development and production deployments:

- **`deployment/docker-compose.yml`** - Development environment with debug settings and local volume mounts
- **`deployment/docker-compose.prod.yml`** - Production environment with resource limits, security hardening, and scaling

### Service Stack

| Service | Purpose | Ports | Storage |
|---------|---------|-------|---------|
| **Vespa** | Vector database and search engine | 8080, 19071, 19092 | vespa-data |
| **Runtime** | FastAPI backend service | 8000 | runtime-data, runtime-logs (production only) |
| **Dashboard** | Streamlit UI | 8501 | Read-only access to runtime data |
| **Phoenix** | Telemetry and experiments | 6006 | phoenix-data |
| **Ollama** | Local LLM inference | 11434 | ollama-data |
| **OTEL Collector** | Telemetry collection | 4317, 4318, 8888 | None |
| **Nginx** (optional) | Reverse proxy | 80, 443 | nginx-logs (production only) |

---

## Prerequisites

### System Requirements

**Development:**

- CPU: 4+ cores

- RAM: 16GB minimum

- Storage: 50GB free space

- OS: Linux, macOS, Windows (with WSL2)

**Production:**

- CPU: 8+ cores

- RAM: 32GB minimum

- Storage: 100GB+ free space

- GPU: NVIDIA GPU with CUDA support (recommended for Ollama)

### Required Software

```bash
# Docker Engine 20.10+ and Docker Compose 2.0+
docker --version
docker-compose --version

# UV package manager (for initialization scripts)
pip install uv

# NVIDIA Container Toolkit (for GPU support)
# Follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

---

## Quick Start

### 1. Clone Repository

```bash
# Clone repository (replace with actual repository URL)
git clone https://github.com/your-org/cogniverse.git
cd cogniverse
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional for development)
nano .env
```

### 3. Start Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

### 4. Initialize Services

```bash
# Wait for services to be healthy and initialize
./scripts/docker/init_services.sh

# This will:
# - Wait for all services to be healthy
# - Deploy Vespa schemas for default tenant
# - Pull default Ollama models (llama2:7b, mistral:7b)
# - Create Phoenix projects
# - Verify connectivity
```

### 5. Verify Deployment

```bash
# Run health checks
./scripts/docker/health_check.sh --verbose

# Access services
open http://localhost:8000/docs       # API documentation
open http://localhost:8501            # Dashboard
open http://localhost:6006            # Phoenix
```

---

## Development Deployment

### Starting Development Environment

```bash
# Start with build
docker-compose up -d --build

# View specific service logs
docker-compose logs -f runtime
docker-compose logs -f dashboard

# Follow all logs
docker-compose logs -f
```

### Development Features

- **Debug Logging:** LOG_LEVEL=DEBUG for detailed logs
- **Local Volumes:** Data and config files mounted for easy access
- **GPU Passthrough:** Automatic GPU access for models
- **Note:** Code is baked into Docker images - rebuild containers after code changes

### Configuration

Edit `.env` file for development settings:

```bash
# Development settings
LOG_LEVEL=DEBUG
RUNTIME_WORKERS=2
OLLAMA_NUM_PARALLEL=2
CACHE_ENABLED=true
```

### Working with Local Code

```bash
# Development deployment/docker-compose.yml mounts:
volumes:
  - ./configs/config.json:/app/configs/config.json:ro
  - ./data:/data
  - ./outputs/logs:/logs

# Code is baked into the image - rebuild after code or dependency changes
docker-compose up -d --build runtime
```

---

## Production Deployment

### Preparation

#### 1. Environment Configuration

```bash
# Copy production template
cp deployment/.env.prod.example deployment/.env.prod

# Edit with production values
nano .env.prod

# Key settings to configure:
# - VERSION: Use specific version tag (e.g., 2.0.0)
# - API_KEY: Strong API key from secrets manager
# - JWT_SECRET_KEY: Strong JWT secret
# - MEM0_API_KEY: Mem0 API credentials
# - CORS_ORIGINS: Your production domain(s)
# - LOG_LEVEL: INFO or WARNING
```

#### 2. SSL/TLS Certificates

```bash
# Create SSL directory structure (not included in repository)
# Note: These directories don't exist by default and must be created
mkdir -p configs/nginx/ssl
mkdir -p configs/nginx/conf.d

# Generate self-signed certificate (for testing)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout configs/nginx/ssl/cogniverse.key \
  -out configs/nginx/ssl/cogniverse.crt

# For production, use Let's Encrypt or your CA
# You will also need to create configs/nginx/nginx.conf and appropriate conf.d files
```

#### 3. Production Configuration Files

```bash
# Create production config (optional - configs/config.json can be used for production too)
cp configs/config.json configs/config.prod.json
nano configs/config.prod.json

# Phoenix accepts OTLP gRPC natively on port 4317 — no separate collector needed
```

### Deployment

#### Start Production Stack

```bash
# Pull latest images
docker compose -f deployment/docker-compose.prod.yml pull

# Start services
docker compose -f deployment/docker-compose.prod.yml up -d

# With Nginx reverse proxy
docker compose -f deployment/docker-compose.prod.yml --profile with-nginx up -d

# Initialize services with multiple tenants
TENANTS="default acme_corp globex_inc" \
OLLAMA_MODELS="mistral:7b-instruct llama2:13b" \
./scripts/docker/init_services.sh
```

#### Scaling Services

```bash
# Scale runtime instances
docker compose -f deployment/docker-compose.prod.yml up -d --scale runtime=4

# Scale with resource limits
docker compose -f deployment/docker-compose.prod.yml up -d \
  --scale runtime=4 \
  --scale dashboard=2
```

#### Health Verification

```bash
# Comprehensive health check
./scripts/docker/health_check.sh --verbose

# JSON output for monitoring
./scripts/docker/health_check.sh --format json

# Prometheus metrics
./scripts/docker/health_check.sh --format prometheus
```

### Production Features

- **Resource Limits:** CPU and memory limits for all services
- **Health Checks:** Stricter health check intervals
- **Logging:** Structured JSON logs with rotation
- **Security:** XSRF protection, restricted CORS
- **Monitoring:** Prometheus metrics endpoints
- **SSL/TLS:** Nginx reverse proxy with SSL termination
- **Scaling:** Load balanced runtime replicas

---

## Service Configuration

### Vespa

**Configuration:**
```yaml
vespa:
  image: vespaengine/vespa:8.259.19  # Pinned for production
  environment:
    - VESPA_MEMORY_OPTIONS=-Xms4g -Xmx16g
  deploy:
    resources:
      limits:
        cpus: '8'
        memory: 20G
```

**Schema Deployment:**
```bash
# Deploy individual schema
uv run python scripts/deploy_json_schema.py \
  configs/schemas/video_colpali_smol500_mv_frame_schema.json

# Deploy all schemas
uv run python scripts/deploy_all_schemas.py
```

**Data Management:**
```bash
# Backup Vespa data
docker run --rm \
  -v cogniverse_vespa-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar czf /backup/vespa-$(date +%Y%m%d).tar.gz /data

# Restore Vespa data
docker run --rm \
  -v cogniverse_vespa-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/vespa-20251015.tar.gz -C /
```

### Phoenix

**Configuration:**
```yaml
phoenix:
  image: arizephoenix/phoenix:4.11.0
  environment:
    - PHOENIX_WORKING_DIR=/phoenix-data
  volumes:
    - phoenix-data:/phoenix-data
```

**Project Management:**
```bash
# Phoenix projects are auto-created per tenant
# Access dashboard: http://localhost:6006

# Export experiment data (production example - use cogniverse-phoenix for development)
docker exec cogniverse-phoenix-prod \
  tar czf /tmp/phoenix-export.tar.gz /phoenix-data

docker cp cogniverse-phoenix-prod:/tmp/phoenix-export.tar.gz \
  ./backups/
```

### Ollama

**Configuration:**
```yaml
ollama:
  image: ollama/ollama:0.1.22
  environment:
    - OLLAMA_NUM_PARALLEL=4
    - OLLAMA_MAX_LOADED_MODELS=2
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

**Model Management:**
```bash
# Pull models (production example - use cogniverse-ollama for development)
docker exec cogniverse-ollama-prod ollama pull mistral:7b-instruct
docker exec cogniverse-ollama-prod ollama pull llama2:13b

# List models
docker exec cogniverse-ollama-prod ollama list

# Remove models
docker exec cogniverse-ollama-prod ollama rm llama2:7b

# Test model
curl http://localhost:11434/api/generate -d '{
  "model": "mistral:7b-instruct",
  "prompt": "What is machine learning?",
  "stream": false
}'
```

**GPU Configuration:**
```bash
# Verify GPU access (production example - use cogniverse-ollama for development)
docker exec cogniverse-ollama-prod nvidia-smi

# Check Ollama GPU usage
docker exec cogniverse-ollama-prod ollama ps
```

### Runtime API

**Configuration:**
```yaml
runtime:
  environment:
    - RUNTIME_WORKERS=4
    - LOG_LEVEL=INFO
  deploy:
    replicas: 2
```

**Testing:**
```bash
# Health check
curl http://localhost:8000/health

# Search API
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning tutorial",
    "top_k": 10,
    "tenant_id": "acme_corp"
  }'

# API documentation
open http://localhost:8000/docs
```

### Dashboard

**Configuration:**
```yaml
dashboard:
  environment:
    - STREAMLIT_SERVER_ENABLE_CORS=false
    - STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

**Access:**
```bash
# Local access
open http://localhost:8501

# Behind Nginx (production)
open https://dashboard.your-domain.com
```

---

## Multi-Tenant Setup

### Initialize Multiple Tenants

```bash
# Deploy schemas for multiple tenants
TENANTS="acme_corp globex_inc initech" ./scripts/docker/init_services.sh

# Or deploy all schemas
uv run python scripts/deploy_all_schemas.py
```

### Tenant-Specific Configuration

```yaml
# Runtime configuration for tenant isolation
runtime:
  environment:
    - DEFAULT_TENANT_ID=default
    - TENANT_MAX_STORAGE_GB=100
    - TENANT_MAX_VIDEOS=10000
    - TENANT_MAX_QPS=100
```

### Tenant Operations

```bash
# Ingest videos for specific tenant
# Note: Ensure profile name matches deployed schema
#       Available schemas in configs/schemas/: video_colpali_smol500_mv_frame,
#       video_colqwen_omni_mv_chunk_30s, video_videoprism_base_mv_chunk_30s, etc.
uv run python scripts/run_ingestion.py \
  --video_dir /data/acme_corp/videos \
  --profile video_colpali_smol500_mv_frame \
  --tenant-id acme_corp \
  --backend vespa

# Query as specific tenant
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "quarterly results", "tenant_id": "acme_corp"}'

# View tenant-specific Phoenix dashboard
open http://localhost:6006/projects/acme_corp_project
```

---

## Operations

### Start/Stop Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d runtime

# Stop all services
docker-compose stop

# Stop specific service
docker-compose stop runtime

# Restart service
docker-compose restart runtime

# Remove all containers (keeps volumes)
docker-compose down

# Remove containers and volumes (DATA LOSS)
docker-compose down -v
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f runtime

# Last 100 lines
docker-compose logs --tail 100 runtime

# Since timestamp
docker-compose logs --since 2025-10-15T10:00:00 runtime

# Export logs
docker-compose logs --no-color runtime > runtime-logs.txt
```

### Update Services

```bash
# Pull latest images
docker-compose pull

# Rebuild and restart
docker-compose up -d --build

# Rolling update (zero downtime)
docker-compose up -d --no-deps --build runtime

# Update specific version
VERSION=2.1.0 docker compose -f deployment/docker-compose.prod.yml up -d
```

### Data Management

```bash
# List volumes
docker volume ls | grep cogniverse

# Inspect volume
docker volume inspect cogniverse_vespa-data

# Backup all volumes
for vol in $(docker volume ls -q | grep cogniverse); do
  docker run --rm \
    -v $vol:/data \
    -v $(pwd)/backups:/backup \
    alpine tar czf /backup/${vol}-$(date +%Y%m%d).tar.gz /data
done

# Restore volume
docker run --rm \
  -v cogniverse_vespa-data:/data \
  -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/cogniverse_vespa-data-20251015.tar.gz -C /
```

### Resource Monitoring

```bash
# Container resource usage
docker stats

# Service-specific stats (use -prod suffix for production containers)
docker stats cogniverse-runtime  # development
docker stats cogniverse-runtime-prod  # production

# Disk usage
docker system df
docker system df -v

# Network inspection
docker network inspect cogniverse
```

---

## Monitoring & Health Checks

### Automated Health Checks

```bash
# Run health check script
./scripts/docker/health_check.sh --verbose

# JSON output for monitoring systems
./scripts/docker/health_check.sh --format json | jq

# Prometheus metrics
./scripts/docker/health_check.sh --format prometheus

# Scheduled health checks (cron)
*/5 * * * * /path/to/cogniverse/scripts/docker/health_check.sh --format json >> /var/log/cogniverse-health.log
```

### Manual Health Checks

```bash
# Vespa
curl http://localhost:8080/ApplicationStatus
curl http://localhost:19071/state/v1/health

# Runtime
curl http://localhost:8000/health

# Phoenix
curl http://localhost:6006/health

# Ollama
curl http://localhost:11434/api/tags

# Dashboard
curl http://localhost:8501/_stcore/health

# OTEL Collector (health check port only available in production)
curl http://localhost:13133/
```

### Docker Health Status

```bash
# View health status
docker-compose ps

# Detailed health info (adjust container name: development uses no suffix, production uses -prod)
docker inspect \
  --format='{{.State.Health.Status}}' \
  cogniverse-runtime-prod  # or cogniverse-runtime for development

# Health check logs
docker inspect \
  --format='{{range .State.Health.Log}}{{.Output}}{{end}}' \
  cogniverse-runtime-prod  # or cogniverse-runtime for development
```

### Prometheus Metrics

```bash
# OTEL Collector metrics
curl http://localhost:8888/metrics

# Vespa metrics
curl http://localhost:19092/metrics
```

---

## Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check logs
docker-compose logs vespa

# Common causes:
# - Port conflicts
# - Insufficient resources
# - Missing volumes
# - Permission issues

# Verify ports are available
sudo lsof -i :8080
sudo lsof -i :8000

# Check disk space
df -h

# Check memory
free -h
```

#### Vespa Not Responding

```bash
# Check Vespa status
curl http://localhost:8080/ApplicationStatus

# Restart Vespa
docker-compose restart vespa

# Check Vespa logs
docker-compose logs vespa | grep ERROR

# Common issues:
# - Insufficient memory
# - Schema deployment failures
# - Data corruption

# Reset Vespa (DATA LOSS)
docker-compose down
docker volume rm cogniverse_vespa-data
docker-compose up -d vespa
./scripts/docker/init_services.sh
```

#### Ollama GPU Issues

```bash
# Verify GPU access (use cogniverse-ollama for dev, cogniverse-ollama-prod for production)
docker exec cogniverse-ollama nvidia-smi  # development
docker exec cogniverse-ollama-prod nvidia-smi  # production

# Check CUDA availability
docker exec cogniverse-ollama nvidia-smi --query-gpu=name,driver_version --format=csv  # adjust container name

# Restart with GPU
docker-compose restart ollama

# Fallback to CPU mode (slower)
# Edit deployment/docker-compose.yml and remove GPU reservation
```

#### Runtime API Errors

```bash
# Check runtime logs
docker-compose logs -f runtime

# Test health endpoint
curl -v http://localhost:8000/health

# Check environment variables
docker exec cogniverse-runtime env | grep VESPA

# Restart runtime
docker-compose restart runtime

# Rebuild if code changed
docker-compose up -d --build runtime
```

#### Phoenix Data Issues

```bash
# Check Phoenix logs
docker-compose logs phoenix

# Verify Phoenix volume
docker volume inspect cogniverse_phoenix-data

# Reset Phoenix (DATA LOSS)
docker-compose down
docker volume rm cogniverse_phoenix-data
docker-compose up -d phoenix
```

### Debug Mode

```bash
# Enable debug logging
echo "LOG_LEVEL=DEBUG" >> .env
docker-compose up -d --build runtime

# Access container shell (adjust for dev/prod: cogniverse-runtime or cogniverse-runtime-prod)
docker exec -it cogniverse-runtime bash

# Check Python packages
docker exec cogniverse-runtime uv pip list

# Test imports
docker exec cogniverse-runtime python -c "from cogniverse_foundation.config.unified_config import SystemConfig"
```

### Network Issues

```bash
# Check network
docker network inspect cogniverse

# Test connectivity between services
docker exec cogniverse-runtime curl http://vespa:8080/ApplicationStatus
docker exec cogniverse-runtime curl http://phoenix:6006/health

# Recreate network
docker-compose down
docker network rm cogniverse
docker-compose up -d
```

---

## Best Practices

### Security

1. **Never commit `.env.prod` with actual secrets**
   ```bash
   # Add to .gitignore
   echo ".env.prod" >> .gitignore
   ```

2. **Use secrets management in production**
   ```bash
   # AWS Secrets Manager
   aws secretsmanager get-secret-value --secret-id cogniverse/api-key

   # HashiCorp Vault
   vault kv get secret/cogniverse/api-key
   ```

3. **Restrict CORS origins**
   ```bash
   CORS_ORIGINS=https://your-domain.com
   ```

4. **Enable SSL/TLS**
   ```bash
   docker compose -f deployment/docker-compose.prod.yml --profile with-nginx up -d
   ```

5. **Use specific version tags**
   ```bash
   VERSION=2.0.0 docker compose -f deployment/docker-compose.prod.yml up -d
   ```

### Performance

1. **Resource Allocation**
   ```yaml
   # Adjust based on workload
   deploy:
     resources:
       limits:
         cpus: '8'
         memory: 20G
   ```

2. **Enable Caching**
   ```bash
   CACHE_ENABLED=true
   CACHE_TTL=3600  # seconds
   CACHE_MAX_SIZE=1000  # entries
   ```

3. **Optimize Ollama**
   ```bash
   OLLAMA_NUM_PARALLEL=4
   OLLAMA_MAX_LOADED_MODELS=2
   ```

4. **Scale Runtime**
   ```bash
   docker compose -f deployment/docker-compose.prod.yml up -d --scale runtime=4
   ```

### Monitoring

1. **Set up health check cron**
   ```bash
   */5 * * * * /path/to/scripts/docker/health_check.sh --format json >> /var/log/cogniverse-health.log
   ```

2. **Configure alerting**
   ```bash
   # Slack webhook
   if [ $(./scripts/docker/health_check.sh | grep UNHEALTHY | wc -l) -gt 0 ]; then
     curl -X POST $SLACK_WEBHOOK_URL -d '{"text":"Cogniverse health check failed"}'
   fi
   ```

3. **Log aggregation**
   ```yaml
   # Use ELK, Datadog, or CloudWatch
   logging:
     driver: "json-file"
     options:
       max-size: "50m"
       max-file: "5"
   ```

### Backup & Recovery

1. **Regular backups**
   ```bash
   # Daily backup script
   #!/bin/bash
   DATE=$(date +%Y%m%d)
   for vol in vespa-data phoenix-data ollama-data; do
     docker run --rm \
       -v cogniverse_${vol}:/data \
       -v /backups:/backup \
       alpine tar czf /backup/${vol}-${DATE}.tar.gz /data
   done
   ```

2. **Test restores**
   ```bash
   # Regularly test backup restoration
   docker-compose -f docker-compose.test.yml up -d
   # Restore backup
   # Verify functionality
   docker-compose -f docker-compose.test.yml down
   ```

3. **Version control**
   ```bash
   # Commit configuration changes
   git add deployment/docker-compose.prod.yml deployment/.env.prod.example
   git commit -m "Update production configuration"
   ```

---

## Related Documentation

- [Setup & Installation](setup-installation.md) - Initial UV workspace setup
- [Configuration Guide](configuration.md) - System configuration details
- [Multi-Tenant Operations](multi-tenant-ops.md) - Tenant lifecycle management
- [Kubernetes Deployment](kubernetes-deployment.md) - K8s and Helm charts
- [Testing Guide](../testing/pytest-best-practices.md) - Testing practices

---

## Support

- **Health Issues:** Run `./scripts/docker/health_check.sh --verbose`
- **Logs:** Check `docker-compose logs -f [service]`
- **Documentation:** See `docs/` directory
- **Issues:** GitHub Issues

---
