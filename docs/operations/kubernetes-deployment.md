# Kubernetes Deployment Guide

**Last Updated:** 2025-10-15
**Architecture:** Helm-based Kubernetes deployment with multi-tenant support
**Purpose:** Complete guide for deploying Cogniverse on Kubernetes/K3s

---

## Overview

Cogniverse provides production-ready Helm charts for Kubernetes deployment with:
- **Helm Chart**: `charts/cogniverse/`
- **StatefulSets**: Vespa, Phoenix, Ollama (with persistent storage)
- **Deployments**: Runtime, Dashboard
- **Auto-scaling**: HPA for Runtime
- **Ingress**: NGINX with TLS/SSL
- **Init Jobs**: Schema deployment, model pulling

---

## Prerequisites

### Cluster Requirements

**Minimum:**
- Kubernetes 1.24+
- 3 nodes (1 master, 2 workers)
- 32GB RAM per node
- 100GB+ storage per node

**Recommended:**
- Kubernetes 1.27+
- 5+ nodes
- 64GB RAM per node
- GPU nodes for Ollama
- NVMe/SSD storage

### Required Tools

```bash
# Helm 3.x
helm version

# kubectl
kubectl version --client

# Optional: K3s (lightweight Kubernetes)
curl -sfL https://get.k3s.io | sh -
```

### Storage Class

```bash
# Check available storage classes
kubectl get storageclass

# For K3s, use local-path (default)
# For EKS, use gp3
# For GKE, use standard-rwo
```

---

## Quick Start

### 1. Add Helm Repository (if published)

```bash
# Add Cogniverse Helm repo
helm repo add cogniverse https://charts.cogniverse.ai
helm repo update

# Or use local charts
cd cogniverse/charts
```

### 2. Install with Default Values

```bash
# Create namespace
kubectl create namespace cogniverse

# Install chart
helm install cogniverse ./cogniverse \
  --namespace cogniverse \
  --create-namespace

# Check status
helm status cogniverse -n cogniverse
kubectl get pods -n cogniverse
```

### 3. Access Services

```bash
# Port-forward Runtime API
kubectl port-forward -n cogniverse svc/cogniverse-runtime 8000:8000

# Port-forward Dashboard
kubectl port-forward -n cogniverse svc/cogniverse-dashboard 8501:8501

# Access
open http://localhost:8000/docs
open http://localhost:8501
```

---

## Production Deployment

### Create Production Values

Create `values.prod.yaml`:

```yaml
# Production configuration
image:
  tag: "2.0.0"
  pullPolicy: IfNotPresent

global:
  imageRegistry: "your-registry.io"
  imagePullSecrets:
    - name: regcred

# Vespa configuration
vespa:
  replicaCount: 3
  persistence:
    enabled: true
    storageClass: "fast-ssd"
    size: "200Gi"
  resources:
    requests:
      cpu: "4"
      memory: "16Gi"
    limits:
      cpu: "8"
      memory: "32Gi"

# Runtime auto-scaling
runtime:
  replicaCount: 3
  autoscaling:
    enabled: true
    minReplicas: 3
    maxReplicas: 20
    targetCPUUtilizationPercentage: 70

# Ingress with SSL
ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: cogniverse.your-domain.com
      paths:
        - path: /api
          pathType: Prefix
          service: runtime
          port: 8000
        - path: /
          pathType: Prefix
          service: dashboard
          port: 8501
  tls:
    - secretName: cogniverse-tls
      hosts:
        - cogniverse.your-domain.com

# GPU configuration for Ollama
ollama:
  nodeSelector:
    nvidia.com/gpu: "true"
  resources:
    limits:
      nvidia.com/gpu: "1"
```

### Deploy to Production

```bash
# Deploy with production values
helm install cogniverse ./cogniverse \
  --namespace cogniverse \
  --create-namespace \
  --values values.prod.yaml

# Verify deployment
helm test cogniverse -n cogniverse
kubectl get all -n cogniverse
```

---

## K3s Local Deployment

### Overview

K3s is a lightweight Kubernetes distribution perfect for local development, testing, and edge deployments. It:
- Runs on minimal resources (single node with 4GB RAM)
- Includes built-in storage (local-path provisioner)
- Has Traefik ingress controller by default
- Supports full Kubernetes APIs

**Use Cases:**
- Local development and testing
- CI/CD pipelines
- Edge deployments
- Learning Kubernetes

### Quick Start with K3s

The easiest way to deploy to K3s is using the deployment script:

```bash
# Install K3s and deploy Cogniverse (all-in-one)
./scripts/deploy_k3s.sh --install-k3s

# Deploy to existing K3s
./scripts/deploy_k3s.sh

# Full local setup with Argo Workflows (recommended for testing batch workflows)
./scripts/deploy_k3s.sh --install-k3s --install-argo

# With custom values
./scripts/deploy_k3s.sh --values my-values.yaml
```

**Argo Workflows on K3s:**

Argo Workflows works perfectly on K3s for local testing of batch processing workflows:

```bash
# Deploy K3s + Cogniverse + Argo
./scripts/deploy_k3s.sh --install-k3s --install-argo

# Access Argo UI locally
kubectl port-forward -n argo svc/argo-server 2746:2746
open http://localhost:2746

# Submit a workflow (e.g., video ingestion)
argo submit workflows/video-ingestion.yaml \
  -n cogniverse \
  --parameter video-dir="/data/videos" \
  --parameter tenant-id="default" \
  --parameter profiles="video_colpali_smol500_mv_frame"

# Watch workflow progress
argo watch <workflow-name> -n cogniverse

# List all workflows
argo list -n cogniverse
```

This gives you the full Kubernetes + Argo experience locally without needing a cloud cluster!

### Manual K3s Installation

If you prefer manual setup:

```bash
# Install K3s
curl -sfL https://get.k3s.io | sh -s - \
  --write-kubeconfig-mode 644 \
  --disable traefik  # Optional: disable if using custom ingress

# Wait for K3s to be ready
sudo k3s kubectl get nodes

# Setup kubeconfig
mkdir -p $HOME/.kube
sudo cp /etc/rancher/k3s/k3s.yaml $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# Verify
kubectl get nodes
```

### K3s-Specific Configuration

Create `values.k3s.yaml` for optimized K3s deployment:

```yaml
# K3s-specific Helm values
# Optimized for local development

# Use local-path storage (K3s default)
global:
  storageClass: "local-path"

# Reduced resources for single-node deployment
vespa:
  replicaCount: 1
  persistence:
    enabled: true
    storageClass: "local-path"
    size: "20Gi"
  resources:
    requests:
      cpu: "1"
      memory: "4Gi"
    limits:
      cpu: "2"
      memory: "8Gi"

runtime:
  replicaCount: 1
  autoscaling:
    enabled: false  # Disable for single node
  resources:
    requests:
      cpu: "500m"
      memory: "2Gi"
    limits:
      cpu: "1"
      memory: "4Gi"

dashboard:
  replicaCount: 1
  resources:
    requests:
      cpu: "250m"
      memory: "1Gi"
    limits:
      cpu: "500m"
      memory: "2Gi"

phoenix:
  replicaCount: 1
  persistence:
    enabled: true
    storageClass: "local-path"
    size: "10Gi"
  resources:
    requests:
      cpu: "500m"
      memory: "1Gi"
    limits:
      cpu: "1"
      memory: "2Gi"

ollama:
  replicaCount: 1
  persistence:
    enabled: true
    storageClass: "local-path"
    size: "20Gi"
  resources:
    requests:
      cpu: "1"
      memory: "4Gi"
    limits:
      cpu: "2"
      memory: "8Gi"
  # GPU support (if available on local machine)
  nodeSelector: {}
  tolerations: []

# Ingress with Traefik (K3s default)
ingress:
  enabled: true
  className: "traefik"
  hosts:
    - host: cogniverse.local
      paths:
        - path: /api
          pathType: Prefix
          service: runtime
          port: 8000
        - path: /
          pathType: Prefix
          service: dashboard
          port: 8501
  tls: []  # No TLS for local development

# Local development tenant
config:
  tenants:
    - id: "default"
      name: "Default Tenant"
  ollamaModels:
    - "mistral:7b-instruct"

# Enable init jobs
initJobs:
  schemaDeployment:
    enabled: true
  modelPulling:
    enabled: true
```

### Deploy to K3s

```bash
# Create namespace
kubectl create namespace cogniverse

# Deploy with K3s values
helm install cogniverse ./charts/cogniverse \
  --namespace cogniverse \
  --values values.k3s.yaml \
  --wait \
  --timeout 10m

# Check status
kubectl get pods -n cogniverse -w
```

### Local Access Setup

For domain-based access (optional):

```bash
# Add entry to /etc/hosts
echo "127.0.0.1 cogniverse.local" | sudo tee -a /etc/hosts

# Access via domain
open http://cogniverse.local
open http://cogniverse.local/api/docs
```

Or use port-forwarding:

```bash
# Runtime API
kubectl port-forward -n cogniverse svc/cogniverse-runtime 8000:8000
open http://localhost:8000/docs

# Dashboard
kubectl port-forward -n cogniverse svc/cogniverse-dashboard 8501:8501
open http://localhost:8501

# Phoenix
kubectl port-forward -n cogniverse svc/cogniverse-phoenix 6006:6006
open http://localhost:6006
```

### K3s Operations

**View K3s logs:**
```bash
sudo journalctl -u k3s -f
```

**Restart K3s:**
```bash
sudo systemctl restart k3s
```

**Stop K3s:**
```bash
sudo systemctl stop k3s
```

**Uninstall K3s:**
```bash
# Uninstall Cogniverse first
helm uninstall cogniverse -n cogniverse
kubectl delete namespace cogniverse

# Uninstall K3s
/usr/local/bin/k3s-uninstall.sh
```

### K3s Troubleshooting

**Issue: Pods stuck in Pending**
```bash
# Check node resources
kubectl describe nodes

# K3s typically runs on single node, check if resources exhausted
kubectl top nodes
kubectl top pods -A

# Consider reducing resource requests in values.k3s.yaml
```

**Issue: Storage provisioning fails**
```bash
# Check local-path provisioner
kubectl get pods -n kube-system | grep local-path

# Check storage class
kubectl get storageclass

# Verify PVC
kubectl get pvc -n cogniverse
kubectl describe pvc <pvc-name> -n cogniverse

# Local-path uses /var/lib/rancher/k3s/storage
sudo ls -la /var/lib/rancher/k3s/storage/
```

**Issue: Cannot connect to cluster**
```bash
# Check K3s service
sudo systemctl status k3s

# Check kubeconfig
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
sudo kubectl get nodes

# Or copy to ~/.kube/config
sudo cp /etc/rancher/k3s/k3s.yaml ~/.kube/config
sudo chown $(id -u):$(id -g) ~/.kube/config
```

**Issue: Ingress not working**
```bash
# Check Traefik (K3s default ingress)
kubectl get pods -n kube-system | grep traefik

# View Traefik logs
kubectl logs -n kube-system -l app.kubernetes.io/name=traefik

# Check ingress resources
kubectl get ingress -n cogniverse
kubectl describe ingress cogniverse -n cogniverse
```

### K3s vs Full Kubernetes

| Feature | K3s | Full Kubernetes |
|---------|-----|----------------|
| Resource Usage | 512MB RAM minimum | 2GB+ RAM minimum |
| Installation | Single binary | Multi-component |
| Storage | local-path (built-in) | Requires CSI driver |
| Ingress | Traefik (built-in) | Requires installation |
| Use Case | Development, Edge | Production, Scale |
| Node Requirement | 1+ nodes | 3+ nodes (HA) |

### When to Use K3s

**✅ Good for:**
- Local development and testing
- CI/CD testing pipelines
- Learning Kubernetes concepts
- Edge deployments
- Small-scale production (single tenant)

**❌ Not ideal for:**
- Large-scale production (100+ tenants)
- High-availability requirements
- Heavy resource workloads
- Multi-region deployments

---

## Configuration

### Multi-Tenant Setup

Update `values.yaml`:

```yaml
config:
  tenants:
    - id: "acme_corp"
      name: "Acme Corporation"
    - id: "globex_inc"
      name: "Globex Inc"

# Enable multi-tenant resource quotas
multiTenant:
  enabled: true
  quotas:
    cpu: "10"
    memory: "20Gi"
    storage: "100Gi"
```

### GPU Configuration

For Ollama with GPU support:

```yaml
ollama:
  enabled: true
  nodeSelector:
    nvidia.com/gpu: "true"
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
  resources:
    limits:
      nvidia.com/gpu: "1"
      memory: "16Gi"
    requests:
      cpu: "4"
      memory: "8Gi"
```

### Persistent Storage

Configure storage for stateful services:

```yaml
vespa:
  persistence:
    enabled: true
    storageClass: "fast-ssd"  # Your storage class
    size: "200Gi"
    accessMode: ReadWriteOnce

phoenix:
  persistence:
    enabled: true
    storageClass: "standard"
    size: "50Gi"

ollama:
  persistence:
    enabled: true
    storageClass: "standard"
    size: "100Gi"
```

---

## Operations

### Upgrade Deployment

```bash
# Update values
nano values.prod.yaml

# Upgrade with new values
helm upgrade cogniverse ./cogniverse \
  --namespace cogniverse \
  --values values.prod.yaml \
  --reuse-values

# Check upgrade status
helm history cogniverse -n cogniverse
```

### Rollback

```bash
# List releases
helm history cogniverse -n cogniverse

# Rollback to previous version
helm rollback cogniverse -n cogniverse

# Rollback to specific revision
helm rollback cogniverse 3 -n cogniverse
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment cogniverse-runtime \
  --replicas=5 \
  -n cogniverse

# Or update via Helm
helm upgrade cogniverse ./cogniverse \
  --namespace cogniverse \
  --set runtime.replicaCount=5 \
  --reuse-values
```

### Backup & Restore

**Backup:**
```bash
# Backup Helm values
helm get values cogniverse -n cogniverse > backup-values.yaml

# Backup PVCs
kubectl get pvc -n cogniverse -o yaml > backup-pvcs.yaml

# Snapshot volumes (cloud-specific)
# AWS EBS: Use EBS snapshots
# GCP: Use persistent disk snapshots
```

**Restore:**
```bash
# Restore from backup
helm install cogniverse ./cogniverse \
  --namespace cogniverse \
  --values backup-values.yaml
```

---

## Monitoring

### Check Pod Status

```bash
# Get all pods
kubectl get pods -n cogniverse -o wide

# Describe pod
kubectl describe pod cogniverse-runtime-xxx -n cogniverse

# View logs
kubectl logs -f cogniverse-runtime-xxx -n cogniverse

# Previous logs
kubectl logs --previous cogniverse-runtime-xxx -n cogniverse
```

### Resource Usage

```bash
# Node resources
kubectl top nodes

# Pod resources
kubectl top pods -n cogniverse

# Detailed pod metrics
kubectl describe node <node-name>
```

### Health Checks

```bash
# Check all services
kubectl get svc -n cogniverse

# Test health endpoints
kubectl run curl --image=curlimages/curl -i --rm --restart=Never -- \
  curl http://cogniverse-runtime:8000/health

# Check HPA status
kubectl get hpa -n cogniverse
```

---

## Troubleshooting

### Pods Not Starting

```bash
# Check pod events
kubectl describe pod <pod-name> -n cogniverse

# Common issues:
# 1. Image pull errors
kubectl get events -n cogniverse --sort-by='.lastTimestamp'

# 2. Resource constraints
kubectl describe nodes

# 3. Storage issues
kubectl get pvc -n cogniverse
kubectl describe pvc <pvc-name> -n cogniverse
```

### Init Jobs Failing

```bash
# Check init job status
kubectl get jobs -n cogniverse

# View init job logs
kubectl logs job/cogniverse-schema-deployment -n cogniverse

# Delete and re-run
kubectl delete job cogniverse-schema-deployment -n cogniverse
helm upgrade cogniverse ./cogniverse -n cogniverse --reuse-values
```

### Service Connection Issues

```bash
# Test service connectivity
kubectl run test-pod --image=busybox -i --rm --restart=Never -- \
  wget -O- http://cogniverse-vespa:8080/ApplicationStatus

# Check service endpoints
kubectl get endpoints -n cogniverse

# Verify DNS
kubectl run test-dns --image=busybox -i --rm --restart=Never -- \
  nslookup cogniverse-vespa.cogniverse.svc.cluster.local
```

---

## Best Practices

### Resource Management

1. **Set resource requests and limits**
   ```yaml
   resources:
     requests:
       cpu: "2"
       memory: "4Gi"
     limits:
       cpu: "4"
       memory: "8Gi"
   ```

2. **Use Pod Disruption Budgets**
   ```yaml
   apiVersion: policy/v1
   kind: PodDisruptionBudget
   metadata:
     name: runtime-pdb
   spec:
     minAvailable: 1
     selector:
       matchLabels:
         app.kubernetes.io/component: runtime
   ```

### Security

1. **Use Network Policies**
   ```yaml
   networkPolicy:
     enabled: true
   ```

2. **Enable RBAC**
   ```yaml
   rbac:
     create: true
   ```

3. **Use Secrets for sensitive data**
   ```bash
   kubectl create secret generic cogniverse-secrets \
     --from-literal=api-key=xxx \
     -n cogniverse
   ```

### High Availability

1. **Multi-replica deployments**
2. **Pod anti-affinity rules**
3. **Health checks configured**
4. **Auto-scaling enabled**
5. **PVC backup strategy**

---

## Deployment Scripts

For automated deployment, use the provided scripts:

**Local Docker Compose:**
```bash
./scripts/deploy_local_docker.sh [--production] [--logs]
```

**Local K3s:**
```bash
./scripts/deploy_k3s.sh [--install-k3s] [--install-argo]
```

**Remote Kubernetes:**
```bash
./scripts/deploy_kubernetes.sh \
  --cloud-provider aws \
  --domain cogniverse.example.com \
  --install-ingress \
  --install-cert-manager
```

See script help for full options: `./scripts/deploy_*.sh --help`

---

## Related Documentation

- [Docker Deployment](docker-deployment.md) - Docker Compose setup
- [Argo Workflows](argo-workflows.md) - Batch processing workflows
- [Multi-Tenant Operations](multi-tenant-ops.md) - Tenant management
- [SDK Architecture](../architecture/sdk-architecture.md) - System architecture

---

**Version:** 2.0.0
**Last Updated:** 2025-10-15
**Status:** Production Ready
