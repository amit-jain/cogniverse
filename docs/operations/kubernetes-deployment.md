# Kubernetes Deployment Guide

---

## Overview

Cogniverse provides production-ready Helm charts for Kubernetes deployment with:

- **Helm Chart**: `charts/cogniverse/`

- **StatefulSets**: Vespa, Phoenix (with persistent storage)

- **Deployments**: Runtime, Dashboard, vLLM inference sidecars
  (`vllm_llm_student`, `vllm_llm_teacher`, `vllm_colpali` — serves the
  ColQwen3 token-embed model, `vllm_asr`), `colbert_pylate`,
  `code_colbert_pylate`, `denseon`, `videoprism_jax` (remote sidecar)

- **LLM serving**: `llm.engine: ollama` is the chart default (deploys
  `Ollama` as a StatefulSet). Set `llm.engine: vllm` to deploy a
  built-in vLLM Deployment instead (the ROCm GPU overlay,
  `values.rocm.yaml`, switches to `vllm` by default) — or
  `llm.engine: external` to point at an existing endpoint with no pod.
  Configure once via `llm.builtin.enabled` and `llm.engine` in your
  values file.

> See [`models-and-inference.md`](./models-and-inference.md) for the
> canonical list of every model, image source, and deployment style
> (custom-built sidecars vs official vLLM/Ollama images, CPU vs ROCm,
> student vs teacher LLM).

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

- GPU nodes for vLLM inference sidecars (or for Ollama if used)

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
helm install cogniverse ./charts/cogniverse \
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

# GPU configuration for LLM (nodeSelector, tolerations, and the
# nvidia.com/gpu or amd.com/gpu resource request are all wired
# automatically off llm.device — no manual resource block needed)
llm:
  engine: vllm
  device: cuda
  gpuCount: 1
```

### Deploy to Production

```bash
# Deploy with production values
helm install cogniverse ./charts/cogniverse \
  --namespace cogniverse \
  --create-namespace \
  --values values.prod.yaml

# Verify deployment
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

The easiest way to deploy locally is using the CLI:

```bash
# Start all services via k3d/Helm
cogniverse up

# Check status
cogniverse status
```

**Argo Workflows on K3s:**

Argo Workflows works perfectly on K3s for local testing of batch processing workflows:

```bash
# Deploy with Argo Workflows
cogniverse up

# Access Argo UI locally
kubectl port-forward -n cogniverse svc/cogniverse-argo-workflows-server 2746:2746
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

llm:
  engine: ollama
  device: cpu
  nodeSelector: {}
  tolerations: []
  ollama:
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
    models:
      - "gemma3:4b"

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
  llmModels:
    - "gemma3:4b"

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

# Deploy with K3s values (CPU host)
helm install cogniverse ./charts/cogniverse \
  --namespace cogniverse \
  --values values.k3s.yaml \
  --wait \
  --timeout 10m

# Deploy with K3s + ROCm overlay (AMD GPU host)
helm install cogniverse ./charts/cogniverse \
  --namespace cogniverse \
  --values values.k3s.yaml \
  --values values.rocm.yaml \
  --wait \
  --timeout 10m

# Check status
kubectl get pods -n cogniverse -w
```

`cogniverse up` (the dev CLI) composes these layers automatically when
it detects a GPU host: see
[`development/scripts-operations.md`](../development/scripts-operations.md)
for the auto-detection path.

### GPU passthrough (ROCm + CUDA)

**ROCm (AMD)**: GPU access is via `hostPath` volume mounts of
`/dev/kfd` and `/dev/dri` into the pods, not the legacy k8s device
plugin. This means:

- The k3d cluster must be created with the host devices bind-mounted.
  `cogniverse up` does this automatically when `/dev/kfd` is detected
  on the host:

  ```
  k3d cluster create cogniverse \
    --volume /dev/kfd:/dev/kfd@server:0 \
    --volume /dev/dri:/dev/dri@server:0 \
    ...
  ```

- The k3d node must carry the label `amd.com/gpu.present=true` so
  the chart's `nodeSelector` schedules vLLM pods. `cogniverse up`
  applies this via `kubectl label`. For manual `helm install`,
  apply it once: `kubectl label node --all amd.com/gpu.present=true --overwrite`.

- Pods need `supplementalGroups` for the host's `render` and `video`
  group ids (default 992 and 44 on Debian/Ubuntu). Override per
  service for distros with different ids:

  ```yaml
  inference:
    vllm_colpali:
      rocm:
        supplementalGroups: [109, 18]   # Fedora render=109, video=18
  ```

  The chart resolves `supplementalGroups` via `dig()` so an absent
  `inference.<svc>.rocm` block falls through to the default `[992, 44]`
  rather than nil-derefing.

**CUDA (NVIDIA)**: still uses `nvidia.com/gpu` resource requests via
the NVIDIA device plugin. Apply the `nvidia.com/gpu.present=true`
label on the node and set GPU resource requests on the relevant pods
in `values.cuda.yaml`.

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

For a GPU-backed LLM pod, set `llm.device` — the chart wires the
matching `nodeSelector`, `tolerations`, and `nvidia.com/gpu` /
`amd.com/gpu` resource request automatically:

```yaml
llm:
  engine: ollama          # or vllm
  device: cuda            # cpu | cuda | rocm | mps (vllm only)
  gpuCount: 1
  ollama:
    resources:
      limits:
        memory: "16Gi"
      requests:
        cpu: "4"
        memory: "8Gi"
```

### Persistent Storage

See [persistence-and-backup.md](persistence-and-backup.md) for the full
durability matrix, per-component storage knobs, the dev/cloud config
recipes, the backup destination switch (in-cluster MinIO vs Cloudflare
R2 / Backblaze B2 / AWS S3), and the restore procedure.

Quick reference:

```yaml
# Cloud / multi-node prod — replicated CSI for primary, S3 for backup
vespa:    {persistence: {storageClass: "gp3", size: "1Ti"}}
phoenix:  {persistence: {storageClass: "gp3", size: "200Gi"}}
minio:    {persistence: {storageClass: "gp3", size: "5Ti"}}
hfCache:  {persistence: {enabled: true, storageClass: "gp3", size: "100Gi"}}
hostStorage:
  backup:
    enabled: true
    s3:
      endpoint: "https://s3.us-east-1.amazonaws.com"
      existingSecret: "cogniverse-aws-creds"
```

```yaml
# Single-laptop dev — hostStorage everywhere, in-cluster MinIO is the
# backup target. Survives ``k3d cluster delete``.
hostStorage:
  enabled: true
  backup: {enabled: true}
minio:
  persistence:
    hostPath: /host-data/minio
```

---

## Operations

### Upgrade Deployment

```bash
# Update values
nano values.prod.yaml

# Upgrade with new values
helm upgrade cogniverse ./charts/cogniverse \
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
helm upgrade cogniverse ./charts/cogniverse \
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
helm install cogniverse ./charts/cogniverse \
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
helm upgrade cogniverse ./charts/cogniverse -n cogniverse --reuse-values
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

For automated deployment, use the CLI:

```bash
# Start all services via k3d/Helm
cogniverse up

# Check status
cogniverse status
```

See `cogniverse --help` for full options.

---

## Related Documentation

- [Deployment](deployment.md) - Deployment overview (use `cogniverse up`)
- [Argo Workflows](argo-workflows.md) - Batch processing workflows
- [Multi-Tenant Operations](multi-tenant-ops.md) - Tenant management
- [SDK Architecture](../architecture/sdk-architecture.md) - System architecture

---
