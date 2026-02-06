# Argo Workflows Guide

---

## Overview

Cogniverse uses Argo Workflows for batch processing and scheduled maintenance:

- **Video Ingestion**: Bulk video processing workflows
- **Batch Optimization**: DSPy model optimization at scale
- **Tenant Provisioning**: Automated tenant setup
- **Scheduled Maintenance**: Cron-based cleanup, backups, reports

---

## Prerequisites

### Quick Install with K3s (Local Development)

For local testing of Argo Workflows, use K3s:

```bash
# One-command local setup with K3s + Argo
./scripts/deploy_k3s.sh --install-k3s --install-argo

# This installs:
# - K3s (lightweight Kubernetes)
# - Argo Workflows controller
# - Cogniverse with Helm
# - All workflow templates

# Access Argo UI
kubectl port-forward -n argo svc/argo-server 2746:2746
open http://localhost:2746
```

**Why K3s for local Argo testing?**
- ✅ Full Kubernetes API compatibility
- ✅ Runs on laptop/desktop (4GB RAM)
- ✅ Perfect for testing workflows before production
- ✅ Same workflow YAML works on K3s and production K8s

### Manual Install (Production K8s)

For production Kubernetes clusters:

```bash
# Install Argo Workflows controller
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.0/install.yaml

# Install Argo CLI
curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.5.0/argo-linux-amd64.gz
gunzip argo-linux-amd64.gz
chmod +x argo-linux-amd64
sudo mv argo-linux-amd64 /usr/local/bin/argo

# Verify installation
argo version
```

### Setup Workflow Templates

```bash
# Apply workflow templates
kubectl apply -f workflows/video-ingestion.yaml
kubectl apply -f workflows/batch-optimization.yaml
kubectl apply -f workflows/tenant-provisioning.yaml
kubectl apply -f workflows/scheduled-maintenance.yaml
kubectl apply -f workflows/scheduled-optimization.yaml

# Verify templates
argo template list -n cogniverse
```

---

## Video Ingestion Workflow

### Purpose
Bulk ingestion of videos with multi-profile support.

### Workflow Steps
1. **Validate Input**: Check tenant and video directory
2. **Prepare Videos**: Create batch manifest
3. **Ingest Videos**: Process videos with embeddings
4. **Verify Ingestion**: Confirm documents in Vespa
5. **Notify Completion**: Send status notification

### Submit Workflow

```bash
# Basic ingestion
argo submit workflows/video-ingestion.yaml \
  -n cogniverse \
  --parameter video-dir="/data/videos" \
  --parameter tenant-id="acme_corp" \
  --parameter profiles="video_colpali_smol500_mv_frame"

# With custom parameters
argo submit workflows/video-ingestion.yaml \
  -n cogniverse \
  --parameter video-dir="/data/acme_corp/videos" \
  --parameter tenant-id="acme_corp" \
  --parameter profiles="video_colpali_smol500_mv_frame,video_videoprism_base_mv_chunk_30s" \
  --parameter batch-size="20" \
  --parameter max-workers="8"
```

### Monitor Progress

```bash
# List workflows
argo list -n cogniverse

# Watch workflow
argo watch <workflow-name> -n cogniverse

# Get logs
argo logs <workflow-name> -n cogniverse

# Get workflow details
argo get <workflow-name> -n cogniverse
```

---

## Batch Optimization Workflow

### Purpose
Run DSPy optimization experiments at scale.

### Workflow Steps
1. **Validate Config**: Check optimizer and dataset
2. **Prepare Dataset**: Load evaluation data
3. **Run Optimization**: Execute DSPy optimizer
4. **Evaluate Results**: Compare baseline vs optimized
5. **Deploy Model**: Deploy if improvement > 5%
6. **Notify Completion**: Send results

### Submit Workflow

```bash
# Basic DSPy optimization
argo submit workflows/batch-optimization.yaml \
  -n cogniverse \
  --parameter tenant-id="acme_corp" \
  --parameter optimizer-category="dspy" \
  --parameter optimizer-type="GEPA" \
  --parameter dataset-name="golden_eval_v1"

# Advanced DSPy optimization
argo submit workflows/batch-optimization.yaml \
  -n cogniverse \
  --parameter tenant-id="acme_corp" \
  --parameter optimizer-category="dspy" \
  --parameter optimizer-type="GEPA" \
  --parameter dataset-name="golden_eval_v1" \
  --parameter profiles="video_colpali_smol500_mv_frame" \
  --parameter max-iterations="200" \
  --parameter learning-rate="0.001"
```

### Check Results

```bash
# Get output parameters
argo get <workflow-name> -n cogniverse -o json | jq '.status.outputs.parameters'

# View improvement
argo get <workflow-name> -n cogniverse -o json | \
  jq -r '.status.nodes | .[] | select(.displayName=="run-optimization") | .outputs.parameters[] | select(.name=="improvement") | .value'
```

---

## Tenant Provisioning Workflow

### Purpose
Automated setup of new tenants with complete isolation.

### Workflow Steps
1. **Validate Tenant**: Check tenant ID format
2. **Create Namespace**: K8s namespace for tenant
3. **Deploy Schemas**: Vespa schemas for all profiles
4. **Create Phoenix Project**: Isolated telemetry
5. **Setup Resource Quotas**: CPU/memory/storage limits
6. **Create Storage**: Persistent volume claims
7. **Initialize Memory**: Mem0 memory system
8. **Verify Tenant**: Confirm all resources
9. **Notify Completion**: Send notification

### Submit Workflow

```bash
# Provision new tenant
argo submit workflows/tenant-provisioning.yaml \
  -n cogniverse \
  --parameter tenant-id="newcorp_inc" \
  --parameter tenant-name="NewCorp Inc" \
  --parameter profiles="video_colpali_smol500_mv_frame,video_videoprism_base_mv_chunk_30s" \
  --parameter storage-quota="200Gi" \
  --parameter cpu-quota="20" \
  --parameter memory-quota="40Gi"
```

### Verify Tenant

```bash
# Check namespace
kubectl get namespace cogniverse-newcorp_inc

# Check schemas
curl "http://cogniverse-vespa:8080/search/?yql=select+*+from+video_colpali_smol500_mv_frame_newcorp_inc+where+true+limit+0"

# Check resource quota
kubectl get resourcequota -n cogniverse-newcorp_inc
```

---

## Scheduled Maintenance Workflows

### Daily Backup (2 AM UTC)

Backs up Vespa and Phoenix data daily.

```bash
# View cron workflow
kubectl get cronworkflow daily-backup -n cogniverse -o yaml

# Trigger manually
argo cron create workflows/scheduled-maintenance.yaml -n cogniverse
argo submit --from cronwf/daily-backup -n cogniverse

# View history
argo list -n cogniverse --selector workflows.argoproj.io/cron-workflow=daily-backup
```

### Weekly DSPy Optimization (Sunday 3 AM UTC)

Runs DSPy optimization for all tenants weekly.

```bash
# View schedule
kubectl get cronworkflow weekly-dspy-optimization -n cogniverse

# Modify schedule
kubectl edit cronworkflow weekly-dspy-optimization -n cogniverse

# Suspend/resume
argo cron suspend weekly-dspy-optimization -n cogniverse
argo cron resume weekly-dspy-optimization -n cogniverse
```

### Daily Cleanup (4 AM UTC)

Cleans up logs, temp files, and databases daily.

```bash
# View workflow
kubectl get cronworkflow daily-cleanup -n cogniverse

# Check last run
argo list -n cogniverse --selector workflows.argoproj.io/cron-workflow=daily-cleanup --limit 1
```

### Monthly Reports (1st of month, 5 AM UTC)

Generates usage and performance reports monthly.

```bash
# View workflow
kubectl get cronworkflow monthly-reports -n cogniverse

# Trigger manually for testing
argo submit --from cronwf/monthly-reports -n cogniverse
```

---

## Scheduled Module Optimization Workflows

Cogniverse includes automated optimization workflows that run on a schedule to continuously improve system performance.

### Weekly Module Optimization (Sunday 3 AM UTC)

Comprehensive optimization of all routing/workflow modules when sufficient annotations are collected.

**What it does:**

- Checks Phoenix for annotation count (threshold: 50)

- Runs optimization for all modules if threshold met:
  - Modality optimizer (per-modality routing)
  - Cross-modal optimizer (fusion decisions)
  - Routing optimizer (entity-based routing)
  - Workflow optimizer (orchestration)
  - DSPy optimizer (GEPA by default)
- Generates synthetic training data from backend storage using DSPy modules
- Auto-selects DSPy optimizer based on data size

```bash
# View schedule
kubectl get cronworkflow weekly-optimization -n cogniverse

# Check last run
argo list -n cogniverse --selector workflows.argoproj.io/cron-workflow=weekly-optimization --limit 1

# Trigger manually
argo submit --from cronwf/weekly-optimization -n cogniverse

# Modify parameters
kubectl edit cronworkflow weekly-optimization -n cogniverse
```

**Configuration:**

- `annotation-threshold`: Minimum annotations needed (default: 50)

- `improvement-threshold`: Minimum improvement % to deploy (default: 5%)

- `tenant-id`: Target tenant (default: "default")

### Daily Optimization Check (4 AM UTC)

Lightweight routing optimization triggered by new annotations.

**What it does:**

- Checks Phoenix for annotations in last 24 hours (threshold: 20)

- Runs quick routing optimization if threshold met

- Uses synthetic data generation for training

- Faster than weekly optimization (single module)

```bash
# View schedule
kubectl get cronworkflow daily-optimization-check -n cogniverse

# Check last run
argo list -n cogniverse --selector workflows.argoproj.io/cron-workflow=daily-optimization-check --limit 1

# Suspend/resume
argo cron suspend daily-optimization-check -n cogniverse
argo cron resume daily-optimization-check -n cogniverse
```

**Configuration:**

- `annotation-threshold`: Minimum new annotations (default: 20)

- `tenant-id`: Target tenant (default: "default")

### Module Optimization vs DSPy Optimization

The workflows support two optimizer categories:

**Module Optimization** (`optimizer-category: routing`):

- **What gets optimized**: modality, cross_modal, routing, workflow, unified modules

- **How they get optimized**: Auto-selected DSPy optimizer (Bootstrap/SIMBA/MIPRO/GEPA)

- **Data source**: Phoenix traces + synthetic data generation

- **Use case**: Optimize routing decisions and workflow planning

**DSPy Optimization** (`optimizer-category: dspy`):

- **What gets optimized**: DSPy modules (prompt templates, reasoning chains)

- **How they get optimized**: Explicit DSPy optimizer (GEPA/Bootstrap/SIMBA/MIPRO)

- **Data source**: Golden evaluation datasets

- **Use case**: Teacher-student distillation for local models

### Manual Workflow Submission

Submit optimization workflows on-demand:

```bash
# Submit module optimization
argo submit workflows/batch-optimization.yaml \
  -n cogniverse \
  --parameter tenant-id="acme_corp" \
  --parameter optimizer-category="routing" \
  --parameter optimizer-type="modality" \
  --parameter max-iterations="100" \
  --parameter use-synthetic-data="true"

# Submit DSPy optimization
argo submit workflows/batch-optimization.yaml \
  -n cogniverse \
  --parameter tenant-id="acme_corp" \
  --parameter optimizer-category="dspy" \
  --parameter optimizer-type="GEPA" \
  --parameter dataset-name="golden_eval_v1" \
  --parameter max-iterations="100"
```

### Monitoring Optimization Workflows

```bash
# List optimization workflows
argo list -n cogniverse --selector workflow-type=optimization

# Get workflow results
argo get <workflow-name> -n cogniverse -o json | \
  jq '.status.nodes | .[] | select(.displayName=="run-optimization") | .outputs.parameters'

# View improvement metrics
argo get <workflow-name> -n cogniverse -o json | \
  jq -r '.status.outputs.parameters[] | select(.name=="improvement") | .value'

# Check annotation count
argo get <workflow-name> -n cogniverse -o json | \
  jq -r '.status.outputs.parameters[] | select(.name=="annotation-count") | .value'
```

---

## Operations

### List All Workflows

```bash
# All workflows
argo list -n cogniverse

# Running workflows
argo list -n cogniverse --status Running

# Failed workflows
argo list -n cogniverse --status Failed

# Completed workflows
argo list -n cogniverse --status Succeeded
```

### Delete Workflows

```bash
# Delete specific workflow
argo delete <workflow-name> -n cogniverse

# Delete all completed workflows
argo delete --completed -n cogniverse

# Delete workflows older than 7 days
argo delete --older 7d -n cogniverse
```

### Retry Failed Workflows

```bash
# Retry workflow
argo retry <workflow-name> -n cogniverse

# Retry from specific node
argo retry <workflow-name> --node-field-selector displayName=ingest-videos -n cogniverse
```

### Suspend/Resume Workflows

```bash
# Suspend running workflow
argo suspend <workflow-name> -n cogniverse

# Resume suspended workflow
argo resume <workflow-name> -n cogniverse
```

---

## Monitoring

### View Logs

```bash
# All logs
argo logs <workflow-name> -n cogniverse

# Follow logs
argo logs <workflow-name> -f -n cogniverse

# Logs for specific step
argo logs <workflow-name> --node-field-selector displayName=ingest-videos -n cogniverse
```

### Workflow Status

```bash
# Detailed status
argo get <workflow-name> -n cogniverse

# Watch workflow progress
argo watch <workflow-name> -n cogniverse

# JSON output
argo get <workflow-name> -n cogniverse -o json
```

### Resource Usage

```bash
# Pod resource usage
kubectl top pods -n cogniverse -l workflows.argoproj.io/workflow=<workflow-name>

# Workflow metrics
kubectl get workflow <workflow-name> -n cogniverse -o jsonpath='{.status.resourcesDuration}'
```

---

## Best Practices

### Resource Management

1. **Set appropriate resource limits**
   ```yaml
   resources:
     requests:
       memory: "8Gi"
       cpu: "4"
     limits:
       memory: "16Gi"
       cpu: "8"
   ```

2. **Use volume claim templates for large data**
   ```yaml
   volumeClaimTemplates:
   - metadata:
       name: workspace
     spec:
       accessModes: ["ReadWriteOnce"]
       resources:
         requests:
           storage: 100Gi
   ```

### Error Handling

1. **Set backoff limits**
   ```yaml
   spec:
     backoffLimit: 3
     restartPolicy: OnFailure
   ```

2. **Use retryStrategy**
   ```yaml
   retryStrategy:
     limit: 3
     retryPolicy: "Always"
     backoff:
       duration: "1m"
       factor: 2
       maxDuration: "10m"
   ```

### Notifications

Configure workflow notifications:

```yaml
metadata:
  annotations:
    notifications.argoproj.io/subscribe.on-workflow-completion: |
      slack:your-channel
```

---

## Troubleshooting

### Workflow Stuck

```bash
# Check workflow status
argo get <workflow-name> -n cogniverse

# Check pod status
kubectl get pods -n cogniverse -l workflows.argoproj.io/workflow=<workflow-name>

# Delete stuck workflow
argo delete <workflow-name> -n cogniverse --force
```

### Pod Failures

```bash
# View pod logs
kubectl logs <pod-name> -n cogniverse

# Describe pod
kubectl describe pod <pod-name> -n cogniverse

# Check events
kubectl get events -n cogniverse --sort-by='.lastTimestamp'
```

### Storage Issues

```bash
# Check PVC status
kubectl get pvc -n cogniverse

# Describe PVC
kubectl describe pvc <pvc-name> -n cogniverse

# Check storage class
kubectl get storageclass
```

---

## Related Documentation

- [Kubernetes Deployment](kubernetes-deployment.md) - K8s setup
- [Docker Deployment](docker-deployment.md) - Docker Compose
- [Multi-Tenant Operations](multi-tenant-ops.md) - Tenant management

---
