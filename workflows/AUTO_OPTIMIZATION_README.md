# Auto-Optimization

Automated background optimization for routing modules using Phoenix traces.

## Overview

Auto-optimization runs automatically on a schedule (default: hourly) and optimizes routing modules based on real production traces collected in Phoenix. It supports **multi-tenancy** with independent workflow tracking per tenant. Optimization only runs when specific conditions are met, ensuring optimization happens when there's sufficient new data.

## Multi-Tenant Architecture

**Key Feature:** Each tenant gets a **separate workflow** for independent monitoring, retry, and history.

```
CronWorkflow (hourly)
  ↓
[Discover Tenants] → ["default", "tenant-a", "tenant-b"]
  ↓
[Spawn Workflows]
  ├─ Workflow: auto-opt-default-xyz     (independent)
  ├─ Workflow: auto-opt-tenant-a-abc    (independent)
  └─ Workflow: auto-opt-tenant-b-def    (independent)

Each workflow optimizes: modality + routing + cross_modal in parallel
```

**Benefits:**
- ✅ **Independent workflow history** per tenant
- ✅ **Per-tenant monitoring** and debugging
- ✅ **Isolated failures** - one tenant doesn't block others
- ✅ **Individual retry policies** per tenant
- ✅ **Clear audit trail** with workflow labels

## Architecture

```
Argo CronWorkflow (hourly)
  ↓
auto_optimization_trigger.py (checks conditions)
  ↓ if conditions met
run_module_optimization.py (runs optimization)
  ↓
Optimizer Classes (ModalityOptimizer, etc.)
  ↓
Phoenix Traces (training data)
```

## Configuration

Auto-optimization is configured via the Routing Config in the Configuration Management tab:

### Settings

- **Enable Auto-Optimization**: Turn automatic optimization on/off
- **Optimization Interval**: How often to run (in minutes, default: 60)
- **Min Samples for Optimization**: Minimum Phoenix traces required (default: 100)

### Example Configuration

```python
routing_config = RoutingConfigUnified(
    tenant_id="default",
    enable_auto_optimization=True,
    optimization_interval_seconds=3600,  # 1 hour
    min_samples_for_optimization=100
)
```

## Deployment

### 1. Deploy Multi-Tenant Auto-Optimization

```bash
# Deploy WorkflowTemplate + CronWorkflow
kubectl apply -f workflows/auto-optimization-multi-tenant.yaml

# Verify deployment
kubectl get cronworkflows -n cogniverse
kubectl get cronworkflow auto-optimization-multi-tenant -n cogniverse -o yaml

# Verify WorkflowTemplate
kubectl get workflowtemplate tenant-auto-optimization -n cogniverse
```

### Alternative: Single-Tenant (Legacy)

If you only need single-tenant:
```bash
kubectl apply -f workflows/auto-optimization-cron.yaml
```

### 2. Configure Docker Image

The CronWorkflow uses `cogniverse/optimization:latest`. Build and push:

```bash
# Build optimization image
docker build -t cogniverse/optimization:latest -f docker/Dockerfile.optimization .

# Push to registry
docker push cogniverse/optimization:latest
```

### 3. Configure Secrets

Create Kubernetes secret with config store URL:

```bash
kubectl create secret generic cogniverse-config \
  --from-literal=store-url="redis://redis.cogniverse.svc.cluster.local:6379" \
  -n cogniverse
```

## How It Works

### Condition Checking

The trigger script checks three conditions before running optimization:

1. **Auto-optimization enabled**: `enable_auto_optimization` = `True` in routing config
2. **Time interval met**: Enough time has passed since last optimization
3. **Sufficient traces**: Phoenix has >= `min_samples_for_optimization` traces

### Optimization Flow

When all conditions are met:

1. **Trigger**: CronWorkflow runs `auto_optimization_trigger.py`
2. **Check**: Script verifies conditions using ConfigManager and Phoenix API
3. **Execute**: If conditions met, calls `run_module_optimization.py`
4. **Train**: Optimizer pulls Phoenix traces, trains model, saves results
5. **Mark**: Updates marker file to prevent duplicate runs

### Marker Files

Optimization timestamps are stored in `/tmp/auto_opt_{tenant}_{module}.marker` to track last run time.

## Monitoring

### Multi-Tenant Workflow Monitoring

```bash
# List all optimization workflows
kubectl get workflows -n cogniverse --sort-by=.metadata.creationTimestamp

# Filter by tenant
kubectl get workflows -n cogniverse -l tenant-id=default
kubectl get workflows -n cogniverse -l tenant-id=tenant-a

# Filter by optimization type
kubectl get workflows -n cogniverse -l optimization-type=auto

# Get specific tenant workflow
kubectl get workflow auto-opt-default-abc123 -n cogniverse -o yaml

# View logs for specific tenant
kubectl logs -n cogniverse -l workflows.argoproj.io/workflow=auto-opt-default-abc123
```

### Tenant Discovery Monitoring

```bash
# Check which tenants were discovered in last run
kubectl logs -n cogniverse -l workflows.argoproj.io/workflow-template=auto-optimization-multi-tenant \
  -c discover-tenants --tail=50
```

### Check Trigger Logs

```bash
# View trigger script logs
kubectl logs -n cogniverse -l app=auto-optimization -c trigger
```

### Phoenix Dashboard

View optimization results in Phoenix dashboard:
- Go to http://localhost:8501/#module-optimization
- Check "Metrics Dashboard" tab for performance trends

## Manual Testing

Test auto-optimization locally before deploying:

```bash
# Test trigger script
uv run python scripts/auto_optimization_trigger.py \
  --tenant-id default \
  --module routing \
  --phoenix-endpoint http://localhost:6006

# Expected output:
# - Checks routing config
# - Checks Phoenix trace count
# - Triggers optimization if conditions met
# - Exit code: 0=success, 1=skipped, 2=error
```

## Customization

### Adjust Schedule

Edit `workflows/auto-optimization-cron.yaml`:

```yaml
spec:
  schedule: "0 */2 * * *"  # Every 2 hours
  # schedule: "0 0 * * *"  # Daily at midnight
  # schedule: "*/30 * * * *"  # Every 30 minutes
```

### Adjust Resource Limits

```yaml
resources:
  requests:
    memory: "4Gi"  # Increase for large datasets
    cpu: "2"
  limits:
    memory: "16Gi"
    cpu: "8"
```

### Add More Modules

Add workflow or unified modules to the CronWorkflow:

```yaml
- name: workflow-optimization
  template: trigger-optimization
  arguments:
    parameters:
    - name: module
      value: "workflow"
    - name: tenant-id
      value: "default"
```

## Troubleshooting

### Auto-optimization not running

1. **Check CronWorkflow is active**:
   ```bash
   kubectl get cronworkflow auto-optimization -n cogniverse
   ```

2. **Check routing config**:
   - Verify `enable_auto_optimization` is `True`
   - Check interval and min_samples settings

3. **Check Phoenix traces**:
   - Ensure Phoenix is running
   - Verify traces are being collected
   - Check project name: `{tenant_id}-cogniverse-routing`

### Optimization failing

1. **Check logs**:
   ```bash
   kubectl logs -n cogniverse <workflow-pod-name>
   ```

2. **Check Phoenix connectivity**:
   - Verify Phoenix endpoint is accessible
   - Test: `curl http://phoenix.cogniverse.svc.cluster.local:6006`

3. **Check training data**:
   - Verify sufficient traces exist
   - Check trace quality (not all empty/malformed)

### Optimization too frequent

Increase `optimization_interval_seconds` in routing config:

```python
routing_config.optimization_interval_seconds = 7200  # 2 hours
```

## Best Practices

1. **Start conservative**: Begin with high `min_samples` (200+) and long intervals (2+ hours)
2. **Monitor performance**: Track optimization impact in Phoenix dashboard
3. **Review logs**: Check trigger logs regularly to understand skip reasons
4. **Gradual tuning**: Adjust parameters based on trace volume and model performance
5. **Alert on failures**: Set up notifications for failed optimizations

## Related Documentation

- [Module Optimization](../scripts/run_module_optimization.py)
- [Routing Configuration](../libs/core/cogniverse_core/config/unified_config.py)
- [Phoenix Trace Collection](../libs/agents/cogniverse_agents/routing/modality_span_collector.py)
- [Argo Workflows](https://argoproj.github.io/argo-workflows/)
