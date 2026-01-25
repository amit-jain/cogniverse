# Istio Service Mesh Guide

**Last Updated:** 2026-01-25
**Architecture:** UV Workspace with 11 packages in layered architecture
**Purpose:** Deploy Istio service mesh for mTLS, traffic control, and observability on K3s/K8s
**Telemetry:** Integrated with Phoenix (no Jaeger needed)

---

## Overview

Istio provides production-grade service mesh capabilities for Cogniverse:

### Key Benefits

**Security**:
- Automatic mTLS between all services (zero code changes)
- Certificate rotation and management
- Fine-grained access control policies

**Resilience**:
- Automatic retries with exponential backoff
- Circuit breakers to prevent cascade failures
- Timeouts and connection pooling

**Traffic Control**:
- Canary deployments (send 10% traffic to new version)
- A/B testing (route by headers)
- DNS-based multi-cluster routing

**Observability**:
- Distributed tracing via Phoenix (already deployed)
- Service topology visualization via Kiali
- Metrics collection (optional with default/production profiles)

---

## Prerequisites

### Local K3s Setup

```bash
# Install K3s (lightweight Kubernetes)
curl -sfL https://get.k3s.io | sh -

# Verify K3s is running
sudo k3s kubectl get nodes

# Create kubeconfig for kubectl
mkdir -p ~/.kube
sudo k3s kubectl config view --raw > ~/.kube/config
chmod 600 ~/.kube/config

# Verify kubectl access
kubectl get nodes
```

### Resource Requirements

**Minimal Profile** (recommended for home/development):
- **CPU**: 2 cores minimum
- **RAM**: 8-12GB total (Istio uses ~700MB)
- **Disk**: 20GB
- **Services**: Istiod (control plane), Ingress Gateway

**Default Profile** (production-like):
- **CPU**: 4 cores minimum
- **RAM**: 16GB total (Istio uses ~2GB)
- **Disk**: 40GB
- **Services**: Istiod, Ingress Gateway, Egress Gateway, Prometheus, Grafana

**Production Profile** (enterprise scale):
- **CPU**: 8+ cores
- **RAM**: 32GB+ total (Istio uses ~4GB)
- **Disk**: 100GB+
- **Services**: All services with high availability (multi-replica)

---

## Minimal Profile Installation (Recommended)

The minimal profile provides core service mesh features with lowest resource footprint.

### Step 1: Install istioctl

```bash
# Download Istio 1.20+
curl -L https://istio.io/downloadIstio | sh -
cd istio-1.*

# Add istioctl to PATH
export PATH=$PWD/bin:$PATH

# Verify installation
istioctl version
```

### Step 2: Install Istio with Minimal Profile + Phoenix Integration

Create Istio configuration with Phoenix tracing:

```yaml
# istio-config-minimal-phoenix.yaml
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  namespace: istio-system
  name: cogniverse-istio
spec:
  profile: minimal

  # Core mesh configuration
  meshConfig:
    # Enable access logging for debugging
    accessLogFile: /dev/stdout
    accessLogEncoding: JSON

    # Default tracing configuration (Phoenix OTLP)
    defaultConfig:
      tracing:
        sampling: 100.0  # 100% sampling for development (reduce for production)
        openCensusAgent:
          address: phoenix.default.svc.cluster.local:4317
          context: [W3C_TRACE_CONTEXT]

    # Extension providers (Phoenix for tracing)
    extensionProviders:
    - name: phoenix-tracing
      opentelemetry:
        service: phoenix.default.svc.cluster.local
        port: 4317

  # Component configuration
  components:
    # Control plane (istiod)
    pilot:
      enabled: true
      k8s:
        resources:
          requests:
            cpu: 100m
            memory: 512Mi
          limits:
            cpu: 500m
            memory: 1Gi
        hpaSpec:
          minReplicas: 1
          maxReplicas: 2

    # Ingress gateway
    ingressGateways:
    - name: istio-ingressgateway
      enabled: true
      k8s:
        service:
          type: LoadBalancer
          ports:
          - name: http
            port: 80
            targetPort: 8080
          - name: https
            port: 443
            targetPort: 8443
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
        hpaSpec:
          minReplicas: 1
          maxReplicas: 3

    # Egress gateway (disabled in minimal)
    egressGateways:
    - name: istio-egressgateway
      enabled: false

  # Values overrides
  values:
    global:
      # Proxy configuration (sidecar injected into each pod)
      proxy:
        resources:
          requests:
            cpu: 50m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi

      # mTLS configuration
      mtls:
        enabled: true
        auto: true  # Automatic mTLS (permissive mode initially)
```

### Step 3: Deploy Istio

```bash
# Create istio-system namespace
kubectl create namespace istio-system

# Install Istio with minimal profile
istioctl install -f istio-config-minimal-phoenix.yaml -y

# Verify installation
kubectl get pods -n istio-system

# Expected output:
# NAME                                    READY   STATUS    RESTARTS   AGE
# istiod-xxxxxxxxx-xxxxx                  1/1     Running   0          2m
# istio-ingressgateway-xxxxxxxxx-xxxxx    1/1     Running   0          2m
```

### Step 4: Enable Sidecar Injection

Enable automatic sidecar injection for the `default` namespace (where Cogniverse services run):

```bash
# Label namespace for automatic sidecar injection
kubectl label namespace default istio-injection=enabled

# Verify label
kubectl get namespace -L istio-injection

# Expected output:
# NAME              STATUS   AGE   ISTIO-INJECTION
# default           Active   10d   enabled
# istio-system      Active   2m
```

### Step 5: Deploy Phoenix (if not already deployed)

Phoenix must be deployed for distributed tracing:

```bash
# Create Phoenix deployment
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: phoenix
  namespace: default
spec:
  selector:
    app: phoenix
  ports:
  - name: http
    port: 6006
    targetPort: 6006
  - name: otlp-grpc
    port: 4317
    targetPort: 4317
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: phoenix
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: phoenix
  template:
    metadata:
      labels:
        app: phoenix
    spec:
      containers:
      - name: phoenix
        image: arizephoenix/phoenix:latest
        ports:
        - containerPort: 6006
          name: http
        - containerPort: 4317
          name: otlp-grpc
        env:
        - name: PHOENIX_WORKING_DIR
          value: /data
        volumeMounts:
        - name: phoenix-data
          mountPath: /data
        resources:
          requests:
            cpu: 100m
            memory: 512Mi
          limits:
            cpu: 500m
            memory: 1Gi
      volumes:
      - name: phoenix-data
        persistentVolumeClaim:
          claimName: phoenix-data
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: phoenix-data
  namespace: default
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
EOF

# Verify Phoenix is running
kubectl get pods -l app=phoenix

# Check Phoenix logs
kubectl logs -l app=phoenix -f
```

### Step 6: Restart Services for Sidecar Injection

Restart existing Cogniverse services to inject Istio sidecars:

```bash
# Restart all deployments in default namespace
kubectl rollout restart deployment -n default

# Verify sidecars are injected (should show 2/2 READY)
kubectl get pods -n default

# Expected output:
# NAME                              READY   STATUS    RESTARTS   AGE
# cogniverse-runtime-xxxxx-xxxxx    2/2     Running   0          1m
# cogniverse-dashboard-xxxxx-xxxxx  2/2     Running   0          1m
# vespa-0                           2/2     Running   0          1m
# phoenix-xxxxx-xxxxx               2/2     Running   0          1m
# ollama-0                          2/2     Running   0          1m
```

**Explanation of 2/2 READY:**
- Container 1: Application (cogniverse-runtime, etc.)
- Container 2: Envoy proxy sidecar (injected by Istio)

### Step 7: Enable mTLS (Strict Mode)

After verifying services work with sidecars, enable strict mTLS:

```bash
# Apply strict mTLS policy for all services in default namespace
kubectl apply -f - <<EOF
apiVersion: security.istio.io/v1beta1
kind: PeerAuthentication
metadata:
  name: default-mtls-strict
  namespace: default
spec:
  mtls:
    mode: STRICT
EOF

# Verify mTLS is enabled
istioctl authn tls-check $(kubectl get pod -l app=cogniverse-runtime -o jsonpath='{.items[0].metadata.name}')
```

### Step 8: Configure Tracing (Phoenix Integration)

Enable tracing for all services in the mesh:

```bash
# Apply Telemetry resource to enable tracing
kubectl apply -f - <<EOF
apiVersion: telemetry.istio.io/v1alpha1
kind: Telemetry
metadata:
  name: mesh-tracing
  namespace: istio-system
spec:
  tracing:
  - providers:
    - name: phoenix-tracing
    randomSamplingPercentage: 100.0  # 100% for dev, reduce to 1-10% for production
EOF

# Verify telemetry configuration
kubectl get telemetry -n istio-system
```

### Step 9: Verify Tracing Works

Test that traces are being sent to Phoenix:

```bash
# Generate some traffic
kubectl exec -it $(kubectl get pod -l app=cogniverse-runtime -o jsonpath='{.items[0].metadata.name}') -c cogniverse-runtime -- curl -v http://localhost:8000/health

# Port-forward Phoenix UI
kubectl port-forward svc/phoenix 6006:6006

# Open browser to http://localhost:6006
# Navigate to "Traces" section
# You should see distributed traces from Istio service mesh
```

**What you'll see in Phoenix:**
- Service-to-service calls with timing
- HTTP headers, status codes, methods
- Trace spans showing Envoy proxy routing
- Service topology graph

---

## Observability Stack

### Kiali Dashboard (Service Mesh Visualization)

Kiali provides real-time service mesh topology and health visualization:

```bash
# Install Kiali
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/kiali.yaml

# Wait for Kiali to be ready
kubectl rollout status deployment/kiali -n istio-system

# Access Kiali dashboard
istioctl dashboard kiali

# Or port-forward manually
kubectl port-forward svc/kiali -n istio-system 20001:20001

# Open http://localhost:20001
```

**Kiali Features:**
- Service topology graph (visual map of all services)
- Traffic flow visualization (request rates, error rates)
- Configuration validation (detect misconfigurations)
- Distributed tracing integration (links to Phoenix traces)
- mTLS status indicators (see which services have mTLS enabled)

### Resource Monitoring

For minimal profile, use kubectl to monitor resources:

```bash
# Check resource usage
kubectl top pods -n istio-system
kubectl top pods -n default

# Check Istio proxy memory/CPU per pod
kubectl exec -it $(kubectl get pod -l app=cogniverse-runtime -o jsonpath='{.items[0].metadata.name}') -c istio-proxy -- sh -c 'curl -s http://localhost:15000/stats/prometheus | grep -E "(memory|cpu)"'
```

---

## DNS-Based Multi-Cluster Routing

Use Istio for **zero-code multi-cluster routing** based on tenant headers.

### Architecture

```
┌─────────────────────────────────────────────────┐
│         Load Balancer (tenant-aware)            │
│  Reads X-Tenant-ID header, routes to cluster    │
└─────────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐        ┌──────────────┐
│  Cluster US  │        │  Cluster EU  │
│  (tenant_a)  │        │  (tenant_b)  │
│              │        │              │
│  Istio VirtualService │  Istio VirtualService
│  Routes internally    │  Routes internally
└──────────────┘        └──────────────┘
```

**Key Points:**
- ✅ Load balancer routes based on `X-Tenant-ID` header
- ✅ Istio handles internal routing within cluster
- ✅ **Zero application code changes** needed
- ✅ Each cluster has separate Vespa, Phoenix, Ollama instances

### Load Balancer Configuration

Configure your load balancer (NGINX, HAProxy, Cloud LB) to route by tenant:

**Example NGINX config:**
```nginx
# /etc/nginx/nginx.conf
upstream cluster_us {
    server us-cluster.example.com:443;
}

upstream cluster_eu {
    server eu-cluster.example.com:443;
}

map $http_x_tenant_id $backend {
    default         cluster_us;
    "tenant_a"      cluster_us;
    "tenant_b"      cluster_eu;
    "acme_corp"     cluster_us;
    "globex_inc"    cluster_eu;
}

server {
    listen 443 ssl;
    server_name api.cogniverse.com;

    location / {
        proxy_pass https://$backend;
        proxy_set_header X-Tenant-ID $http_x_tenant_id;
        proxy_set_header Host $host;
    }
}
```

**Example AWS ALB config (JSON):**
```json
{
  "Type": "forward",
  "ForwardConfig": {
    "TargetGroups": [
      {
        "TargetGroupArn": "arn:aws:elasticloadbalancing:us-east-1:123456789012:targetgroup/cluster-us",
        "Weight": 1
      },
      {
        "TargetGroupArn": "arn:aws:elasticloadbalancing:eu-west-1:123456789012:targetgroup/cluster-eu",
        "Weight": 0
      }
    ],
    "TargetGroupStickinessConfig": {
      "Enabled": false
    }
  },
  "Conditions": [
    {
      "Field": "http-header",
      "HttpHeaderConfig": {
        "HttpHeaderName": "X-Tenant-ID",
        "Values": ["tenant_a", "acme_corp"]
      }
    }
  ]
}
```

### Istio VirtualService (Per-Cluster Routing)

Within each cluster, Istio routes requests to appropriate services:

```yaml
# Apply this in each cluster
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: cogniverse-routing
  namespace: default
spec:
  hosts:
  - api.cogniverse.com
  gateways:
  - cogniverse-gateway
  http:
  # Route /search to runtime service
  - match:
    - uri:
        prefix: /search
    route:
    - destination:
        host: cogniverse-runtime
        port:
          number: 8000
      weight: 100

  # Route /dashboard to dashboard service
  - match:
    - uri:
        prefix: /dashboard
    route:
    - destination:
        host: cogniverse-dashboard
        port:
          number: 8501
      weight: 100

  # Canary deployment example (10% traffic to new version)
  - match:
    - uri:
        prefix: /api
    route:
    - destination:
        host: cogniverse-runtime
        subset: v1
      weight: 90
    - destination:
        host: cogniverse-runtime
        subset: v2-canary
      weight: 10
---
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: cogniverse-gateway
  namespace: default
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: cogniverse-tls-cert
    hosts:
    - api.cogniverse.com
```

### Testing Multi-Cluster Routing

```bash
# Test US cluster routing (tenant_a)
curl -H "X-Tenant-ID: tenant_a" https://api.cogniverse.com/search?q=test

# Test EU cluster routing (tenant_b)
curl -H "X-Tenant-ID: tenant_b" https://api.cogniverse.com/search?q=test

# Verify in Kiali - should see traffic routing correctly
```

---

## Traffic Control Examples

### Canary Deployment (Gradual Rollout)

Deploy new version alongside old, send small percentage of traffic:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: cogniverse-runtime-canary
  namespace: default
spec:
  hosts:
  - cogniverse-runtime
  http:
  - match:
    - headers:
        x-canary-user:
          exact: "true"
    route:
    - destination:
        host: cogniverse-runtime
        subset: v2-canary
      weight: 100

  - route:
    - destination:
        host: cogniverse-runtime
        subset: v1-stable
      weight: 95
    - destination:
        host: cogniverse-runtime
        subset: v2-canary
      weight: 5
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: cogniverse-runtime
  namespace: default
spec:
  host: cogniverse-runtime
  subsets:
  - name: v1-stable
    labels:
      version: v1
  - name: v2-canary
    labels:
      version: v2
```

### Circuit Breaker (Prevent Cascade Failures)

Automatically stop sending traffic to unhealthy services:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: vespa-circuit-breaker
  namespace: default
spec:
  host: vespa
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
      http:
        http1MaxPendingRequests: 50
        http2MaxRequests: 100
        maxRequestsPerConnection: 2
    outlierDetection:
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 40
```

### Retry and Timeout Policies

Automatic retries with exponential backoff:

```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vespa-resilience
  namespace: default
spec:
  hosts:
  - vespa
  http:
  - route:
    - destination:
        host: vespa
    timeout: 10s
    retries:
      attempts: 3
      perTryTimeout: 3s
      retryOn: 5xx,reset,connect-failure,refused-stream
```

---

## Production Profile

Upgrade to production profile when you need:

- ✅ High availability (multi-replica control plane)
- ✅ Dedicated egress gateway
- ✅ Integrated Prometheus + Grafana
- ✅ Resource guarantees (QoS)

### Production Profile Installation

```bash
# Install with production profile
istioctl install --set profile=production -y

# Or use custom configuration
cat <<EOF | istioctl install -f -
apiVersion: install.istio.io/v1alpha1
kind: IstioOperator
metadata:
  namespace: istio-system
  name: cogniverse-istio-production
spec:
  profile: production

  meshConfig:
    defaultConfig:
      tracing:
        sampling: 1.0  # 1% sampling for production
        openCensusAgent:
          address: phoenix.default.svc.cluster.local:4317
          context: [W3C_TRACE_CONTEXT]
    extensionProviders:
    - name: phoenix-tracing
      opentelemetry:
        service: phoenix.default.svc.cluster.local
        port: 4317

  components:
    pilot:
      k8s:
        hpaSpec:
          minReplicas: 2
          maxReplicas: 5
        resources:
          requests:
            cpu: 500m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi

    ingressGateways:
    - name: istio-ingressgateway
      enabled: true
      k8s:
        hpaSpec:
          minReplicas: 2
          maxReplicas: 5
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 2000m
            memory: 2Gi

    egressGateways:
    - name: istio-egressgateway
      enabled: true
      k8s:
        hpaSpec:
          minReplicas: 2
          maxReplicas: 5
EOF
```

### Production Monitoring (Prometheus + Grafana)

```bash
# Install Prometheus
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/prometheus.yaml

# Install Grafana with Istio dashboards
kubectl apply -f https://raw.githubusercontent.com/istio/istio/release-1.20/samples/addons/grafana.yaml

# Access Grafana
istioctl dashboard grafana

# Pre-built dashboards available:
# - Istio Mesh Dashboard (overall mesh health)
# - Istio Service Dashboard (per-service metrics)
# - Istio Workload Dashboard (per-pod metrics)
# - Istio Performance Dashboard (latency, throughput)
```

---

## Troubleshooting

### Sidecar Not Injected

```bash
# Check namespace label
kubectl get namespace default -L istio-injection

# If missing, add label
kubectl label namespace default istio-injection=enabled

# Restart deployment to inject sidecar
kubectl rollout restart deployment/cogniverse-runtime
```

### mTLS Connection Errors

```bash
# Check mTLS status for a pod
istioctl authn tls-check $(kubectl get pod -l app=cogniverse-runtime -o jsonpath='{.items[0].metadata.name}')

# Check peer authentication policy
kubectl get peerauthentication -n default

# View Envoy proxy logs
kubectl logs -l app=cogniverse-runtime -c istio-proxy
```

### Tracing Not Appearing in Phoenix

```bash
# Check Telemetry resource
kubectl get telemetry -n istio-system

# Check Phoenix is receiving traces
kubectl logs -l app=phoenix | grep -i otlp

# Verify Envoy is sending traces
kubectl exec -it $(kubectl get pod -l app=cogniverse-runtime -o jsonpath='{.items[0].metadata.name}') -c istio-proxy -- curl -s http://localhost:15000/config_dump | grep tracing

# Test manual trace generation
kubectl exec -it $(kubectl get pod -l app=cogniverse-runtime -o jsonpath='{.items[0].metadata.name}') -c cogniverse-runtime -- curl -v http://localhost:8000/health
```

### High Resource Usage

```bash
# Check Istio component resource usage
kubectl top pods -n istio-system

# Check per-pod sidecar usage
kubectl top pods -n default --containers | grep istio-proxy

# Reduce sidecar resources if needed
kubectl annotate deployment cogniverse-runtime sidecar.istio.io/proxyCPU="50m"
kubectl annotate deployment cogniverse-runtime sidecar.istio.io/proxyMemory="128Mi"
```

### Gateway Not Accessible

```bash
# Check gateway status
kubectl get gateway -n default
kubectl get virtualservice -n default

# Check ingress gateway service
kubectl get svc istio-ingressgateway -n istio-system

# Get external IP
kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}'

# For K3s with Traefik, you might need to use NodePort or port-forward
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
```

---

## Summary

**Minimal Profile** (recommended for home/development):
- ✅ Core service mesh features (mTLS, retries, circuit breakers)
- ✅ Phoenix integration (no Jaeger needed)
- ✅ Kiali for visualization
- ✅ ~700MB resource usage
- ✅ Single replica for all components

**Production Profile** (enterprise scale):
- ✅ All minimal features
- ✅ High availability (multi-replica)
- ✅ Egress gateway
- ✅ Prometheus + Grafana
- ✅ ~4GB resource usage

**Key Takeaways:**
1. **Phoenix replaces Jaeger** - single telemetry stack
2. **DNS-based multi-cluster** - zero code changes needed
3. **Automatic mTLS** - secure by default
4. **Traffic control** - canary, A/B testing, circuit breakers
5. **Kiali visualization** - see your mesh in real-time

---

## Related Documentation

- [Deployment Guide](deployment.md) - Overall deployment strategy
- [Kubernetes Deployment](kubernetes-deployment.md) - K8s setup without Istio
- [Configuration](configuration.md) - Multi-tenant configuration
- [Performance & Monitoring](performance-monitoring.md) - Performance tuning
