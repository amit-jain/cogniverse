#!/bin/bash
# K3s Local Kubernetes Deployment Script
# Deploys Cogniverse to local K3s cluster with Helm

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RELEASE_NAME="${RELEASE_NAME:-cogniverse}"
NAMESPACE="${NAMESPACE:-cogniverse}"
VALUES_FILE="${VALUES_FILE:-values.k3s.yaml}"
INSTALL_K3S="${INSTALL_K3S:-false}"
INSTALL_HELM="${INSTALL_HELM:-false}"
INSTALL_ARGO="${INSTALL_ARGO:-false}"

print_header() {
    echo ""
    echo "=========================================="
    echo "Cogniverse K3s Deployment"
    echo "=========================================="
    echo "Release: $RELEASE_NAME"
    echo "Namespace: $NAMESPACE"
    echo "Values: $VALUES_FILE"
    echo "=========================================="
    echo ""
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        log_info "Install: https://kubernetes.io/docs/tasks/tools/"
        exit 1
    fi

    # Check helm
    if ! command -v helm &> /dev/null; then
        if [ "$INSTALL_HELM" = true ]; then
            install_helm
        else
            log_error "Helm is not installed"
            log_info "Install with: --install-helm or visit https://helm.sh/docs/intro/install/"
            exit 1
        fi
    fi

    log_success "Prerequisites check passed"
}

install_k3s() {
    log_info "Installing K3s..."

    if command -v k3s &> /dev/null; then
        log_info "K3s already installed"
        return 0
    fi

    # Install K3s
    curl -sfL https://get.k3s.io | sh -s - \
        --write-kubeconfig-mode 644 \
        --disable traefik

    # Wait for K3s to be ready
    log_info "Waiting for K3s to be ready..."
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if sudo k3s kubectl get nodes &> /dev/null; then
            log_success "K3s is ready"
            break
        fi
        attempt=$((attempt + 1))
        sleep 2
    done

    # Setup kubeconfig
    if [ ! -d "$HOME/.kube" ]; then
        mkdir -p "$HOME/.kube"
    fi

    sudo cp /etc/rancher/k3s/k3s.yaml "$HOME/.kube/config"
    sudo chown $(id -u):$(id -g) "$HOME/.kube/config"

    log_success "K3s installed successfully"
}

install_helm() {
    log_info "Installing Helm..."

    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

    log_success "Helm installed successfully"
}

check_k3s() {
    log_info "Checking K3s cluster..."

    if ! kubectl cluster-info &> /dev/null; then
        if [ "$INSTALL_K3S" = true ]; then
            install_k3s
        else
            log_error "K3s cluster not accessible"
            log_info "Install K3s with: --install-k3s"
            exit 1
        fi
    fi

    # Check node status
    if ! kubectl get nodes | grep -q "Ready"; then
        log_error "K3s node not ready"
        kubectl get nodes
        exit 1
    fi

    log_success "K3s cluster is ready"
}

create_values_file() {
    log_info "Creating K3s-specific values file..."

    local values_path="$PROJECT_ROOT/$VALUES_FILE"

    if [ -f "$values_path" ]; then
        log_info "Values file already exists: $VALUES_FILE"
        return 0
    fi

    cat > "$values_path" <<'EOF'
# K3s-specific Helm values for Cogniverse
# Optimized for local development with K3s

# Use local-path storage class (K3s default)
global:
  storageClass: "local-path"

# Reduced resource requirements for local deployment
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
    enabled: false
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
  # GPU support (if available)
  nodeSelector: {}
  tolerations: []

otelCollector:
  resources:
    requests:
      cpu: "100m"
      memory: "256Mi"
    limits:
      cpu: "200m"
      memory: "512Mi"

# Ingress with K3s
ingress:
  enabled: true
  className: "traefik"  # K3s comes with Traefik by default
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

# Local development configuration
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
EOF

    log_success "Created K3s values file: $VALUES_FILE"
}

create_namespace() {
    log_info "Creating namespace: $NAMESPACE..."

    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace already exists"
    else
        kubectl create namespace "$NAMESPACE"
        log_success "Namespace created"
    fi
}

install_argo_workflows() {
    log_info "Installing Argo Workflows..."

    # Create argo namespace
    if ! kubectl get namespace argo &> /dev/null; then
        kubectl create namespace argo
    fi

    # Install Argo Workflows
    kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.0/install.yaml

    # Wait for Argo to be ready
    log_info "Waiting for Argo Workflows to be ready..."
    kubectl wait --for=condition=available --timeout=300s \
        deployment/argo-server -n argo

    log_success "Argo Workflows installed"

    # Install Argo CLI if not present
    if ! command -v argo &> /dev/null; then
        log_info "Installing Argo CLI..."
        curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.5.0/argo-linux-amd64.gz
        gunzip argo-linux-amd64.gz
        chmod +x argo-linux-amd64
        sudo mv argo-linux-amd64 /usr/local/bin/argo
        log_success "Argo CLI installed"
    fi
}

deploy_helm_chart() {
    log_info "Deploying Cogniverse Helm chart..."

    cd "$PROJECT_ROOT"

    # Check if release exists
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        log_info "Release exists, upgrading..."
        helm upgrade "$RELEASE_NAME" charts/cogniverse/ \
            --namespace "$NAMESPACE" \
            --values "$VALUES_FILE" \
            --wait \
            --timeout 10m
    else
        log_info "Installing new release..."
        helm install "$RELEASE_NAME" charts/cogniverse/ \
            --namespace "$NAMESPACE" \
            --values "$VALUES_FILE" \
            --wait \
            --timeout 10m
    fi

    log_success "Helm chart deployed"
}

deploy_argo_workflows() {
    log_info "Deploying Argo workflow templates..."

    cd "$PROJECT_ROOT"

    kubectl apply -f workflows/video-ingestion.yaml -n "$NAMESPACE"
    kubectl apply -f workflows/batch-optimization.yaml -n "$NAMESPACE"
    kubectl apply -f workflows/tenant-provisioning.yaml -n "$NAMESPACE"
    kubectl apply -f workflows/scheduled-maintenance.yaml -n "$NAMESPACE"

    log_success "Argo workflows deployed"
}

wait_for_pods() {
    log_info "Waiting for pods to be ready..."

    local max_attempts=60
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local pending=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running,status.phase!=Succeeded -o json | jq '.items | length')

        if [ "$pending" -eq 0 ]; then
            log_success "All pods are ready"
            return 0
        fi

        attempt=$((attempt + 1))
        log_info "Waiting for $pending pods... (attempt $attempt/$max_attempts)"
        sleep 10
    done

    log_warning "Some pods may not be ready yet"
    kubectl get pods -n "$NAMESPACE"
}

setup_local_access() {
    log_info "Setting up local access..."

    # Add entry to /etc/hosts if not present
    if ! grep -q "cogniverse.local" /etc/hosts; then
        log_info "Adding cogniverse.local to /etc/hosts..."
        echo "127.0.0.1 cogniverse.local" | sudo tee -a /etc/hosts
    fi

    log_success "Local access configured"
}

show_access_info() {
    echo ""
    echo "=========================================="
    log_success "K3s deployment completed!"
    echo "=========================================="
    echo ""
    echo "Access services:"
    echo ""
    echo "Via Ingress (if configured):"
    echo "  • Dashboard:       http://cogniverse.local"
    echo "  • Runtime API:     http://cogniverse.local/api"
    echo ""
    echo "Via Port-Forward:"
    echo "  # Runtime"
    echo "  kubectl port-forward -n $NAMESPACE svc/$RELEASE_NAME-runtime 8000:8000"
    echo "  open http://localhost:8000/docs"
    echo ""
    echo "  # Dashboard"
    echo "  kubectl port-forward -n $NAMESPACE svc/$RELEASE_NAME-dashboard 8501:8501"
    echo "  open http://localhost:8501"
    echo ""
    echo "  # Phoenix"
    echo "  kubectl port-forward -n $NAMESPACE svc/$RELEASE_NAME-phoenix 6006:6006"
    echo "  open http://localhost:6006"
    echo ""
    echo "Check status:"
    echo "  kubectl get pods -n $NAMESPACE"
    echo "  kubectl get svc -n $NAMESPACE"
    echo "  helm status $RELEASE_NAME -n $NAMESPACE"
    echo ""
    echo "View logs:"
    echo "  kubectl logs -f deployment/$RELEASE_NAME-runtime -n $NAMESPACE"
    echo ""
    if [ "$INSTALL_ARGO" = true ]; then
        echo "Argo Workflows:"
        echo "  argo list -n $NAMESPACE"
        echo "  kubectl port-forward -n argo svc/argo-server 2746:2746"
        echo "  open http://localhost:2746"
        echo ""
    fi
    echo "Uninstall:"
    echo "  helm uninstall $RELEASE_NAME -n $NAMESPACE"
    echo "  kubectl delete namespace $NAMESPACE"
    echo ""
    echo "=========================================="
}

main() {
    print_header
    check_prerequisites
    check_k3s
    create_values_file
    create_namespace

    if [ "$INSTALL_ARGO" = true ]; then
        install_argo_workflows
    fi

    deploy_helm_chart
    wait_for_pods

    if [ "$INSTALL_ARGO" = true ]; then
        deploy_argo_workflows
    fi

    setup_local_access
    show_access_info
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        cat <<EOF
Cogniverse K3s Deployment Script

Usage: $0 [OPTIONS]

Options:
  --install-k3s       Install K3s if not present
  --install-helm      Install Helm if not present
  --install-argo      Install and configure Argo Workflows
  --release NAME      Helm release name (default: cogniverse)
  --namespace NS      Kubernetes namespace (default: cogniverse)
  --values FILE       Values file (default: values.k3s.yaml)
  --help, -h          Show this help message

Environment Variables:
  INSTALL_K3S         Same as --install-k3s
  INSTALL_HELM        Same as --install-helm
  INSTALL_ARGO        Same as --install-argo
  RELEASE_NAME        Helm release name
  NAMESPACE           Kubernetes namespace
  VALUES_FILE         Values file path

Examples:
  # Basic K3s deployment
  ./scripts/deploy_k3s.sh

  # Full setup (K3s + Argo Workflows)
  ./scripts/deploy_k3s.sh --install-k3s --install-argo

  # Custom release and namespace
  ./scripts/deploy_k3s.sh --release my-cogniverse --namespace my-namespace

  # With custom values
  ./scripts/deploy_k3s.sh --values values.custom.yaml
EOF
        exit 0
        ;;
    --install-k3s)
        INSTALL_K3S=true
        shift
        main "$@"
        ;;
    --install-helm)
        INSTALL_HELM=true
        shift
        main "$@"
        ;;
    --install-argo)
        INSTALL_ARGO=true
        shift
        main "$@"
        ;;
    --release)
        RELEASE_NAME="$2"
        shift 2
        main "$@"
        ;;
    --namespace)
        NAMESPACE="$2"
        shift 2
        main "$@"
        ;;
    --values)
        VALUES_FILE="$2"
        shift 2
        main "$@"
        ;;
    *)
        main "$@"
        ;;
esac
