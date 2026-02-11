#!/bin/bash
# Remote Kubernetes Deployment Script
# Deploys Cogniverse to remote K8s clusters (EKS, GKE, AKS, etc.)

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
VALUES_FILE="${VALUES_FILE:-values.yaml}"
INSTALL_HELM="${INSTALL_HELM:-false}"
INSTALL_ARGO="${INSTALL_ARGO:-false}"
INSTALL_CERT_MANAGER="${INSTALL_CERT_MANAGER:-false}"
INSTALL_INGRESS="${INSTALL_INGRESS:-false}"
CREATE_STORAGE_CLASS="${CREATE_STORAGE_CLASS:-false}"
DOMAIN="${DOMAIN:-}"
CLOUD_PROVIDER="${CLOUD_PROVIDER:-auto}"  # auto, aws, gcp, azure, generic

print_header() {
    echo ""
    echo "==========================================="
    echo "Cogniverse Remote Kubernetes Deployment"
    echo "==========================================="
    echo "Release: $RELEASE_NAME"
    echo "Namespace: $NAMESPACE"
    echo "Values: $VALUES_FILE"
    echo "Cloud Provider: $CLOUD_PROVIDER"
    echo "==========================================="
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

install_helm() {
    log_info "Installing Helm..."

    curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

    log_success "Helm installed successfully"
}

verify_cluster_connection() {
    log_info "Verifying cluster connection..."

    # Get current context
    local context=$(kubectl config current-context 2>/dev/null || echo "none")
    log_info "Current context: $context"

    # Verify cluster access
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        log_info "Please ensure kubectl is configured correctly"
        log_info "Run: kubectl config get-contexts"
        exit 1
    fi

    # Get cluster info
    local cluster_version=$(kubectl version --short 2>/dev/null | grep "Server Version" || echo "unknown")
    log_info "Cluster version: $cluster_version"

    # Check node status
    local nodes=$(kubectl get nodes --no-headers | wc -l)
    local ready_nodes=$(kubectl get nodes --no-headers | grep -c " Ready " || echo "0")
    log_info "Cluster nodes: $ready_nodes/$nodes ready"

    if [ "$ready_nodes" -eq 0 ]; then
        log_error "No ready nodes found in cluster"
        kubectl get nodes
        exit 1
    fi

    # Confirm deployment
    echo ""
    log_warning "You are about to deploy to: $context"
    log_warning "Cluster: $(kubectl cluster-info | head -n1)"
    echo ""
    read -p "Continue with deployment? (yes/no): " confirm

    if [ "$confirm" != "yes" ]; then
        log_info "Deployment cancelled"
        exit 0
    fi

    log_success "Cluster connection verified"
}

detect_cloud_provider() {
    if [ "$CLOUD_PROVIDER" != "auto" ]; then
        log_info "Using specified cloud provider: $CLOUD_PROVIDER"
        return 0
    fi

    log_info "Detecting cloud provider..."

    # Check nodes for cloud provider labels
    local provider_label=$(kubectl get nodes -o json | jq -r '.items[0].metadata.labels["node.kubernetes.io/instance-type"]' 2>/dev/null || echo "")

    if kubectl get nodes -o json | jq -r '.items[0].spec.providerID' | grep -q "aws"; then
        CLOUD_PROVIDER="aws"
        log_info "Detected AWS EKS"
    elif kubectl get nodes -o json | jq -r '.items[0].spec.providerID' | grep -q "gce"; then
        CLOUD_PROVIDER="gcp"
        log_info "Detected GCP GKE"
    elif kubectl get nodes -o json | jq -r '.items[0].spec.providerID' | grep -q "azure"; then
        CLOUD_PROVIDER="azure"
        log_info "Detected Azure AKS"
    else
        CLOUD_PROVIDER="generic"
        log_info "Using generic Kubernetes configuration"
    fi
}

setup_storage_class() {
    if [ "$CREATE_STORAGE_CLASS" != true ]; then
        return 0
    fi

    log_info "Setting up storage class..."

    local storage_class_name="cogniverse-storage"

    case "$CLOUD_PROVIDER" in
        aws)
            cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: $storage_class_name
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  fsType: ext4
  encrypted: "true"
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
EOF
            ;;
        gcp)
            cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: $storage_class_name
provisioner: pd.csi.storage.gke.io
parameters:
  type: pd-balanced
  fsType: ext4
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
EOF
            ;;
        azure)
            cat <<EOF | kubectl apply -f -
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: $storage_class_name
provisioner: disk.csi.azure.com
parameters:
  storageaccounttype: Premium_LRS
  kind: Managed
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true
EOF
            ;;
        *)
            log_warning "Storage class creation skipped for generic provider"
            ;;
    esac

    log_success "Storage class configured"
}

install_ingress_controller() {
    if [ "$INSTALL_INGRESS" != true ]; then
        return 0
    fi

    log_info "Installing NGINX Ingress Controller..."

    # Add helm repo
    helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
    helm repo update

    # Install ingress controller
    helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
        --namespace ingress-nginx \
        --create-namespace \
        --set controller.service.type=LoadBalancer \
        --wait \
        --timeout 10m

    # Wait for LoadBalancer IP
    log_info "Waiting for LoadBalancer IP..."
    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local lb_ip=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
        local lb_hostname=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)

        if [ -n "$lb_ip" ] || [ -n "$lb_hostname" ]; then
            log_success "Ingress controller ready"
            log_info "LoadBalancer: ${lb_ip:-$lb_hostname}"
            export INGRESS_IP="${lb_ip:-$lb_hostname}"
            return 0
        fi

        attempt=$((attempt + 1))
        sleep 10
    done

    log_warning "LoadBalancer IP not assigned yet, continuing..."
}

install_cert_manager() {
    if [ "$INSTALL_CERT_MANAGER" != true ]; then
        return 0
    fi

    log_info "Installing cert-manager..."

    # Add helm repo
    helm repo add jetstack https://charts.jetstack.io
    helm repo update

    # Install CRDs
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.crds.yaml

    # Install cert-manager
    helm upgrade --install cert-manager jetstack/cert-manager \
        --namespace cert-manager \
        --create-namespace \
        --version v1.13.0 \
        --wait \
        --timeout 10m

    log_success "cert-manager installed"
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

create_secrets() {
    log_info "Setting up secrets..."

    # Create image pull secret if credentials provided
    if [ -n "$DOCKER_USERNAME" ] && [ -n "$DOCKER_PASSWORD" ] && [ -n "$DOCKER_REGISTRY" ]; then
        kubectl create secret docker-registry regcred \
            --docker-server="$DOCKER_REGISTRY" \
            --docker-username="$DOCKER_USERNAME" \
            --docker-password="$DOCKER_PASSWORD" \
            --namespace "$NAMESPACE" \
            --dry-run=client -o yaml | kubectl apply -f -

        log_success "Image pull secret created"
    fi

    # Create other secrets from environment
    if [ -n "$ROUTER_OPTIMIZER_TEACHER_KEY" ]; then
        kubectl create secret generic cogniverse-secrets \
            --from-literal=teacher-api-key="$ROUTER_OPTIMIZER_TEACHER_KEY" \
            --namespace "$NAMESPACE" \
            --dry-run=client -o yaml | kubectl apply -f -

        log_success "API secrets created"
    fi
}

prepare_values() {
    log_info "Preparing Helm values..."

    local values_path="$PROJECT_ROOT/$VALUES_FILE"

    # If using default values.yaml, create production-ready override
    if [ "$VALUES_FILE" = "values.yaml" ] && [ ! -f "$values_path" ]; then
        log_warning "No custom values file found, using chart defaults"
        log_info "Consider creating a custom values file for production"
        return 0
    fi

    # Override storage class if created
    if [ "$CREATE_STORAGE_CLASS" = true ]; then
        log_info "Using custom storage class: cogniverse-storage"
        export STORAGE_CLASS="cogniverse-storage"
    fi

    # Override domain if provided
    if [ -n "$DOMAIN" ]; then
        log_info "Using domain: $DOMAIN"
        export DOMAIN="$DOMAIN"
    fi

    log_success "Values prepared"
}

install_argo_workflows() {
    if [ "$INSTALL_ARGO" != true ]; then
        return 0
    fi

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
}

deploy_helm_chart() {
    log_info "Deploying Cogniverse Helm chart..."

    cd "$PROJECT_ROOT"

    # Build helm command
    local helm_cmd="helm"
    local helm_args=()

    # Check if release exists
    if helm list -n "$NAMESPACE" | grep -q "$RELEASE_NAME"; then
        log_info "Release exists, upgrading..."
        helm_cmd="$helm_cmd upgrade"
    else
        log_info "Installing new release..."
        helm_cmd="$helm_cmd install"
    fi

    helm_args+=("$RELEASE_NAME" "charts/cogniverse/")
    helm_args+=("--namespace" "$NAMESPACE")

    # Add values file if exists
    if [ -f "$VALUES_FILE" ]; then
        helm_args+=("--values" "$VALUES_FILE")
    fi

    # Add overrides
    if [ -n "$STORAGE_CLASS" ]; then
        helm_args+=("--set" "global.storageClass=$STORAGE_CLASS")
    fi

    if [ -n "$DOMAIN" ]; then
        helm_args+=("--set" "ingress.enabled=true")
        helm_args+=("--set" "ingress.hosts[0].host=$DOMAIN")
    fi

    helm_args+=("--wait")
    helm_args+=("--timeout" "15m")

    # Execute helm command
    $helm_cmd "${helm_args[@]}"

    log_success "Helm chart deployed"
}

deploy_argo_workflows() {
    if [ "$INSTALL_ARGO" != true ]; then
        return 0
    fi

    log_info "Deploying Argo workflow templates..."

    cd "$PROJECT_ROOT"

    # Check if workflow files exist
    if [ -d "workflows" ]; then
        kubectl apply -f workflows/video-ingestion.yaml -n "$NAMESPACE" || true
        kubectl apply -f workflows/batch-optimization.yaml -n "$NAMESPACE" || true
        kubectl apply -f workflows/tenant-provisioning.yaml -n "$NAMESPACE" || true
        kubectl apply -f workflows/scheduled-maintenance.yaml -n "$NAMESPACE" || true

        log_success "Argo workflows deployed"
    else
        log_warning "Workflow templates not found, skipping"
    fi
}

wait_for_pods() {
    log_info "Waiting for pods to be ready..."

    local max_attempts=60
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        local pending=$(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase!=Running,status.phase!=Succeeded -o json 2>/dev/null | jq '.items | length' || echo "0")

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

get_access_info() {
    log_info "Gathering access information..."

    # Get LoadBalancer IPs
    if [ -n "$DOMAIN" ]; then
        echo "Domain: $DOMAIN"
    fi

    # Get service endpoints
    local runtime_endpoint=$(kubectl get svc "$RELEASE_NAME-runtime" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    local dashboard_endpoint=$(kubectl get svc "$RELEASE_NAME-dashboard" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

    echo "Runtime: $runtime_endpoint"
    echo "Dashboard: $dashboard_endpoint"
}

show_access_info() {
    echo ""
    echo "==========================================="
    log_success "Kubernetes deployment completed!"
    echo "==========================================="
    echo ""
    echo "Access services:"
    echo ""

    if [ -n "$DOMAIN" ]; then
        echo "Via Domain (Ingress):"
        echo "  • Dashboard:       https://$DOMAIN"
        echo "  • Runtime API:     https://$DOMAIN/api"
        echo ""
        echo "DNS Configuration:"
        echo "  Point $DOMAIN to LoadBalancer IP:"

        local lb_ip=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
        local lb_hostname=$(kubectl get svc ingress-nginx-controller -n ingress-nginx -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)

        if [ -n "$lb_ip" ]; then
            echo "  $DOMAIN IN A $lb_ip"
        elif [ -n "$lb_hostname" ]; then
            echo "  $DOMAIN IN CNAME $lb_hostname"
        fi
        echo ""
    fi

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
    echo "==========================================="
}

main() {
    print_header
    check_prerequisites
    verify_cluster_connection
    detect_cloud_provider
    setup_storage_class
    install_ingress_controller
    install_cert_manager
    create_namespace
    create_secrets
    prepare_values

    if [ "$INSTALL_ARGO" = true ]; then
        install_argo_workflows
    fi

    deploy_helm_chart
    wait_for_pods

    if [ "$INSTALL_ARGO" = true ]; then
        deploy_argo_workflows
    fi

    show_access_info
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        cat <<EOF
Cogniverse Remote Kubernetes Deployment Script

Usage: $0 [OPTIONS]

Options:
  --install-helm              Install Helm if not present
  --install-argo              Install and configure Argo Workflows
  --install-cert-manager      Install cert-manager for TLS certificates
  --install-ingress           Install NGINX Ingress Controller
  --create-storage-class      Create cloud-specific storage class
  --release NAME              Helm release name (default: cogniverse)
  --namespace NS              Kubernetes namespace (default: cogniverse)
  --values FILE               Values file (default: values.yaml)
  --domain DOMAIN             Domain for ingress (e.g., cogniverse.example.com)
  --cloud-provider PROVIDER   Cloud provider: aws, gcp, azure, generic (default: auto)
  --help, -h                  Show this help message

Environment Variables:
  INSTALL_HELM                Same as --install-helm
  INSTALL_ARGO                Same as --install-argo
  INSTALL_CERT_MANAGER        Same as --install-cert-manager
  INSTALL_INGRESS             Same as --install-ingress
  CREATE_STORAGE_CLASS        Same as --create-storage-class
  RELEASE_NAME                Helm release name
  NAMESPACE                   Kubernetes namespace
  VALUES_FILE                 Values file path
  DOMAIN                      Domain for ingress
  CLOUD_PROVIDER              Cloud provider
  DOCKER_USERNAME             Docker registry username
  DOCKER_PASSWORD             Docker registry password
  DOCKER_REGISTRY             Docker registry URL
  ROUTER_OPTIMIZER_TEACHER_KEY API key for DSPy teacher model

Examples:
  # Basic deployment (assumes kubectl configured)
  ./scripts/deploy_kubernetes.sh

  # Full AWS EKS deployment with all components
  ./scripts/deploy_kubernetes.sh \
    --install-argo \
    --install-cert-manager \
    --install-ingress \
    --create-storage-class \
    --cloud-provider aws \
    --domain cogniverse.mycompany.com

  # GCP GKE deployment with custom values
  ./scripts/deploy_kubernetes.sh \
    --values values.production.yaml \
    --cloud-provider gcp \
    --namespace cogniverse-prod

  # Azure AKS deployment
  ./scripts/deploy_kubernetes.sh \
    --install-ingress \
    --cloud-provider azure \
    --domain cogniverse.azure.mycompany.com

  # With Docker registry and API secrets
  DOCKER_USERNAME=myuser \
  DOCKER_PASSWORD=mypass \
  DOCKER_REGISTRY=registry.example.com \
  ROUTER_OPTIMIZER_TEACHER_KEY=your-api-key \
    ./scripts/deploy_kubernetes.sh
EOF
        exit 0
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
    --install-cert-manager)
        INSTALL_CERT_MANAGER=true
        shift
        main "$@"
        ;;
    --install-ingress)
        INSTALL_INGRESS=true
        shift
        main "$@"
        ;;
    --create-storage-class)
        CREATE_STORAGE_CLASS=true
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
    --domain)
        DOMAIN="$2"
        shift 2
        main "$@"
        ;;
    --cloud-provider)
        CLOUD_PROVIDER="$2"
        shift 2
        main "$@"
        ;;
    *)
        main "$@"
        ;;
esac
