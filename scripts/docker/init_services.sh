#!/bin/bash
# Cogniverse Service Initialization Script
# Initializes all Docker Compose services with proper ordering and health checks

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VESPA_URL=${VESPA_URL:-"http://localhost:8080"}
VESPA_CONFIG_PORT=${VESPA_CONFIG_PORT:-19071}
PHOENIX_URL=${PHOENIX_URL:-"http://localhost:6006"}
OLLAMA_URL=${OLLAMA_URL:-"http://localhost:11434"}
RUNTIME_URL=${RUNTIME_URL:-"http://localhost:8000"}
DASHBOARD_URL=${DASHBOARD_URL:-"http://localhost:8501"}

MAX_WAIT=${MAX_WAIT:-300}  # Maximum wait time in seconds
CHECK_INTERVAL=5

# Default tenants to initialize
TENANTS=${TENANTS:-"default"}

# Default Ollama models to pull
OLLAMA_MODELS=${OLLAMA_MODELS:-"llama2:7b mistral:7b"}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Wait for service to be healthy
wait_for_service() {
    local service_name=$1
    local health_url=$2
    local timeout=${3:-$MAX_WAIT}

    log_info "Waiting for $service_name to be healthy..."

    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -sf "$health_url" > /dev/null 2>&1; then
            log_success "$service_name is healthy"
            return 0
        fi

        sleep $CHECK_INTERVAL
        elapsed=$((elapsed + CHECK_INTERVAL))

        if [ $((elapsed % 30)) -eq 0 ]; then
            log_info "Still waiting for $service_name... (${elapsed}s elapsed)"
        fi
    done

    log_error "$service_name failed to become healthy within ${timeout}s"
    return 1
}

# Deploy Vespa schemas for all tenants
deploy_schemas() {
    log_info "Deploying Vespa schemas for tenants: $TENANTS"

    # Find all schema files
    local schema_dir="configs/schemas"
    if [ ! -d "$schema_dir" ]; then
        log_warning "Schema directory $schema_dir not found, skipping schema deployment"
        return 0
    fi

    local schema_count=0
    for tenant in $TENANTS; do
        log_info "Deploying schemas for tenant: $tenant"

        for schema_file in "$schema_dir"/*.json; do
            if [ -f "$schema_file" ]; then
                local schema_name=$(basename "$schema_file" .json)
                log_info "  Deploying schema: $schema_name for tenant $tenant"

                if uv run python scripts/deploy_json_schema.py "$schema_file" --tenant-id "$tenant" 2>&1 | tee /tmp/schema_deploy.log; then
                    log_success "  Schema $schema_name deployed successfully for tenant $tenant"
                    schema_count=$((schema_count + 1))
                else
                    log_warning "  Failed to deploy schema $schema_name for tenant $tenant"
                    log_warning "  Check /tmp/schema_deploy.log for details"
                fi
            fi
        done
    done

    if [ $schema_count -eq 0 ]; then
        log_warning "No schemas were deployed"
    else
        log_success "Deployed $schema_count schemas across all tenants"
    fi
}

# Pull Ollama models
pull_ollama_models() {
    log_info "Pulling Ollama models: $OLLAMA_MODELS"

    for model in $OLLAMA_MODELS; do
        log_info "  Pulling model: $model"

        if curl -sf -X POST "$OLLAMA_URL/api/pull" \
            -H "Content-Type: application/json" \
            -d "{\"name\": \"$model\"}" > /dev/null 2>&1; then
            log_success "  Model $model pulled successfully"
        else
            log_warning "  Failed to pull model $model (may already exist or network issue)"
        fi

        # Give Ollama time to process
        sleep 5
    done

    # List available models
    log_info "Available Ollama models:"
    if curl -sf "$OLLAMA_URL/api/tags" 2>/dev/null | python3 -m json.tool; then
        log_success "Ollama models listed successfully"
    else
        log_warning "Could not list Ollama models"
    fi
}

# Create Phoenix projects for tenants
create_phoenix_projects() {
    log_info "Creating Phoenix projects for tenants: $TENANTS"

    for tenant in $TENANTS; do
        local project_name="${tenant}_project"
        log_info "  Creating Phoenix project: $project_name"

        # Phoenix projects are created automatically when first span is sent
        # We just verify Phoenix is accessible
        if curl -sf "$PHOENIX_URL/health" > /dev/null 2>&1; then
            log_success "  Phoenix is accessible for tenant $tenant"
        else
            log_warning "  Phoenix may not be accessible for tenant $tenant"
        fi
    done
}

# Verify all services are accessible
verify_services() {
    log_info "Verifying all services are accessible..."

    local all_healthy=true

    # Vespa
    if curl -sf "$VESPA_URL/ApplicationStatus" > /dev/null 2>&1; then
        log_success "  Vespa: OK ($VESPA_URL)"
    else
        log_error "  Vespa: FAILED ($VESPA_URL)"
        all_healthy=false
    fi

    # Phoenix
    if curl -sf "$PHOENIX_URL/health" > /dev/null 2>&1; then
        log_success "  Phoenix: OK ($PHOENIX_URL)"
    else
        log_error "  Phoenix: FAILED ($PHOENIX_URL)"
        all_healthy=false
    fi

    # Ollama
    if curl -sf "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
        log_success "  Ollama: OK ($OLLAMA_URL)"
    else
        log_error "  Ollama: FAILED ($OLLAMA_URL)"
        all_healthy=false
    fi

    # Runtime
    if curl -sf "$RUNTIME_URL/health" > /dev/null 2>&1; then
        log_success "  Runtime: OK ($RUNTIME_URL)"
    else
        log_warning "  Runtime: Not yet available ($RUNTIME_URL)"
        log_info "  Note: Runtime may take longer to start after initialization"
    fi

    # Dashboard
    if curl -sf "$DASHBOARD_URL/_stcore/health" > /dev/null 2>&1; then
        log_success "  Dashboard: OK ($DASHBOARD_URL)"
    else
        log_warning "  Dashboard: Not yet available ($DASHBOARD_URL)"
        log_info "  Note: Dashboard may take longer to start after initialization"
    fi

    if $all_healthy; then
        log_success "Core services verification passed"
        return 0
    else
        log_error "Some core services failed verification"
        return 1
    fi
}

# Print service access information
print_access_info() {
    echo ""
    echo "=========================================="
    log_success "Cogniverse services initialized!"
    echo "=========================================="
    echo ""
    echo "Access services at:"
    echo "  • Runtime API:     $RUNTIME_URL"
    echo "  • API Docs:        $RUNTIME_URL/docs"
    echo "  • Dashboard:       $DASHBOARD_URL"
    echo "  • Phoenix:         $PHOENIX_URL"
    echo "  • Vespa:           $VESPA_URL"
    echo "  • Ollama:          $OLLAMA_URL"
    echo ""
    echo "Initialized tenants: $TENANTS"
    echo ""
    echo "Next steps:"
    echo "  1. Ingest videos:"
    echo "     uv run python scripts/run_ingestion.py \\"
    echo "       --video_dir data/videos \\"
    echo "       --profile video_colpali_smol500_mv_frame \\"
    echo "       --tenant default"
    echo ""
    echo "  2. Run queries:"
    echo "     curl -X POST $RUNTIME_URL/api/v1/search \\"
    echo "       -H 'Content-Type: application/json' \\"
    echo "       -H 'X-Tenant-ID: default' \\"
    echo "       -d '{\"query\": \"machine learning tutorial\"}'"
    echo ""
    echo "  3. View Phoenix dashboard:"
    echo "     open $PHOENIX_URL"
    echo ""
    echo "=========================================="
}

# Main initialization flow
main() {
    log_info "Starting Cogniverse service initialization..."
    log_info "Configuration:"
    log_info "  Tenants: $TENANTS"
    log_info "  Ollama models: $OLLAMA_MODELS"
    log_info "  Max wait time: ${MAX_WAIT}s"
    echo ""

    # Step 1: Wait for core services
    log_info "Step 1/5: Waiting for core services..."
    wait_for_service "Vespa" "$VESPA_URL/ApplicationStatus" || exit 1
    wait_for_service "Vespa Config" "http://localhost:$VESPA_CONFIG_PORT/state/v1/health" || exit 1
    wait_for_service "Phoenix" "$PHOENIX_URL/health" || exit 1
    wait_for_service "Ollama" "$OLLAMA_URL/api/tags" || exit 1
    log_success "Core services are healthy"
    echo ""

    # Step 2: Deploy Vespa schemas
    log_info "Step 2/5: Deploying Vespa schemas..."
    deploy_schemas
    echo ""

    # Step 3: Pull Ollama models
    log_info "Step 3/5: Pulling Ollama models..."
    pull_ollama_models
    echo ""

    # Step 4: Create Phoenix projects
    log_info "Step 4/5: Setting up Phoenix projects..."
    create_phoenix_projects
    echo ""

    # Step 5: Verify all services
    log_info "Step 5/5: Verifying all services..."
    verify_services
    echo ""

    # Print access information
    print_access_info

    log_success "Initialization complete!"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Cogniverse Service Initialization Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h              Show this help message"
        echo ""
        echo "Environment variables:"
        echo "  TENANTS                 Space-separated list of tenant IDs (default: 'default')"
        echo "  OLLAMA_MODELS           Space-separated list of Ollama models (default: 'llama2:7b mistral:7b')"
        echo "  MAX_WAIT                Maximum wait time for services in seconds (default: 300)"
        echo "  VESPA_URL               Vespa URL (default: http://localhost:8080)"
        echo "  PHOENIX_URL             Phoenix URL (default: http://localhost:6006)"
        echo "  OLLAMA_URL              Ollama URL (default: http://localhost:11434)"
        echo ""
        echo "Examples:"
        echo "  # Initialize with default settings"
        echo "  ./scripts/docker/init_services.sh"
        echo ""
        echo "  # Initialize with custom tenants"
        echo "  TENANTS='acme_corp globex_inc' ./scripts/docker/init_services.sh"
        echo ""
        echo "  # Initialize with specific Ollama models"
        echo "  OLLAMA_MODELS='llama2:13b mistral:7b-instruct' ./scripts/docker/init_services.sh"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac
