#!/bin/bash
# Local Docker Compose Deployment Script
# Deploys Cogniverse using Docker Compose for local development/testing

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
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
PRODUCTION="${PRODUCTION:-false}"
TENANTS="${TENANTS:-default}"
OLLAMA_MODELS="${OLLAMA_MODELS:-mistral:7b-instruct}"

print_header() {
    echo ""
    echo "=========================================="
    echo "Cogniverse Local Docker Deployment"
    echo "=========================================="
    echo "Mode: $([ "$PRODUCTION" = true ] && echo "Production" || echo "Development")"
    echo "Compose file: $COMPOSE_FILE"
    echo "Tenants: $TENANTS"
    echo "=========================================="
    echo ""
}

check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

create_env_file() {
    log_info "Setting up environment configuration..."

    if [ "$PRODUCTION" = true ]; then
        if [ ! -f "$PROJECT_ROOT/.env.prod" ]; then
            if [ -f "$PROJECT_ROOT/.env.prod.example" ]; then
                log_warning ".env.prod not found, copying from example"
                cp "$PROJECT_ROOT/.env.prod.example" "$PROJECT_ROOT/.env.prod"
                log_warning "Please edit .env.prod with your configuration"
                exit 1
            else
                log_error ".env.prod.example not found"
                exit 1
            fi
        fi
        export ENV_FILE=".env.prod"
    else
        if [ ! -f "$PROJECT_ROOT/.env" ]; then
            if [ -f "$PROJECT_ROOT/.env.example" ]; then
                log_info "Creating .env from example"
                cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
            fi
        fi
        export ENV_FILE=".env"
    fi

    log_success "Environment configuration ready"
}

start_services() {
    log_info "Starting Docker Compose services..."

    cd "$PROJECT_ROOT"

    # Determine compose command
    if docker compose version &> /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi

    # Start services
    if [ "$PRODUCTION" = true ]; then
        $COMPOSE_CMD -f docker-compose.prod.yml up -d
    else
        $COMPOSE_CMD up -d
    fi

    log_success "Services started"
}

wait_for_services() {
    log_info "Waiting for services to be healthy..."

    # Run health check script
    if [ -f "$PROJECT_ROOT/scripts/docker/health_check.sh" ]; then
        local max_attempts=30
        local attempt=0

        while [ $attempt -lt $max_attempts ]; do
            if "$PROJECT_ROOT/scripts/docker/health_check.sh" &> /dev/null; then
                log_success "All services are healthy"
                return 0
            fi

            attempt=$((attempt + 1))
            log_info "Waiting for services... (attempt $attempt/$max_attempts)"
            sleep 10
        done

        log_warning "Some services may not be fully healthy yet"
    else
        log_warning "Health check script not found, waiting 60 seconds..."
        sleep 60
    fi
}

initialize_services() {
    log_info "Initializing services..."

    if [ -f "$PROJECT_ROOT/scripts/docker/init_services.sh" ]; then
        TENANTS="$TENANTS" OLLAMA_MODELS="$OLLAMA_MODELS" \
            "$PROJECT_ROOT/scripts/docker/init_services.sh"
    else
        log_warning "Initialization script not found, skipping"
    fi
}

show_access_info() {
    echo ""
    echo "=========================================="
    log_success "Deployment completed successfully!"
    echo "=========================================="
    echo ""
    echo "Access services at:"
    echo "  • Runtime API:     http://localhost:8000"
    echo "  • API Docs:        http://localhost:8000/docs"
    echo "  • Dashboard:       http://localhost:8501"
    echo "  • Phoenix:         http://localhost:6006"
    echo "  • Vespa:           http://localhost:8080"
    echo "  • Ollama:          http://localhost:11434"
    echo ""
    echo "Initialized tenants: $TENANTS"
    echo ""
    echo "Next steps:"
    echo "  1. Check service status:"
    echo "     docker-compose ps"
    echo ""
    echo "  2. View logs:"
    echo "     docker-compose logs -f runtime"
    echo ""
    echo "  3. Ingest videos:"
    echo "     uv run python scripts/run_ingestion.py \\"
    echo "       --video_dir data/videos \\"
    echo "       --profile video_colpali_smol500_mv_frame \\"
    echo "       --tenant default"
    echo ""
    echo "  4. Stop services:"
    echo "     docker-compose down"
    echo ""
    echo "=========================================="
}

show_logs() {
    log_info "Showing service logs (Ctrl+C to exit)..."
    cd "$PROJECT_ROOT"

    if docker compose version &> /dev/null 2>&1; then
        docker compose logs -f
    else
        docker-compose logs -f
    fi
}

main() {
    print_header
    check_prerequisites
    create_env_file
    start_services
    wait_for_services
    initialize_services
    show_access_info

    # Optionally show logs
    if [ "${SHOW_LOGS:-false}" = true ]; then
        show_logs
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        cat <<EOF
Cogniverse Local Docker Deployment Script

Usage: $0 [OPTIONS]

Options:
  --production        Use production configuration (docker-compose.prod.yml)
  --logs              Show logs after deployment
  --help, -h          Show this help message

Environment Variables:
  PRODUCTION          Same as --production
  TENANTS             Space-separated tenant IDs (default: "default")
  OLLAMA_MODELS       Space-separated Ollama models (default: "mistral:7b-instruct")
  SHOW_LOGS           Same as --logs

Examples:
  # Development deployment
  ./scripts/deploy_local_docker.sh

  # Production deployment
  ./scripts/deploy_local_docker.sh --production

  # With custom tenants and models
  TENANTS="default acme_corp" OLLAMA_MODELS="llama2:13b mistral:7b-instruct" \\
    ./scripts/deploy_local_docker.sh

  # With logs
  ./scripts/deploy_local_docker.sh --logs
EOF
        exit 0
        ;;
    --production)
        PRODUCTION=true
        COMPOSE_FILE="docker-compose.prod.yml"
        shift
        main "$@"
        ;;
    --logs)
        SHOW_LOGS=true
        shift
        main "$@"
        ;;
    *)
        main "$@"
        ;;
esac
