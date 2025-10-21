#!/bin/bash
# Cogniverse Health Check Script
# Checks health status of all Docker Compose services

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
OTEL_URL=${OTEL_URL:-"http://localhost:13133"}

TIMEOUT=${TIMEOUT:-5}
VERBOSE=${VERBOSE:-false}
OUTPUT_FORMAT=${OUTPUT_FORMAT:-"text"}  # text, json, prometheus

# Logging functions
log_info() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

log_success() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${GREEN}[SUCCESS]${NC} $1"
    fi
}

log_warning() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${YELLOW}[WARNING]${NC} $1"
    fi
}

log_error() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${RED}[ERROR]${NC} $1"
    fi
}

# Check service health
check_service() {
    local service_name=$1
    local health_url=$2
    local timeout=${3:-$TIMEOUT}

    log_info "Checking $service_name at $health_url"

    local start_time=$(date +%s%N)
    local response_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$timeout" "$health_url" 2>/dev/null || echo "000")
    local end_time=$(date +%s%N)
    local response_time=$(( (end_time - start_time) / 1000000 ))  # Convert to milliseconds

    if [ "$response_code" -ge 200 ] && [ "$response_code" -lt 300 ]; then
        log_success "$service_name is healthy (${response_time}ms)"
        echo "HEALTHY|$response_code|$response_time"
        return 0
    else
        log_error "$service_name is unhealthy (HTTP $response_code)"
        echo "UNHEALTHY|$response_code|$response_time"
        return 1
    fi
}

# Check Docker container status
check_container_status() {
    local container_name=$1

    if command -v docker &> /dev/null; then
        local status=$(docker inspect -f '{{.State.Status}}' "$container_name" 2>/dev/null || echo "not_found")
        local health=$(docker inspect -f '{{.State.Health.Status}}' "$container_name" 2>/dev/null || echo "unknown")

        if [ "$status" = "running" ]; then
            if [ "$health" = "healthy" ] || [ "$health" = "unknown" ]; then
                return 0
            fi
        fi
    fi
    return 1
}

# Detailed service checks
check_vespa() {
    log_info "Checking Vespa..."

    # Check application status
    local app_status=$(check_service "Vespa Application" "$VESPA_URL/ApplicationStatus" "$TIMEOUT")
    local app_health=$(echo "$app_status" | cut -d'|' -f1)
    local app_code=$(echo "$app_status" | cut -d'|' -f2)
    local app_time=$(echo "$app_status" | cut -d'|' -f3)

    # Check config server
    local config_status=$(check_service "Vespa Config" "http://localhost:$VESPA_CONFIG_PORT/state/v1/health" "$TIMEOUT")
    local config_health=$(echo "$config_status" | cut -d'|' -f1)
    local config_code=$(echo "$config_status" | cut -d'|' -f2)
    local config_time=$(echo "$config_status" | cut -d'|' -f3)

    if [ "$app_health" = "HEALTHY" ] && [ "$config_health" = "HEALTHY" ]; then
        echo "vespa|HEALTHY|$app_code|$app_time"
        return 0
    else
        echo "vespa|UNHEALTHY|$app_code|$app_time"
        return 1
    fi
}

check_phoenix() {
    log_info "Checking Phoenix..."

    local status=$(check_service "Phoenix" "$PHOENIX_URL/health" "$TIMEOUT")
    local health=$(echo "$status" | cut -d'|' -f1)
    local code=$(echo "$status" | cut -d'|' -f2)
    local time=$(echo "$status" | cut -d'|' -f3)

    echo "phoenix|$health|$code|$time"
    [ "$health" = "HEALTHY" ]
}

check_ollama() {
    log_info "Checking Ollama..."

    local status=$(check_service "Ollama" "$OLLAMA_URL/api/tags" "$TIMEOUT")
    local health=$(echo "$status" | cut -d'|' -f1)
    local code=$(echo "$status" | cut -d'|' -f2)
    local time=$(echo "$status" | cut -d'|' -f3)

    # Count available models
    local model_count=0
    if [ "$health" = "HEALTHY" ]; then
        model_count=$(curl -sf "$OLLAMA_URL/api/tags" 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(len(data.get('models', [])))" 2>/dev/null || echo "0")
        log_info "Ollama has $model_count models available"
    fi

    echo "ollama|$health|$code|$time|$model_count"
    [ "$health" = "HEALTHY" ]
}

check_runtime() {
    log_info "Checking Runtime..."

    local status=$(check_service "Runtime" "$RUNTIME_URL/health" "$TIMEOUT")
    local health=$(echo "$status" | cut -d'|' -f1)
    local code=$(echo "$status" | cut -d'|' -f2)
    local time=$(echo "$status" | cut -d'|' -f3)

    echo "runtime|$health|$code|$time"
    [ "$health" = "HEALTHY" ]
}

check_dashboard() {
    log_info "Checking Dashboard..."

    local status=$(check_service "Dashboard" "$DASHBOARD_URL/_stcore/health" "$TIMEOUT")
    local health=$(echo "$status" | cut -d'|' -f1)
    local code=$(echo "$status" | cut -d'|' -f2)
    local time=$(echo "$status" | cut -d'|' -f3)

    echo "dashboard|$health|$code|$time"
    [ "$health" = "HEALTHY" ]
}

check_otel() {
    log_info "Checking OTEL Collector..."

    local status=$(check_service "OTEL" "$OTEL_URL" "$TIMEOUT")
    local health=$(echo "$status" | cut -d'|' -f1)
    local code=$(echo "$status" | cut -d'|' -f2)
    local time=$(echo "$status" | cut -d'|' -f3)

    echo "otel|$health|$code|$time"
    [ "$health" = "HEALTHY" ]
}

# Output formatters
output_text() {
    local results=("$@")

    echo ""
    echo "=========================================="
    echo "Cogniverse Health Check"
    echo "=========================================="
    echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    local total=0
    local healthy=0
    local unhealthy=0

    for result in "${results[@]}"; do
        IFS='|' read -r service health code time extra <<< "$result"
        total=$((total + 1))

        if [ "$health" = "HEALTHY" ]; then
            echo -e "${GREEN}✓${NC} $service: $health (${time}ms, HTTP $code)"
            if [ -n "$extra" ]; then
                echo "  Extra info: $extra"
            fi
            healthy=$((healthy + 1))
        else
            echo -e "${RED}✗${NC} $service: $health (${time}ms, HTTP $code)"
            unhealthy=$((unhealthy + 1))
        fi
    done

    echo ""
    echo "Summary: $healthy/$total services healthy"
    echo "=========================================="
    echo ""

    if [ $unhealthy -gt 0 ]; then
        return 1
    fi
    return 0
}

output_json() {
    local results=("$@")

    local timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    local services_json="["

    local first=true
    for result in "${results[@]}"; do
        IFS='|' read -r service health code time extra <<< "$result"

        if [ "$first" = false ]; then
            services_json="$services_json,"
        fi
        first=false

        services_json="$services_json{\"name\":\"$service\",\"status\":\"$health\",\"http_code\":$code,\"response_time_ms\":$time"
        if [ -n "$extra" ]; then
            services_json="$services_json,\"extra\":\"$extra\""
        fi
        services_json="$services_json}"
    done

    services_json="$services_json]"

    local healthy_count=0
    local total_count=0
    for result in "${results[@]}"; do
        IFS='|' read -r service health code time extra <<< "$result"
        total_count=$((total_count + 1))
        if [ "$health" = "HEALTHY" ]; then
            healthy_count=$((healthy_count + 1))
        fi
    done

    local overall_status="HEALTHY"
    if [ $healthy_count -lt $total_count ]; then
        overall_status="UNHEALTHY"
    fi

    cat <<EOF
{
  "timestamp": "$timestamp",
  "overall_status": "$overall_status",
  "healthy_count": $healthy_count,
  "total_count": $total_count,
  "services": $services_json
}
EOF

    [ "$overall_status" = "HEALTHY" ]
}

output_prometheus() {
    local results=("$@")

    echo "# HELP cogniverse_service_up Service health status (1=healthy, 0=unhealthy)"
    echo "# TYPE cogniverse_service_up gauge"

    for result in "${results[@]}"; do
        IFS='|' read -r service health code time extra <<< "$result"

        local status_value=0
        if [ "$health" = "HEALTHY" ]; then
            status_value=1
        fi

        echo "cogniverse_service_up{service=\"$service\"} $status_value"
    done

    echo ""
    echo "# HELP cogniverse_service_response_time_ms Service response time in milliseconds"
    echo "# TYPE cogniverse_service_response_time_ms gauge"

    for result in "${results[@]}"; do
        IFS='|' read -r service health code time extra <<< "$result"
        echo "cogniverse_service_response_time_ms{service=\"$service\"} $time"
    done

    echo ""
    echo "# HELP cogniverse_service_http_code Service HTTP response code"
    echo "# TYPE cogniverse_service_http_code gauge"

    for result in "${results[@]}"; do
        IFS='|' read -r service health code time extra <<< "$result"
        echo "cogniverse_service_http_code{service=\"$service\"} $code"
    done
}

# Main health check
main() {
    local results=()

    # Check all services
    results+=("$(check_vespa)")
    results+=("$(check_phoenix)")
    results+=("$(check_ollama)")
    results+=("$(check_otel)")
    results+=("$(check_runtime)")
    results+=("$(check_dashboard)")

    # Output results in requested format
    case "$OUTPUT_FORMAT" in
        json)
            output_json "${results[@]}"
            ;;
        prometheus)
            output_prometheus "${results[@]}"
            return 0  # Prometheus output is always successful
            ;;
        *)
            output_text "${results[@]}"
            ;;
    esac
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --format|-f)
            OUTPUT_FORMAT="$2"
            shift 2
            ;;
        --timeout|-t)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            cat <<EOF
Cogniverse Health Check Script

Usage: $0 [OPTIONS]

Options:
  --verbose, -v           Enable verbose output
  --format, -f FORMAT     Output format: text, json, prometheus (default: text)
  --timeout, -t SECONDS   Health check timeout in seconds (default: 5)
  --help, -h              Show this help message

Environment variables:
  VESPA_URL               Vespa URL (default: http://localhost:8080)
  VESPA_CONFIG_PORT       Vespa config port (default: 19071)
  PHOENIX_URL             Phoenix URL (default: http://localhost:6006)
  OLLAMA_URL              Ollama URL (default: http://localhost:11434)
  RUNTIME_URL             Runtime URL (default: http://localhost:8000)
  DASHBOARD_URL           Dashboard URL (default: http://localhost:8501)
  OTEL_URL                OTEL Collector URL (default: http://localhost:13133)

Examples:
  # Basic health check
  ./scripts/docker/health_check.sh

  # Verbose output
  ./scripts/docker/health_check.sh --verbose

  # JSON output
  ./scripts/docker/health_check.sh --format json

  # Prometheus metrics
  ./scripts/docker/health_check.sh --format prometheus

  # Custom timeout
  ./scripts/docker/health_check.sh --timeout 10

Exit codes:
  0 - All services healthy
  1 - One or more services unhealthy
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run health check
main
exit $?
