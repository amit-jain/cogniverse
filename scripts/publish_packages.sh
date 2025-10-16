#!/bin/bash
# Cogniverse SDK Package Publishing Script
# Publishes packages to PyPI or TestPyPI with validation and safety checks

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DIST_DIR="$PROJECT_ROOT/dist"

# Publishing options
TEST_PYPI=${TEST_PYPI:-false}
DRY_RUN=${DRY_RUN:-false}
VERBOSE=${VERBOSE:-false}
SKIP_EXISTING=${SKIP_EXISTING:-true}

# PyPI URLs
PYPI_URL="https://upload.pypi.org/legacy/"
TEST_PYPI_URL="https://test.pypi.org/legacy/"

# Packages in dependency order
PACKAGES=(
    "cogniverse-core"
    "cogniverse-agents"
    "cogniverse-vespa"
    "cogniverse-runtime"
    "cogniverse-dashboard"
)

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

# Print header
print_header() {
    local target="PyPI"
    if [ "$TEST_PYPI" = true ]; then
        target="TestPyPI"
    fi

    echo ""
    echo "=========================================="
    echo "Cogniverse SDK Package Publishing"
    echo "=========================================="
    echo "Target: $target"
    echo "Distribution directory: $DIST_DIR"
    if [ "$DRY_RUN" = true ]; then
        echo "Mode: DRY RUN (no actual publishing)"
    fi
    echo "=========================================="
    echo ""
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        log_error "uv is not installed"
        log_error "Install with: pip install uv"
        exit 1
    fi
}

# Check if twine is available
check_twine() {
    if ! uv run python -c "import twine" 2>/dev/null; then
        log_info "Installing twine..."
        uv pip install twine
    fi
}

# Check if dist directory exists and has packages
check_dist_directory() {
    if [ ! -d "$DIST_DIR" ]; then
        log_error "Distribution directory not found: $DIST_DIR"
        log_error "Run ./scripts/build_packages.sh first"
        exit 1
    fi

    local wheel_count=$(find "$DIST_DIR" -name "*.whl" | wc -l)
    local sdist_count=$(find "$DIST_DIR" -name "*.tar.gz" | wc -l)

    if [ $wheel_count -eq 0 ] && [ $sdist_count -eq 0 ]; then
        log_error "No packages found in $DIST_DIR"
        log_error "Run ./scripts/build_packages.sh first"
        exit 1
    fi

    log_info "Found $wheel_count wheel(s) and $sdist_count source distribution(s)"
}

# Validate package distributions
validate_distributions() {
    log_info "Validating distributions with twine..."

    local files_to_check=()
    for file in "$DIST_DIR"/*.whl "$DIST_DIR"/*.tar.gz; do
        if [ -f "$file" ]; then
            files_to_check+=("$file")
        fi
    done

    if [ ${#files_to_check[@]} -eq 0 ]; then
        log_error "No distributions to validate"
        return 1
    fi

    if uv run twine check "${files_to_check[@]}"; then
        log_success "All distributions are valid"
        return 0
    else
        log_error "Distribution validation failed"
        return 1
    fi
}

# Check PyPI credentials
check_credentials() {
    local target="PyPI"
    if [ "$TEST_PYPI" = true ]; then
        target="TestPyPI"
    fi

    log_info "Checking $target credentials..."

    # Check for API token in environment
    if [ "$TEST_PYPI" = true ]; then
        if [ -z "${TEST_PYPI_TOKEN:-}" ]; then
            log_warning "TEST_PYPI_TOKEN environment variable not set"
            log_info "You will be prompted for credentials during upload"
        else
            log_success "TEST_PYPI_TOKEN is set"
        fi
    else
        if [ -z "${PYPI_TOKEN:-}" ]; then
            log_warning "PYPI_TOKEN environment variable not set"
            log_info "You will be prompted for credentials during upload"
        else
            log_success "PYPI_TOKEN is set"
        fi
    fi

    # Check .pypirc
    if [ -f "$HOME/.pypirc" ]; then
        log_info ".pypirc file found"
    else
        log_warning ".pypirc file not found"
        log_info "You may need to configure PyPI credentials"
    fi
}

# Check if package already exists on PyPI
check_package_exists() {
    local package_name=$1
    local version=$2
    local target_url

    if [ "$TEST_PYPI" = true ]; then
        target_url="https://test.pypi.org/pypi/${package_name}/${version}/json"
    else
        target_url="https://pypi.org/pypi/${package_name}/${version}/json"
    fi

    if curl -sf "$target_url" > /dev/null 2>&1; then
        return 0  # Package exists
    else
        return 1  # Package doesn't exist
    fi
}

# Extract version from filename
extract_version() {
    local filename=$1

    # Extract version from wheel: package-1.0.0-py3-none-any.whl
    # Or from sdist: package-1.0.0.tar.gz
    if [[ "$filename" =~ -([0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?)(-|\.tar) ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "unknown"
    fi
}

# Publish a single package
publish_package() {
    local package_name=$1

    log_info "Publishing package: $package_name"

    # Find distributions for this package
    local wheel_file=$(find "$DIST_DIR" -name "${package_name//-/_}-*.whl" | head -n 1)
    local sdist_file=$(find "$DIST_DIR" -name "${package_name}-*.tar.gz" | head -n 1)

    if [ -z "$wheel_file" ] && [ -z "$sdist_file" ]; then
        log_warning "No distributions found for $package_name"
        return 1
    fi

    # Extract version
    local version=$(extract_version "$(basename "$wheel_file")")
    log_info "  Version: $version"

    # Check if already published
    if check_package_exists "$package_name" "$version"; then
        if [ "$SKIP_EXISTING" = true ]; then
            log_warning "  Package already exists, skipping"
            return 0
        else
            log_error "  Package already exists"
            return 1
        fi
    fi

    # Prepare files to upload
    local files_to_upload=()
    if [ -n "$wheel_file" ]; then
        files_to_upload+=("$wheel_file")
        log_info "  Wheel: $(basename "$wheel_file")"
    fi
    if [ -n "$sdist_file" ]; then
        files_to_upload+=("$sdist_file")
        log_info "  Source: $(basename "$sdist_file")"
    fi

    if [ "$DRY_RUN" = true ]; then
        log_info "  [DRY RUN] Would upload ${#files_to_upload[@]} file(s)"
        return 0
    fi

    # Build twine upload command
    local upload_args=()

    if [ "$TEST_PYPI" = true ]; then
        upload_args+=("--repository" "testpypi")
        # Use token if available
        if [ -n "${TEST_PYPI_TOKEN:-}" ]; then
            upload_args+=("--username" "__token__")
            upload_args+=("--password" "$TEST_PYPI_TOKEN")
        fi
    else
        upload_args+=("--repository" "pypi")
        # Use token if available
        if [ -n "${PYPI_TOKEN:-}" ]; then
            upload_args+=("--username" "__token__")
            upload_args+=("--password" "$PYPI_TOKEN")
        fi
    fi

    if [ "$SKIP_EXISTING" = true ]; then
        upload_args+=("--skip-existing")
    fi

    if [ "$VERBOSE" = true ]; then
        upload_args+=("--verbose")
    fi

    # Upload
    log_info "  Uploading..."
    if uv run twine upload "${upload_args[@]}" "${files_to_upload[@]}"; then
        log_success "  Published successfully: $package_name v$version"
        return 0
    else
        log_error "  Failed to publish $package_name"
        return 1
    fi
}

# Publish all packages
publish_all_packages() {
    log_info "Publishing packages in dependency order..."
    echo ""

    local published_count=0
    local failed_count=0
    local skipped_count=0

    for package in "${PACKAGES[@]}"; do
        if publish_package "$package"; then
            published_count=$((published_count + 1))
        else
            if [ "$SKIP_EXISTING" = true ]; then
                skipped_count=$((skipped_count + 1))
            else
                failed_count=$((failed_count + 1))

                if [ "${CONTINUE_ON_ERROR:-false}" != true ]; then
                    log_error "Stopping due to failure (use CONTINUE_ON_ERROR=true to continue)"
                    exit 1
                fi
            fi
        fi
        echo ""
    done

    # Print summary
    echo ""
    log_info "Publishing Summary:"
    log_info "  Published: $published_count packages"
    if [ $skipped_count -gt 0 ]; then
        log_warning "  Skipped: $skipped_count packages (already exist)"
    fi
    if [ $failed_count -gt 0 ]; then
        log_error "  Failed: $failed_count packages"
    fi

    if [ $failed_count -gt 0 ]; then
        return 1
    fi

    return 0
}

# Print post-publish instructions
print_post_publish() {
    local target="PyPI"
    local url="https://pypi.org"

    if [ "$TEST_PYPI" = true ]; then
        target="TestPyPI"
        url="https://test.pypi.org"
    fi

    echo ""
    echo "=========================================="
    log_success "Publishing completed!"
    echo "=========================================="
    echo ""
    echo "Packages published to: $target"
    echo ""
    echo "View packages:"
    for package in "${PACKAGES[@]}"; do
        echo "  $url/project/$package/"
    done
    echo ""

    if [ "$TEST_PYPI" = true ]; then
        echo "Test installation:"
        echo "  pip install --index-url https://test.pypi.org/simple/ cogniverse-core"
        echo ""
        echo "To publish to production PyPI:"
        echo "  ./scripts/publish_packages.sh"
    else
        echo "Install packages:"
        echo "  pip install cogniverse-core"
        echo "  pip install cogniverse-agents"
        echo "  pip install cogniverse-vespa"
        echo "  pip install cogniverse-runtime"
        echo "  pip install cogniverse-dashboard"
        echo ""
        echo "Or install all:"
        echo "  pip install cogniverse-runtime  # Includes core, agents, vespa"
    fi
    echo ""
}

# Main publishing process
main() {
    print_header

    # Pre-flight checks
    log_info "Running pre-flight checks..."
    check_uv
    check_twine
    check_dist_directory
    check_credentials
    echo ""

    # Validate distributions
    if ! validate_distributions; then
        log_error "Validation failed, aborting"
        exit 1
    fi
    echo ""

    # Confirm publishing
    if [ "$DRY_RUN" != true ]; then
        local target="PyPI"
        if [ "$TEST_PYPI" = true ]; then
            target="TestPyPI"
        fi

        echo "WARNING: You are about to publish packages to $target"
        echo "This action cannot be undone!"
        echo ""
        read -p "Continue? (yes/no): " confirm

        if [ "$confirm" != "yes" ]; then
            log_info "Publishing cancelled"
            exit 0
        fi
        echo ""
    fi

    # Publish packages
    if publish_all_packages; then
        print_post_publish
        exit 0
    else
        log_error "Publishing failed"
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        cat <<EOF
Cogniverse SDK Package Publishing Script

Usage: $0 [OPTIONS]

Options:
  --test              Publish to TestPyPI instead of PyPI
  --dry-run           Preview publishing without uploading
  --verbose           Enable verbose output
  --no-skip-existing  Fail if package version already exists
  --continue          Continue publishing even if a package fails
  --help, -h          Show this help message

Environment Variables:
  TEST_PYPI           Same as --test
  DRY_RUN             Same as --dry-run
  VERBOSE             Same as --verbose
  SKIP_EXISTING       Skip packages that already exist (default: true)
  CONTINUE_ON_ERROR   Same as --continue
  PYPI_TOKEN          PyPI API token
  TEST_PYPI_TOKEN     TestPyPI API token

Examples:
  # Test publishing to TestPyPI
  ./scripts/publish_packages.sh --test --dry-run

  # Publish to TestPyPI
  TEST_PYPI_TOKEN="your-token" ./scripts/publish_packages.sh --test

  # Publish to production PyPI
  PYPI_TOKEN="your-token" ./scripts/publish_packages.sh

  # Dry run for production
  ./scripts/publish_packages.sh --dry-run
EOF
        exit 0
        ;;
    --test)
        TEST_PYPI=true
        shift
        main "$@"
        ;;
    --dry-run)
        DRY_RUN=true
        shift
        main "$@"
        ;;
    --verbose)
        VERBOSE=true
        shift
        main "$@"
        ;;
    --no-skip-existing)
        SKIP_EXISTING=false
        shift
        main "$@"
        ;;
    --continue)
        CONTINUE_ON_ERROR=true
        shift
        main "$@"
        ;;
    *)
        main "$@"
        ;;
esac
