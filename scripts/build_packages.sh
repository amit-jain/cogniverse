#!/bin/bash
# Cogniverse SDK Package Build Script
# Builds all 5 SDK packages in dependency order with validation

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
LIBS_DIR="$PROJECT_ROOT/libs"
DIST_DIR="$PROJECT_ROOT/dist"
BUILD_DIR="$PROJECT_ROOT/build"

# Package build order (respects dependencies)
PACKAGES=(
    "core"           # cogniverse_core (no dependencies)
    "agents"         # cogniverse_agents (depends on core)
    "vespa"          # cogniverse_vespa (depends on core)
    "runtime"        # cogniverse_runtime (depends on core, agents, vespa)
    "dashboard"      # cogniverse_dashboard (depends on core, agents)
)

# Build options
CLEAN=${CLEAN:-false}
VERBOSE=${VERBOSE:-false}
CHECK_VERSION=${CHECK_VERSION:-true}
RUN_TESTS=${RUN_TESTS:-false}

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
    echo ""
    echo "=========================================="
    echo "Cogniverse SDK Package Build"
    echo "=========================================="
    echo "Build Directory: $DIST_DIR"
    echo "Packages: ${PACKAGES[*]}"
    echo "Clean Build: $CLEAN"
    echo "Run Tests: $RUN_TESTS"
    echo "=========================================="
    echo ""
}

# Clean build artifacts
clean_build() {
    log_info "Cleaning build artifacts..."

    # Remove dist directory
    if [ -d "$DIST_DIR" ]; then
        log_info "  Removing $DIST_DIR"
        rm -rf "$DIST_DIR"
    fi

    # Remove build directory
    if [ -d "$BUILD_DIR" ]; then
        log_info "  Removing $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi

    # Clean each package
    for package in "${PACKAGES[@]}"; do
        local package_dir="$LIBS_DIR/$package"

        if [ -d "$package_dir/dist" ]; then
            log_info "  Removing $package_dir/dist"
            rm -rf "$package_dir/dist"
        fi

        if [ -d "$package_dir/build" ]; then
            log_info "  Removing $package_dir/build"
            rm -rf "$package_dir/build"
        fi

        if [ -d "$package_dir/*.egg-info" ]; then
            log_info "  Removing $package_dir/*.egg-info"
            rm -rf "$package_dir"/*.egg-info
        fi
    done

    log_success "Build artifacts cleaned"
}

# Check if package directory exists
check_package_exists() {
    local package=$1
    local package_dir="$LIBS_DIR/$package"

    if [ ! -d "$package_dir" ]; then
        log_error "Package directory not found: $package_dir"
        return 1
    fi

    if [ ! -f "$package_dir/pyproject.toml" ]; then
        log_error "pyproject.toml not found in $package_dir"
        return 1
    fi

    return 0
}

# Extract version from pyproject.toml
get_package_version() {
    local package=$1
    local package_dir="$LIBS_DIR/$package"
    local pyproject="$package_dir/pyproject.toml"

    # Extract version using Python
    python3 -c "
import tomli
with open('$pyproject', 'rb') as f:
    data = tomli.load(f)
    print(data['project']['version'])
" 2>/dev/null || echo "unknown"
}

# Get package name
get_package_name() {
    local package=$1
    local package_dir="$LIBS_DIR/$package"
    local pyproject="$package_dir/pyproject.toml"

    # Extract name using Python
    python3 -c "
import tomli
with open('$pyproject', 'rb') as f:
    data = tomli.load(f)
    print(data['project']['name'])
" 2>/dev/null || echo "unknown"
}

# Validate package version format
validate_version() {
    local version=$1

    # Check semantic versioning format (X.Y.Z or X.Y.Z-prerelease)
    if ! echo "$version" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$'; then
        log_error "Invalid version format: $version (expected X.Y.Z or X.Y.Z-prerelease)"
        return 1
    fi

    return 0
}

# Build a single package
build_package() {
    local package=$1
    local package_dir="$LIBS_DIR/$package"

    log_info "Building package: $package"

    # Check package exists
    if ! check_package_exists "$package"; then
        return 1
    fi

    # Get package info
    local package_name=$(get_package_name "$package")
    local package_version=$(get_package_version "$package")

    log_info "  Package name: $package_name"
    log_info "  Version: $package_version"

    # Validate version
    if [ "$CHECK_VERSION" = true ]; then
        if ! validate_version "$package_version"; then
            return 1
        fi
    fi

    # Change to package directory
    cd "$package_dir"

    # Build with uv
    log_info "  Running uv build..."
    if [ "$VERBOSE" = true ]; then
        uv build --verbose
    else
        uv build 2>&1 | grep -v "^Resolving" || true
    fi

    # Check if build succeeded
    if [ ! -d "$package_dir/dist" ]; then
        log_error "  Build failed: dist directory not created"
        return 1
    fi

    # Count built artifacts
    local wheel_count=$(find "$package_dir/dist" -name "*.whl" | wc -l)
    local sdist_count=$(find "$package_dir/dist" -name "*.tar.gz" | wc -l)

    if [ $wheel_count -eq 0 ] || [ $sdist_count -eq 0 ]; then
        log_error "  Build incomplete: expected wheel and sdist"
        return 1
    fi

    log_success "  Built: $wheel_count wheel(s), $sdist_count sdist(s)"

    # List built files
    if [ "$VERBOSE" = true ]; then
        log_info "  Built files:"
        find "$package_dir/dist" -type f -exec basename {} \; | sed 's/^/    /'
    fi

    # Run package tests if requested
    if [ "$RUN_TESTS" = true ]; then
        log_info "  Running tests for $package..."
        cd "$PROJECT_ROOT"

        if JAX_PLATFORM_NAME=cpu timeout 300 uv run pytest "tests/${package}/" -v 2>&1 | tee "/tmp/build-test-${package}.log"; then
            log_success "  Tests passed for $package"
        else
            log_warning "  Some tests failed for $package (see /tmp/build-test-${package}.log)"
            if [ "${STRICT:-false}" = true ]; then
                return 1
            fi
        fi
    fi

    cd "$PROJECT_ROOT"
    log_success "Package built successfully: $package_name v$package_version"
    echo ""

    return 0
}

# Copy all distributions to central dist directory
collect_distributions() {
    log_info "Collecting distributions to $DIST_DIR..."

    # Create dist directory
    mkdir -p "$DIST_DIR"

    # Copy from each package
    for package in "${PACKAGES[@]}"; do
        local package_dir="$LIBS_DIR/$package"

        if [ -d "$package_dir/dist" ]; then
            log_info "  Copying from $package..."
            cp "$package_dir/dist"/* "$DIST_DIR/"
        fi
    done

    # Count total artifacts
    local total_wheels=$(find "$DIST_DIR" -name "*.whl" | wc -l)
    local total_sdist=$(find "$DIST_DIR" -name "*.tar.gz" | wc -l)
    local total_size=$(du -sh "$DIST_DIR" | cut -f1)

    log_success "Collected $total_wheels wheel(s) and $total_sdist sdist(s) (Total: $total_size)"

    # List all distributions
    if [ "$VERBOSE" = true ]; then
        log_info "All distributions in $DIST_DIR:"
        ls -lh "$DIST_DIR" | tail -n +2 | awk '{print "    " $9 " (" $5 ")"}'
    fi
}

# Generate build manifest
generate_manifest() {
    log_info "Generating build manifest..."

    local manifest_file="$DIST_DIR/BUILD_MANIFEST.txt"

    cat > "$manifest_file" <<EOF
Cogniverse SDK Build Manifest
Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')
Build Host: $(hostname)
Build User: $(whoami)

Packages Built:
EOF

    for package in "${PACKAGES[@]}"; do
        local package_name=$(get_package_name "$package")
        local package_version=$(get_package_version "$package")

        echo "  - $package_name v$package_version" >> "$manifest_file"
    done

    echo "" >> "$manifest_file"
    echo "Distributions:" >> "$manifest_file"
    find "$DIST_DIR" -type f \( -name "*.whl" -o -name "*.tar.gz" \) -exec basename {} \; | sort | sed 's/^/  - /' >> "$manifest_file"

    echo "" >> "$manifest_file"
    echo "Build completed successfully" >> "$manifest_file"

    log_success "Build manifest generated: $manifest_file"

    if [ "$VERBOSE" = true ]; then
        cat "$manifest_file"
    fi
}

# Verify all packages
verify_packages() {
    log_info "Verifying packages..."

    local failed=0

    for package in "${PACKAGES[@]}"; do
        local package_name=$(get_package_name "$package")
        local package_version=$(get_package_version "$package")

        # Check wheel exists
        local wheel_file=$(find "$DIST_DIR" -name "${package_name//-/_}-${package_version}-*.whl" | head -n 1)
        if [ -z "$wheel_file" ]; then
            log_error "  Wheel not found for $package_name"
            failed=$((failed + 1))
            continue
        fi

        # Check sdist exists
        local sdist_file=$(find "$DIST_DIR" -name "${package_name}-${package_version}.tar.gz" | head -n 1)
        if [ -z "$sdist_file" ]; then
            log_error "  Source distribution not found for $package_name"
            failed=$((failed + 1))
            continue
        fi

        # Verify wheel metadata
        if command -v unzip &> /dev/null; then
            local wheel_metadata=$(unzip -l "$wheel_file" 2>/dev/null | grep -c "METADATA" || echo "0")
            if [ "$wheel_metadata" -eq 0 ]; then
                log_warning "  Wheel metadata missing for $package_name"
            fi
        fi

        log_success "  Verified: $package_name v$package_version"
    done

    if [ $failed -gt 0 ]; then
        log_error "Verification failed for $failed package(s)"
        return 1
    fi

    log_success "All packages verified successfully"
    return 0
}

# Main build process
main() {
    print_header

    # Clean if requested
    if [ "$CLEAN" = true ]; then
        clean_build
        echo ""
    fi

    # Change to project root
    cd "$PROJECT_ROOT"

    # Build each package in order
    local built_count=0
    local failed_count=0

    for package in "${PACKAGES[@]}"; do
        if build_package "$package"; then
            built_count=$((built_count + 1))
        else
            log_error "Failed to build package: $package"
            failed_count=$((failed_count + 1))

            if [ "${CONTINUE_ON_ERROR:-false}" != true ]; then
                log_error "Stopping build due to failure (use CONTINUE_ON_ERROR=true to continue)"
                exit 1
            fi
        fi
    done

    echo ""
    log_info "Build Summary:"
    log_info "  Built: $built_count packages"
    if [ $failed_count -gt 0 ]; then
        log_warning "  Failed: $failed_count packages"
    fi
    echo ""

    # Collect distributions
    collect_distributions
    echo ""

    # Verify packages
    if ! verify_packages; then
        log_error "Package verification failed"
        exit 1
    fi
    echo ""

    # Generate manifest
    generate_manifest
    echo ""

    # Print success message
    echo "=========================================="
    log_success "Build completed successfully!"
    echo "=========================================="
    echo ""
    echo "Distribution directory: $DIST_DIR"
    echo "Total packages: ${#PACKAGES[@]}"
    echo ""
    echo "Next steps:"
    echo "  1. Test packages: uv pip install dist/*.whl"
    echo "  2. Publish to TestPyPI: ./scripts/publish_packages.sh --test"
    echo "  3. Publish to PyPI: ./scripts/publish_packages.sh"
    echo ""

    if [ $failed_count -gt 0 ]; then
        exit 1
    fi
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        cat <<EOF
Cogniverse SDK Package Build Script

Usage: $0 [OPTIONS]

Options:
  --clean             Clean build artifacts before building
  --verbose           Enable verbose output
  --no-version-check  Skip version format validation
  --test              Run tests after building each package
  --strict            Fail if any test fails (with --test)
  --continue          Continue building even if a package fails
  --help, -h          Show this help message

Environment Variables:
  CLEAN               Same as --clean
  VERBOSE             Same as --verbose
  CHECK_VERSION       Enable version validation (default: true)
  RUN_TESTS           Same as --test
  STRICT              Same as --strict
  CONTINUE_ON_ERROR   Same as --continue

Examples:
  # Basic build
  ./scripts/build_packages.sh

  # Clean build with tests
  ./scripts/build_packages.sh --clean --test

  # Verbose build
  ./scripts/build_packages.sh --verbose

  # Continue on errors
  CONTINUE_ON_ERROR=true ./scripts/build_packages.sh
EOF
        exit 0
        ;;
    --clean)
        CLEAN=true
        shift
        main "$@"
        ;;
    --verbose)
        VERBOSE=true
        shift
        main "$@"
        ;;
    --no-version-check)
        CHECK_VERSION=false
        shift
        main "$@"
        ;;
    --test)
        RUN_TESTS=true
        shift
        main "$@"
        ;;
    --strict)
        STRICT=true
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
