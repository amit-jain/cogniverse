#!/bin/bash
# Cogniverse SDK Package Build Script
# Builds the release package set in dependency order through the build backend
# and writes dist/BUILD_MANIFEST.json describing exactly the artifacts built.

set -euo pipefail

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
MANIFEST_NAME="BUILD_MANIFEST.json"

# Release set in dependency order: the published packages (core, agents, vespa,
# runtime, dashboard) plus every internal package they require.
PACKAGES=(
    "sdk"
    "foundation"
    "core"
    "evaluation"
    "synthetic"
    "vespa"
    "agents"
    "telemetry-phoenix"
    "runtime"
    "dashboard"
)

# Build options
CLEAN=${CLEAN:-false}
VERBOSE=${VERBOSE:-false}
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
    echo -e "${RED}[ERROR]${NC} $1" >&2
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

# Build a single package into its own staging directory
build_package() {
    local package=$1
    local package_dir="$LIBS_DIR/$package"
    local out_dir="$STAGE_DIR/$package"

    log_info "Building package: $package"

    if [ ! -f "$package_dir/pyproject.toml" ]; then
        log_error "pyproject.toml not found in $package_dir"
        return 1
    fi

    local uv_args=(build --no-sources "$package_dir" --out-dir "$out_dir")
    if [ "$VERBOSE" = true ]; then
        uv_args+=(--verbose)
    fi
    if ! uv "${uv_args[@]}"; then
        log_error "  uv build failed for $package"
        return 1
    fi

    if [ "$RUN_TESTS" = true ]; then
        log_info "  Running tests for $package..."
        local test_log="$TEST_LOG_DIR/${package}.log"
        if (cd "$PROJECT_ROOT" && JAX_PLATFORM_NAME=cpu timeout 300 uv run pytest "tests/${package}/" -v 2>&1 | tee "$test_log"); then
            log_success "  Tests passed for $package"
        else
            log_warning "  Some tests failed for $package (see $test_log)"
            if [ "${STRICT:-false}" = true ]; then
                return 1
            fi
        fi
    fi

    log_success "Package built: $package"
    echo ""
    return 0
}

# Copy this invocation's artifacts into DIST_DIR without replacing different bytes
collect_distributions() {
    log_info "Collecting distributions to $DIST_DIR..."
    if ! mkdir -p "$DIST_DIR"; then
        log_error "Failed to create $DIST_DIR"
        return 1
    fi

    local artifacts=()
    local artifact
    for package in "${PACKAGES[@]}"; do
        for artifact in "$STAGE_DIR/$package"/*.whl "$STAGE_DIR/$package"/*.tar.gz; do
            artifacts+=("$artifact")
        done
    done

    local conflicts=0
    local name
    local target
    local partial
    for artifact in "${artifacts[@]}"; do
        target="$DIST_DIR/${artifact##*/}"
        if [ -e "$target" ] && ! cmp -s "$artifact" "$target"; then
            log_error "$target differs from the artifact built by this invocation"
            conflicts=$((conflicts + 1))
        fi
    done
    if [ $conflicts -gt 0 ]; then
        log_error "Refusing to replace $conflicts existing artifact(s); use --clean or remove them"
        return 1
    fi

    for artifact in "${artifacts[@]}" "$STAGE_DIR/$MANIFEST_NAME"; do
        name="${artifact##*/}"
        target="$DIST_DIR/$name"
        if [ "$name" != "$MANIFEST_NAME" ] && [ -e "$target" ]; then
            continue
        fi
        partial="$DIST_DIR/.$name.$$.partial"
        if ! cp "$artifact" "$partial" || ! mv "$partial" "$target"; then
            rm -f "$partial"
            log_error "Failed to copy $name into $DIST_DIR"
            return 1
        fi
    done
    log_success "Collected ${#artifacts[@]} artifact(s) listed in $DIST_DIR/$MANIFEST_NAME"
}

# Main build process
main() {
    print_header

    if [ "$CLEAN" = true ]; then
        log_info "Removing $DIST_DIR"
        rm -rf "$DIST_DIR"
    fi
    rm -f "$DIST_DIR/$MANIFEST_NAME"

    STAGE_DIR=$(mktemp -d)
    trap 'rm -rf "$STAGE_DIR"' EXIT
    if [ "$RUN_TESTS" = true ]; then
        TEST_LOG_DIR=$(mktemp -d "${TMPDIR:-/tmp}/cogniverse-build-tests.XXXXXX")
        log_info "Test logs: $TEST_LOG_DIR"
    fi

    cd "$PROJECT_ROOT"

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
        log_error "  Failed: $failed_count packages"
        exit 1
    fi
    echo ""

    log_info "Validating artifacts and writing manifest..."
    if ! uv run --no-sync python "$PROJECT_ROOT/scripts/release_manifest.py" \
        --libs-dir "$LIBS_DIR" \
        --stage-dir "$STAGE_DIR" \
        --output "$STAGE_DIR/$MANIFEST_NAME" \
        "${PACKAGES[@]}"; then
        log_error "Artifact validation failed"
        exit 1
    fi

    if ! collect_distributions; then
        exit 1
    fi
    echo ""

    if [ "$VERBOSE" = true ]; then
        cat "$DIST_DIR/$MANIFEST_NAME"
    fi

    echo "=========================================="
    log_success "Build completed successfully!"
    echo "=========================================="
    echo ""
    echo "Distribution directory: $DIST_DIR"
    echo "Manifest: $DIST_DIR/$MANIFEST_NAME"
    echo "Total packages: ${#PACKAGES[@]}"
    echo ""
    echo "Next steps:"
    echo "  1. Test packages: uv pip install dist/*.whl"
    echo "  2. Publish to TestPyPI: ./scripts/publish_packages.sh --test"
    echo "  3. Publish to PyPI: ./scripts/publish_packages.sh"
    echo ""
}

usage() {
    cat <<EOF
Cogniverse SDK Package Build Script

Usage: $0 [OPTIONS]

Options:
  --clean             Remove dist/ before building
  --verbose           Enable verbose output
  --test              Run tests after building each package
  --strict            Fail if any test fails (with --test)
  --continue          Continue building even if a package fails
  --help, -h          Show this help message

Environment Variables:
  CLEAN               Same as --clean
  VERBOSE             Same as --verbose
  RUN_TESTS           Same as --test
  STRICT              Same as --strict
  CONTINUE_ON_ERROR   Same as --continue

Requires a synced project environment (uv sync) for artifact validation.

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
}

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --clean)
            CLEAN=true
            ;;
        --verbose)
            VERBOSE=true
            ;;
        --test)
            RUN_TESTS=true
            ;;
        --strict)
            STRICT=true
            ;;
        --continue)
            CONTINUE_ON_ERROR=true
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

main
