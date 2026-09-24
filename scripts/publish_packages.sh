#!/bin/bash
# Cogniverse SDK Package Publishing Script
# Publishes exactly the artifacts listed in dist/BUILD_MANIFEST.json to PyPI or
# TestPyPI, then verifies the index serves each of them with the manifest's sha256.

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DIST_DIR="$PROJECT_ROOT/dist"
MANIFEST="$DIST_DIR/BUILD_MANIFEST.json"
RELEASE_MANIFEST="$PROJECT_ROOT/scripts/release_manifest.py"
TWINE_REQUIREMENT="twine==7.0.0"

# Publishing options
TEST_PYPI=${TEST_PYPI:-false}
DRY_RUN=${DRY_RUN:-false}
VERBOSE=${VERBOSE:-false}
SKIP_EXISTING=${SKIP_EXISTING:-true}
CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR:-false}
ASSUME_YES=${ASSUME_YES:-false}
VERIFY_TIMEOUT=${VERIFY_TIMEOUT:-300}

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

join_names() {
    local joined=""
    local name
    for name in "$@"; do
        joined="${joined:+$joined, }$name"
    done
    echo "$joined"
}

# Python with the pinned Twine (and packaging) installed, independent of the project environment
release_python() {
    uv run --no-project --with "$TWINE_REQUIREMENT" python "$@"
}

configure_target() {
    if [ "$TEST_PYPI" = true ]; then
        TARGET="TestPyPI"
        REPOSITORY="testpypi"
        INDEX_URL="https://test.pypi.org/simple/"
        PROJECT_URL="https://test.pypi.org/project"
        TOKEN="${TEST_PYPI_TOKEN:-}"
        TOKEN_NAME="TEST_PYPI_TOKEN"
    else
        TARGET="PyPI"
        REPOSITORY="pypi"
        INDEX_URL="https://pypi.org/simple/"
        PROJECT_URL="https://pypi.org/project"
        TOKEN="${PYPI_TOKEN:-}"
        TOKEN_NAME="PYPI_TOKEN"
    fi
}

# Print header
print_header() {
    echo ""
    echo "=========================================="
    echo "Cogniverse SDK Package Publishing"
    echo "=========================================="
    echo "Target: $TARGET"
    echo "Manifest: $MANIFEST"
    echo "Twine: $TWINE_REQUIREMENT"
    if [ "$DRY_RUN" = true ]; then
        echo "Mode: DRY RUN (verifies artifacts; uploads nothing and contacts no registry)"
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

# Read the upload plan: the manifest's packages, verified against dist/
load_release() {
    local plan
    if ! plan=$(release_python "$RELEASE_MANIFEST" publishable --dist-dir "$DIST_DIR"); then
        log_error "Nothing was uploaded: the artifacts in $DIST_DIR are not publishable"
        return 1
    fi

    PACKAGE_NAMES=()
    PACKAGE_VERSIONS=()
    WHEELS=()
    SDISTS=()
    local name version wheel sdist
    while IFS=$'\t' read -r name version wheel sdist; do
        PACKAGE_NAMES+=("$name")
        PACKAGE_VERSIONS+=("$version")
        WHEELS+=("$wheel")
        SDISTS+=("$sdist")
    done <<< "$plan"
    log_info "Manifest lists ${#PACKAGE_NAMES[@]} package(s), version ${PACKAGE_VERSIONS[0]}"
}

# Validate the manifest's distributions (and only those) with twine
check_distributions() {
    local files=()
    local i
    for i in "${!PACKAGE_NAMES[@]}"; do
        files+=("${WHEELS[$i]}" "${SDISTS[$i]}")
    done

    log_info "Validating ${#files[@]} distribution(s) with twine..."
    if ! (cd "$DIST_DIR" && release_python -m twine check "${files[@]}"); then
        log_error "Distribution validation failed; nothing was uploaded"
        return 1
    fi
    log_success "All distributions are valid"
}

check_credentials() {
    if [ -n "$TOKEN" ]; then
        log_info "$TOKEN_NAME is set"
    else
        log_warning "$TOKEN_NAME is not set; twine uses ~/.pypirc or prompts for credentials"
    fi
}

confirm_publish() {
    if [ "$ASSUME_YES" = true ]; then
        return 0
    fi
    echo "WARNING: You are about to publish ${#PACKAGE_NAMES[@]} package(s) to $TARGET"
    echo "This action cannot be undone!"
    echo ""
    local confirm=""
    read -r -p "Continue? (yes/no): " confirm || true
    if [ "$confirm" != "yes" ]; then
        log_error "Publishing cancelled: nothing was uploaded"
        exit 1
    fi
    echo ""
}

report_dry_run() {
    local i
    for i in "${!PACKAGE_NAMES[@]}"; do
        echo "[DRY RUN] Would upload ${WHEELS[$i]}"
        echo "[DRY RUN] Would upload ${SDISTS[$i]}"
    done
    echo ""
    echo "DRY RUN complete: nothing was uploaded and no registry was contacted"
}

# Publish one package's wheel and sdist; the twine exit status is the result
publish_package() {
    local name=$1 version=$2 wheel=$3 sdist=$4

    log_info "Publishing package: $name $version"
    log_info "  Wheel: $wheel"
    log_info "  Source: $sdist"

    local upload_args=(upload --repository "$REPOSITORY")
    if [ "$SKIP_EXISTING" = true ]; then
        upload_args+=("--skip-existing")
    fi
    if [ "$VERBOSE" = true ]; then
        upload_args+=("--verbose")
    fi
    upload_args+=("$DIST_DIR/$wheel" "$DIST_DIR/$sdist")

    local status=0
    if [ -n "$TOKEN" ]; then
        TWINE_USERNAME="__token__" TWINE_PASSWORD="$TOKEN" \
            release_python -m twine "${upload_args[@]}" || status=$?
    else
        release_python -m twine "${upload_args[@]}" || status=$?
    fi

    if [ $status -ne 0 ]; then
        log_error "Failed to publish $name $version (twine exited $status)"
        return 1
    fi
    log_success "Uploaded or already present: $name $version"
}

# Publish all packages in manifest (dependency) order
publish_all_packages() {
    log_info "Publishing packages in dependency order..."
    echo ""

    local published=()
    local failed=()
    local not_attempted=()
    local i
    for i in "${!PACKAGE_NAMES[@]}"; do
        if [ ${#failed[@]} -gt 0 ] && [ "$CONTINUE_ON_ERROR" != true ]; then
            not_attempted+=("${PACKAGE_NAMES[$i]}")
            continue
        fi
        if publish_package "${PACKAGE_NAMES[$i]}" "${PACKAGE_VERSIONS[$i]}" "${WHEELS[$i]}" "${SDISTS[$i]}"; then
            published+=("${PACKAGE_NAMES[$i]}")
        else
            failed+=("${PACKAGE_NAMES[$i]}")
        fi
        echo ""
    done

    log_info "Publishing Summary:"
    log_info "  Uploaded or already present: ${#published[@]} package(s)"
    if [ ${#failed[@]} -gt 0 ]; then
        log_error "  Failed: ${#failed[@]} package(s): $(join_names "${failed[@]}")"
    fi
    if [ ${#not_attempted[@]} -gt 0 ]; then
        log_error "  Not attempted: ${#not_attempted[@]} package(s): $(join_names "${not_attempted[@]}")"
    fi
    [ ${#failed[@]} -eq 0 ]
}

# Confirm the index serves every manifest artifact with the manifest's sha256
verify_index() {
    log_info "Verifying $INDEX_URL serves every manifest artifact..."
    if ! release_python "$RELEASE_MANIFEST" check-index --dist-dir "$DIST_DIR" \
        --index-url "$INDEX_URL" --timeout "$VERIFY_TIMEOUT"; then
        log_error "$TARGET does not serve the manifest's artifacts unchanged"
        return 1
    fi
}

print_post_publish() {
    echo ""
    echo "=========================================="
    log_success "Publishing completed!"
    echo "=========================================="
    echo ""
    echo "Packages published to $TARGET:"
    local name
    for name in "${PACKAGE_NAMES[@]}"; do
        echo "  $PROJECT_URL/$name/${PACKAGE_VERSIONS[0]}/"
    done
    echo ""
}

# Main publishing process
main() {
    configure_target
    print_header

    log_info "Running pre-flight checks..."
    check_uv
    if ! load_release; then
        exit 1
    fi
    if ! check_distributions; then
        exit 1
    fi
    echo ""

    if [ "$DRY_RUN" = true ]; then
        report_dry_run
        exit 0
    fi

    check_credentials
    confirm_publish

    if ! publish_all_packages; then
        log_error "Publishing failed"
        exit 1
    fi
    if ! verify_index; then
        log_error "Publishing failed"
        exit 1
    fi

    print_post_publish
}

usage() {
    cat <<EOF
Cogniverse SDK Package Publishing Script

Usage: $0 [OPTIONS]

Publishes the artifacts listed in dist/BUILD_MANIFEST.json (written by
./scripts/build_packages.sh), in manifest order, then verifies the index serves
each with the manifest's sha256.

Options:
  --test              Publish to TestPyPI instead of PyPI
  --dry-run           Verify artifacts and list the uploads without uploading
  --verbose           Enable verbose output
  --no-skip-existing  Fail if a file already exists on the index
  --continue          Continue publishing even if a package fails (still exits nonzero)
  --yes               Do not ask for confirmation
  --help, -h          Show this help message

Environment Variables:
  TEST_PYPI           Same as --test
  DRY_RUN             Same as --dry-run
  VERBOSE             Same as --verbose
  SKIP_EXISTING       Pass --skip-existing to twine (default: true)
  CONTINUE_ON_ERROR   Same as --continue
  ASSUME_YES          Same as --yes
  VERIFY_TIMEOUT      Seconds to wait for the index to serve the uploads (default: 300)
  PYPI_TOKEN          PyPI API token
  TEST_PYPI_TOKEN     TestPyPI API token

Examples:
  # Verify what would be published to TestPyPI
  ./scripts/publish_packages.sh --test --dry-run

  # Publish to TestPyPI
  TEST_PYPI_TOKEN="your-token" ./scripts/publish_packages.sh --test

  # Publish to production PyPI
  PYPI_TOKEN="your-token" ./scripts/publish_packages.sh
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --test)
            TEST_PYPI=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --verbose)
            VERBOSE=true
            ;;
        --no-skip-existing)
            SKIP_EXISTING=false
            ;;
        --continue)
            CONTINUE_ON_ERROR=true
            ;;
        --yes)
            ASSUME_YES=true
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
