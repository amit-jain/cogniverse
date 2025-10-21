# Package Publishing Guide

**Last Updated:** 2025-10-15
**Architecture:** UV Workspace with 5 SDK Packages
**Purpose:** Complete guide for building and publishing Cogniverse SDK packages to PyPI

---

## Table of Contents

- [Overview](#overview)
- [Package Structure](#package-structure)
- [Prerequisites](#prerequisites)
- [Version Management](#version-management)
- [Building Packages](#building-packages)
- [Testing Packages](#testing-packages)
- [Publishing to TestPyPI](#publishing-to-testpypi)
- [Publishing to PyPI](#publishing-to-pypi)
- [Automated Publishing (CI/CD)](#automated-publishing-cicd)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Overview

Cogniverse consists of **5 independent SDK packages** published to PyPI:

| Package | Description | Dependencies |
|---------|-------------|--------------|
| **cogniverse-core** | Core configuration, telemetry, evaluation | None (foundation) |
| **cogniverse-agents** | Agent implementations, routing, ingestion | cogniverse-core |
| **cogniverse-vespa** | Vespa backend integration | cogniverse-core |
| **cogniverse-runtime** | FastAPI server runtime | core, agents, vespa |
| **cogniverse-dashboard** | Streamlit UI dashboard | core, agents |

### Publishing Workflow

```
Version Bump → Build → Test → Publish → Release
     ↓           ↓       ↓        ↓         ↓
  version_    build_   pytest  publish_  GitHub
  bump.py   packages          packages  Release
           .sh                .sh
```

---

## Package Structure

Each package follows UV workspace structure:

```
libs/
├── core/                    # cogniverse-core
│   ├── cogniverse_core/
│   │   ├── __init__.py
│   │   ├── config/
│   │   ├── telemetry/
│   │   └── ...
│   ├── pyproject.toml       # Package metadata
│   └── README.md
├── agents/                  # cogniverse-agents
│   ├── cogniverse_agents/
│   ├── pyproject.toml
│   └── README.md
└── ...
```

**Key File: `pyproject.toml`**

```toml
[project]
name = "cogniverse-core"
version = "2.0.0"
description = "Core utilities for Cogniverse SDK"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.0",
    "pyyaml>=6.0"
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## Prerequisites

### Required Tools

```bash
# UV package manager
pip install uv

# Python dependencies for versioning
pip install tomli tomli-w

# Twine for PyPI uploads (optional - installed by scripts)
pip install twine
```

### PyPI Account Setup

#### 1. Create PyPI Accounts

- **PyPI (Production):** https://pypi.org/account/register/
- **TestPyPI (Testing):** https://test.pypi.org/account/register/

#### 2. Generate API Tokens

**PyPI:**
1. Go to https://pypi.org/manage/account/token/
2. Create token with scope: "Entire account"
3. Save token securely (starts with `pypi-`)

**TestPyPI:**
1. Go to https://test.pypi.org/manage/account/token/
2. Create token with scope: "Entire account"
3. Save token securely (starts with `pypi-`)

#### 3. Configure Credentials

**Option A: Environment Variables (Recommended for CI/CD)**

```bash
export PYPI_TOKEN="pypi-your-production-token"
export TEST_PYPI_TOKEN="pypi-your-test-token"
```

**Option B: .pypirc File (Local Development)**

```ini
# ~/.pypirc
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-your-production-token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-your-test-token
```

**Security Note:** Never commit `.pypirc` or tokens to version control!

---

## Version Management

### Semantic Versioning

Cogniverse follows [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH[-PRERELEASE]
  ↓     ↓     ↓         ↓
  2  .  1  .  3  -  alpha.0

MAJOR: Breaking changes
MINOR: New features (backward compatible)
PATCH: Bug fixes
PRERELEASE: alpha, beta, rc
```

### Version Bump Script

**Script:** `scripts/version_bump.py`

#### Basic Usage

```bash
# Patch bump (2.0.0 → 2.0.1)
./scripts/version_bump.py patch

# Minor bump (2.0.1 → 2.1.0)
./scripts/version_bump.py minor

# Major bump (2.1.0 → 3.0.0)
./scripts/version_bump.py major

# Prerelease (2.0.0 → 2.0.1-alpha.0)
./scripts/version_bump.py prerelease --prerelease-suffix alpha
```

#### Advanced Options

```bash
# Dry run (preview changes)
./scripts/version_bump.py patch --dry-run

# Bump specific package only
./scripts/version_bump.py patch --package core

# With git commit and tag
./scripts/version_bump.py minor --commit --tag

# Force (ignore git status)
./scripts/version_bump.py patch --force
```

#### Workflow Example

```bash
# 1. Preview version bump
./scripts/version_bump.py minor --dry-run

# 2. Apply version bump with commit and tag
./scripts/version_bump.py minor --commit --tag

# Output:
# [INFO] Current version: 2.0.0
# [INFO] New version: 2.1.0
# [SUCCESS] Updated core: 2.0.0 → 2.1.0
# [SUCCESS] Updated agents: 2.0.0 → 2.1.0
# ...
# [SUCCESS] Committed version bump: 2.1.0
# [SUCCESS] Created git tag: v2.1.0
```

---

## Building Packages

### Build Script

**Script:** `scripts/build_packages.sh`

#### Basic Build

```bash
# Build all packages
./scripts/build_packages.sh

# Clean build (remove previous artifacts)
./scripts/build_packages.sh --clean

# Verbose output
./scripts/build_packages.sh --verbose
```

#### Advanced Options

```bash
# Build with tests
./scripts/build_packages.sh --test

# Strict mode (fail on test failures)
./scripts/build_packages.sh --test --strict

# Continue on errors
CONTINUE_ON_ERROR=true ./scripts/build_packages.sh
```

#### Build Output

```
dist/
├── cogniverse_core-2.1.0-py3-none-any.whl
├── cogniverse_core-2.1.0.tar.gz
├── cogniverse_agents-2.1.0-py3-none-any.whl
├── cogniverse_agents-2.1.0.tar.gz
├── cogniverse_vespa-2.1.0-py3-none-any.whl
├── cogniverse_vespa-2.1.0.tar.gz
├── cogniverse_runtime-2.1.0-py3-none-any.whl
├── cogniverse_runtime-2.1.0.tar.gz
├── cogniverse_dashboard-2.1.0-py3-none-any.whl
├── cogniverse_dashboard-2.1.0.tar.gz
└── BUILD_MANIFEST.txt
```

#### Build Process

1. **Validation:** Checks package structure and version format
2. **Dependency Order:** Builds packages respecting dependencies
3. **Artifact Generation:** Creates wheel (.whl) and source (.tar.gz) distributions
4. **Verification:** Validates metadata and contents
5. **Manifest:** Generates build manifest with details

---

## Testing Packages

### Local Testing

#### Install from Built Distributions

```bash
# Create test environment
python -m venv test-env
source test-env/bin/activate

# Install packages in dependency order
pip install dist/cogniverse_core-*.whl
pip install dist/cogniverse_agents-*.whl
pip install dist/cogniverse_vespa-*.whl
pip install dist/cogniverse_runtime-*.whl
pip install dist/cogniverse_dashboard-*.whl

# Verify imports
python -c "from cogniverse_core.config import SystemConfig"
python -c "from cogniverse_agents.agents import RoutingAgent"
```

#### Run Test Suite

```bash
# Run all tests
JAX_PLATFORM_NAME=cpu uv run pytest

# Run package-specific tests
JAX_PLATFORM_NAME=cpu uv run pytest tests/common/
JAX_PLATFORM_NAME=cpu uv run pytest tests/agents/
```

---

## Publishing to TestPyPI

### Why TestPyPI?

- **Safe testing environment** for PyPI without affecting production
- **Validate package metadata** before production release
- **Test installation** from PyPI-like repository

### Manual Publishing

```bash
# 1. Build packages
./scripts/build_packages.sh --clean

# 2. Publish to TestPyPI (with dry run first)
TEST_PYPI_TOKEN="your-test-token" ./scripts/publish_packages.sh --test --dry-run

# 3. Actual publish
TEST_PYPI_TOKEN="your-test-token" ./scripts/publish_packages.sh --test

# Output:
# [INFO] Publishing package: cogniverse-core
# [INFO]   Version: 2.1.0
# [INFO]   Uploading...
# [SUCCESS] Published successfully: cogniverse-core v2.1.0
```

### Test Installation

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            cogniverse-core

# Note: --extra-index-url allows dependencies from production PyPI
```

### Verification

```bash
# Test package functionality
python -c "
from cogniverse_core.config import SystemConfig
config = SystemConfig(tenant_id='test')
print(f'Successfully imported: {config.tenant_id}')
"
```

---

## Publishing to PyPI

### Pre-publish Checklist

- [ ] All tests passing
- [ ] Version bumped correctly
- [ ] Git tag created (`v2.1.0`)
- [ ] CHANGELOG updated
- [ ] README accurate
- [ ] License included
- [ ] Tested on TestPyPI

### Manual Publishing

```bash
# 1. Final build
./scripts/build_packages.sh --clean --test

# 2. Publish to PyPI (with dry run first)
PYPI_TOKEN="your-production-token" ./scripts/publish_packages.sh --dry-run

# 3. Confirm and publish
PYPI_TOKEN="your-production-token" ./scripts/publish_packages.sh

# WARNING: You are about to publish packages to PyPI
# This action cannot be undone!
#
# Continue? (yes/no): yes

# Output:
# [SUCCESS] Published successfully: cogniverse-core v2.1.0
# [SUCCESS] Published successfully: cogniverse-agents v2.1.0
# ...
# [SUCCESS] Publishing completed!
```

### Verify Publication

```bash
# Check PyPI page
open https://pypi.org/project/cogniverse-core/

# Test installation
pip install cogniverse-core==2.1.0

# Test functionality
python -c "from cogniverse_core.config import SystemConfig; print('Success!')"
```

---

## Automated Publishing (CI/CD)

### GitHub Actions Workflow

**File:** `.github/workflows/publish-packages.yml`

#### Trigger: Version Tags

```bash
# Create and push version tag
git tag -a v2.1.0 -m "Release 2.1.0"
git push origin v2.1.0

# GitHub Actions automatically:
# 1. Builds packages
# 2. Runs tests
# 3. Publishes to PyPI
# 4. Creates GitHub Release
```

#### Trigger: Manual Workflow Dispatch

```bash
# Via GitHub UI:
# 1. Go to Actions → "Publish SDK Packages"
# 2. Click "Run workflow"
# 3. Select target: testpypi or pypi
# 4. Optional: Enable dry run
```

### Workflow Stages

```
┌──────────┐
│  Build   │ → Build all 5 packages
└────┬─────┘
     ↓
┌────┴─────┐
│   Test   │ → Run test suite with Vespa
└────┬─────┘
     ↓
   ┌─┴──────────┐
   │            │
┌──┴────┐  ┌───┴────┐
│TestPyPI│  │  PyPI  │ → Publish based on tag type
└───┬────┘  └───┬────┘
    ↓           ↓
┌───┴───────────┴────┐
│  GitHub Release     │ → Create release with notes
└────────────────────┘
```

### Tag-Based Publishing Rules

| Tag Format | Target | Example |
|------------|--------|---------|
| `v*.*.*` | PyPI | `v2.1.0` |
| `v*.*.*-alpha.*` | TestPyPI | `v2.1.0-alpha.0` |
| `v*.*.*-beta.*` | TestPyPI | `v2.1.0-beta.1` |
| `v*.*.*-rc.*` | TestPyPI | `v2.1.0-rc.1` |

### Required GitHub Secrets

```
Settings → Secrets → Actions:

PYPI_TOKEN           Production PyPI API token
TEST_PYPI_TOKEN      TestPyPI API token
```

---

## Troubleshooting

### Common Issues

#### Version Already Exists on PyPI

**Error:**
```
HTTPError: 400 Bad Request from https://upload.pypi.org/legacy/
File already exists
```

**Solution:**
```bash
# PyPI doesn't allow overwriting versions
# You must bump the version
./scripts/version_bump.py patch --commit --tag

# Then rebuild and republish
./scripts/build_packages.sh --clean
./scripts/publish_packages.sh
```

#### Missing Dependencies in Build

**Error:**
```
ModuleNotFoundError: No module named 'cogniverse_core'
```

**Solution:**
```bash
# Sync workspace dependencies
uv sync

# Rebuild
./scripts/build_packages.sh --clean
```

#### Twine Upload Fails

**Error:**
```
twine.exceptions.TwineException: Invalid or non-existent authentication information
```

**Solution:**
```bash
# Check credentials
cat ~/.pypirc

# Or use environment variables
export PYPI_TOKEN="pypi-your-token"
./scripts/publish_packages.sh
```

#### Package Import Fails After Install

**Error:**
```
ImportError: cannot import name 'SystemConfig' from 'cogniverse_core.config'
```

**Solution:**
```bash
# Check package contents
unzip -l dist/cogniverse_core-*.whl | grep config

# Verify __init__.py exports
cat libs/core/cogniverse_core/__init__.py

# Rebuild with correct exports
./scripts/build_packages.sh --clean
```

---

## Best Practices

### Version Management

1. **Use semantic versioning strictly**
   ```bash
   # Breaking changes
   ./scripts/version_bump.py major

   # New features
   ./scripts/version_bump.py minor

   # Bug fixes
   ./scripts/version_bump.py patch
   ```

2. **Always test prereleases**
   ```bash
   # Create prerelease
   ./scripts/version_bump.py prerelease --prerelease-suffix alpha

   # Publish to TestPyPI
   ./scripts/publish_packages.sh --test

   # Test thoroughly before production
   ```

3. **Use git tags for releases**
   ```bash
   ./scripts/version_bump.py minor --commit --tag
   git push origin --tags
   ```

### Building

1. **Always use clean builds for releases**
   ```bash
   ./scripts/build_packages.sh --clean
   ```

2. **Run tests before publishing**
   ```bash
   ./scripts/build_packages.sh --clean --test --strict
   ```

3. **Verify build artifacts**
   ```bash
   # Check wheel contents
   unzip -l dist/*.whl

   # Validate with twine
   twine check dist/*
   ```

### Publishing

1. **Always test on TestPyPI first**
   ```bash
   # Test publish
   ./scripts/publish_packages.sh --test

   # Verify installation
   pip install --index-url https://test.pypi.org/simple/ cogniverse-core

   # Then publish to production
   ./scripts/publish_packages.sh
   ```

2. **Use dry runs for validation**
   ```bash
   ./scripts/publish_packages.sh --dry-run
   ```

3. **Document changes in CHANGELOG**
   ```markdown
   ## [2.1.0] - 2025-10-15
   ### Added
   - New routing strategy: GLiNER-based
   - Multi-tenant Phoenix projects

   ### Changed
   - Improved memory performance

   ### Fixed
   - Vespa schema deployment bug
   ```

### Security

1. **Never commit secrets**
   ```bash
   # Add to .gitignore
   echo "*.pypirc" >> .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use environment variables in CI/CD**
   ```yaml
   # GitHub Actions
   env:
     PYPI_TOKEN: ${{ secrets.PYPI_TOKEN }}
   ```

3. **Rotate API tokens regularly**
   - PyPI tokens should be rotated every 6 months
   - Use scoped tokens when possible

### Quality Assurance

1. **Maintain package READMEs**
   - Each package should have a clear README
   - Include installation and usage examples

2. **Keep pyproject.toml updated**
   - Accurate dependencies
   - Correct Python version requirements
   - Proper classifiers

3. **Test cross-package dependencies**
   ```bash
   # Install only runtime (should pull dependencies)
   pip install cogniverse-runtime

   # Verify all dependencies installed
   pip list | grep cogniverse
   ```

---

## Complete Publishing Workflow

### Standard Release Process

```bash
# 1. Update code and tests
git add .
git commit -m "Add new routing feature"

# 2. Bump version
./scripts/version_bump.py minor --dry-run    # Preview
./scripts/version_bump.py minor --commit --tag

# 3. Build packages
./scripts/build_packages.sh --clean --verbose

# 4. Test packages
JAX_PLATFORM_NAME=cpu uv run pytest

# 5. Publish to TestPyPI
TEST_PYPI_TOKEN="your-token" ./scripts/publish_packages.sh --test

# 6. Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ cogniverse-core

# 7. Publish to production PyPI
PYPI_TOKEN="your-token" ./scripts/publish_packages.sh

# 8. Push tags to trigger CI/CD
git push origin --tags

# 9. Verify on PyPI
open https://pypi.org/project/cogniverse-core/

# 10. Create GitHub release notes
# (Automated by CI/CD workflow)
```

---

## Related Documentation

- [Package Development](package-dev.md) - SDK development guide
- [SDK Architecture](../architecture/sdk-architecture.md) - Package structure details
- [Testing Guide](../testing/pytest-best-practices.md) - Testing practices

---

## Support

- **PyPI Issues:** Check [PyPI Help](https://pypi.org/help/)
- **Build Issues:** Review build logs in `dist/BUILD_MANIFEST.txt`
- **CI/CD Issues:** Check GitHub Actions logs

---

**Version:** 2.0.0
**Last Updated:** 2025-10-15
**Status:** Production Ready
