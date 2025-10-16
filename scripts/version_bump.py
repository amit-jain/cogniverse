#!/usr/bin/env python3
"""
Cogniverse SDK Version Bump Script

Automatically bumps package versions across all SDK packages following semantic versioning.
Supports major, minor, patch, and prerelease version bumps with Git tagging.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import tomli
    import tomli_w
except ImportError:
    print("Error: tomli and tomli_w are required")
    print("Install with: pip install tomli tomli-w")
    sys.exit(1)


# Color codes for terminal output
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'  # No Color


def log_info(message: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {message}")


def log_success(message: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {message}")


def log_warning(message: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {message}")


def log_error(message: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {message}")


# Package build order (dependency order)
PACKAGES = [
    "core",        # cogniverse_core
    "agents",      # cogniverse_agents
    "vespa",       # cogniverse_vespa
    "runtime",     # cogniverse_runtime
    "dashboard",   # cogniverse_dashboard
]


def get_project_root() -> Path:
    """Get the project root directory."""
    script_dir = Path(__file__).parent
    return script_dir.parent


def get_package_dir(package: str) -> Path:
    """Get the directory path for a package."""
    return get_project_root() / "libs" / package


def get_pyproject_path(package: str) -> Path:
    """Get the pyproject.toml path for a package."""
    return get_package_dir(package) / "pyproject.toml"


def parse_version(version: str) -> Tuple[int, int, int, str]:
    """
    Parse a semantic version string.

    Returns: (major, minor, patch, prerelease)
    """
    # Match semantic version: X.Y.Z or X.Y.Z-prerelease
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(-[a-zA-Z0-9.]+)?$', version)

    if not match:
        raise ValueError(f"Invalid version format: {version}")

    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3))
    prerelease = match.group(4) or ""

    return major, minor, patch, prerelease


def format_version(major: int, minor: int, patch: int, prerelease: str = "") -> str:
    """Format version components into a semantic version string."""
    version = f"{major}.{minor}.{patch}"
    if prerelease:
        version += prerelease
    return version


def bump_version(version: str, bump_type: str, prerelease_suffix: str = None) -> str:
    """
    Bump version according to semantic versioning.

    Args:
        version: Current version string
        bump_type: One of 'major', 'minor', 'patch', 'prerelease'
        prerelease_suffix: Suffix for prerelease versions (e.g., 'alpha', 'beta', 'rc')

    Returns:
        New version string
    """
    major, minor, patch, current_prerelease = parse_version(version)

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
        prerelease = ""
    elif bump_type == "minor":
        minor += 1
        patch = 0
        prerelease = ""
    elif bump_type == "patch":
        patch += 1
        prerelease = ""
    elif bump_type == "prerelease":
        if prerelease_suffix:
            # Add or increment prerelease suffix
            if current_prerelease:
                # Extract number from current prerelease
                match = re.match(r'-(\w+)\.(\d+)', current_prerelease)
                if match and match.group(1) == prerelease_suffix:
                    # Increment existing prerelease
                    num = int(match.group(2)) + 1
                    prerelease = f"-{prerelease_suffix}.{num}"
                else:
                    # New prerelease type
                    prerelease = f"-{prerelease_suffix}.0"
            else:
                # First prerelease
                patch += 1
                prerelease = f"-{prerelease_suffix}.0"
        else:
            log_error("Prerelease bump requires --prerelease-suffix")
            sys.exit(1)
    else:
        log_error(f"Invalid bump type: {bump_type}")
        sys.exit(1)

    return format_version(major, minor, patch, prerelease)


def get_package_version(package: str) -> str:
    """Get current version of a package."""
    pyproject_path = get_pyproject_path(package)

    if not pyproject_path.exists():
        log_error(f"pyproject.toml not found for package: {package}")
        sys.exit(1)

    with open(pyproject_path, 'rb') as f:
        data = tomli.load(f)

    return data['project']['version']


def set_package_version(package: str, new_version: str, dry_run: bool = False):
    """Set version for a package in its pyproject.toml."""
    pyproject_path = get_pyproject_path(package)

    if not pyproject_path.exists():
        log_error(f"pyproject.toml not found for package: {package}")
        sys.exit(1)

    # Read current pyproject.toml
    with open(pyproject_path, 'rb') as f:
        data = tomli.load(f)

    old_version = data['project']['version']

    if dry_run:
        log_info(f"  Would update {package}: {old_version} → {new_version}")
        return

    # Update version
    data['project']['version'] = new_version

    # Write updated pyproject.toml
    with open(pyproject_path, 'wb') as f:
        tomli_w.dump(data, f)

    log_success(f"  Updated {package}: {old_version} → {new_version}")


def get_package_name(package: str) -> str:
    """Get the package name from pyproject.toml."""
    pyproject_path = get_pyproject_path(package)

    with open(pyproject_path, 'rb') as f:
        data = tomli.load(f)

    return data['project']['name']


def check_git_status() -> bool:
    """Check if git working directory is clean."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True
        )
        return len(result.stdout.strip()) == 0
    except subprocess.CalledProcessError:
        return False


def git_commit_version_bump(new_version: str, packages: List[str], dry_run: bool = False):
    """Commit version bump changes."""
    if dry_run:
        log_info(f"Would commit version bump to {new_version}")
        return

    try:
        # Stage pyproject.toml files
        for package in packages:
            pyproject_path = get_pyproject_path(package)
            subprocess.run(["git", "add", str(pyproject_path)], check=True)

        # Commit
        commit_message = f"Bump version to {new_version}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        log_success(f"Committed version bump: {new_version}")
    except subprocess.CalledProcessError as e:
        log_error(f"Git commit failed: {e}")
        sys.exit(1)


def git_tag_version(new_version: str, dry_run: bool = False):
    """Create git tag for new version."""
    tag_name = f"v{new_version}"

    if dry_run:
        log_info(f"Would create git tag: {tag_name}")
        return

    try:
        subprocess.run(
            ["git", "tag", "-a", tag_name, "-m", f"Release {new_version}"],
            check=True
        )
        log_success(f"Created git tag: {tag_name}")
    except subprocess.CalledProcessError as e:
        log_error(f"Git tag failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Bump version for Cogniverse SDK packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bump patch version (0.1.0 -> 0.1.1)
  ./scripts/version_bump.py patch

  # Bump minor version (0.1.1 -> 0.2.0)
  ./scripts/version_bump.py minor

  # Bump major version (0.2.0 -> 1.0.0)
  ./scripts/version_bump.py major

  # Create prerelease version (0.1.0 -> 0.1.1-alpha.0)
  ./scripts/version_bump.py prerelease --prerelease-suffix alpha

  # Dry run (preview changes)
  ./scripts/version_bump.py patch --dry-run

  # Bump specific package only
  ./scripts/version_bump.py patch --package core

  # With git commit and tag
  ./scripts/version_bump.py minor --commit --tag
        """
    )

    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch", "prerelease"],
        help="Type of version bump"
    )

    parser.add_argument(
        "--package",
        "-p",
        help="Bump specific package only (default: all packages)"
    )

    parser.add_argument(
        "--prerelease-suffix",
        "-s",
        choices=["alpha", "beta", "rc"],
        help="Prerelease suffix (required for prerelease bump)"
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview changes without modifying files"
    )

    parser.add_argument(
        "--commit",
        "-c",
        action="store_true",
        help="Commit version bump changes"
    )

    parser.add_argument(
        "--tag",
        "-t",
        action="store_true",
        help="Create git tag for new version"
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force bump even if git working directory is not clean"
    )

    args = parser.parse_args()

    # Print header
    print()
    print("=" * 50)
    print("Cogniverse SDK Version Bump")
    print("=" * 50)
    print(f"Bump type: {args.bump_type}")
    if args.prerelease_suffix:
        print(f"Prerelease suffix: {args.prerelease_suffix}")
    if args.dry_run:
        print("Mode: DRY RUN (no changes will be made)")
    print("=" * 50)
    print()

    # Check git status
    if (args.commit or args.tag) and not args.force:
        if not check_git_status():
            log_error("Git working directory is not clean")
            log_error("Commit or stash changes before version bump")
            log_error("Use --force to override")
            sys.exit(1)

    # Determine packages to bump
    if args.package:
        if args.package not in PACKAGES:
            log_error(f"Unknown package: {args.package}")
            log_error(f"Available packages: {', '.join(PACKAGES)}")
            sys.exit(1)
        packages_to_bump = [args.package]
    else:
        packages_to_bump = PACKAGES

    # Get current version (use core as reference)
    reference_package = packages_to_bump[0]
    current_version = get_package_version(reference_package)

    log_info(f"Current version: {current_version}")

    # Calculate new version
    new_version = bump_version(current_version, args.bump_type, args.prerelease_suffix)

    log_info(f"New version: {new_version}")
    print()

    # Bump each package
    log_info("Bumping package versions...")
    for package in packages_to_bump:
        set_package_version(package, new_version, dry_run=args.dry_run)

    print()

    if args.dry_run:
        log_warning("Dry run completed - no files were modified")
        print()
        print("To apply changes, run without --dry-run:")
        cmd = f"./scripts/version_bump.py {args.bump_type}"
        if args.prerelease_suffix:
            cmd += f" --prerelease-suffix {args.prerelease_suffix}"
        if args.commit:
            cmd += " --commit"
        if args.tag:
            cmd += " --tag"
        print(f"  {cmd}")
        return

    # Git commit
    if args.commit:
        log_info("Committing version bump...")
        git_commit_version_bump(new_version, packages_to_bump, dry_run=args.dry_run)
        print()

    # Git tag
    if args.tag:
        log_info("Creating git tag...")
        git_tag_version(new_version, dry_run=args.dry_run)
        print()

    # Print summary
    print("=" * 50)
    log_success("Version bump completed!")
    print("=" * 50)
    print()
    print(f"Version: {current_version} → {new_version}")
    print(f"Packages updated: {len(packages_to_bump)}")
    print()
    print("Next steps:")
    print("  1. Review changes: git diff")
    if not args.commit:
        print("  2. Commit changes: git add libs/*/pyproject.toml && git commit")
    if not args.tag:
        print(f"  3. Create tag: git tag -a v{new_version} -m 'Release {new_version}'")
    print("  4. Build packages: ./scripts/build_packages.sh --clean")
    print("  5. Publish packages: ./scripts/publish_packages.sh")
    print()


if __name__ == "__main__":
    main()
