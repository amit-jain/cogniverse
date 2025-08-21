#!/usr/bin/env python3
"""
Test runner script for ingestion module with marker-based filtering.

Usage examples:
  # Run all CI-safe unit tests
  python scripts/test_ingestion.py --unit --ci-safe
  
  # Run local-only integration tests with heavy models
  python scripts/test_ingestion.py --integration --local-only
  
  # Run only Vespa tests
  python scripts/test_ingestion.py --requires-vespa
  
  # Run all tests except heavy models
  python scripts/test_ingestion.py --exclude-heavy
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.utils.markers import get_test_environment_info, is_ci_environment


def build_pytest_command(args) -> List[str]:
    """Build pytest command with appropriate markers."""
    cmd = ["uv", "run", "python", "-m", "pytest"]
    
    # Base test paths
    test_paths = []
    if args.unit:
        test_paths.append("tests/ingestion/unit/")
    if args.integration:
        test_paths.append("tests/ingestion/integration/")
    
    if not test_paths:
        test_paths = ["tests/ingestion/"]
    
    cmd.extend(test_paths)
    
    # Build marker expression
    markers = []
    
    if args.unit:
        markers.append("unit")
    if args.integration:
        markers.append("integration")
    
    if args.ci_safe:
        markers.append("ci_safe")
    if args.local_only:
        markers.append("local_only")
    
    if args.requires_vespa:
        markers.append("requires_vespa")
    if args.requires_colpali:
        markers.append("requires_colpali")
    if args.requires_videoprism:
        markers.append("requires_videoprism")
    if args.requires_colqwen:
        markers.append("requires_colqwen")
    
    # Exclude heavy models in CI by default
    if args.exclude_heavy or (is_ci_environment() and not args.include_heavy):
        markers.append("not local_only")
    
    if markers:
        marker_expr = " and ".join(markers)
        cmd.extend(["-m", marker_expr])
    
    # Add common options
    cmd.extend([
        "-v",
        "--tb=short",
        "--cov=src/app/ingestion/processors",
        "--cov-report=term-missing"
    ])
    
    if args.coverage_fail_under:
        cmd.extend([f"--cov-fail-under={args.coverage_fail_under}"])
    
    if args.verbose:
        cmd.append("-s")
    
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run ingestion tests with marker filtering")
    
    # Test type selection
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    
    # Environment markers
    parser.add_argument("--ci-safe", action="store_true", help="Run only CI-safe tests")
    parser.add_argument("--local-only", action="store_true", help="Run local-only tests")
    
    # Backend requirements
    parser.add_argument("--requires-vespa", action="store_true", help="Run Vespa tests")
    parser.add_argument("--requires-docker", action="store_true", help="Run Docker tests")
    
    # Model requirements
    parser.add_argument("--requires-colpali", action="store_true", help="Run ColPali tests")
    parser.add_argument("--requires-videoprism", action="store_true", help="Run VideoPrism tests")
    parser.add_argument("--requires-colqwen", action="store_true", help="Run ColQwen tests")
    
    # Heavy model control
    parser.add_argument("--exclude-heavy", action="store_true", help="Exclude heavy model tests")
    parser.add_argument("--include-heavy", action="store_true", help="Include heavy model tests")
    
    # Coverage options
    parser.add_argument("--coverage-fail-under", type=int, default=80, help="Coverage threshold")
    
    # Other options
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running")
    parser.add_argument("--env-info", action="store_true", help="Show environment info")
    
    args = parser.parse_args()
    
    # Show environment info if requested
    if args.env_info:
        env_info = get_test_environment_info()
        print("🔍 Test Environment Information:")
        print("="*50)
        for key, value in env_info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {'✅' if sub_value else '❌'}")
            else:
                print(f"{key}: {'✅' if value else '❌'}")
        print("="*50)
        print()
    
    # Build and run command
    cmd = build_pytest_command(args)
    
    if args.dry_run:
        print("Would run command:")
        print(" ".join(cmd))
        return 0
    
    print(f"🧪 Running tests: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n❌ Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())