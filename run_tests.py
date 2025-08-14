#!/usr/bin/env python3
"""
Test runner for the comprehensive routing system.
Provides options to run different test suites.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"✅ {description} passed")
    else:
        print(f"❌ {description} failed")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run routing system tests")
    parser.add_argument(
        "--suite",
        choices=["unit", "integration", "all", "demo"],
        default="all",
        help="Which test suite to run"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ["uv", "run", "pytest"]
    
    if args.verbose:
        base_cmd.append("-vv")
    else:
        base_cmd.append("-v")
    
    if args.coverage:
        base_cmd.extend([
            "--cov=src/routing",
            "--cov-report=term-missing",
            "--cov-report=html"
        ])
    
    # Track return codes
    return_codes = []
    
    if args.suite == "unit" or args.suite == "all":
        # Run unit tests
        cmd = base_cmd + ["tests/unit/", "-m", "not integration"]
        if not args.slow:
            cmd.extend(["-m", "not slow"])
        
        rc = run_command(cmd, "Unit Tests")
        return_codes.append(rc)
    
    if args.suite == "integration" or args.suite == "all":
        # Run integration tests
        cmd = base_cmd + ["tests/integration/", "-m", "not slow"]
        if args.slow:
            cmd = base_cmd + ["tests/integration/"]
        
        rc = run_command(cmd, "Integration Tests")
        return_codes.append(rc)
    
    if args.suite == "demo":
        # Run the demonstration script
        print("\n" + "="*60)
        print("Running: Tier Demonstration")
        print("="*60)
        
        demo_script = Path(__file__).parent / "demo_routing_unified.py"
        if demo_script.exists():
            # Add --verbose flag if requested
            cmd = ["uv", "run", "python", str(demo_script)]
            if args.verbose:
                cmd.append("--verbose")
            
            rc = subprocess.run(cmd, capture_output=False).returncode
            return_codes.append(rc)
        else:
            print(f"❌ Demo script not found: {demo_script}")
            return_codes.append(1)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if all(rc == 0 for rc in return_codes):
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())