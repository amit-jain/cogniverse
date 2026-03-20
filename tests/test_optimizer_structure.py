#!/usr/bin/env python3
"""
Test New Structure Script

Validates that the new src/ structure imports work correctly.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all new structure imports work."""

    print("🧪 Testing New src/ Structure Imports")
    print("=" * 60)

    # Add parent directory to path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

    tests_passed = 0
    total_tests = 0

    # Test 1: Core schemas
    total_tests += 1
    try:
        from cogniverse_agents.optimizer.schemas import (  # noqa: F401
            AgenticRouter,
            RoutingDecision,
        )

        print("✅ schemas: RoutingDecision, AgenticRouter")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ schemas: {e}")

    # Test 2: Router optimizer
    total_tests += 1
    try:
        from cogniverse_agents.optimizer.router_optimizer import (  # noqa: F401
            OptimizedRouter,
            RouterModule,
        )

        print("✅ router_optimizer: RouterModule, OptimizedRouter")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ router_optimizer: {e}")

    # Test 3: DSPy agent optimizer
    total_tests += 1
    try:
        from cogniverse_agents.optimizer.dspy_agent_optimizer import (  # noqa: F401
            DSPyAgentPromptOptimizer,
        )

        print("✅ dspy_agent_optimizer: DSPyAgentPromptOptimizer")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ dspy_agent_optimizer: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 IMPORT TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {tests_passed}/{total_tests}")
    print(f"📈 Success Rate: {(tests_passed / total_tests) * 100:.1f}%")

    return tests_passed == total_tests


def test_structure_integrity():
    """Test that the directory structure is correct."""

    print("\n🗂️ Testing Directory Structure")
    print("=" * 60)

    parent_dir = Path(__file__).parent.parent
    required_paths = [
        "libs/agents/cogniverse_agents/optimizer/schemas.py",
        "libs/agents/cogniverse_agents/optimizer/router_optimizer.py",
        "libs/agents/cogniverse_agents/optimizer/dspy_agent_optimizer.py",
    ]

    all_exist = True

    for path in required_paths:
        full_path = parent_dir / path
        if full_path.exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - MISSING")
            all_exist = False

    print(f"\n📊 Structure: {'✅ Complete' if all_exist else '❌ Missing files'}")
    return all_exist


def test_old_files_status():
    """Check status of old files."""

    print("\n🔄 Checking Old Files Status")
    print("=" * 60)

    parent_dir = Path(__file__).parent.parent
    # Verify dead code has been cleaned up
    dead_files = [
        "libs/agents/cogniverse_agents/optimizer/orchestrator.py",
        "libs/agents/cogniverse_agents/text_agent_server.py",
    ]

    for file_path in dead_files:
        full_path = parent_dir / file_path
        if full_path.exists():
            print(f"⚠️ {file_path} - Dead code still present, should be deleted")
        else:
            print(f"✅ {file_path} - Cleaned up")


def main():
    """Main test function."""

    print("🔍 Agentic Router Structure Validation")
    print("=" * 80)

    try:
        # Test imports
        imports_ok = test_imports()

        # Test directory structure
        structure_ok = test_structure_integrity()

        # Check old files
        test_old_files_status()

        # Overall result
        success = imports_ok and structure_ok

        print("\n" + "=" * 80)
        print(f"🎯 OVERALL RESULT: {'✅ SUCCESS' if success else '❌ ISSUES FOUND'}")
        print("=" * 80)

        if success:
            print("Optimizer structure validation passed.")
        else:
            print("🔧 Please fix the issues above before proceeding.")

        return success

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
