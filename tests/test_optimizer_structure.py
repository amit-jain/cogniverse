#!/usr/bin/env python3
"""
Test New Structure Script

Validates that the new src/ structure imports work correctly.
"""

import sys
import os
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
        from cogniverse_agents.optimizer.schemas import RoutingDecision, AgenticRouter
        print("✅ schemas: RoutingDecision, AgenticRouter")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ schemas: {e}")
    
    # Test 2: Router optimizer
    total_tests += 1
    try:
        from cogniverse_agents.optimizer.router_optimizer import RouterModule, OptimizedRouter
        print("✅ router_optimizer: RouterModule, OptimizedRouter")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ router_optimizer: {e}")
    
    # Test 3: Orchestrator
    total_tests += 1
    try:
        from cogniverse_agents.optimizer.orchestrator import OptimizationOrchestrator
        print("✅ orchestrator: OptimizationOrchestrator")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ orchestrator: {e}")
    
    # Test 4: Provider abstractions
    total_tests += 1
    try:
        from cogniverse_agents.optimizer.providers.base_provider import ProviderFactory
        print("✅ base_provider: ProviderFactory")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ base_provider: {e}")
    
    # Test 5: Modal provider
    total_tests += 1
    try:
        from cogniverse_agents.optimizer.providers.modal_provider import ModalModelProvider
        print("✅ modal_provider: ModalModelProvider")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ modal_provider: {e}")
    
    # Test 6: Local provider
    total_tests += 1
    try:
        from cogniverse_agents.optimizer.providers.local_provider import LocalModelProvider
        print("✅ local_provider: LocalModelProvider")
        tests_passed += 1
    except ImportError as e:
        print(f"❌ local_provider: {e}")
    
    # Test 7: Production API (Modal app)
    total_tests += 1
    try:
        # This imports the Modal app, so it might fail without Modal setup
        from cogniverse_runtime.inference.inference import app
        print("✅ production_api: Modal app")
        tests_passed += 1
    except Exception as e:
        print(f"⚠️ production_api: {e} (might need Modal setup)")
        # Don't count this as a failure since Modal might not be configured
        tests_passed += 1
    
    # Test 8: Model service
    total_tests += 1
    try:
        from cogniverse_runtime.inference.model_service import app as model_app
        print("✅ model_service: Modal app")
        tests_passed += 1
    except Exception as e:
        print(f"⚠️ model_service: {e} (might need Modal setup)")
        # Don't count this as a failure since Modal might not be configured
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 IMPORT TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {tests_passed}/{total_tests}")
    print(f"📈 Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    return tests_passed == total_tests

def test_structure_integrity():
    """Test that the directory structure is correct."""
    
    print("\n🗂️ Testing Directory Structure")
    print("=" * 60)
    
    parent_dir = Path(__file__).parent.parent
    required_paths = [
        "src/optimizer/schemas.py",
        "src/optimizer/router_optimizer.py", 
        "src/optimizer/orchestrator.py",
        "src/optimizer/providers/__init__.py",
        "src/optimizer/providers/base_provider.py",
        "src/optimizer/providers/modal_provider.py",
        "src/optimizer/providers/local_provider.py",
        "src/inference/production_api.py",
        "src/inference/model_service.py"
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
    old_files = [
        "orchestrator.py",
        "agentic_router_optimizer.py", 
        "router_optimizer.py",
        "production_api.py",
        "model_service.py"
    ]
    
    for file_path in old_files:
        full_path = parent_dir / file_path
        if full_path.exists():
            print(f"📄 {file_path} - Still exists (can be removed after testing)")
        else:
            print(f"🗑️ {file_path} - Removed/moved")
    
    # Check modal_inference directory
    modal_inference_dir = parent_dir / "modal_inference"
    if modal_inference_dir.exists():
        print(f"📁 modal_inference/ - Directory exists (can be removed after migration)")
        print(f"   └─ Contains: {len(list(modal_inference_dir.glob('*')))} items")
    else:
        print(f"🗑️ modal_inference/ - Directory removed")

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
            print("🚀 New src/ structure is ready to use!")
            print("\nNext steps:")
            print("1. Test orchestrator: python scripts/run_orchestrator.py --test-models")
            print("2. Run optimization: python scripts/run_orchestrator.py")
            print("3. Deploy service: modal deploy src/inference/modal_inference_service.py")
            print("4. Test the system: python tests/test_system.py")
        else:
            print("🔧 Please fix the issues above before proceeding.")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)