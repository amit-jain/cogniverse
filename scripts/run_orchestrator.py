#!/usr/bin/env python3
"""
Run Orchestrator Script

Simple script to run the optimization orchestrator with the new src/ structure.
"""

import subprocess
import sys
from pathlib import Path


def run_orchestrator(config_path: str = "config.json", **kwargs) -> bool:
    """
    Run the orchestrator using the new src/ structure.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments for the orchestrator
        
    Returns:
        True if successful
    """
    print("🚀 Running Agentic Router Orchestrator")
    print("=" * 60)
    
    # Build command
    cmd = [sys.executable, "-m", "src.optimizer.orchestrator", "--config", config_path]
    
    # Add optional arguments
    if kwargs.get("setup_only"):
        cmd.append("--setup-only")
    if kwargs.get("test_models"):
        cmd.append("--test-models")
    
    try:
        parent_dir = Path(__file__).parent.parent
        
        print(f"📁 Working directory: {parent_dir}")
        print(f"🔧 Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=parent_dir,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        if result.returncode == 0:
            print("\n✅ Orchestrator completed successfully")
            return True
        else:
            print(f"\n❌ Orchestrator failed with exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running orchestrator: {e}")
        return False

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Agentic Router Orchestrator")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--setup-only", action="store_true", help="Only setup services, don't run optimization")
    parser.add_argument("--test-models", action="store_true", help="Test model connections")
    
    args = parser.parse_args()
    
    # Run orchestrator
    success = run_orchestrator(
        config_path=args.config,
        setup_only=args.setup_only,
        test_models=args.test_models
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
