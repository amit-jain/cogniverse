#!/usr/bin/env python3
"""
Calculate ingestion-only coverage from the coverage report.
"""

import subprocess
import sys
import re

def calculate_ingestion_coverage():
    """Calculate coverage for ingestion module only."""
    
    # Run tests with coverage
    cmd = [
        "uv", "run", "python", "-m", "pytest", 
        "tests/ingestion/unit", 
        "-v", "-m", "unit", 
        "--cov=src/app/ingestion", 
        "--cov-report=term-missing", 
        "--tb=no"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env={"JAX_PLATFORM_NAME": "cpu"})
        output = result.stdout
        
        # Parse the coverage output
        lines = output.split('\n')
        
        # Find coverage section
        coverage_started = False
        ingestion_lines = []
        total_stmts = 0
        total_miss = 0
        
        for line in lines:
            if "coverage:" in line and "platform" in line:
                coverage_started = True
                continue
            
            if coverage_started and line.startswith("src/app/ingestion/"):
                # Parse the coverage line
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        stmts = int(parts[1])
                        miss = int(parts[2])
                        total_stmts += stmts
                        total_miss += miss
                        ingestion_lines.append(line)
                    except ValueError:
                        continue
            
            # Stop when we hit non-ingestion modules
            if coverage_started and (line.startswith("src/app/routing/") or 
                                   line.startswith("src/evaluation/") or
                                   line.startswith("TOTAL")):
                break
        
        # Calculate coverage
        if total_stmts > 0:
            covered = total_stmts - total_miss
            coverage_percent = (covered / total_stmts) * 100
            
            print("="*60)
            print("INGESTION MODULE COVERAGE REPORT")
            print("="*60)
            print("\nIngestion module files covered:")
            for line in ingestion_lines:
                print(f"  {line}")
            
            print(f"\nIngestion Coverage Summary:")
            print(f"  Total statements: {total_stmts}")
            print(f"  Covered statements: {covered}")
            print(f"  Missed statements: {total_miss}")
            print(f"  Coverage: {coverage_percent:.1f}%")
            
            if coverage_percent >= 80:
                print(f"\n✅ Coverage target of 80% ACHIEVED!")
                return True
            else:
                print(f"\n❌ Coverage target of 80% NOT reached. Current: {coverage_percent:.1f}%")
                return False
        else:
            print("❌ No ingestion coverage data found")
            return False
            
    except Exception as e:
        print(f"Error calculating coverage: {e}")
        return False

if __name__ == "__main__":
    success = calculate_ingestion_coverage()
    sys.exit(0 if success else 1)