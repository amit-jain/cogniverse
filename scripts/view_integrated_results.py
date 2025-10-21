#!/usr/bin/env python3
"""
View integrated results from both quantitative tests and evaluation experiments
"""

import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from scripts.generate_integrated_evaluation_report import generate_integrated_report


def main():
    parser = argparse.ArgumentParser(description="View integrated evaluation results")
    parser.add_argument("--open", action="store_true", help="Open report in browser")
    parser.add_argument("--test-results", help="Path to specific test results JSON")
    parser.add_argument("--experiments-dir", help="Path to experiments directory")
    args = parser.parse_args()
    
    print("🔍 Looking for latest results...")
    
    # Generate the integrated report
    output_file = generate_integrated_report(
        test_results_file=args.test_results,
        experiment_results_dir=args.experiments_dir
    )
    
    if args.open and output_file:
        print("\n🌐 Opening report in browser...")
        webbrowser.open(f"file://{output_file.absolute()}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
