#!/usr/bin/env python3
"""
Setup and display output directory structure
"""

from src.common.utils.output_manager import get_output_manager
import json


def main():
    """Setup output directories and display structure"""
    
    print("Setting up output directory structure...")
    
    # Get output manager (this creates all directories)
    output_manager = get_output_manager()
    
    # Print the structure
    output_manager.print_structure()
    
    # Save structure to file for reference
    structure = output_manager.get_structure()
    structure_file = output_manager.get_path("metadata", "directory_structure.json")
    
    with open(structure_file, 'w') as f:
        json.dump(structure, f, indent=2)
    
    print(f"\nDirectory structure saved to: {structure_file}")
    
    # Create example files to show usage
    examples = [
        ("logs", "example.log", "Example log file"),
        ("test_results", "test_run_20250123.csv", "Test results CSV"),
        ("optimization", "best_model.json", "Optimization results"),
        ("processing/embeddings", "video1_embeddings.json", "Video embeddings")
    ]
    
    print("\nCreating example files:")
    for component, filename, content in examples:
        if "/" in component:
            # Handle subdirectory paths
            parts = component.split("/")
            file_path = output_manager.get_processing_dir(parts[1]) / filename
        else:
            file_path = output_manager.get_path(component, filename)
        
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  Created: {file_path}")
    
    print("\n✅ Output directory structure setup complete!")
    print("\nTo use in your code:")
    print("  from src.common.utils.output_manager import get_output_manager")
    print("  om = get_output_manager()")
    print("  log_file = om.get_logs_dir() / 'my_log.log'")


if __name__ == "__main__":
    main()