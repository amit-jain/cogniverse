#!/usr/bin/env python3
"""
Generate Vespa schema from template based on configuration
"""

import sys
import json
from pathlib import Path
from string import Template

sys.path.append(str(Path(__file__).parent.parent))

from src.common.config_utils import get_config


def generate_schema(profile_name: str, profile_config: dict) -> str:
    """Generate schema content from template"""
    
    # Check if schema_config exists
    if "schema_config" not in profile_config:
        print(f"Warning: No schema_config for profile '{profile_name}', skipping")
        return None
    
    schema_config = profile_config["schema_config"]
    
    # Read template
    template_path = Path(__file__).parent.parent / "schemas" / "video_multimodal_template.sd"
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    
    with open(template_path, 'r') as f:
        template_content = f.read()
    
    # Create template and substitute values
    template = Template(template_content)
    schema_content = template.substitute(
        SCHEMA_NAME=schema_config["schema_name"],
        MODEL_NAME=schema_config["model_name"],
        NUM_PATCHES=schema_config["num_patches"],
        EMBEDDING_DIM=schema_config["embedding_dim"],
        BINARY_DIM=schema_config["binary_dim"]
    )
    
    return schema_content


def main():
    """Generate schemas for all profiles with schema_config"""
    
    config = get_config()
    profiles = config.get("video_processing_profiles", {})
    
    generated_count = 0
    
    for profile_name, profile_config in profiles.items():
        if "schema_config" not in profile_config:
            continue
        
        schema_name = profile_config["schema_config"]["schema_name"]
        print(f"\nGenerating schema for profile '{profile_name}':")
        print(f"  Schema name: {schema_name}")
        print(f"  Model: {profile_config['schema_config']['model_name']}")
        print(f"  Patches: {profile_config['schema_config']['num_patches']}")
        print(f"  Dimensions: {profile_config['schema_config']['embedding_dim']}")
        
        # Generate schema
        schema_content = generate_schema(profile_name, profile_config)
        if schema_content:
            # Write to file
            output_path = Path(__file__).parent.parent / "schemas" / f"{schema_name}.sd"
            with open(output_path, 'w') as f:
                f.write(schema_content)
            print(f"  ✅ Generated: {output_path}")
            generated_count += 1
    
    print(f"\n✅ Generated {generated_count} schemas")
    
    # Show deployment commands
    if generated_count > 0:
        print("\nTo deploy these schemas:")
        for profile_name, profile_config in profiles.items():
            if "schema_config" in profile_config:
                schema_name = profile_config["schema_config"]["schema_name"]
                print(f"  # For {profile_config['schema_config']['model_name']}:")
                print(f"  vespa deploy schemas/{schema_name}.sd")
                print()


if __name__ == "__main__":
    main()