#!/usr/bin/env python3
"""
Update video_frame schema to use consistent field names:
- colpali_embedding -> embedding
- colpali_binary -> embedding_binary
"""

import json
import sys
from pathlib import Path

def update_schema_field_names(schema_path: Path):
    """Update field names in schema for consistency"""
    
    # Read the schema
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Update field names in document fields
    for field in schema.get("document", {}).get("fields", []):
        if field.get("name") == "colpali_embedding":
            field["name"] = "embedding"
            print(f"Updated field: colpali_embedding -> embedding")
        elif field.get("name") == "colpali_binary":
            field["name"] = "embedding_binary"
            print(f"Updated field: colpali_binary -> embedding_binary")
    
    # Update references in rank profiles
    for profile in schema.get("rank_profiles", []):
        # Update functions
        for func in profile.get("functions", []):
            if "colpali_embedding" in func.get("expression", ""):
                func["expression"] = func["expression"].replace("colpali_embedding", "embedding")
                print(f"Updated function expression in profile '{profile['name']}'")
            if "colpali_binary" in func.get("expression", ""):
                func["expression"] = func["expression"].replace("colpali_binary", "embedding_binary")
                print(f"Updated function expression in profile '{profile['name']}'")
        
        # Update first_phase
        if "colpali_embedding" in profile.get("first_phase", ""):
            profile["first_phase"] = profile["first_phase"].replace("colpali_embedding", "embedding")
            print(f"Updated first_phase in profile '{profile['name']}'")
        if "colpali_binary" in profile.get("first_phase", ""):
            profile["first_phase"] = profile["first_phase"].replace("colpali_binary", "embedding_binary")
            print(f"Updated first_phase in profile '{profile['name']}'")
            
        # Update second_phase
        if profile.get("second_phase"):
            if "colpali_embedding" in profile["second_phase"].get("expression", ""):
                profile["second_phase"]["expression"] = profile["second_phase"]["expression"].replace("colpali_embedding", "embedding")
                print(f"Updated second_phase in profile '{profile['name']}'")
            if "colpali_binary" in profile["second_phase"].get("expression", ""):
                profile["second_phase"]["expression"] = profile["second_phase"]["expression"].replace("colpali_binary", "embedding_binary")
                print(f"Updated second_phase in profile '{profile['name']}'")
    
    # Save updated schema
    output_path = schema_path.with_suffix('.json.updated')
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    print(f"\nUpdated schema saved to: {output_path}")
    print(f"To apply changes, run: mv {output_path} {schema_path}")
    
    return output_path

def main():
    schema_path = Path("configs/schemas/video_frame_schema.json")
    
    if not schema_path.exists():
        print(f"Schema file not found: {schema_path}")
        sys.exit(1)
    
    print(f"Updating schema: {schema_path}")
    update_schema_field_names(schema_path)

if __name__ == "__main__":
    main()