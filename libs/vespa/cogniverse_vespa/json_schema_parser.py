"""
JSON Schema Parser for Vespa
Converts JSON schema definitions to PyVespa objects
"""

import json
import logging
from typing import Any, Dict, List

from vespa.package import (
    Document,
    Field,
    FieldSet,
    FirstPhaseRanking,
    Function,
    RankProfile,
    Schema,
    SecondPhaseRanking,
)


class JsonSchemaParser:
    """Parser to convert JSON schema definitions to PyVespa objects"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_field(self, field_config: Dict[str, Any]) -> Field:
        """Parse a field configuration from JSON to PyVespa Field object"""
        
        field_params = {
            'name': field_config['name'],
            'type': field_config['type'],
            'indexing': field_config.get('indexing', [])
        }
        
        # Add optional parameters
        if 'attribute' in field_config:
            field_params['attribute'] = field_config['attribute']
        
        if 'index' in field_config:
            field_params['index'] = field_config['index']
            
        if 'summary' in field_config:
            field_params['summary'] = field_config['summary']
            
        return Field(**field_params)
    
    def parse_function(self, func_config: Dict[str, Any]) -> Function:
        """Parse a function configuration from JSON to PyVespa Function object"""
        
        return Function(
            name=func_config['name'],
            expression=func_config['expression']
        )
    
    def parse_fieldset(self, fieldset_config: Dict[str, Any]) -> FieldSet:
        """Parse a fieldset configuration from JSON to PyVespa FieldSet object"""
        
        return FieldSet(
            name=fieldset_config['name'],
            fields=fieldset_config['fields']
        )
    
    def parse_rank_profile(self, rp_config: Dict[str, Any]) -> RankProfile:
        """Parse a rank profile configuration from JSON to PyVespa RankProfile object"""
        
        rank_params = {
            'name': rp_config['name']
        }
        
        # Parse inputs
        if 'inputs' in rp_config:
            rank_params['inputs'] = [
                (inp['name'], inp['type']) for inp in rp_config['inputs']
            ]
        
        # Parse functions
        if 'functions' in rp_config:
            rank_params['functions'] = [
                self.parse_function(func_config) for func_config in rp_config['functions']
            ]
        
        # Parse first phase
        if 'first_phase' in rp_config:
            first_phase = rp_config['first_phase']
            if isinstance(first_phase, dict):
                # Complex first phase with expression
                rank_params['first_phase'] = FirstPhaseRanking(
                    expression=first_phase['expression'],
                    keep_rank_count=first_phase.get('keep_rank_count'),
                    rank_score_drop_limit=first_phase.get('rank_score_drop_limit')
                )
            else:
                # Simple first phase expression
                rank_params['first_phase'] = FirstPhaseRanking(
                    expression=first_phase
                )
        
        # Parse second phase
        if 'second_phase' in rp_config:
            second_phase = rp_config['second_phase']
            if isinstance(second_phase, dict):
                # Complex second phase with rerank count
                rank_params['second_phase'] = SecondPhaseRanking(
                    expression=second_phase['expression'],
                    rerank_count=second_phase.get('rerank_count', 100)
                )
            else:
                # Simple second phase expression
                rank_params['second_phase'] = SecondPhaseRanking(
                    expression=second_phase,
                    rerank_count=100
                )
        
        # Parse inheritance
        if 'inherits' in rp_config:
            rank_params['inherits'] = rp_config['inherits']
        
        # Parse other optional parameters
        for param in ['constants', 'summary_features', 'match_features', 'num_threads_per_search', 'timeout']:
            if param in rp_config:
                rank_params[param] = rp_config[param]
        
        return RankProfile(**rank_params)
    
    def parse_document(self, doc_config: Dict[str, Any]) -> Document:
        """Parse a document configuration from JSON to PyVespa Document object"""
        
        fields = [self.parse_field(field_config) for field_config in doc_config['fields']]
        
        return Document(
            fields=fields,
            inherits=doc_config.get('inherits'),
            structs=doc_config.get('structs', [])
        )
    
    def parse_schema(self, schema_config: Dict[str, Any]) -> Schema:
        """Parse a complete schema configuration from JSON to PyVespa Schema object"""
        
        # Parse document
        document = self.parse_document(schema_config['document'])
        
        # Parse rank profiles
        rank_profiles = []
        if 'rank_profiles' in schema_config:
            rank_profiles = [
                self.parse_rank_profile(rp_config) 
                for rp_config in schema_config['rank_profiles']
            ]
        
        # Parse fieldsets
        fieldsets = []
        if 'fieldsets' in schema_config:
            fieldsets = [
                self.parse_fieldset(fieldset_config)
                for fieldset_config in schema_config['fieldsets']
            ]
        
        # Create schema first without rank profiles
        schema_params = {
            'name': schema_config['name'],
            'document': document
        }
        
        # Add fieldsets if present
        if fieldsets:
            schema_params['fieldsets'] = fieldsets
        
        # Add other optional schema-level parameters
        for param in ['imports', 'annotations']:
            if param in schema_config:
                schema_params[param] = schema_config[param]
        
        schema = Schema(**schema_params)
        
        # Add rank profiles using add_rank_profile method
        for rank_profile in rank_profiles:
            schema.add_rank_profile(rank_profile)
        
        return schema
    
    def load_schema_from_json_file(self, json_file_path: str) -> Schema:
        """Load and parse schema from JSON file"""
        
        try:
            with open(json_file_path, 'r') as f:
                schema_config = json.load(f)
            
            schema = self.parse_schema(schema_config)
            self.logger.info(f"Successfully loaded schema '{schema.name}' from {json_file_path}")
            return schema
            
        except Exception as e:
            error_msg = f"Failed to load schema from {json_file_path}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_schema_from_json_string(self, json_string: str) -> Schema:
        """Load and parse schema from JSON string"""
        
        try:
            schema_config = json.loads(json_string)
            return self.parse_schema(schema_config)
            
        except Exception as e:
            error_msg = f"Failed to parse JSON schema: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def validate_schema_config(self, schema_config: Dict[str, Any]) -> List[str]:
        """Validate schema configuration and return list of errors"""
        
        errors = []
        
        # Check required fields
        if 'name' not in schema_config:
            errors.append("Schema must have a 'name' field")
        
        if 'document' not in schema_config:
            errors.append("Schema must have a 'document' field")
        else:
            doc_config = schema_config['document']
            if 'fields' not in doc_config:
                errors.append("Document must have a 'fields' array")
            elif not isinstance(doc_config['fields'], list):
                errors.append("Document 'fields' must be an array")
        
        # Validate rank profiles
        if 'rank_profiles' in schema_config:
            if not isinstance(schema_config['rank_profiles'], list):
                errors.append("'rank_profiles' must be an array")
            else:
                for i, rp in enumerate(schema_config['rank_profiles']):
                    if 'name' not in rp:
                        errors.append(f"Rank profile {i} must have a 'name' field")
        
        return errors
