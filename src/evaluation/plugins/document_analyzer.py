"""
Document/text search analyzer plugin.

Example plugin showing how non-video domains work with the evaluation system.
"""

import re
from typing import Dict, Any, Optional
from src.evaluation.core.schema_analyzer import SchemaAnalyzer


class DocumentSchemaAnalyzer(SchemaAnalyzer):
    """Analyzer for document/text search schemas."""
    
    def can_handle(self, schema_name: str, schema_fields: Dict[str, Any]) -> bool:
        """Check if this is a document schema."""
        # Check schema name
        if any(term in schema_name.lower() for term in ['document', 'text', 'article', 'page']):
            return True
        
        # Check for document-specific fields
        doc_indicators = [
            'document_id', 'doc_id', 'page_number', 
            'title', 'author', 'content', 'abstract', 'body'
        ]
        
        all_fields = []
        for field_list in schema_fields.values():
            if isinstance(field_list, list):
                all_fields.extend(field_list)
        
        return any(indicator in all_fields for indicator in doc_indicators)
    
    def analyze_query(self, query: str, schema_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document-specific queries."""
        query_lower = query.lower()
        
        constraints = {
            "query_type": "document",
            "field_constraints": {},
            "author_constraints": {},
            "date_constraints": {},
            "available_fields": schema_fields
        }
        
        # Extract author constraints
        author_pattern = r'author:\s*"?([^"\s]+)"?'
        match = re.search(author_pattern, query)
        if match:
            constraints["author_constraints"]["author"] = match.group(1)
            constraints["query_type"] = "document_author"
        
        # Extract document type constraints
        doc_types = ['pdf', 'docx', 'txt', 'html', 'markdown']
        for doc_type in doc_types:
            if f'type:{doc_type}' in query_lower or f'filetype:{doc_type}' in query_lower:
                constraints["field_constraints"]["document_type"] = doc_type
        
        # Extract title search
        title_pattern = r'title:\s*"([^"]+)"'
        match = re.search(title_pattern, query)
        if match:
            constraints["field_constraints"]["title"] = match.group(1)
            constraints["query_type"] = "document_title"
        
        # Extract date constraints (for documents)
        date_patterns = [
            (r'after:(\d{4}-\d{2}-\d{2})', 'after_date'),
            (r'before:(\d{4}-\d{2}-\d{2})', 'before_date'),
            (r'published:(\d{4})', 'publication_year')
        ]
        
        for pattern, constraint_type in date_patterns:
            match = re.search(pattern, query_lower)
            if match:
                constraints["date_constraints"][constraint_type] = match.group(1)
                if constraints["query_type"] == "document":
                    constraints["query_type"] = "document_temporal"
        
        return constraints
    
    def extract_item_id(self, document: Any) -> Optional[str]:
        """Extract document ID from dict or object."""
        # Handle dict format (most common in our pipeline)
        if isinstance(document, dict):
            # Try document-specific fields
            for field in ['document_id', 'doc_id', 'article_id', 'page_id', 'id']:
                if field in document:
                    return str(document[field])
            # Check metadata if present
            if 'metadata' in document:
                for field in ['document_id', 'doc_id', 'article_id', 'page_id']:
                    if field in document['metadata']:
                        return str(document['metadata'][field])
        
        # Handle object format (for compatibility)
        elif hasattr(document, 'metadata'):
            for field in ['document_id', 'doc_id', 'article_id', 'page_id']:
                if field in document.metadata:
                    return document.metadata[field]
        
        # Try direct attributes
        if hasattr(document, 'doc_id'):
            return document.doc_id
        if hasattr(document, 'document_id'):
            return document.document_id
        
        # Fallback to generic ID
        if hasattr(document, 'id'):
            return document.id
        
        return None
    
    def get_expected_field_name(self) -> str:
        """Document-specific expected field name."""
        return "expected_documents"


class ImageSchemaAnalyzer(SchemaAnalyzer):
    """Analyzer for image search schemas."""
    
    def can_handle(self, schema_name: str, schema_fields: Dict[str, Any]) -> bool:
        """Check if this is an image schema."""
        # Check schema name
        if any(term in schema_name.lower() for term in ['image', 'photo', 'picture', 'visual']):
            return True
        
        # Check for image-specific fields
        image_indicators = [
            'image_id', 'image_path', 'width', 'height',
            'format', 'caption', 'alt_text', 'visual_features'
        ]
        
        all_fields = []
        for field_list in schema_fields.values():
            if isinstance(field_list, list):
                all_fields.extend(field_list)
        
        return any(indicator in all_fields for indicator in image_indicators)
    
    def analyze_query(self, query: str, schema_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image-specific queries."""
        query_lower = query.lower()
        
        constraints = {
            "query_type": "image",
            "visual_constraints": {},
            "format_constraints": {},
            "size_constraints": {},
            "available_fields": schema_fields
        }
        
        # Extract visual features
        colors = ['red', 'blue', 'green', 'yellow', 'black', 'white', 'colorful', 'monochrome']
        found_colors = [c for c in colors if c in query_lower]
        if found_colors:
            constraints["visual_constraints"]["colors"] = found_colors
            constraints["query_type"] = "image_visual"
        
        # Extract image format constraints
        formats = ['jpg', 'png', 'gif', 'webp', 'svg']
        for fmt in formats:
            if fmt in query_lower:
                constraints["format_constraints"]["format"] = fmt
        
        # Extract size constraints
        size_patterns = [
            (r'(\d+)x(\d+)', 'exact_size'),
            (r'larger than (\d+)', 'min_size'),
            (r'smaller than (\d+)', 'max_size'),
            (r'(thumbnail|small|medium|large|xlarge)', 'size_category')
        ]
        
        for pattern, constraint_type in size_patterns:
            match = re.search(pattern, query_lower)
            if match:
                constraints["size_constraints"][constraint_type] = match.groups()
        
        # Extract style/content
        styles = ['portrait', 'landscape', 'abstract', 'realistic', 'cartoon', 'sketch']
        found_styles = [s for s in styles if s in query_lower]
        if found_styles:
            constraints["visual_constraints"]["styles"] = found_styles
            constraints["query_type"] = "image_style"
        
        return constraints
    
    def extract_item_id(self, document: Any) -> Optional[str]:
        """Extract image ID from dict or object."""
        # Handle dict format (most common in our pipeline)
        if isinstance(document, dict):
            # Try image-specific fields
            for field in ['image_id', 'photo_id', 'picture_id', 'visual_id', 'id']:
                if field in document:
                    return str(document[field])
            # Check metadata if present
            if 'metadata' in document:
                for field in ['image_id', 'photo_id', 'picture_id', 'visual_id']:
                    if field in document['metadata']:
                        return str(document['metadata'][field])
                # Try to extract from path
                if 'image_path' in document['metadata']:
                    path = document['metadata']['image_path']
                    import os
                    return os.path.splitext(os.path.basename(path))[0]
        
        # Handle object format (for compatibility)
        elif hasattr(document, 'metadata'):
            for field in ['image_id', 'photo_id', 'picture_id', 'visual_id']:
                if field in document.metadata:
                    return document.metadata[field]
            
            # Try to extract from path
            if 'image_path' in document.metadata:
                path = document.metadata['image_path']
                # Extract filename without extension as ID
                import os
                return os.path.splitext(os.path.basename(path))[0]
        
        # Fallback to generic
        if hasattr(document, 'id'):
            return document.id
        
        return None
    
    def get_expected_field_name(self) -> str:
        """Image-specific expected field name."""
        return "expected_images"