import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vespa.package import (
    ApplicationPackage,
    Document,
    Field,
    FirstPhaseRanking,
    Function,
    RankProfile,
    Schema,
    SecondPhaseRanking,
)

from cogniverse_core.config.utils import get_config


class VespaSchemaManager:
    """
    A class to read Vespa schema files (.sd) and upload them to Vespa.
    
    Supports both native .sd files and JSON schema definitions.
    """
    
    def __init__(self, vespa_endpoint: str = None, vespa_port: int = None):
        config = get_config()
        self.vespa_endpoint = vespa_endpoint or f"{config['vespa_url']}:{config['vespa_port']}"
        self.vespa_port = vespa_port or 19071  # Default deployment port
        self._logger = logging.getLogger(self.__class__.__name__)
    
    def read_sd_file(self, sd_file_path: str) -> str:
        """
        Read a Vespa schema definition (.sd) file.
        
        Args:
            sd_file_path: Path to the .sd file
            
        Returns:
            String content of the schema file
        """
        if not os.path.exists(sd_file_path):
            raise FileNotFoundError(f"Schema file not found: {sd_file_path}")
        
        with open(sd_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self._logger.info(f"Read schema file: {sd_file_path}")
        return content
    
    def parse_sd_schema(self, sd_content: str) -> Schema:
        """
        Parse a Vespa schema definition (.sd) content and convert to pyvespa Schema.
        
        Args:
            sd_content: String content of the .sd file
            
        Returns:
            pyvespa Schema object
        """
        # Extract schema name
        schema_match = re.search(r'schema\s+(\w+)\s*{', sd_content)
        if not schema_match:
            raise ValueError("Could not find schema name in .sd file")
        
        schema_name = schema_match.group(1)
        self._logger.info(f"Parsing schema: {schema_name}")
        
        # Parse document fields
        document_fields = self._parse_document_fields(sd_content)

        # Create document with fields
        document = Document(fields=document_fields)
        
        # Create schema
        schema = Schema(name=schema_name, document=document)
        
        # Parse and add rank profiles
        rank_profiles = self._parse_rank_profiles(sd_content)
        for rank_profile in rank_profiles:
            schema.add_rank_profile(rank_profile)
        
        self._logger.info(f"Successfully parsed schema: {schema_name}")
        return schema
    
    def _find_balanced_brackets(self, text: str, start_pattern: str) -> Optional[str]:
        """Find content between balanced brackets starting with a pattern."""
        match = re.search(start_pattern, text)
        if not match:
            return None
        
        start_pos = match.start()
        brace_pos = text.find('{', start_pos)
        if brace_pos == -1:
            return None
        
        # Find the matching closing brace
        brace_count = 1
        pos = brace_pos + 1
        
        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count == 0:
            return text[brace_pos + 1:pos - 1]
        
        return None
    
    def _parse_document_fields(self, sd_content: str) -> List[Field]:
        """Parse document fields from .sd content."""
        fields = []
        
        # Find document block using balanced bracket matching
        document_content = self._find_balanced_brackets(sd_content, r'document\s+\w+\s*\{')
        if not document_content:
            self._logger.warning("Could not find document definition in .sd file")
            return fields
        
        # Parse individual fields
        field_pattern = r'field\s+(\w+)\s+type\s+([^{]+?)\s*\{([^}]*)\}'
        field_matches = re.findall(field_pattern, document_content, re.DOTALL)
        
        for field_name, field_type, field_config in field_matches:
            field_type = field_type.strip()
            
            # Parse field configuration
            indexing = self._parse_field_indexing(field_config)
            attributes = self._parse_field_attributes(field_config)
            
            field = Field(
                name=field_name,
                type=field_type,
                indexing=indexing,
                attribute=attributes
            )
            
            fields.append(field)
            self._logger.debug(f"Parsed field: {field_name} ({field_type})")
        
        return fields
    
    def _parse_field_indexing(self, field_config: str) -> List[str]:
        """Parse indexing configuration from field config."""
        indexing_match = re.search(r'indexing:\s*([^\n]+)', field_config)
        if not indexing_match:
            return []
        
        indexing_str = indexing_match.group(1).strip()
        indexing_parts = [part.strip() for part in indexing_str.split('|')]
        return indexing_parts
    
    def _parse_field_attributes(self, field_config: str) -> List[str]:
        """Parse attribute configuration from field config."""
        attributes = []
        
        # Look for attribute specifications
        attribute_matches = re.findall(r'attribute:\s*([^}]+)', field_config)
        for attr_match in attribute_matches:
            attr_parts = [part.strip() for part in attr_match.split()]
            attributes.extend(attr_parts)
        
        return attributes
    
    def _parse_structs(self, sd_content: str) -> Dict[str, List[Field]]:
        """Parse struct definitions from .sd content."""
        structs = {}
        
        # Find struct blocks
        struct_pattern = r'struct\s+(\w+)\s*{(.*?)^\s*}'
        struct_matches = re.findall(struct_pattern, sd_content, re.DOTALL | re.MULTILINE)
        
        for struct_name, struct_content in struct_matches:
            struct_fields = []
            
            # Parse struct fields
            field_pattern = r'field\s+(\w+)\s+type\s+([^{]+)\s*{([^}]*)}'
            field_matches = re.findall(field_pattern, struct_content)
            
            for field_name, field_type, field_config in field_matches:
                field_type = field_type.strip()
                
                indexing = self._parse_field_indexing(field_config)
                attributes = self._parse_field_attributes(field_config)
                
                field = Field(
                    name=field_name,
                    type=field_type,
                    indexing=indexing,
                    attribute=attributes
                )
                
                struct_fields.append(field)
            
            structs[struct_name] = struct_fields
            self._logger.debug(f"Parsed struct: {struct_name} with {len(struct_fields)} fields")
        
        return structs
    
    def _parse_rank_profiles(self, sd_content: str) -> List[RankProfile]:
        """Parse rank profiles from .sd content."""
        rank_profiles = []
        
        # Find rank-profile blocks by name first
        rank_profile_name_pattern = r'rank-profile\s+(\w+)\s*{'
        rank_profile_names = re.findall(rank_profile_name_pattern, sd_content)
        
        for profile_name in rank_profile_names:
            # Use balanced brackets to find the full content
            pattern = rf'rank-profile\s+{profile_name}\s*{{'
            profile_content = self._find_balanced_brackets(sd_content, pattern)
            
            if profile_content:
                # Parse inputs
                inputs = self._parse_rank_inputs(profile_content)
                
                # Parse first-phase
                first_phase = self._parse_first_phase(profile_content)
                
                # Parse second-phase
                second_phase = self._parse_second_phase(profile_content)
                
                # Parse functions
                functions = self._parse_functions(profile_content)
                
                rank_profile = RankProfile(
                    name=profile_name,
                    inputs=inputs,
                    first_phase=first_phase,
                    second_phase=second_phase,
                    functions=functions
                )
                
                rank_profiles.append(rank_profile)
                self._logger.debug(f"Parsed rank profile: {profile_name}")
        
        return rank_profiles
    
    def _parse_rank_inputs(self, profile_content: str) -> List[Tuple[str, str]]:
        """Parse inputs from rank profile content."""
        inputs = []
        
        # Find inputs block using balanced brackets
        inputs_content = self._find_balanced_brackets(profile_content, r'inputs\s*{')
        if inputs_content:
            # Parse individual inputs
            input_pattern = r'query\(([^)]+)\)\s+([^\n]+)'
            input_matches = re.findall(input_pattern, inputs_content)
            
            for input_name, input_type in input_matches:
                inputs.append((f"query({input_name})", input_type.strip()))
        
        return inputs
    
    def _parse_first_phase(self, profile_content: str) -> Optional[FirstPhaseRanking]:
        """Parse first-phase ranking from profile content."""
        first_phase_content = self._find_balanced_brackets(profile_content, r'first-phase\s*{')
        if first_phase_content:
            expr_match = re.search(r'expression:\s*(.+)', first_phase_content, re.DOTALL)
            if expr_match:
                expression = expr_match.group(1).strip()
                # Clean up multi-line expressions
                expression = re.sub(r'\s+', ' ', expression)
                return FirstPhaseRanking(expression=expression)
        return None
    
    def _parse_second_phase(self, profile_content: str) -> Optional[SecondPhaseRanking]:
        """Parse second-phase ranking from profile content."""
        second_phase_content = self._find_balanced_brackets(profile_content, r'second-phase\s*{')
        if second_phase_content:
            # Parse expression
            expr_match = re.search(r'expression:\s*([^\n]+)', second_phase_content)
            expression = expr_match.group(1).strip() if expr_match else "firstPhase"
            
            # Parse rerank-count
            rerank_match = re.search(r'rerank-count:\s*(\d+)', second_phase_content)
            rerank_count = int(rerank_match.group(1)) if rerank_match else 100
            
            return SecondPhaseRanking(expression=expression, rerank_count=rerank_count)
        return None
    
    def _parse_functions(self, profile_content: str) -> List[Function]:
        """Parse functions from rank profile content."""
        functions = []
        
        # Find function blocks
        function_pattern = r'function\s+(\w+)\s*{[^}]*expression:\s*([^}]+)}'
        function_matches = re.findall(function_pattern, profile_content, re.DOTALL)
        
        for func_name, func_expression in function_matches:
            expression = re.sub(r'\s+', ' ', func_expression.strip())
            functions.append(Function(name=func_name, expression=expression))
        
        return functions
    
    def create_application_package(self, schema: Schema, app_name: str = "video_search") -> ApplicationPackage:
        """
        Create an ApplicationPackage from a schema.
        
        Args:
            schema: The pyvespa Schema object
            app_name: Name of the application
            
        Returns:
            ApplicationPackage ready for deployment
        """
        app_package = ApplicationPackage(name=app_name, schema=[schema])
        self._logger.info(f"Created application package: {app_name}")
        return app_package
    
    def upload_image_content_schema(self, app_name: str = "imagesearch") -> None:
        """
        Create and upload image_content schema with ColPali multi-vector embeddings.

        Uses ColPali for image similarity search (same approach as video frames).

        Args:
            app_name: Name of the application
        """
        try:
            from vespa.package import (
                ApplicationPackage,
                Document,
                Field,
                RankProfile,
                Schema,
                SecondPhaseRanking,
            )

            image_content_schema = Schema(
                name='image_content',
                document=Document(
                    fields=[
                        Field(name='image_id', type='string', indexing=['summary', 'attribute'], attribute=['fast-search']),
                        Field(name='image_title', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                        Field(name='source_url', type='string', indexing=['summary', 'attribute']),
                        Field(name='creation_timestamp', type='long', indexing=['summary', 'attribute']),
                        Field(name='image_description', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                        Field(name='detected_objects', type='array<string>', indexing=['summary', 'attribute']),
                        Field(name='detected_scenes', type='array<string>', indexing=['summary', 'attribute']),
                        # ColPali multi-vector embedding (same as video frames)
                        Field(
                            name='colpali_embedding',
                            type='tensor<float>(x[1024],d[128])',
                            indexing=['attribute'],
                            attribute=['distance-metric:prenormalized-angular']
                        ),
                    ]
                ),
                rank_profiles=[
                    RankProfile(
                        name='colpali_similarity',
                        inputs=[('query(q)', 'tensor<float>(x[1024],d[128])')],
                        first_phase='sum(reduce(sum(query(q) * attribute(colpali_embedding), d), max, x))'
                    ),
                    RankProfile(
                        name='hybrid_image',
                        inputs=[('query(q)', 'tensor<float>(x[1024],d[128])')],
                        first_phase='bm25(image_description)',
                        second_phase=SecondPhaseRanking(
                            expression='sum(reduce(sum(query(q) * attribute(colpali_embedding), d), max, x))',
                            rerank_count=100
                        )
                    ),
                ]
            )

            app_package = ApplicationPackage(name=app_name, schema=[image_content_schema])
            self._deploy_package(app_package)

            self._logger.info("Successfully uploaded image_content schema with ColPali")

        except Exception as e:
            self._logger.error(f"Failed to upload image_content schema: {str(e)}")
            raise

    def upload_audio_content_schema(self, app_name: str = "audiosearch") -> None:
        """
        Create and upload audio_content schema for Phase 8.

        Supports acoustic + transcript-based search.

        Args:
            app_name: Name of the application
        """
        try:
            from vespa.package import (
                ApplicationPackage,
                Document,
                Field,
                RankProfile,
                Schema,
                SecondPhaseRanking,
            )

            audio_content_schema = Schema(
                name='audio_content',
                document=Document(
                    fields=[
                        Field(name='audio_id', type='string', indexing=['summary', 'attribute'], attribute=['fast-search']),
                        Field(name='audio_title', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                        Field(name='source_url', type='string', indexing=['summary', 'attribute']),
                        Field(name='duration', type='double', indexing=['summary', 'attribute']),
                        Field(name='transcript', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                        Field(name='speaker_labels', type='array<string>', indexing=['summary', 'attribute']),
                        Field(name='detected_events', type='array<string>', indexing=['summary', 'attribute']),
                        Field(name='language', type='string', indexing=['summary', 'attribute']),
                        # Acoustic embeddings (512 dims)
                        Field(
                            name='audio_embedding',
                            type='tensor<float>(d[512])',
                            indexing=['attribute', 'index'],
                            attribute=['distance-metric:angular']
                        ),
                        # Transcript semantic embeddings (768 dims)
                        Field(
                            name='semantic_embedding',
                            type='tensor<float>(d[768])',
                            indexing=['attribute', 'index'],
                            attribute=['distance-metric:angular']
                        ),
                    ]
                ),
                rank_profiles=[
                    # Acoustic similarity
                    RankProfile(
                        name='acoustic_similarity',
                        inputs=[('query(q)', 'tensor<float>(d[512])')],
                        first_phase='closeness(field, audio_embedding)'
                    ),
                    # Transcript BM25 search
                    RankProfile(
                        name='transcript_search',
                        first_phase='bm25(transcript)'
                    ),
                    # Hybrid: BM25 recall -> semantic reranking
                    RankProfile(
                        name='hybrid_audio',
                        inputs=[('query(q)', 'tensor<float>(d[768])')],
                        first_phase='bm25(transcript)',
                        second_phase=SecondPhaseRanking(
                            expression='closeness(field, semantic_embedding)',
                            rerank_count=100
                        )
                    ),
                ]
            )

            app_package = ApplicationPackage(name=app_name, schema=[audio_content_schema])
            self._deploy_package(app_package)

            self._logger.info("Successfully uploaded audio_content schema")

        except Exception as e:
            self._logger.error(f"Failed to upload audio_content schema: {str(e)}")
            raise

    def upload_content_type_schemas(self, app_name: str = "contenttypes", schemas: list = None) -> None:
        """
        Upload multiple content type schemas together in one application package.

        This avoids schema removal errors when deploying multiple schemas.

        Args:
            app_name: Name of the application
            schemas: List of schema names to deploy. Defaults to all content types
        """
        if schemas is None:
            schemas = ['image_content', 'audio_content', 'document_visual', 'document_text']

        try:
            from vespa.package import (
                ApplicationPackage,
                Document,
                Field,
                RankProfile,
                Schema,
                SecondPhaseRanking,
            )

            schema_objects = []

            # Build image_content schema
            if 'image_content' in schemas:
                image_content_schema = Schema(
                    name='image_content',
                    document=Document(
                        fields=[
                            Field(name='image_id', type='string', indexing=['summary', 'attribute'], attribute=['fast-search']),
                            Field(name='image_title', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                            Field(name='source_url', type='string', indexing=['summary', 'attribute']),
                            Field(name='creation_timestamp', type='long', indexing=['summary', 'attribute']),
                            Field(name='image_description', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                            Field(name='detected_objects', type='array<string>', indexing=['summary', 'attribute']),
                            Field(name='detected_scenes', type='array<string>', indexing=['summary', 'attribute']),
                            # ColPali multi-vector embedding (same as video frames)
                            Field(
                                name='colpali_embedding',
                                type='tensor<float>(x[1024],d[128])',
                                indexing=['attribute'],
                                attribute=['distance-metric:prenormalized-angular']
                            ),
                        ]
                    ),
                    rank_profiles=[
                        RankProfile(
                            name='colpali_similarity',
                            inputs=[('query(q)', 'tensor<float>(x[1024],d[128])')],
                            first_phase='sum(reduce(sum(query(q) * attribute(colpali_embedding), d), max, x))'
                        ),
                        RankProfile(
                            name='hybrid_image',
                            inputs=[('query(q)', 'tensor<float>(x[1024],d[128])')],
                            first_phase='bm25(image_description)',
                            second_phase=SecondPhaseRanking(
                                expression='sum(reduce(sum(query(q) * attribute(colpali_embedding), d), max, x))',
                                rerank_count=100
                            )
                        ),
                    ]
                )
                schema_objects.append(image_content_schema)

            # Build audio_content schema
            if 'audio_content' in schemas:
                audio_content_schema = Schema(
                    name='audio_content',
                    document=Document(
                        fields=[
                            Field(name='audio_id', type='string', indexing=['summary', 'attribute'], attribute=['fast-search']),
                            Field(name='audio_title', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                            Field(name='source_url', type='string', indexing=['summary', 'attribute']),
                            Field(name='duration', type='double', indexing=['summary', 'attribute']),
                            Field(name='transcript', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                            Field(name='speaker_labels', type='array<string>', indexing=['summary', 'attribute']),
                            Field(name='detected_events', type='array<string>', indexing=['summary', 'attribute']),
                            Field(name='language', type='string', indexing=['summary', 'attribute']),
                            # Acoustic embeddings (512 dims)
                            Field(
                                name='audio_embedding',
                                type='tensor<float>(d[512])',
                                indexing=['attribute', 'index'],
                                attribute=['distance-metric:angular']
                            ),
                            # Transcript semantic embeddings (768 dims)
                            Field(
                                name='semantic_embedding',
                                type='tensor<float>(d[768])',
                                indexing=['attribute', 'index'],
                                attribute=['distance-metric:angular']
                            ),
                        ]
                    ),
                    rank_profiles=[
                        # Acoustic similarity search
                        RankProfile(
                            name='acoustic_similarity',
                            inputs=[('query(q)', 'tensor<float>(d[512])')],
                            first_phase='closeness(field, audio_embedding)'
                        ),
                        # Transcript BM25 search
                        RankProfile(
                            name='transcript_search',
                            first_phase='bm25(transcript)'
                        ),
                        # Hybrid: BM25 + semantic embeddings
                        RankProfile(
                            name='hybrid_audio',
                            inputs=[('query(q)', 'tensor<float>(d[768])')],
                            first_phase='bm25(transcript)',
                            second_phase=SecondPhaseRanking(
                                expression='closeness(field, semantic_embedding)',
                                rerank_count=100
                            )
                        ),
                    ]
                )
                schema_objects.append(audio_content_schema)

            # Build document_visual schema (ColPali page-as-image)
            if 'document_visual' in schemas:

                document_visual_schema = Schema(
                    name='document_visual',
                    document=Document(
                        fields=[
                            Field(name='document_id', type='string', indexing=['summary', 'attribute'], attribute=['fast-search']),
                            Field(name='document_title', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                            Field(name='document_type', type='string', indexing=['summary', 'attribute']),
                            Field(name='page_number', type='int', indexing=['summary', 'attribute'], attribute=['fast-search']),
                            Field(name='page_count', type='int', indexing=['summary', 'attribute']),
                            Field(name='source_url', type='string', indexing=['summary', 'attribute']),
                            Field(name='creation_timestamp', type='long', indexing=['summary', 'attribute']),
                            # ColPali multi-vector embeddings per page (same as video frames)
                            Field(
                                name='colpali_embedding',
                                type='tensor<float>(x[1024],d[128])',
                                indexing=['attribute'],
                                attribute=['distance-metric:prenormalized-angular']
                            ),
                        ]
                    ),
                    rank_profiles=[
                        # Pure ColPali max similarity (identical to video frames)
                        RankProfile(
                            name='colpali',
                            inputs=[('query(qt)', 'tensor<float>(x[1024],d[128])')],
                            first_phase='sum(reduce(sum(query(qt) * attribute(colpali_embedding), d), max, x))'
                        ),
                        # Hybrid: visual recall -> text re-ranking
                        RankProfile(
                            name='hybrid_visual_text',
                            inputs=[('query(qt)', 'tensor<float>(x[1024],d[128])')],
                            first_phase='sum(reduce(sum(query(qt) * attribute(colpali_embedding), d), max, x))',
                            second_phase=SecondPhaseRanking(
                                expression='bm25(document_title)',
                                rerank_count=100
                            )
                        ),
                    ]
                )
                schema_objects.append(document_visual_schema)

            # Build document_text schema (traditional text extraction)
            if 'document_text' in schemas:
                document_text_schema = Schema(
                    name='document_text',
                    document=Document(
                        fields=[
                            Field(name='document_id', type='string', indexing=['summary', 'attribute'], attribute=['fast-search']),
                            Field(name='document_title', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                            Field(name='document_type', type='string', indexing=['summary', 'attribute']),
                            Field(name='page_count', type='int', indexing=['summary', 'attribute']),
                            Field(name='source_url', type='string', indexing=['summary', 'attribute']),
                            Field(name='creation_timestamp', type='long', indexing=['summary', 'attribute']),
                            # Extracted text content
                            Field(name='full_text', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                            Field(name='section_headings', type='array<string>', indexing=['summary', 'attribute']),
                            Field(name='key_entities', type='array<string>', indexing=['summary', 'attribute']),
                            # Dense semantic embeddings (768 dims from sentence-transformers)
                            Field(
                                name='document_embedding',
                                type='tensor<float>(d[768])',
                                indexing=['attribute', 'index'],
                                attribute=['distance-metric:angular']
                            ),
                        ]
                    ),
                    rank_profiles=[
                        # Pure BM25 keyword search
                        RankProfile(
                            name='bm25',
                            first_phase='bm25(document_title) + bm25(full_text)'
                        ),
                        # Pure semantic search
                        RankProfile(
                            name='semantic',
                            inputs=[('query(q)', 'tensor<float>(d[768])')],
                            first_phase='closeness(field, document_embedding)'
                        ),
                        # Hybrid: BM25 recall -> semantic re-ranking
                        RankProfile(
                            name='hybrid_bm25_semantic',
                            inputs=[('query(q)', 'tensor<float>(d[768])')],
                            first_phase='bm25(full_text)',
                            second_phase=SecondPhaseRanking(
                                expression='closeness(field, document_embedding)',
                                rerank_count=100
                            )
                        ),
                    ]
                )
                schema_objects.append(document_text_schema)

            # Deploy all schemas together
            app_package = ApplicationPackage(name=app_name, schema=schema_objects)
            self._deploy_package(app_package)

            self._logger.info(f"Successfully uploaded content type schemas: {schemas}")

        except Exception as e:
            self._logger.error(f"Failed to upload content type schemas: {str(e)}")
            raise

    def upload_frame_schema(self, app_name: str = "videosearch") -> None:
        """
        Create and upload document-per-frame schema using PyVespa directly.

        Args:
            app_name: Name of the application
        """
        try:
            from vespa.package import (
                ApplicationPackage,
                Document,
                Field,
                Function,
                RankProfile,
                Schema,
                SecondPhaseRanking,
            )
            
            # Create document-per-frame schema with ranking profiles
            video_frame_schema = Schema(
                name='video_frame',
                document=Document(
                    fields=[
                        Field(name='video_id', type='string', indexing=['summary', 'attribute'], attribute=['fast-search']),
                        Field(name='video_title', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                        Field(name='creation_timestamp', type='long', indexing=['summary', 'attribute'], attribute=['fast-search']),
                        Field(name='frame_id', type='int', indexing=['summary', 'attribute'], attribute=['fast-search']),
                        Field(name='start_time', type='double', indexing=['summary', 'attribute'], attribute=['fast-search']),
                        Field(name='end_time', type='double', indexing=['summary', 'attribute'], attribute=['fast-search']),
                        Field(name='frame_description', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                        Field(name='audio_transcript', type='string', indexing=['summary', 'index'], index='enable-bm25'),
                        Field(
                            name='colpali_embedding',
                            type='tensor<float>(patch{}, v[128])',
                            indexing=['attribute'],
                            attribute=['distance-metric:dotproduct']
                        ),
                        Field(
                            name='embedding_binary', 
                            type='tensor<int8>(patch{}, v[16])',
                            indexing=['attribute', 'index'],
                            attribute=['distance-metric:hamming']
                        )
                    ]
                ),
                rank_profiles=[
                    # Text-only BM25 ranking
                    RankProfile(
                        name='text_only',
                        first_phase='bm25(video_title) + bm25(frame_description) + bm25(audio_transcript)'
                    ),
                    
                    # Pure ColPali max similarity ranking
                    RankProfile(
                        name='colpali',
                        inputs=[
                            ('query(qt)', 'tensor<float>(querytoken{}, v[128])')
                        ],
                        functions=[
                            Function(
                                name='max_sim',
                                expression='sum(reduce(sum(query(qt) * attribute(embedding), v), max, patch), querytoken)'
                            )
                        ],
                        first_phase='max_sim'
                    ),
                    
                    # Binary-first with float re-ranking (efficient retrieval)
                    RankProfile(
                        name='binary_float',
                        inputs=[
                            ('query(qt)', 'tensor<float>(querytoken{}, v[128])'),
                            ('query(qtb)', 'tensor<int8>(querytoken{}, v[16])')
                        ],
                        functions=[
                            Function(
                                name='max_sim_float',
                                expression='sum(reduce(sum(query(qt) * attribute(embedding), v), max, patch), querytoken)'
                            ),
                            Function(
                                name='max_sim_binary',
                                expression='sum(reduce(sum(query(qtb) * attribute(embedding_binary), v), max, patch), querytoken)'
                            )
                        ],
                        first_phase='max_sim_binary',
                        second_phase=SecondPhaseRanking(expression='max_sim_float', rerank_count=1000)
                    ),
                    
                    # True hybrid: visual recall -> text re-ranking
                    RankProfile(
                        name='hybrid_search',
                        inputs=[
                            ('query(qt)', 'tensor<float>(querytoken{}, v[128])')
                        ],
                        functions=[
                            Function(
                                name='visual_sim',
                                expression='sum(reduce(sum(query(qt) * attribute(embedding), v), max, patch), querytoken)'
                            ),
                            Function(
                                name='text_sim',
                                expression='bm25(video_title) + bm25(frame_description) + bm25(audio_transcript)'
                            )
                        ],
                        first_phase='visual_sim',
                        second_phase=SecondPhaseRanking(expression='text_sim', rerank_count=100)
                    )
                ]
            )
            
            # Create application package
            app_package = ApplicationPackage(name=app_name, schema=[video_frame_schema])
            
            # Deploy to Vespa
            self._deploy_package(app_package)
            
            self._logger.info("Successfully uploaded document-per-frame schema")
            
        except Exception as e:
            self._logger.error(f"Failed to upload schema: {str(e)}")
            raise
    
    def upload_schema_from_json_file(self, json_file_path: str, app_name: str = "videosearch") -> None:
        """
        Load schema from JSON file and deploy to Vespa using JSON parser.
        
        Args:
            json_file_path: Path to the JSON schema file
            app_name: Name of the application
        """
        try:
            from vespa.package import ApplicationPackage

            from .json_schema_parser import JsonSchemaParser
            
            # Parse JSON schema to PyVespa objects
            parser = JsonSchemaParser()
            schema = parser.load_schema_from_json_file(json_file_path)
            
            # Validate the schema
            with open(json_file_path, 'r') as f:
                import json
                schema_config = json.load(f)
            
            errors = parser.validate_schema_config(schema_config)
            if errors:
                raise ValueError(f"Schema validation errors: {'; '.join(errors)}")
            
            # Create application package
            app_package = ApplicationPackage(name=app_name, schema=[schema])
            
            # Deploy to Vespa
            self._deploy_package(app_package)
            
            self._logger.info(f"Successfully uploaded schema from JSON file: {json_file_path}")
            
        except Exception as e:
            self._logger.error(f"Failed to upload schema from JSON: {str(e)}")
            raise
    
    def upload_schema_from_sd_file(self, sd_file_path: str, app_name: str = "videosearch") -> None:
        """
        Read a .sd file and upload it to Vespa using the original parsing approach.
        
        Args:
            sd_file_path: Path to the .sd schema file
            app_name: Name of the application
        """
        try:
            # Read the .sd file
            sd_content = self.read_sd_file(sd_file_path)
            
            # Parse the schema
            schema = self.parse_sd_schema(sd_content)
            
            # Create application package
            app_package = self.create_application_package(schema, app_name)
            
            # Upload to Vespa
            self._deploy_package(app_package)
            
            self._logger.info(f"Successfully uploaded schema from {sd_file_path}")
            
        except Exception as e:
            self._logger.error(f"Failed to upload schema: {str(e)}")
            raise
    
    def upload_schema_from_directory(self, schemas_dir: str, app_name: str = "video_search") -> None:
        """
        Read all .sd files from a directory and upload them to Vespa.
        
        Args:
            schemas_dir: Directory containing .sd files
            app_name: Name of the application
        """
        schemas_path = Path(schemas_dir)
        if not schemas_path.exists():
            raise FileNotFoundError(f"Schemas directory not found: {schemas_dir}")
        
        sd_files = list(schemas_path.glob("*.sd"))
        if not sd_files:
            raise ValueError(f"No .sd files found in {schemas_dir}")
        
        schemas = []
        for sd_file in sd_files:
            try:
                sd_content = self.read_sd_file(str(sd_file))
                schema = self.parse_sd_schema(sd_content)
                schemas.append(schema)
                self._logger.info(f"Parsed schema from {sd_file.name}")
            except Exception as e:
                self._logger.error(f"Failed to parse {sd_file.name}: {str(e)}")
                raise
        
        # Create application package with all schemas
        app_package = ApplicationPackage(name=app_name, schema=schemas)
        
        # Upload to Vespa
        self._deploy_package(app_package)
        
        self._logger.info(f"Successfully uploaded {len(schemas)} schemas from {schemas_dir}")
    
    def _deploy_package(self, app_package: ApplicationPackage, allow_field_type_change: bool = False) -> None:
        """
        Deploy an application package to Vespa.
        
        Args:
            app_package: The ApplicationPackage to deploy
            allow_field_type_change: If True, adds validation override for field type changes
        """
        import json

        import requests
        from vespa.package import Validation, ValidationID
        
        # Add validation override if requested
        if allow_field_type_change:
            from datetime import datetime, timedelta
            # Set validation until 29 days from now (to stay within 30-day limit)
            until_date = (datetime.now() + timedelta(days=29)).strftime("%Y-%m-%d")
            validation = Validation(
                validation_id=ValidationID.fieldTypeChange,
                until=until_date,
                comment="Allow field type changes for schema updates"
            )
            if app_package.validations is None:
                app_package.validations = []
            app_package.validations.append(validation)
        
        # Create the deployment URL
        deploy_url = f"{self.vespa_endpoint.replace('8080', str(self.vespa_port))}/application/v2/tenant/default/prepareandactivate"
        
        try:
            # Generate the ZIP package
            app_zip = app_package.to_zip()
            
            # Deploy via HTTP
            response = requests.post(
                deploy_url,
                headers={"Content-Type": "application/zip"},
                data=app_zip,
                verify=False
            )
            
            if response.status_code == 200:
                self._logger.info("Successfully deployed application package")
            else:
                error_msg = f"Deployment failed with status {response.status_code}"
                try:
                    error_detail = json.loads(response.content.decode('utf-8'))
                    error_msg += f": {error_detail}"
                except Exception:
                    error_msg += f": {response.content.decode('utf-8')}"
                
                raise RuntimeError(error_msg)
                
        except Exception as e:
            self._logger.error(f"Failed to deploy package: {str(e)}")
            raise

    def upload_metadata_schemas(self, app_name: str = "metadata") -> None:
        """
        Deploy organization and tenant metadata schemas for multi-tenant management.

        These schemas store org/tenant metadata and are used by the tenant management API.

        Args:
            app_name: Name of the application
        """
        try:
            from vespa.package import (
                ApplicationPackage,
                Document,
                Field,
                Schema,
            )

            # Organization metadata schema
            organization_metadata_schema = Schema(
                name='organization_metadata',
                document=Document(
                    fields=[
                        Field(
                            name='org_id',
                            type='string',
                            indexing=['summary', 'attribute'],
                            attribute=['fast-search']
                        ),
                        Field(
                            name='org_name',
                            type='string',
                            indexing=['summary', 'index']
                        ),
                        Field(
                            name='created_at',
                            type='long',
                            indexing=['summary', 'attribute']
                        ),
                        Field(
                            name='created_by',
                            type='string',
                            indexing=['summary', 'attribute']
                        ),
                        Field(
                            name='status',
                            type='string',
                            indexing=['summary', 'attribute'],
                            attribute=['fast-search']
                        ),
                        Field(
                            name='tenant_count',
                            type='int',
                            indexing=['summary', 'attribute']
                        ),
                    ]
                )
            )

            # Tenant metadata schema
            tenant_metadata_schema = Schema(
                name='tenant_metadata',
                document=Document(
                    fields=[
                        Field(
                            name='tenant_full_id',
                            type='string',
                            indexing=['summary', 'attribute'],
                            attribute=['fast-search']
                        ),
                        Field(
                            name='org_id',
                            type='string',
                            indexing=['summary', 'attribute'],
                            attribute=['fast-search']
                        ),
                        Field(
                            name='tenant_name',
                            type='string',
                            indexing=['summary', 'attribute']
                        ),
                        Field(
                            name='created_at',
                            type='long',
                            indexing=['summary', 'attribute']
                        ),
                        Field(
                            name='created_by',
                            type='string',
                            indexing=['summary', 'attribute']
                        ),
                        Field(
                            name='status',
                            type='string',
                            indexing=['summary', 'attribute'],
                            attribute=['fast-search']
                        ),
                        Field(
                            name='schemas_deployed',
                            type='array<string>',
                            indexing=['summary', 'attribute']
                        ),
                    ]
                )
            )

            # Deploy both schemas together
            app_package = ApplicationPackage(
                name=app_name,
                schema=[organization_metadata_schema, tenant_metadata_schema]
            )
            self._deploy_package(app_package)

            self._logger.info("Successfully deployed organization and tenant metadata schemas")

        except Exception as e:
            self._logger.error(f"Failed to deploy metadata schemas: {str(e)}")
            raise
