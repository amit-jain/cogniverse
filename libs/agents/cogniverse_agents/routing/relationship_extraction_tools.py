"""
Relationship Extraction Tools for DSPy 3.0 Routing System

This module provides tools for extracting relationships from queries using GLiNER and spaCy
for enhanced routing decisions. It integrates with the DSPy 3.0 signatures to provide
structured relationship data that can be used for query enhancement.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.tokens import Doc, Token

logger = logging.getLogger(__name__)


class GLiNERRelationshipExtractor:
    """
    GLiNER-based relationship extractor for entity recognition and relationship inference.

    Uses GLiNER for accurate entity extraction and combines it with linguistic analysis
    to identify relationships between entities.
    """

    def __init__(self, model_name: str = "urchade/gliner_mediumv2.1"):
        """
        Initialize GLiNER relationship extractor.

        Args:
            model_name: GLiNER model to use for entity extraction
        """
        self.model_name = model_name
        self.gliner_model = None
        self._load_gliner_model()

    def _load_gliner_model(self):
        """Load GLiNER model with error handling"""
        try:
            from gliner import GLiNER

            self.gliner_model = GLiNER.from_pretrained(self.model_name)
            logger.info(f"Loaded GLiNER model: {self.model_name}")
        except ImportError:
            logger.warning("GLiNER not installed. Entity extraction will be limited.")
            self.gliner_model = None
        except Exception as e:
            logger.error(f"Failed to load GLiNER model {self.model_name}: {e}")
            self.gliner_model = None

    def extract_entities(
        self, text: str, labels: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from text using GLiNER.

        Args:
            text: Input text to extract entities from
            labels: Optional list of entity labels to look for

        Returns:
            List of entity dictionaries with text, label, score, start, end
        """
        if not self.gliner_model:
            logger.warning("GLiNER model not available, returning empty entities")
            return []

        if labels is None:
            # Default entity types relevant for video/content queries
            labels = [
                "PERSON",
                "ORGANIZATION",
                "LOCATION",
                "EVENT",
                "PRODUCT",
                "TECHNOLOGY",
                "CONCEPT",
                "ACTION",
                "OBJECT",
                "ANIMAL",
                "SPORT",
                "ACTIVITY",
                "TOOL",
                "VEHICLE",
                "MATERIAL",
            ]

        try:
            entities = self.gliner_model.predict_entities(text, labels)

            # Convert to our standard format
            extracted_entities = []
            for entity in entities:
                extracted_entities.append(
                    {
                        "text": entity["text"],
                        "label": entity["label"],
                        "confidence": float(entity["score"]),
                        "start_pos": int(entity["start"]),
                        "end_pos": int(entity["end"]),
                    }
                )

            logger.debug(f"Extracted {len(extracted_entities)} entities from text")
            return extracted_entities

        except Exception as e:
            logger.error(f"GLiNER entity extraction failed: {e}")
            return []

    def infer_relationships_from_entities(
        self, text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Infer relationships between entities using positional and contextual analysis.

        Args:
            text: Original text
            entities: List of extracted entities

        Returns:
            List of relationship tuples
        """
        relationships = []

        if len(entities) < 2:
            return relationships

        # Simple relationship inference based on proximity and common patterns
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i + 1 :], i + 1):
                # Calculate distance between entities
                distance = abs(entity1["start_pos"] - entity2["start_pos"])

                # Only consider relationships between nearby entities
                if distance > 100:  # Skip if entities are too far apart
                    continue

                # Infer relationship type based on entity types and context
                relation_type = self._infer_relation_type(entity1, entity2, text)

                if relation_type:
                    confidence = self._calculate_relationship_confidence(
                        entity1, entity2, distance, text
                    )

                    relationships.append(
                        {
                            "subject": entity1["text"],
                            "relation": relation_type,
                            "object": entity2["text"],
                            "confidence": confidence,
                            "subject_type": entity1["label"],
                            "object_type": entity2["label"],
                            "context": self._extract_context(
                                text, entity1["start_pos"], entity2["end_pos"]
                            ),
                        }
                    )

        # Sort by confidence
        relationships.sort(key=lambda x: x["confidence"], reverse=True)
        logger.debug(f"Inferred {len(relationships)} relationships")

        return relationships

    def _infer_relation_type(
        self, entity1: Dict[str, Any], entity2: Dict[str, Any], text: str
    ) -> Optional[str]:
        """
        Infer relationship type between two entities based on their types and context.

        Args:
            entity1: First entity
            entity2: Second entity
            text: Original text

        Returns:
            Relationship type or None
        """
        type1, type2 = entity1["label"], entity2["label"]

        # Common relationship patterns
        relation_patterns = {
            ("PERSON", "ORGANIZATION"): "works_for",
            ("PERSON", "LOCATION"): "lives_in",
            ("PERSON", "ACTIVITY"): "performs",
            ("PERSON", "SPORT"): "plays",
            ("PERSON", "TOOL"): "uses",
            ("ORGANIZATION", "PRODUCT"): "produces",
            ("ORGANIZATION", "LOCATION"): "located_in",
            ("ANIMAL", "ACTION"): "performs",
            ("VEHICLE", "LOCATION"): "travels_to",
            ("TECHNOLOGY", "ACTIVITY"): "enables",
            ("CONCEPT", "APPLICATION"): "applies_to",
        }

        # Check direct patterns
        if (type1, type2) in relation_patterns:
            return relation_patterns[(type1, type2)]
        elif (type2, type1) in relation_patterns:
            return relation_patterns[(type2, type1)]

        # Context-based inference (simplified)
        context_text = text.lower()

        # Action relationships
        if "playing" in context_text or "plays" in context_text:
            return "plays"
        elif "using" in context_text or "uses" in context_text:
            return "uses"
        elif "showing" in context_text or "shows" in context_text:
            return "shows"
        elif "demonstrating" in context_text:
            return "demonstrates"
        elif "learning" in context_text:
            return "learns"
        elif "teaching" in context_text:
            return "teaches"

        # Spatial relationships
        elif "in" in context_text:
            return "located_in"
        elif "with" in context_text:
            return "associated_with"
        elif "for" in context_text:
            return "intended_for"

        # Default relationship
        return "related_to"

    def _calculate_relationship_confidence(
        self, entity1: Dict[str, Any], entity2: Dict[str, Any], distance: int, text: str
    ) -> float:
        """
        Calculate confidence score for a relationship.

        Args:
            entity1: First entity
            entity2: Second entity
            distance: Distance between entities
            text: Original text

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from entity confidence scores
        base_confidence = (entity1["confidence"] + entity2["confidence"]) / 2

        # Distance penalty (closer entities more likely to be related)
        distance_factor = max(0.1, 1.0 - (distance / 100.0))

        # Context indicators
        context_factor = 1.0
        context_text = text.lower()

        # Boost confidence for explicit relationship indicators
        relationship_indicators = [
            "with",
            "using",
            "showing",
            "playing",
            "performing",
            "demonstrating",
            "teaching",
            "learning",
            "in",
            "at",
            "for",
        ]

        for indicator in relationship_indicators:
            if indicator in context_text:
                context_factor *= 1.2
                break

        # Final confidence calculation
        final_confidence = min(1.0, base_confidence * distance_factor * context_factor)
        return round(final_confidence, 3)

    def _extract_context(self, text: str, start: int, end: int) -> str:
        """
        Extract context around the entities.

        Args:
            text: Original text
            start: Start position of first entity
            end: End position of second entity

        Returns:
            Context string
        """
        # Extract a window around the entities
        window_start = max(0, start - 20)
        window_end = min(len(text), end + 20)

        return text[window_start:window_end].strip()


class SpaCyDependencyAnalyzer:
    """
    spaCy-based dependency parser for linguistic relationship analysis.

    Complements GLiNER entity extraction with grammatical relationship analysis
    to identify more complex linguistic patterns and dependencies.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize spaCy dependency analyzer.

        Args:
            model_name: spaCy model to use
        """
        self.model_name = model_name
        self.nlp = None
        self._load_spacy_model()

    def _load_spacy_model(self):
        """Load spaCy model with error handling"""
        try:
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
        except IOError:
            logger.warning(
                f"spaCy model {self.model_name} not found. Operating without spaCy dependency analysis."
            )
            self.nlp = None
        except Exception as e:
            logger.warning(f"spaCy model unavailable: {e}")
            self.nlp = None

    def analyze_dependencies(self, text: str) -> Dict[str, Any]:
        """
        Analyze grammatical dependencies in text.

        Args:
            text: Input text to analyze

        Returns:
            Dictionary with dependency analysis results
        """
        if not self.nlp:
            logger.warning("spaCy model not available")
            return {"dependencies": [], "structure": "unknown"}

        try:
            doc = self.nlp(text)

            dependencies = []
            for token in doc:
                dependencies.append(
                    {
                        "text": token.text,
                        "lemma": token.lemma_,
                        "pos": token.pos_,
                        "tag": token.tag_,
                        "dep": token.dep_,
                        "head": token.head.text,
                        "children": [child.text for child in token.children],
                    }
                )

            # Analyze overall structure
            structure = self._analyze_sentence_structure(doc)

            return {
                "dependencies": dependencies,
                "structure": structure,
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "noun_phrases": [chunk.text for chunk in doc.noun_chunks],
                "complexity_score": self._calculate_complexity_score(doc),
            }

        except Exception as e:
            logger.error(f"spaCy dependency analysis failed: {e}")
            return {"dependencies": [], "structure": "error"}

    def extract_semantic_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract semantic relationships using dependency parsing.

        Args:
            text: Input text

        Returns:
            List of semantic relationship tuples
        """
        if not self.nlp:
            return []

        try:
            doc = self.nlp(text)
            relationships = []

            # Extract verb-based relationships
            for token in doc:
                if token.pos_ == "VERB":
                    subject, obj = self._find_subject_object(token)
                    if subject and obj:
                        relationships.append(
                            {
                                "subject": subject.text,
                                "relation": token.lemma_,
                                "object": obj.text,
                                "confidence": 0.8,
                                "grammatical_pattern": f"{subject.dep_}-{token.dep_}-{obj.dep_}",
                            }
                        )

            # Extract noun-based relationships through prepositions
            for token in doc:
                if token.dep_ == "prep":
                    head = token.head
                    prep_obj = [
                        child for child in token.children if child.dep_ == "pobj"
                    ]
                    if head and prep_obj:
                        relationships.append(
                            {
                                "subject": head.text,
                                "relation": token.text,  # preposition as relation
                                "object": prep_obj[0].text,
                                "confidence": 0.7,
                                "grammatical_pattern": f"prep-{token.text}",
                            }
                        )

            return relationships

        except Exception as e:
            logger.error(f"Semantic relationship extraction failed: {e}")
            return []

    def _find_subject_object(
        self, verb_token: Token
    ) -> Tuple[Optional[Token], Optional[Token]]:
        """
        Find subject and object for a verb token.

        Args:
            verb_token: Verb token to find subject/object for

        Returns:
            Tuple of (subject_token, object_token)
        """
        subject = None
        obj = None

        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                subject = child
            elif child.dep_ in ["dobj", "pobj"]:
                obj = child

        return subject, obj

    def _analyze_sentence_structure(self, doc: Doc) -> str:
        """
        Analyze overall sentence structure.

        Args:
            doc: spaCy Doc object

        Returns:
            Structure classification
        """
        verbs = [token for token in doc if token.pos_ == "VERB"]
        # nouns = [token for token in doc if token.pos_ in ["NOUN", "PROPN"]]  # Unused for now

        if len(verbs) == 0:
            return "nominal"
        elif len(verbs) == 1:
            return "simple"
        elif len(verbs) > 1:
            return "complex"
        else:
            return "unknown"

    def _calculate_complexity_score(self, doc: Doc) -> float:
        """
        Calculate text complexity score.

        Args:
            doc: spaCy Doc object

        Returns:
            Complexity score between 0 and 1
        """
        # Simple complexity metrics
        factors = {
            "sentence_length": min(1.0, len(doc) / 20.0),
            "unique_pos_tags": len(set(token.pos_ for token in doc))
            / 17.0,  # 17 universal POS tags
            "dependency_depth": self._max_dependency_depth(doc) / 10.0,
            "entity_density": len(doc.ents) / len(doc) if len(doc) > 0 else 0,
        }

        # Weighted average
        weights = {
            "sentence_length": 0.3,
            "unique_pos_tags": 0.3,
            "dependency_depth": 0.25,
            "entity_density": 0.15,
        }

        complexity = sum(factors[key] * weights[key] for key in factors)
        return min(1.0, complexity)

    def _max_dependency_depth(self, doc: Doc) -> int:
        """
        Calculate maximum dependency depth in the sentence.

        Args:
            doc: spaCy Doc object

        Returns:
            Maximum depth
        """

        def get_depth(token, current_depth=0):
            if not list(token.children):
                return current_depth
            return max(get_depth(child, current_depth + 1) for child in token.children)

        return max(
            (get_depth(token) for token in doc if token.dep_ == "ROOT"), default=0
        )


class RelationshipExtractorTool:
    """
    Unified relationship extraction tool that combines GLiNER and spaCy.

    Provides a high-level interface for extracting entities and relationships
    from queries for use in DSPy 3.0 routing decisions.
    """

    def __init__(
        self,
        gliner_model: str = "urchade/gliner_mediumv2.1",
        spacy_model: str = "en_core_web_sm",
    ):
        """
        Initialize the relationship extractor tool.

        Args:
            gliner_model: GLiNER model name
            spacy_model: spaCy model name
        """
        self.gliner_extractor = GLiNERRelationshipExtractor(gliner_model)
        self.spacy_analyzer = SpaCyDependencyAnalyzer(spacy_model)

        logger.info("Relationship extractor tool initialized")

    async def extract_comprehensive_relationships(
        self, text: str, entity_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract comprehensive relationship information from text.

        Args:
            text: Input text to analyze
            entity_labels: Optional entity labels for GLiNER

        Returns:
            Comprehensive relationship analysis
        """
        try:
            # Extract entities with GLiNER
            entities = self.gliner_extractor.extract_entities(text, entity_labels)

            # Infer relationships between entities
            entity_relationships = (
                self.gliner_extractor.infer_relationships_from_entities(text, entities)
            )

            # Perform dependency analysis with spaCy
            dependency_analysis = self.spacy_analyzer.analyze_dependencies(text)

            # Extract semantic relationships with spaCy
            semantic_relationships = self.spacy_analyzer.extract_semantic_relationships(
                text
            )

            # Combine and deduplicate relationships
            all_relationships = entity_relationships + semantic_relationships
            deduplicated_relationships = self._deduplicate_relationships(
                all_relationships
            )

            # Generate relationship summary
            relationship_types = list(
                set(rel["relation"] for rel in deduplicated_relationships)
            )
            semantic_connections = self._generate_semantic_connections(
                entities, deduplicated_relationships
            )

            return {
                "entities": entities,
                "relationships": deduplicated_relationships,
                "relationship_types": relationship_types,
                "semantic_connections": semantic_connections,
                "query_structure": dependency_analysis.get("structure", "unknown"),
                "complexity_indicators": self._identify_complexity_indicators(
                    text, entities, deduplicated_relationships, dependency_analysis
                ),
                "confidence": self._calculate_overall_confidence(
                    entities, deduplicated_relationships
                ),
            }

        except Exception as e:
            logger.error(f"Comprehensive relationship extraction failed: {e}")
            return {
                "entities": [],
                "relationships": [],
                "relationship_types": [],
                "semantic_connections": [],
                "query_structure": "error",
                "complexity_indicators": [],
                "confidence": 0.0,
            }

    def _deduplicate_relationships(
        self, relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate relationships based on subject-relation-object triples.

        Args:
            relationships: List of relationships

        Returns:
            Deduplicated relationships
        """
        seen = set()
        deduplicated = []

        for rel in relationships:
            key = (
                rel["subject"].lower(),
                rel["relation"].lower(),
                rel["object"].lower(),
            )

            if key not in seen:
                seen.add(key)
                deduplicated.append(rel)

        return deduplicated

    def _generate_semantic_connections(
        self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate human-readable semantic connections.

        Args:
            entities: Extracted entities
            relationships: Extracted relationships

        Returns:
            List of semantic connection descriptions
        """
        connections = []

        for rel in relationships[:5]:  # Top 5 relationships
            subject = rel["subject"]
            relation = rel["relation"].replace("_", " ")
            obj = rel["object"]

            connections.append(f"{subject} {relation} {obj}")

        return connections

    def _identify_complexity_indicators(
        self,
        text: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        dependency_analysis: Dict[str, Any],
    ) -> List[str]:
        """
        Identify factors that indicate query complexity.

        Args:
            text: Original text
            entities: Extracted entities
            relationships: Extracted relationships
            dependency_analysis: spaCy analysis results

        Returns:
            List of complexity indicators
        """
        indicators = []

        # Entity density
        if len(entities) > 3:
            indicators.append(f"High entity density ({len(entities)} entities)")

        # Relationship complexity
        if len(relationships) > 2:
            indicators.append(
                f"Multiple relationships ({len(relationships)} relations)"
            )

        # Text length
        if len(text.split()) > 15:
            indicators.append(f"Long query ({len(text.split())} words)")

        # Grammatical complexity
        complexity_score = dependency_analysis.get("complexity_score", 0)
        if complexity_score > 0.7:
            indicators.append(
                f"High grammatical complexity (score: {complexity_score:.2f})"
            )

        # Multiple entity types
        entity_types = set(entity["label"] for entity in entities)
        if len(entity_types) > 3:
            indicators.append(f"Multiple entity types ({len(entity_types)} types)")

        return indicators

    def _calculate_overall_confidence(
        self, entities: List[Dict[str, Any]], relationships: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall confidence in the relationship extraction.

        Args:
            entities: Extracted entities
            relationships: Extracted relationships

        Returns:
            Overall confidence score
        """
        if not entities:
            return 0.0

        # Average entity confidence
        entity_confidence = sum(e["confidence"] for e in entities) / len(entities)

        # Average relationship confidence
        if relationships:
            relationship_confidence = sum(r["confidence"] for r in relationships) / len(
                relationships
            )
        else:
            relationship_confidence = 0.5  # Neutral if no relationships

        # Weighted average
        overall_confidence = 0.6 * entity_confidence + 0.4 * relationship_confidence

        return round(overall_confidence, 3)


# Factory function for easy instantiation
def create_relationship_extractor(
    gliner_model: str = "urchade/gliner_mediumv2.1", spacy_model: str = "en_core_web_sm"
) -> RelationshipExtractorTool:
    """
    Factory function to create a relationship extractor tool.

    Args:
        gliner_model: GLiNER model name
        spacy_model: spaCy model name

    Returns:
        Configured RelationshipExtractorTool instance
    """
    return RelationshipExtractorTool(gliner_model, spacy_model)
