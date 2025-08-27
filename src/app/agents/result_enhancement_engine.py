"""Result Enhancement Engine with Relationship Context."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EnhancementContext:
    """Context for result enhancement"""

    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    query: str
    enhanced_query: Optional[str] = None
    routing_confidence: float = 0.0
    enhancement_metadata: Optional[Dict[str, Any]] = None


@dataclass
class EnhancedResult:
    """Enhanced search result with relationship context"""

    original_result: Dict[str, Any]
    relevance_score: float
    entity_matches: List[Dict[str, Any]]
    relationship_matches: List[Dict[str, Any]]
    contextual_connections: List[Dict[str, Any]]
    enhancement_score: float
    enhancement_metadata: Dict[str, Any]


class ResultEnhancementEngine:
    """Engine for enhancing search results with relationship context"""

    def __init__(self, **kwargs):
        """Initialize result enhancement engine"""
        logger.info("Initializing ResultEnhancementEngine...")

        # Configuration
        self.entity_match_boost = kwargs.get("entity_match_boost", 0.15)
        self.relationship_match_boost = kwargs.get("relationship_match_boost", 0.25)
        self.contextual_connection_boost = kwargs.get(
            "contextual_connection_boost", 0.10
        )
        self.max_total_boost = kwargs.get("max_total_boost", 0.50)

        # Thresholds
        self.entity_confidence_threshold = kwargs.get(
            "entity_confidence_threshold", 0.5
        )
        self.relationship_confidence_threshold = kwargs.get(
            "relationship_confidence_threshold", 0.6
        )
        self.connection_strength_threshold = kwargs.get(
            "connection_strength_threshold", 0.3
        )

        logger.info("ResultEnhancementEngine initialization complete")

    def enhance_results(
        self, results: List[Dict[str, Any]], context: EnhancementContext
    ) -> List[EnhancedResult]:
        """
        Enhance search results with relationship context

        Args:
            results: Original search results
            context: Enhancement context with entities and relationships

        Returns:
            List of enhanced results with relationship context
        """
        logger.info(f"Enhancing {len(results)} results with relationship context")

        enhanced_results = []

        for result in results:
            try:
                enhanced_result = self._enhance_single_result(result, context)
                enhanced_results.append(enhanced_result)
            except Exception as e:
                logger.warning(f"Failed to enhance result: {e}")
                # Fallback to unenhanced result
                enhanced_results.append(self._create_fallback_enhanced_result(result))

        # Sort by enhanced relevance
        enhanced_results.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(f"Enhanced {len(enhanced_results)} results")
        return enhanced_results

    def _enhance_single_result(
        self, result: Dict[str, Any], context: EnhancementContext
    ) -> EnhancedResult:
        """Enhance a single search result"""

        # Extract text fields for analysis
        result_text = self._extract_result_text(result)

        # Find entity matches
        entity_matches = self._find_entity_matches(result_text, context.entities)

        # Find relationship matches
        relationship_matches = self._find_relationship_matches(
            result_text, context.relationships
        )

        # Find contextual connections
        contextual_connections = self._find_contextual_connections(
            result, entity_matches, relationship_matches, context
        )

        # Calculate enhancement score
        enhancement_score = self._calculate_enhancement_score(
            entity_matches, relationship_matches, contextual_connections
        )

        # Calculate boosted relevance score
        original_score = result.get("score", result.get("relevance", 0.0))
        boost = min(enhancement_score * self.max_total_boost, self.max_total_boost)
        boosted_score = min(original_score + boost, 1.0)

        # Create enhancement metadata
        enhancement_metadata = {
            "original_score": original_score,
            "boost_applied": boost,
            "entity_matches": len(entity_matches),
            "relationship_matches": len(relationship_matches),
            "contextual_connections": len(contextual_connections),
            "enhancement_score": enhancement_score,
            "enhancement_factors": {
                "entity_boost": len(entity_matches) * self.entity_match_boost,
                "relationship_boost": len(relationship_matches)
                * self.relationship_match_boost,
                "connection_boost": len(contextual_connections)
                * self.contextual_connection_boost,
            },
        }

        return EnhancedResult(
            original_result=result,
            relevance_score=boosted_score,
            entity_matches=entity_matches,
            relationship_matches=relationship_matches,
            contextual_connections=contextual_connections,
            enhancement_score=enhancement_score,
            enhancement_metadata=enhancement_metadata,
        )

    def _extract_result_text(self, result: Dict[str, Any]) -> str:
        """Extract all relevant text from a result for analysis"""
        text_parts = []

        # Standard text fields
        for field in ["title", "description", "content", "transcript", "summary"]:
            if field in result and result[field]:
                text_parts.append(str(result[field]))

        # Frame descriptions
        if "frame_descriptions" in result and isinstance(
            result["frame_descriptions"], list
        ):
            text_parts.extend([str(desc) for desc in result["frame_descriptions"]])

        # Segment descriptions
        if "segment_descriptions" in result and isinstance(
            result["segment_descriptions"], list
        ):
            text_parts.extend([str(desc) for desc in result["segment_descriptions"]])

        # Audio transcript
        if "audio_transcript" in result and result["audio_transcript"]:
            text_parts.append(str(result["audio_transcript"]))

        return " ".join(text_parts).lower()

    def _find_entity_matches(
        self, result_text: str, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find entity matches in result text"""
        matches = []

        for entity in entities:
            entity_text = entity.get("text", "").lower().strip()
            entity_confidence = entity.get("confidence", 0.0)

            if (
                entity_text
                and entity_confidence >= self.entity_confidence_threshold
                and entity_text in result_text
            ):

                # Calculate match strength based on context
                match_strength = self._calculate_entity_match_strength(
                    entity_text, result_text, entity_confidence
                )

                matches.append(
                    {
                        "entity": entity,
                        "match_text": entity_text,
                        "match_strength": match_strength,
                        "entity_type": entity.get("label", "unknown"),
                        "entity_confidence": entity_confidence,
                    }
                )

        # Sort by match strength
        matches.sort(key=lambda x: x["match_strength"], reverse=True)
        return matches

    def _find_relationship_matches(
        self, result_text: str, relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Find relationship matches in result text"""
        matches = []

        for relationship in relationships:
            subject = relationship.get("subject", "").lower().strip()
            relation = relationship.get("relation", "").lower().strip()
            obj = relationship.get("object", "").lower().strip()

            if not (subject and relation and obj):
                continue

            # Check if all relationship components are present
            subject_present = subject in result_text
            relation_present = relation in result_text
            object_present = obj in result_text

            # Calculate relationship match score
            components_present = sum(
                [subject_present, relation_present, object_present]
            )

            if components_present >= 2:  # At least 2 out of 3 components present
                match_strength = self._calculate_relationship_match_strength(
                    subject_present, relation_present, object_present, result_text
                )

                if match_strength >= self.relationship_confidence_threshold:
                    matches.append(
                        {
                            "relationship": relationship,
                            "subject_present": subject_present,
                            "relation_present": relation_present,
                            "object_present": object_present,
                            "match_strength": match_strength,
                            "components_matched": components_present,
                        }
                    )

        # Sort by match strength
        matches.sort(key=lambda x: x["match_strength"], reverse=True)
        return matches

    def _find_contextual_connections(
        self,
        result: Dict[str, Any],
        entity_matches: List[Dict[str, Any]],
        relationship_matches: List[Dict[str, Any]],
        context: EnhancementContext,
    ) -> List[Dict[str, Any]]:
        """Find contextual connections between entities and relationships"""
        connections = []

        # Connect entities through relationships
        for rel_match in relationship_matches:
            relationship = rel_match["relationship"]
            subject = relationship.get("subject", "").lower()
            obj = relationship.get("object", "").lower()

            # Find matching entities for this relationship
            connected_entities = []
            for entity_match in entity_matches:
                entity_text = entity_match["entity"].get("text", "").lower()
                if entity_text in [subject, obj]:
                    connected_entities.append(entity_match)

            if len(connected_entities) >= 2:
                # Strong connection - multiple entities from same relationship
                connection_strength = 0.8
            elif len(connected_entities) == 1:
                # Moderate connection - one entity from relationship
                connection_strength = 0.6
            else:
                continue

            if connection_strength >= self.connection_strength_threshold:
                connections.append(
                    {
                        "type": "entity_relationship_connection",
                        "relationship": relationship,
                        "connected_entities": connected_entities,
                        "strength": connection_strength,
                        "description": f"Entities connected through '{relationship.get('relation')}' relationship",
                    }
                )

        # Connect entities through co-occurrence
        if len(entity_matches) >= 2:
            result_text = self._extract_result_text(result)
            for i, entity_match_1 in enumerate(entity_matches):
                for entity_match_2 in entity_matches[i + 1 :]:
                    # Calculate co-occurrence strength
                    strength = self._calculate_cooccurrence_strength(
                        entity_match_1, entity_match_2, result_text
                    )

                    if strength >= self.connection_strength_threshold:
                        connections.append(
                            {
                                "type": "entity_cooccurrence",
                                "entities": [entity_match_1, entity_match_2],
                                "strength": strength,
                                "description": "Co-occurring entities in result context",
                            }
                        )

        # Sort by connection strength
        connections.sort(key=lambda x: x["strength"], reverse=True)
        return connections

    def _calculate_entity_match_strength(
        self, entity_text: str, result_text: str, entity_confidence: float
    ) -> float:
        """Calculate entity match strength"""
        base_strength = entity_confidence

        # Boost for exact word boundaries
        import re

        if re.search(rf"\b{re.escape(entity_text)}\b", result_text):
            base_strength += 0.1

        # Boost for multiple occurrences
        occurrences = result_text.count(entity_text)
        if occurrences > 1:
            base_strength += min(0.1 * (occurrences - 1), 0.2)

        return min(base_strength, 1.0)

    def _calculate_relationship_match_strength(
        self,
        subject_present: bool,
        relation_present: bool,
        object_present: bool,
        result_text: str,
    ) -> float:
        """Calculate relationship match strength"""
        components_present = sum([subject_present, relation_present, object_present])

        if components_present == 3:
            return 0.9  # All components present
        elif components_present == 2:
            return 0.7  # Two components present
        else:
            return 0.4  # One component present

    def _calculate_cooccurrence_strength(
        self,
        entity_match_1: Dict[str, Any],
        entity_match_2: Dict[str, Any],
        result_text: str,
    ) -> float:
        """Calculate co-occurrence strength between entities"""
        entity_1_text = entity_match_1["entity"].get("text", "").lower()
        entity_2_text = entity_match_2["entity"].get("text", "").lower()

        # Find positions of both entities
        import re

        entity_1_positions = [
            m.start() for m in re.finditer(re.escape(entity_1_text), result_text)
        ]
        entity_2_positions = [
            m.start() for m in re.finditer(re.escape(entity_2_text), result_text)
        ]

        if not entity_1_positions or not entity_2_positions:
            return 0.0

        # Calculate minimum distance between any occurrences
        min_distance = float("inf")
        for pos1 in entity_1_positions:
            for pos2 in entity_2_positions:
                distance = abs(pos1 - pos2)
                min_distance = min(min_distance, distance)

        # Convert distance to strength (closer = stronger)
        if min_distance <= 50:  # Very close
            return 0.8
        elif min_distance <= 100:  # Close
            return 0.6
        elif min_distance <= 200:  # Moderate
            return 0.4
        else:  # Distant
            return 0.2

    def _calculate_enhancement_score(
        self,
        entity_matches: List[Dict[str, Any]],
        relationship_matches: List[Dict[str, Any]],
        contextual_connections: List[Dict[str, Any]],
    ) -> float:
        """Calculate overall enhancement score"""

        # Entity contribution
        entity_score = min(len(entity_matches) * 0.2, 0.6)

        # Relationship contribution (weighted higher)
        relationship_score = min(len(relationship_matches) * 0.3, 0.8)

        # Connection contribution
        connection_score = min(len(contextual_connections) * 0.15, 0.4)

        # Combined score with diminishing returns
        total_score = entity_score + relationship_score + connection_score

        # Apply diminishing returns
        if total_score > 0.8:
            total_score = 0.8 + (total_score - 0.8) * 0.5

        return min(total_score, 1.0)

    def _create_fallback_enhanced_result(
        self, result: Dict[str, Any]
    ) -> EnhancedResult:
        """Create fallback enhanced result when enhancement fails"""
        return EnhancedResult(
            original_result=result,
            relevance_score=result.get("score", result.get("relevance", 0.0)),
            entity_matches=[],
            relationship_matches=[],
            contextual_connections=[],
            enhancement_score=0.0,
            enhancement_metadata={
                "enhancement_failed": True,
                "original_score": result.get("score", result.get("relevance", 0.0)),
                "boost_applied": 0.0,
            },
        )

    def get_enhancement_statistics(
        self, enhanced_results: List[EnhancedResult]
    ) -> Dict[str, Any]:
        """Get statistics about the enhancement process"""
        if not enhanced_results:
            return {"total_results": 0}

        total_results = len(enhanced_results)
        enhanced_count = sum(1 for r in enhanced_results if r.enhancement_score > 0)
        avg_enhancement_score = (
            sum(r.enhancement_score for r in enhanced_results) / total_results
        )
        avg_boost = (
            sum(
                r.enhancement_metadata.get("boost_applied", 0) for r in enhanced_results
            )
            / total_results
        )

        entity_matches = sum(len(r.entity_matches) for r in enhanced_results)
        relationship_matches = sum(
            len(r.relationship_matches) for r in enhanced_results
        )
        contextual_connections = sum(
            len(r.contextual_connections) for r in enhanced_results
        )

        return {
            "total_results": total_results,
            "enhanced_results": enhanced_count,
            "enhancement_rate": enhanced_count / total_results,
            "average_enhancement_score": round(avg_enhancement_score, 3),
            "average_boost_applied": round(avg_boost, 3),
            "total_entity_matches": entity_matches,
            "total_relationship_matches": relationship_matches,
            "total_contextual_connections": contextual_connections,
            "avg_entity_matches_per_result": round(entity_matches / total_results, 2),
            "avg_relationship_matches_per_result": round(
                relationship_matches / total_results, 2
            ),
            "avg_connections_per_result": round(
                contextual_connections / total_results, 2
            ),
        }
