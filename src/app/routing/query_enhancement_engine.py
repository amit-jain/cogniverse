"""
Query Enhancement Engine for DSPy 3.0 Routing System

This module implements sophisticated query enhancement using relationship tuples
to improve search quality. It takes extracted entities and relationships and
rewrites queries to increase the likelihood of better retrieval results.

Phase 6.2: Enhanced with SIMBA (Similarity-Based Memory Augmentation) for 
learning-based query enhancement using patterns from successful transformations.
"""

import logging
from typing import Any, Dict, List, Optional

import dspy

from .dspy_routing_signatures import QueryEnhancementSignature
from .relationship_extraction_tools import RelationshipExtractorTool
from .simba_query_enhancer import SIMBAConfig, SIMBAQueryEnhancer

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Advanced query rewriter that uses relationship tuples for enhancement.
    
    Transforms original queries by incorporating relationship context,
    semantic expansions, and domain-specific optimizations.
    """
    
    def __init__(self):
        """Initialize query rewriter with enhancement strategies."""
        self.enhancement_strategies = {
            "relationship_expansion": self._expand_with_relationships,
            "semantic_enrichment": self._add_semantic_context,
            "domain_specialization": self._apply_domain_knowledge,
            "boolean_optimization": self._optimize_boolean_logic,
            "synonym_expansion": self._expand_with_synonyms
        }
    
    def enhance_query(
        self,
        original_query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        search_context: str = "general"
    ) -> Dict[str, Any]:
        """
        Enhance query using relationship context and semantic expansion.
        
        Args:
            original_query: Original user query
            entities: Extracted entities with metadata
            relationships: Extracted relationship tuples
            search_context: Target search system context
            
        Returns:
            Enhancement results dictionary
        """
        try:
            # Apply all enhancement strategies
            enhancement_results = {}
            
            for strategy_name, strategy_func in self.enhancement_strategies.items():
                try:
                    result = strategy_func(
                        original_query, entities, relationships, search_context
                    )
                    enhancement_results[strategy_name] = result
                except Exception as e:
                    logger.warning(f"Enhancement strategy {strategy_name} failed: {e}")
                    enhancement_results[strategy_name] = {"enhanced_query": original_query, "terms": []}
            
            # Combine enhancements intelligently
            final_enhancement = self._combine_enhancements(
                original_query, enhancement_results, search_context
            )
            
            # Generate metadata
            enhancement_metadata = self._generate_enhancement_metadata(
                original_query, final_enhancement, entities, relationships
            )
            
            return {
                "enhanced_query": final_enhancement["query"],
                "semantic_expansions": final_enhancement["expansions"],
                "relationship_phrases": final_enhancement["relationship_phrases"],
                "enhancement_strategy": final_enhancement["strategy"],
                "search_operators": final_enhancement["operators"],
                "quality_score": enhancement_metadata["quality_score"],
                "enhancement_metadata": enhancement_metadata,
                "strategy_results": enhancement_results
            }
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return {
                "enhanced_query": original_query,
                "semantic_expansions": [],
                "relationship_phrases": [],
                "enhancement_strategy": "fallback",
                "search_operators": [],
                "quality_score": 0.1,
                "enhancement_metadata": {"error": str(e)},
                "strategy_results": {}
            }
    
    def _expand_with_relationships(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        context: str
    ) -> Dict[str, Any]:
        """
        Expand query using relationship tuples.
        
        Args:
            query: Original query
            entities: Extracted entities
            relationships: Relationship tuples
            context: Search context
            
        Returns:
            Enhancement result
        """
        if not relationships:
            return {"enhanced_query": query, "terms": []}
        
        # Sort relationships by confidence
        sorted_relationships = sorted(
            relationships, key=lambda r: r.get("confidence", 0), reverse=True
        )
        
        # Create relationship-based expansions
        relationship_terms = []
        relationship_phrases = []
        
        for rel in sorted_relationships[:3]:  # Top 3 relationships
            if rel.get("confidence", 0) > 0.6:
                # Create natural language phrase
                subject = rel["subject"]
                relation = rel["relation"].replace("_", " ")
                obj = rel["object"]
                
                phrase = f"{subject} {relation} {obj}"
                relationship_phrases.append(phrase)
                
                # Create search terms
                relationship_terms.extend([subject, obj])
                
                # Create relation-specific terms
                if relation in ["uses", "using"]:
                    relationship_terms.extend([f"{subject} with {obj}", f"{obj} in {subject}"])
                elif relation in ["plays", "playing"]:
                    relationship_terms.extend([f"{subject} {relation}", f"{relation} {obj}"])
                elif relation in ["shows", "showing", "demonstrates"]:
                    relationship_terms.extend([f"{obj} demonstration", f"{subject} example"])
        
        # Enhance query with relationship context
        if relationship_terms:
            # Add parenthetical expansions
            enhanced_query = f"{query} ({' OR '.join(relationship_phrases)})"
        else:
            enhanced_query = query
        
        return {
            "enhanced_query": enhanced_query,
            "terms": relationship_terms,
            "phrases": relationship_phrases
        }
    
    def _add_semantic_context(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        context: str
    ) -> Dict[str, Any]:
        """
        Add semantic context based on entity types and domain.
        
        Args:
            query: Original query
            entities: Extracted entities
            relationships: Relationship tuples
            context: Search context
            
        Returns:
            Enhancement result
        """
        semantic_terms = []
        
        # Get entity types for semantic expansion
        entity_types = [e.get("label", "") for e in entities]
        
        # Technology domain expansions
        if any(et in entity_types for et in ["TECHNOLOGY", "PRODUCT"]):
            tech_terms = []
            query_lower = query.lower()
            
            if "robot" in query_lower:
                tech_terms.extend(["robotics", "automation", "autonomous system"])
            if "ai" in query_lower or "artificial intelligence" in query_lower:
                tech_terms.extend(["machine learning", "neural network", "deep learning"])
            if "algorithm" in query_lower:
                tech_terms.extend(["computational method", "software algorithm", "AI technique"])
            
            semantic_terms.extend(tech_terms)
        
        # Sports domain expansions
        if any(et in entity_types for et in ["SPORT", "ACTIVITY"]):
            sports_terms = []
            query_lower = query.lower()
            
            if "soccer" in query_lower:
                sports_terms.extend(["football", "ball game", "team sport"])
            if "playing" in query_lower:
                sports_terms.extend(["game", "competition", "match"])
            
            semantic_terms.extend(sports_terms)
        
        # Action/Activity expansions
        if any(et in entity_types for et in ["ACTION", "ACTIVITY"]):
            action_terms = []
            query_lower = query.lower()
            
            if "demonstration" in query_lower:
                action_terms.extend(["showing", "example", "tutorial"])
            if "learning" in query_lower:
                action_terms.extend(["education", "training", "instruction"])
            
            semantic_terms.extend(action_terms)
        
        # Create enhanced query with semantic terms
        if semantic_terms:
            unique_terms = list(set(semantic_terms))[:5]  # Top 5 unique terms
            enhanced_query = f"{query} OR {' OR '.join(unique_terms)}"
        else:
            enhanced_query = query
        
        return {
            "enhanced_query": enhanced_query,
            "terms": semantic_terms
        }
    
    def _apply_domain_knowledge(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        context: str
    ) -> Dict[str, Any]:
        """
        Apply domain-specific knowledge for enhancement.
        
        Args:
            query: Original query
            entities: Extracted entities
            relationships: Relationship tuples
            context: Search context
            
        Returns:
            Enhancement result
        """
        domain_terms = []
        
        # Classify domain based on entities and query content
        domain = self._classify_query_domain(query, entities)
        
        # Domain-specific enhancements
        if domain == "artificial_intelligence":
            ai_terms = [
                "machine learning", "neural networks", "computer vision",
                "natural language processing", "deep learning", "AI research"
            ]
            domain_terms.extend(ai_terms)
            
        elif domain == "robotics":
            robotics_terms = [
                "autonomous robots", "robotic systems", "robot control",
                "mechanical engineering", "automation", "human-robot interaction"
            ]
            domain_terms.extend(robotics_terms)
            
        elif domain == "sports_technology":
            sports_tech_terms = [
                "sports analytics", "performance analysis", "athletic technology",
                "sports science", "biomechanics", "sports engineering"
            ]
            domain_terms.extend(sports_tech_terms)
            
        elif domain == "education":
            education_terms = [
                "learning", "tutorial", "educational content",
                "instruction", "demonstration", "training materials"
            ]
            domain_terms.extend(education_terms)
        
        # Create domain-enhanced query
        if domain_terms:
            # Select most relevant terms (up to 3)
            relevant_terms = domain_terms[:3]
            enhanced_query = f"{query} ({' OR '.join(relevant_terms)})"
        else:
            enhanced_query = query
        
        return {
            "enhanced_query": enhanced_query,
            "terms": domain_terms,
            "domain": domain
        }
    
    def _optimize_boolean_logic(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        context: str
    ) -> Dict[str, Any]:
        """
        Optimize boolean search logic based on entity relationships.
        
        Args:
            query: Original query
            entities: Extracted entities
            relationships: Relationship tuples
            context: Search context
            
        Returns:
            Enhancement result
        """
        operators = []
        optimized_terms = []
        
        # Identify key entities for AND grouping
        high_confidence_entities = [
            e["text"] for e in entities if e.get("confidence", 0) > 0.8
        ]
        
        if len(high_confidence_entities) >= 2:
            # Use AND for high-confidence entities
            operators.append("AND")
            and_group = " AND ".join(high_confidence_entities[:3])
            optimized_terms.append(f"({and_group})")
        
        # Identify related concepts for OR grouping
        relationship_groups = {}
        for rel in relationships:
            if rel.get("confidence", 0) > 0.6:
                relation_type = rel["relation"]
                if relation_type not in relationship_groups:
                    relationship_groups[relation_type] = []
                relationship_groups[relation_type].extend([rel["subject"], rel["object"]])
        
        # Create OR groups for related concepts
        for relation_type, related_entities in relationship_groups.items():
            if len(related_entities) >= 2:
                operators.append("OR")
                unique_entities = list(set(related_entities))[:3]
                or_group = " OR ".join(unique_entities)
                optimized_terms.append(f"({or_group})")
        
        # Construct optimized query
        if optimized_terms:
            enhanced_query = f"{query} {' '.join(optimized_terms)}"
        else:
            enhanced_query = query
        
        return {
            "enhanced_query": enhanced_query,
            "terms": optimized_terms,
            "operators": operators
        }
    
    def _expand_with_synonyms(
        self,
        query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        context: str
    ) -> Dict[str, Any]:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query
            entities: Extracted entities
            relationships: Relationship tuples
            context: Search context
            
        Returns:
            Enhancement result
        """
        synonym_map = {
            # Technology synonyms
            "robot": ["robotics", "automated system", "mechanical agent"],
            "ai": ["artificial intelligence", "machine intelligence", "AI system"],
            "algorithm": ["method", "procedure", "computational technique"],
            "machine learning": ["ML", "statistical learning", "pattern recognition"],
            
            # Sports synonyms
            "soccer": ["football", "association football"],
            "playing": ["competing", "participating", "engaging in"],
            "game": ["match", "competition", "contest"],
            
            # Action synonyms
            "showing": ["demonstrating", "displaying", "exhibiting"],
            "using": ["utilizing", "employing", "applying"],
            "learning": ["acquiring", "studying", "mastering"],
        }
        
        synonym_terms = []
        query_words = query.lower().split()
        
        # Find synonyms for words in query
        for word in query_words:
            if word in synonym_map:
                synonyms = synonym_map[word][:2]  # Top 2 synonyms
                synonym_terms.extend(synonyms)
        
        # Add synonyms for high-confidence entities
        for entity in entities:
            if entity.get("confidence", 0) > 0.9:
                entity_text = entity["text"].lower()
                if entity_text in synonym_map:
                    synonyms = synonym_map[entity_text][:2]
                    synonym_terms.extend(synonyms)
        
        # Create synonym-enhanced query
        if synonym_terms:
            unique_synonyms = list(set(synonym_terms))[:4]  # Top 4 unique synonyms
            enhanced_query = f"{query} OR {' OR '.join(unique_synonyms)}"
        else:
            enhanced_query = query
        
        return {
            "enhanced_query": enhanced_query,
            "terms": synonym_terms
        }
    
    def _combine_enhancements(
        self,
        original_query: str,
        strategy_results: Dict[str, Dict[str, Any]],
        context: str
    ) -> Dict[str, Any]:
        """
        Combine results from multiple enhancement strategies intelligently.
        
        Args:
            original_query: Original query
            strategy_results: Results from each strategy
            context: Search context
            
        Returns:
            Combined enhancement result
        """
        # Collect all expansion terms
        all_expansions = []
        all_phrases = []
        all_operators = []
        
        for strategy, result in strategy_results.items():
            all_expansions.extend(result.get("terms", []))
            all_phrases.extend(result.get("phrases", []))
            all_operators.extend(result.get("operators", []))
        
        # Remove duplicates while preserving order
        unique_expansions = []
        seen = set()
        for term in all_expansions:
            if term not in seen and term not in original_query:
                unique_expansions.append(term)
                seen.add(term)
        
        # Limit expansions to prevent query explosion
        final_expansions = unique_expansions[:5]
        final_phrases = list(set(all_phrases))[:3]
        final_operators = list(set(all_operators))
        
        # Create final enhanced query
        query_parts = [original_query]
        
        # Add relationship phrases with OR
        if final_phrases:
            phrases_part = f"({' OR '.join(final_phrases)})"
            query_parts.append(phrases_part)
        
        # Add semantic expansions
        if final_expansions:
            expansions_part = f"({' OR '.join(final_expansions)})"
            query_parts.append(expansions_part)
        
        # Combine with appropriate logic
        if len(query_parts) > 1:
            enhanced_query = " ".join(query_parts)
        else:
            enhanced_query = original_query
        
        # Determine primary strategy used
        primary_strategy = "combined"
        if strategy_results.get("relationship_expansion", {}).get("terms"):
            primary_strategy = "relationship_expansion"
        elif strategy_results.get("semantic_enrichment", {}).get("terms"):
            primary_strategy = "semantic_enrichment"
        elif strategy_results.get("domain_specialization", {}).get("terms"):
            primary_strategy = "domain_specialization"
        
        return {
            "query": enhanced_query,
            "expansions": final_expansions,
            "relationship_phrases": final_phrases,
            "operators": final_operators,
            "strategy": primary_strategy
        }
    
    def _generate_enhancement_metadata(
        self,
        original_query: str,
        enhancement_result: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate metadata about the enhancement process.
        
        Args:
            original_query: Original query
            enhancement_result: Final enhancement result
            entities: Extracted entities
            relationships: Extracted relationships
            
        Returns:
            Enhancement metadata
        """
        enhanced_query = enhancement_result["query"]
        
        # Calculate quality score
        quality_factors = {
            "length_increase": min(1.0, len(enhanced_query) / len(original_query) - 1),
            "entity_coverage": len(entities) / 10.0,  # Normalize by expected max
            "relationship_richness": len(relationships) / 5.0,  # Normalize by expected max
            "expansion_diversity": len(enhancement_result["expansions"]) / 5.0,
            "phrase_quality": len(enhancement_result["relationship_phrases"]) / 3.0
        }
        
        # Weighted quality score
        weights = {
            "length_increase": 0.2,
            "entity_coverage": 0.25,
            "relationship_richness": 0.25,
            "expansion_diversity": 0.15,
            "phrase_quality": 0.15
        }
        
        quality_score = sum(
            min(1.0, quality_factors[factor]) * weights[factor]
            for factor in quality_factors
        )
        
        return {
            "quality_score": round(quality_score, 3),
            "enhancement_ratio": len(enhanced_query) / len(original_query),
            "entities_used": len(entities),
            "relationships_used": len(relationships),
            "expansions_added": len(enhancement_result["expansions"]),
            "phrases_added": len(enhancement_result["relationship_phrases"]),
            "primary_strategy": enhancement_result["strategy"],
            "complexity_increase": self._calculate_complexity_increase(
                original_query, enhanced_query
            )
        }
    
    def _classify_query_domain(
        self, 
        query: str, 
        entities: List[Dict[str, Any]]
    ) -> str:
        """
        Classify query domain for targeted enhancement.
        
        Args:
            query: Query text
            entities: Extracted entities
            
        Returns:
            Domain classification
        """
        query_lower = query.lower()
        entity_types = [e.get("label", "") for e in entities]
        
        # AI/Technology domain
        ai_keywords = ["ai", "artificial intelligence", "machine learning", "neural", "algorithm"]
        if any(kw in query_lower for kw in ai_keywords) or "TECHNOLOGY" in entity_types:
            return "artificial_intelligence"
        
        # Robotics domain
        robot_keywords = ["robot", "robotic", "automation", "autonomous"]
        if any(kw in query_lower for kw in robot_keywords):
            return "robotics"
        
        # Sports + Technology
        if ("SPORT" in entity_types or "sport" in query_lower) and "TECHNOLOGY" in entity_types:
            return "sports_technology"
        
        # Sports domain
        sports_keywords = ["sport", "game", "play", "competition", "soccer", "football"]
        if any(kw in query_lower for kw in sports_keywords) or "SPORT" in entity_types:
            return "sports"
        
        # Education domain
        edu_keywords = ["learn", "teach", "tutorial", "education", "instruction"]
        if any(kw in query_lower for kw in edu_keywords):
            return "education"
        
        return "general"
    
    def _calculate_complexity_increase(
        self, 
        original: str, 
        enhanced: str
    ) -> float:
        """
        Calculate complexity increase from enhancement.
        
        Args:
            original: Original query
            enhanced: Enhanced query
            
        Returns:
            Complexity increase ratio
        """
        # Simple complexity based on length and operators
        original_complexity = len(original.split()) + original.count("(") + original.count("OR") + original.count("AND")
        enhanced_complexity = len(enhanced.split()) + enhanced.count("(") + enhanced.count("OR") + enhanced.count("AND")
        
        if original_complexity == 0:
            return 0.0
        
        return (enhanced_complexity - original_complexity) / original_complexity


class DSPyQueryEnhancerModule(dspy.Module):
    """
    DSPy 3.0 module for query enhancement using QueryEnhancementSignature.
    
    Integrates query rewriting with DSPy's structured output generation
    for consistent and optimized query enhancement.
    """
    
    def __init__(self):
        super().__init__()
        self.enhancer = dspy.ChainOfThought(QueryEnhancementSignature)
        self.rewriter = QueryRewriter()
    
    def forward(
        self,
        original_query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        search_context: str = "general"
    ) -> dspy.Prediction:
        """
        Enhance query using DSPy 3.0 + relationship-aware rewriting.
        
        Args:
            original_query: Original user query
            entities: Extracted entities with metadata
            relationships: Extracted relationship tuples
            search_context: Target search system context
            
        Returns:
            DSPy prediction with enhancement results
        """
        try:
            # Use query rewriter for actual enhancement
            enhancement_result = self.rewriter.enhance_query(
                original_query, entities, relationships, search_context
            )
            
            # Create DSPy prediction with results
            prediction = dspy.Prediction()
            prediction.enhanced_query = enhancement_result["enhanced_query"]
            prediction.semantic_expansions = enhancement_result["semantic_expansions"]
            prediction.relationship_phrases = enhancement_result["relationship_phrases"]
            prediction.enhancement_strategy = enhancement_result["enhancement_strategy"]
            prediction.search_operators = enhancement_result["search_operators"]
            prediction.quality_score = enhancement_result["quality_score"]
            
            return prediction
            
        except Exception as e:
            logger.error(f"DSPy query enhancement failed: {e}")
            
            # Return fallback prediction
            prediction = dspy.Prediction()
            prediction.enhanced_query = original_query
            prediction.semantic_expansions = []
            prediction.relationship_phrases = []
            prediction.enhancement_strategy = "error_fallback"
            prediction.search_operators = []
            prediction.quality_score = 0.1
            
            return prediction


class QueryEnhancementPipeline:
    """
    Complete query enhancement pipeline that integrates relationship extraction
    with advanced query rewriting for optimal search performance.
    
    Phase 6.2: Enhanced with SIMBA for learning-based query enhancement.
    """
    
    def __init__(self, enable_simba: bool = True, simba_config: Optional[SIMBAConfig] = None):
        """Initialize the enhancement pipeline."""
        self.relationship_tool = RelationshipExtractorTool()
        self.dspy_enhancer = DSPyQueryEnhancerModule()
        
        # Phase 6.2: Initialize SIMBA enhancer
        self.enable_simba = enable_simba
        if enable_simba:
            self.simba_enhancer = SIMBAQueryEnhancer(
                config=simba_config or SIMBAConfig(),
                storage_dir="data/enhancement"
            )
            logger.info("Query enhancement pipeline initialized with SIMBA")
        else:
            self.simba_enhancer = None
            logger.info("Query enhancement pipeline initialized without SIMBA")
    
    async def enhance_query_with_relationships(
        self,
        query: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
        search_context: str = "general",
        entity_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Complete end-to-end query enhancement with relationship extraction and SIMBA learning.
        
        Args:
            query: Original query to enhance
            entities: Pre-extracted entities (optional, will extract if not provided)
            relationships: Pre-extracted relationships (optional, will extract if not provided) 
            search_context: Target search system context
            entity_labels: Optional entity labels for extraction
            
        Returns:
            Complete enhancement results
        """
        try:
            # Step 1: Extract entities and relationships if not provided
            if entities is None or relationships is None:
                relationship_data = await self.relationship_tool.extract_comprehensive_relationships(
                    query, entity_labels
                )
                entities = relationship_data["entities"]
                relationships = relationship_data["relationships"]
                relationship_types = relationship_data["relationship_types"]
                semantic_connections = relationship_data["semantic_connections"]
                complexity_indicators = relationship_data.get("complexity_indicators", [])
            else:
                # Use provided entities and relationships
                relationship_types = list(set([r.get("relation", "") for r in relationships if r.get("relation")]))
                semantic_connections = []
                complexity_indicators = []
            
            # Step 2: Try SIMBA enhancement first (Phase 6.2)
            simba_result = None
            if self.enable_simba and self.simba_enhancer:
                simba_result = await self.simba_enhancer.enhance_query_with_patterns(
                    original_query=query,
                    entities=entities,
                    relationships=relationships,
                    context=search_context
                )
            
            # Step 3: Use SIMBA result or fallback to DSPy baseline
            if simba_result and simba_result.get("enhanced", False):
                # Use SIMBA enhancement
                enhanced_query = simba_result["enhanced_query"]
                enhancement_strategy = simba_result["enhancement_strategy"]
                quality_score = simba_result["confidence"]
                
                # Create compatible structure for legacy code
                enhancement_prediction_data = {
                    "enhanced_query": enhanced_query,
                    "semantic_expansions": [],
                    "relationship_phrases": [],
                    "enhancement_strategy": enhancement_strategy,
                    "search_operators": [],
                    "quality_score": quality_score
                }
                
                logger.info(
                    f"SIMBA enhanced query '{query[:50]}...' -> "
                    f"'{enhanced_query[:50]}...' "
                    f"(patterns: {simba_result.get('similar_patterns_used', 0)})"
                )
                
            else:
                # Fallback to DSPy baseline enhancement
                enhancement_prediction = self.dspy_enhancer.forward(
                    query, entities, relationships, search_context
                )
                
                enhancement_prediction_data = {
                    "enhanced_query": enhancement_prediction.enhanced_query,
                    "semantic_expansions": enhancement_prediction.semantic_expansions,
                    "relationship_phrases": enhancement_prediction.relationship_phrases,
                    "enhancement_strategy": enhancement_prediction.enhancement_strategy,
                    "search_operators": enhancement_prediction.search_operators,
                    "quality_score": enhancement_prediction.quality_score
                }
                
                logger.info(
                    f"DSPy enhanced query '{query[:50]}...' -> "
                    f"'{enhancement_prediction.enhanced_query[:50]}...' "
                    f"(quality: {enhancement_prediction.quality_score})"
                )
            
            # Step 4: Combine results
            complete_result = {
                # Original extraction data
                "original_query": query,
                "extracted_entities": entities,
                "extracted_relationships": relationships,
                "relationship_types": relationship_types,
                "semantic_connections": semantic_connections,
                
                # Enhancement results
                "enhanced_query": enhancement_prediction_data["enhanced_query"],
                "semantic_expansions": enhancement_prediction_data["semantic_expansions"],
                "relationship_phrases": enhancement_prediction_data["relationship_phrases"],
                "enhancement_strategy": enhancement_prediction_data["enhancement_strategy"],
                "search_operators": enhancement_prediction_data["search_operators"],
                "quality_score": enhancement_prediction_data["quality_score"],
                
                # SIMBA metadata
                "simba_applied": simba_result is not None and simba_result.get("enhanced", False),
                "simba_patterns_used": simba_result.get("similar_patterns_used", 0) if simba_result else 0,
                "pattern_avg_improvement": simba_result.get("pattern_avg_improvement", 0.0) if simba_result else 0.0,
                
                # Metadata
                "search_context": search_context,
                "processing_metadata": {
                    "entities_found": len(entities),
                    "relationships_found": len(relationships),
                    "enhancement_quality": enhancement_prediction_data["quality_score"],
                    "query_complexity": complexity_indicators,
                    "enhancement_method": "simba" if simba_result and simba_result.get("enhanced", False) else "dspy_baseline"
                }
            }
            
            return complete_result
            
        except Exception as e:
            logger.error(f"Query enhancement pipeline failed: {e}")
            
            # Return minimal fallback result
            return {
                "original_query": query,
                "extracted_entities": [],
                "extracted_relationships": [],
                "relationship_types": [],
                "semantic_connections": [],
                "enhanced_query": query,
                "semantic_expansions": [],
                "relationship_phrases": [],
                "enhancement_strategy": "pipeline_error",
                "search_operators": [],
                "quality_score": 0.0,
                "search_context": search_context,
                "processing_metadata": {
                    "error": str(e),
                    "fallback_used": True
                }
            }
    
    async def record_enhancement_outcome(
        self,
        original_query: str,
        enhanced_query: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        enhancement_strategy: str,
        search_quality_improvement: float,
        routing_confidence_improvement: float,
        user_satisfaction: Optional[float] = None
    ) -> None:
        """
        Record outcome of query enhancement for SIMBA learning
        
        Args:
            original_query: Original query
            enhanced_query: Enhanced query
            entities: Entities from query
            relationships: Relationships from query
            enhancement_strategy: Strategy used for enhancement
            search_quality_improvement: Improvement in search quality (0-1)
            routing_confidence_improvement: Improvement in routing confidence  
            user_satisfaction: Optional user feedback (0-1)
        """
        if not self.enable_simba or not self.simba_enhancer:
            return
        
        try:
            await self.simba_enhancer.record_enhancement_outcome(
                original_query=original_query,
                enhanced_query=enhanced_query,
                entities=entities,
                relationships=relationships,
                enhancement_strategy=enhancement_strategy,
                search_quality_improvement=search_quality_improvement,
                routing_confidence_improvement=routing_confidence_improvement,
                user_satisfaction=user_satisfaction
            )
            
            logger.debug(f"Recorded enhancement outcome for SIMBA learning: improvement={search_quality_improvement:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to record enhancement outcome: {e}")
    
    def get_simba_status(self) -> Dict[str, Any]:
        """Get SIMBA enhancement status and metrics"""
        if not self.enable_simba or not self.simba_enhancer:
            return {"simba_enabled": False, "reason": "SIMBA disabled or not initialized"}
        
        try:
            return self.simba_enhancer.get_enhancement_status()
            
        except Exception as e:
            return {
                "simba_enabled": True,
                "error": str(e),
                "status": "error"
            }
    
    async def reset_simba_memory(self) -> bool:
        """Reset SIMBA memory (useful for testing)"""
        if not self.enable_simba or not self.simba_enhancer:
            return False
        
        try:
            await self.simba_enhancer.reset_memory()
            logger.info("SIMBA enhancement memory reset")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset SIMBA memory: {e}")
            return False


# Factory functions

def create_query_rewriter() -> QueryRewriter:
    """Create query rewriter instance."""
    return QueryRewriter()


def create_dspy_query_enhancer() -> DSPyQueryEnhancerModule:
    """Create DSPy query enhancer module."""
    return DSPyQueryEnhancerModule()


def create_enhancement_pipeline(enable_simba: bool = True, simba_config: Optional[SIMBAConfig] = None) -> QueryEnhancementPipeline:
    """Create complete query enhancement pipeline with optional SIMBA integration."""
    return QueryEnhancementPipeline(enable_simba=enable_simba, simba_config=simba_config)
