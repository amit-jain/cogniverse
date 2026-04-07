"""
Strategy Learner — distills execution traces into reusable agent strategies.

Two distillation paths:
1. Pattern extraction: statistical analysis of which profiles/strategies/parameters
   scored best for different query types. No LLM needed.
2. LLM distillation: contrastive analysis of high-scoring vs low-scoring trace pairs
   to identify workflow-level insights. Uses DSPy.

Strategies are stored in Vespa memory via Mem0MemoryManager with type=strategy
metadata, enabling runtime retrieval by agents via MemoryAwareMixin.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

STRATEGY_AGENT_NAME = "_strategy_store"
MIN_TRACES_FOR_PATTERN = 5
DEDUP_SIMILARITY_THRESHOLD = 0.9


@dataclass
class Strategy:
    """A learned strategy distilled from execution traces."""

    text: str
    applies_when: str
    agent: str
    level: str  # "org" | "user"
    confidence: float
    source: str  # "pattern_extraction" | "llm_distillation"
    tenant_id: str
    trace_count: int
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_memory_content(self) -> str:
        """Format as a clear preference statement for Mem0 LLM extraction.

        Mem0's LLM extracts memories from conversational content.
        Framing strategies as user preferences ensures they get stored.
        """
        return (
            f"I prefer the following approach for {self.agent}: {self.text} "
            f"I use this when {self.applies_when}."
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Metadata tags for filtering during retrieval."""
        return {
            "type": "strategy",
            "level": self.level,
            "agent": self.agent,
            "confidence": self.confidence,
            "source": self.source,
            "trace_count": self.trace_count,
            "applies_when": self.applies_when,
            "created_at": self.created_at,
        }


class StrategyLearner:
    """Distills execution traces into reusable strategies.

    Input: Phoenix trigger dataset (scored trace examples from QualityMonitor).
    Output: Strategy objects stored in Vespa memory.
    """

    def __init__(
        self,
        memory_manager,
        tenant_id: str,
        org_id: Optional[str] = None,
        llm_config=None,
    ):
        """
        Args:
            memory_manager: Mem0MemoryManager instance for strategy storage.
            tenant_id: User-level tenant ID for user strategies.
            org_id: Org-level ID for shared strategies. Extracted from tenant_id
                    if not provided (e.g., "acme:alice" → org_id="acme").
            llm_config: LLMEndpointConfig for LLM distillation. If None,
                        only pattern extraction runs.
        """
        self.memory_manager = memory_manager
        self.tenant_id = tenant_id
        self.llm_config = llm_config

        if org_id:
            self.org_id = org_id
        elif ":" in tenant_id:
            from cogniverse_core.common.tenant_utils import parse_tenant_id

            self.org_id, _ = parse_tenant_id(tenant_id)
        else:
            self.org_id = tenant_id

    async def learn_from_trigger_dataset(
        self, trigger_df: pd.DataFrame
    ) -> List[Strategy]:
        """Run both distillation paths on a trigger dataset.

        Args:
            trigger_df: DataFrame from Phoenix trigger dataset with columns:
                agent, category (low_scoring/high_scoring), query, score, output

        Returns:
            List of distilled strategies (already stored in memory).
        """
        all_strategies = []

        # Path A: Pattern extraction (statistical, no LLM)
        pattern_strategies = self._extract_patterns(trigger_df)
        all_strategies.extend(pattern_strategies)
        logger.info(f"Pattern extraction produced {len(pattern_strategies)} strategies")

        # Path B: LLM distillation (contrastive, needs LLM)
        if self.llm_config:
            llm_strategies = await self._distill_with_llm(trigger_df)
            all_strategies.extend(llm_strategies)
            logger.info(f"LLM distillation produced {len(llm_strategies)} strategies")

        # Store all strategies in Vespa memory
        stored_count = 0
        for strategy in all_strategies:
            stored = self._store_strategy(strategy)
            if stored:
                stored_count += 1

        logger.info(
            f"Stored {stored_count}/{len(all_strategies)} strategies "
            f"(deduplication may have reduced count)"
        )

        return all_strategies

    def _extract_patterns(self, trigger_df: pd.DataFrame) -> List[Strategy]:
        """Path A: Extract statistical patterns from scored traces.

        Groups traces by agent and analyzes score distributions to identify
        what works and what doesn't for different query types.
        """
        strategies = []

        for agent_name in trigger_df["agent"].unique():
            agent_df = trigger_df[trigger_df["agent"] == agent_name]
            low = agent_df[agent_df["category"] == "low_scoring"]
            high = agent_df[agent_df["category"] == "high_scoring"]

            total = len(agent_df)
            if total < MIN_TRACES_FOR_PATTERN:
                continue

            # Strategy: overall quality assessment
            if len(high) > 0 and len(low) > 0:
                high_avg = high["score"].astype(float).mean()
                low_avg = low["score"].astype(float).mean()
                score_delta = high_avg - low_avg

                if score_delta > 0.3:
                    strategies.append(
                        self._pattern_strategy(
                            agent=agent_name,
                            text=(
                                f"High-scoring {agent_name} queries average "
                                f"{high_avg:.2f} while low-scoring average "
                                f"{low_avg:.2f}. Focus on improving queries "
                                f"similar to the low-scoring patterns."
                            ),
                            applies_when=f"Processing {agent_name} requests",
                            confidence=min(0.9, total / 50),
                            trace_count=total,
                        )
                    )

            # Strategy: identify query patterns in high vs low scoring
            high_queries = high["query"].tolist() if len(high) > 0 else []
            low_queries = low["query"].tolist() if len(low) > 0 else []

            keyword_strategies = self._extract_keyword_patterns(
                agent_name, high_queries, low_queries
            )
            strategies.extend(keyword_strategies)

            # Strategy: extract output patterns from high-scoring traces
            if len(high) >= MIN_TRACES_FOR_PATTERN:
                output_strategies = self._extract_output_patterns(agent_name, high)
                strategies.extend(output_strategies)

        return strategies

    def _extract_keyword_patterns(
        self,
        agent_name: str,
        high_queries: List[str],
        low_queries: List[str],
    ) -> List[Strategy]:
        """Identify keywords that correlate with high/low scores."""
        strategies = []

        keyword_categories = {
            "temporal": ["when", "before", "after", "during", "timeline", "sequence"],
            "object": ["what", "show", "find", "locate", "identify"],
            "action": ["how", "doing", "performing", "demonstrates"],
            "comparison": ["compare", "difference", "versus", "better", "similar"],
        }

        for category, keywords in keyword_categories.items():
            high_count = sum(
                1 for q in high_queries if any(kw in q.lower() for kw in keywords)
            )
            low_count = sum(
                1 for q in low_queries if any(kw in q.lower() for kw in keywords)
            )

            total = high_count + low_count
            if total < 3:
                continue

            high_ratio = high_count / total if total > 0 else 0

            if high_ratio > 0.7:
                strategies.append(
                    self._pattern_strategy(
                        agent=agent_name,
                        text=(
                            f"{category.title()} queries ({', '.join(keywords[:3])}) "
                            f"perform well with current {agent_name} configuration "
                            f"({high_ratio:.0%} score above threshold)."
                        ),
                        applies_when=f"Query contains {category} keywords",
                        confidence=min(0.85, total / 20),
                        trace_count=total,
                    )
                )
            elif high_ratio < 0.3 and total >= 5:
                strategies.append(
                    self._pattern_strategy(
                        agent=agent_name,
                        text=(
                            f"{category.title()} queries ({', '.join(keywords[:3])}) "
                            f"consistently score poorly ({high_ratio:.0%} above threshold). "
                            f"Consider alternative search strategy or profile for "
                            f"these query types."
                        ),
                        applies_when=f"Query contains {category} keywords",
                        confidence=min(0.85, total / 20),
                        trace_count=total,
                    )
                )

        return strategies

    def _extract_output_patterns(
        self, agent_name: str, high_df: pd.DataFrame
    ) -> List[Strategy]:
        """Extract patterns from high-scoring outputs (what worked)."""
        strategies = []

        if agent_name == "search":
            # Analyze result counts in high-scoring searches
            result_counts = []
            for _, row in high_df.iterrows():
                output = row.get("output", "{}")
                if isinstance(output, str):
                    try:
                        output = json.loads(output)
                    except Exception:
                        continue
                results = output.get("results", [])
                if isinstance(results, list):
                    result_counts.append(len(results))

            if result_counts and len(result_counts) >= MIN_TRACES_FOR_PATTERN:
                avg_count = sum(result_counts) / len(result_counts)
                strategies.append(
                    self._pattern_strategy(
                        agent=agent_name,
                        text=(
                            f"High-scoring searches return an average of "
                            f"{avg_count:.0f} results. Adjust top_k accordingly."
                        ),
                        applies_when="Configuring search result count",
                        confidence=0.7,
                        trace_count=len(result_counts),
                    )
                )

        return strategies

    async def _distill_with_llm(self, trigger_df: pd.DataFrame) -> List[Strategy]:
        """Path B: Contrastive LLM distillation from trace pairs.

        Pairs high-scoring and low-scoring traces for similar query types,
        feeds them to an LLM to identify what made the difference.
        """
        strategies = []

        for agent_name in trigger_df["agent"].unique():
            agent_df = trigger_df[trigger_df["agent"] == agent_name]
            low = agent_df[agent_df["category"] == "low_scoring"]
            high = agent_df[agent_df["category"] == "high_scoring"]

            if low.empty or high.empty:
                continue

            # Create contrastive pairs (up to 5 pairs per agent)
            pairs = []
            for _, low_row in low.head(5).iterrows():
                # Find the most similar high-scoring trace
                best_high = high.iloc[0] if len(high) > 0 else None
                if best_high is not None:
                    pairs.append((low_row, best_high))

            if not pairs:
                continue

            for low_trace, high_trace in pairs:
                strategy = await self._distill_single_pair(
                    agent_name, low_trace, high_trace
                )
                if strategy:
                    strategies.append(strategy)

        return strategies

    async def _distill_single_pair(
        self,
        agent_name: str,
        low_trace: pd.Series,
        high_trace: pd.Series,
    ) -> Optional[Strategy]:
        """Use LLM to analyze a single contrastive trace pair."""
        try:
            import dspy

            from cogniverse_foundation.config.llm_factory import create_dspy_lm

            lm = create_dspy_lm(self.llm_config)

            class ContrastiveStrategyDistillation(dspy.Signature):
                """Analyze why one execution succeeded and another failed.
                Produce a concise, actionable strategy."""

                high_scoring_query = dspy.InputField(
                    desc="Query that produced good results"
                )
                high_scoring_output = dspy.InputField(
                    desc="Output from the successful execution"
                )
                high_score = dspy.InputField(desc="Quality score (0-1)")
                low_scoring_query = dspy.InputField(
                    desc="Query that produced poor results"
                )
                low_scoring_output = dspy.InputField(
                    desc="Output from the failed execution"
                )
                low_score = dspy.InputField(desc="Quality score (0-1)")
                agent_name = dspy.InputField(desc="Agent that processed both")

                strategy_text = dspy.OutputField(
                    desc="Concise actionable strategy (1-2 sentences)"
                )
                applies_when = dspy.OutputField(
                    desc="When this strategy should be applied (short condition)"
                )

            predictor = dspy.Predict(ContrastiveStrategyDistillation)

            high_output = high_trace.get("output", "{}")
            low_output = low_trace.get("output", "{}")
            if isinstance(high_output, str) and len(high_output) > 500:
                high_output = high_output[:500]
            if isinstance(low_output, str) and len(low_output) > 500:
                low_output = low_output[:500]

            with dspy.context(lm=lm):
                result = predictor(
                    high_scoring_query=str(high_trace.get("query", "")),
                    high_scoring_output=str(high_output),
                    high_score=str(high_trace.get("score", 0)),
                    low_scoring_query=str(low_trace.get("query", "")),
                    low_scoring_output=str(low_output),
                    low_score=str(low_trace.get("score", 0)),
                    agent_name=agent_name,
                )

            strategy_text = result.strategy_text
            applies_when = result.applies_when

            if not strategy_text or len(strategy_text) < 10:
                return None

            return Strategy(
                text=strategy_text,
                applies_when=applies_when,
                agent=agent_name,
                level="org",
                confidence=0.6,
                source="llm_distillation",
                tenant_id=self.org_id,
                trace_count=2,
            )

        except Exception as e:
            logger.warning(f"LLM distillation failed for {agent_name}: {e}")
            return None

    def _pattern_strategy(
        self,
        agent: str,
        text: str,
        applies_when: str,
        confidence: float,
        trace_count: int,
    ) -> Strategy:
        """Create an org-level strategy from pattern extraction."""
        return Strategy(
            text=text,
            applies_when=applies_when,
            agent=agent,
            level="org",
            confidence=confidence,
            source="pattern_extraction",
            tenant_id=self.org_id,
            trace_count=trace_count,
        )

    def _store_strategy(self, strategy: Strategy) -> bool:
        """Store strategy in Vespa memory, deduplicating against existing ones."""
        if not self.memory_manager or self.memory_manager.memory is None:
            logger.warning("Memory manager not initialized, skipping strategy storage")
            return False

        # Use an agent-specific Mem0 namespace so Vespa's agent_id field isolates
        # strategies at the storage layer. All strategies share the STRATEGY_AGENT_NAME
        # prefix so they're easy to identify, but each agent gets its own suffix.
        agent_namespace = f"{STRATEGY_AGENT_NAME}_{strategy.agent}"

        # Deduplication: search for similar existing strategies
        try:
            existing = self.memory_manager.search_memory(
                query=strategy.to_memory_content(),
                tenant_id=strategy.tenant_id,
                agent_name=agent_namespace,
                top_k=3,
            )

            for mem in existing:
                mem_text = mem.get("memory", "")
                if "strategy" in mem_text.lower():
                    overlap = _text_overlap(
                        strategy.to_memory_content(), mem_text
                    )
                    if overlap > DEDUP_SIMILARITY_THRESHOLD:
                        logger.debug(
                            f"Strategy dedup: '{strategy.text[:50]}...' overlaps "
                            f"with existing ({overlap:.2f}), skipping"
                        )
                        return False
        except Exception as e:
            logger.debug(f"Dedup search failed (non-fatal): {e}")

        # Store the strategy under the agent-specific namespace
        self.memory_manager.add_memory(
            content=strategy.to_memory_content(),
            tenant_id=strategy.tenant_id,
            agent_name=agent_namespace,
            metadata=strategy.to_metadata(),
        )
        logger.info(
            f"Stored {strategy.level}-level {strategy.source} strategy "
            f"for {strategy.agent}: '{strategy.text[:60]}...'"
        )
        return True

    def get_strategies_for_agent(
        self,
        query: str,
        agent_name: Optional[str],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant strategies for an agent and query.

        Two-level retrieval: user strategies + org strategies. Strategies
        are stored in a single shared namespace (``STRATEGY_AGENT_NAME``)
        and tagged with ``metadata.agent`` at write time. Audit fix #6 —
        the previous version accepted ``agent_name`` but never used it,
        so every agent received every strategy regardless of which agent
        the strategy was learned for. Now retrieval filters by the
        ``agent`` metadata tag so each agent only sees its own strategies.

        Args:
            query: Free-text query to find relevant strategies for.
            agent_name: Filter to strategies tagged for this agent. Pass
                ``None`` or ``"*"`` to return strategies for all agents
                (used by debugging tools).
            top_k: Maximum strategies to return after over-fetching and
                filtering by metadata.
        """
        if not self.memory_manager or self.memory_manager.memory is None:
            return []

        all_strategies: List[Dict[str, Any]] = []

        # Strategies are stored under agent-specific Mem0 namespaces:
        #   "_strategy_store_{agent_name}"
        # Wildcard ("*") and None fall back to the bare prefix so debugging
        # tools can retrieve all strategies across agents.
        if agent_name and agent_name != "*":
            lookup_namespace = f"{STRATEGY_AGENT_NAME}_{agent_name}"
        else:
            lookup_namespace = STRATEGY_AGENT_NAME

        try:
            user_results = self.memory_manager.search_memory(
                query=f"strategy for {query}",
                tenant_id=self.tenant_id,
                agent_name=lookup_namespace,
                top_k=top_k,
            )
            for r in user_results:
                r["_level"] = "user"
                all_strategies.append(r)
        except Exception as e:
            logger.debug(f"User strategy retrieval failed: {e}")

        if self.org_id != self.tenant_id:
            try:
                org_results = self.memory_manager.search_memory(
                    query=f"strategy for {query}",
                    tenant_id=self.org_id,
                    agent_name=lookup_namespace,
                    top_k=top_k,
                )
                for r in org_results:
                    r["_level"] = "org"
                    all_strategies.append(r)
            except Exception as e:
                logger.debug(f"Org strategy retrieval failed: {e}")

        return all_strategies[:top_k]

    @staticmethod
    def format_strategies_for_context(strategies: List[Dict[str, Any]]) -> str:
        """Format retrieved strategies for injection into agent context."""
        if not strategies:
            return ""

        lines = []
        for s in strategies:
            meta = s.get("metadata", {})
            text = s.get("memory", "")
            confidence = float(meta.get("confidence", 0))
            trace_count = int(meta.get("trace_count", 0))
            level = s.get("_level", meta.get("level", ""))

            line = f"- {text} (confidence: {confidence:.2f}"
            if trace_count:
                line += f", from {trace_count} traces"
            if level:
                line += f", {level}-level"
            line += ")"
            lines.append(line)

        return "## Learned Strategies\n" + "\n".join(lines)


def _text_overlap(a: str, b: str) -> float:
    """Simple word-level Jaccard similarity for deduplication."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
