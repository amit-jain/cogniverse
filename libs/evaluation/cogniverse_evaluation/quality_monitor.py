"""
Continuous quality monitor for all agents.

Runs dual evaluation strategy:
1. Golden set evaluation — fixed queries scored with MRR/nDCG/P@5
2. Live traffic evaluation — sampled spans scored with LLM judge

Results stored as Phoenix datasets per tenant. When quality degrades,
packages a trigger dataset and submits an Argo optimization workflow.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd

from cogniverse_evaluation.metrics.custom import calculate_metrics_suite

logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    SEARCH = "search"
    SUMMARY = "summary"
    REPORT = "report"
    ROUTING = "routing"


class Verdict(int, Enum):
    SKIP = 0
    OPTIMIZE = 1
    FULL = 2


# Span names match the convention emitted by AgentBase._process_span():
#   f"{ClassName}.process"
# Audit fix #2 — the previous values (search_service.search, etc.) never
# matched what the runtime actually emits, so live-traffic eval queries
# returned zero spans for every agent. After audit fix #10 wraps every
# AgentBase subclass in a span, these names are stable and discoverable
# from the agent class name alone.
SPAN_NAME_BY_AGENT = {
    AgentType.SEARCH: "SearchAgent.process",
    AgentType.SUMMARY: "SummarizerAgent.process",
    AgentType.REPORT: "DetailedReportAgent.process",
    AgentType.ROUTING: "RoutingAgent.process",
}


@dataclass
class AgentEvalResult:
    """Evaluation result for a single agent."""

    agent: AgentType
    score: float
    baseline_score: Optional[float]
    degradation_pct: float
    sample_count: int
    low_scoring_examples: List[Dict[str, Any]] = field(default_factory=list)
    high_scoring_examples: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class GoldenEvalResult:
    """Result of golden set evaluation."""

    timestamp: datetime
    tenant_id: str
    mean_mrr: float
    mean_ndcg: float
    mean_precision_at_5: float
    query_count: int
    per_query_scores: List[Dict[str, Any]] = field(default_factory=list)
    low_scoring_queries: List[Dict[str, Any]] = field(default_factory=list)
    high_scoring_queries: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LiveEvalResult:
    """Result of live traffic evaluation across all agents."""

    timestamp: datetime
    tenant_id: str
    agent_results: Dict[AgentType, AgentEvalResult] = field(default_factory=dict)


@dataclass
class OptimizationTrigger:
    """Payload for the optimization workflow."""

    timestamp: datetime
    tenant_id: str
    agents_to_optimize: List[AgentType]
    golden_eval: Optional[GoldenEvalResult]
    live_eval: Optional[LiveEvalResult]
    low_scoring_examples: Dict[AgentType, List[Dict[str, Any]]]
    high_scoring_examples: Dict[AgentType, List[Dict[str, Any]]]
    misrouted_queries: List[Dict[str, Any]]


@dataclass
class QualityThresholds:
    """Thresholds for triggering optimization per agent."""

    golden_mrr_drop_pct: float = 0.10
    golden_ndcg_drop_pct: float = 0.10
    live_score_floor: float = 0.5
    error_rate_ceiling: float = 0.05
    latency_p95_ceiling_ms: float = 1000.0
    min_samples_for_verdict: int = 10


class QualityMonitor:
    """
    Continuous quality monitor for all agents.

    Composes existing evaluation infrastructure:
    - SpanEvaluator for pulling/evaluating search spans
    - GoldenDatasetEvaluator for scoring against golden set
    - LLMJudgeBase for live traffic relevance scoring
    - PhoenixDatasetStore for baseline storage
    - RetrievalMonitor for latency/error windows

    Does NOT reimplement any of these — orchestrates them on schedule.
    """

    def __init__(
        self,
        tenant_id: str,
        runtime_url: str,
        phoenix_http_endpoint: str,
        llm_base_url: str,
        llm_model: str,
        golden_dataset_path: str,
        argo_api_url: Optional[str] = None,
        argo_namespace: str = "cogniverse",
        golden_eval_interval_seconds: int = 7200,
        live_eval_interval_seconds: int = 14400,
        live_sample_count: int = 20,
        thresholds: Optional[QualityThresholds] = None,
        telemetry_provider=None,
    ):
        self.tenant_id = tenant_id
        self.runtime_url = runtime_url.rstrip("/")
        self.phoenix_http_endpoint = phoenix_http_endpoint.rstrip("/")
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.golden_dataset_path = golden_dataset_path
        self.argo_api_url = argo_api_url
        self.argo_namespace = argo_namespace
        self.golden_eval_interval = golden_eval_interval_seconds
        self.live_eval_interval = live_eval_interval_seconds
        self.live_sample_count = live_sample_count
        self.thresholds = thresholds or QualityThresholds()
        self._telemetry_provider = telemetry_provider
        self._training_decision_model = None

        self._golden_queries: List[Dict[str, Any]] = []
        self._dataset_store = None
        self._llm_judge = None
        self._http_client = None

    def _get_dataset_store(self):
        """Lazy-load PhoenixDatasetStore."""
        if self._dataset_store is None:
            from cogniverse_telemetry_phoenix.provider import PhoenixDatasetStore

            self._dataset_store = PhoenixDatasetStore(
                http_endpoint=self.phoenix_http_endpoint,
                tenant_id=self.tenant_id,
            )
        return self._dataset_store

    def _get_llm_judge(self):
        """Lazy-load LLM judge."""
        if self._llm_judge is None:
            from cogniverse_evaluation.evaluators.llm_judge import LLMJudgeBase

            self._llm_judge = LLMJudgeBase(
                model_name=self.llm_model,
                base_url=self.llm_base_url,
            )
        return self._llm_judge

    def _get_http_client(self) -> httpx.AsyncClient:
        """Lazy-load async HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    def _load_golden_queries(self) -> List[Dict[str, Any]]:
        """Load golden evaluation queries from JSON."""
        if self._golden_queries:
            return self._golden_queries

        path = Path(self.golden_dataset_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Golden dataset not found: {self.golden_dataset_path}"
            )

        with open(path) as f:
            self._golden_queries = json.load(f)

        logger.info(f"Loaded {len(self._golden_queries)} golden queries")
        return self._golden_queries

    async def force_optimization_cycle(self) -> Dict[str, Any]:
        """Run one full eval + trigger + submit cycle, regardless of thresholds.

        Audit fix #7 — provides a scheduled distillation path independent of
        the quality-drop threshold check. Argo CronWorkflows call this via
        ``quality_monitor_cli --once`` so the system continues learning from
        live traces even when quality is stable. The continuous ``run()``
        loop only triggers on degradation, which left long stable periods
        with no learning happening at all.

        Returns a dict with the cycle outcome so cron jobs can log/alert.
        """
        golden_result: Optional[GoldenEvalResult] = None
        live_result: Optional[LiveEvalResult] = None

        try:
            golden_result = await self.evaluate_golden_set()
            logger.info(
                f"Forced golden eval: MRR={golden_result.mean_mrr:.3f}, "
                f"nDCG={golden_result.mean_ndcg:.3f}"
            )
        except Exception as e:
            logger.warning(f"Forced golden eval failed: {e}")

        try:
            live_result = await self.evaluate_live_traffic()
        except Exception as e:
            logger.warning(f"Forced live eval failed: {e}")

        if not golden_result and not live_result:
            logger.warning(
                "Force cycle: no eval data available — skipping trigger build"
            )
            return {"status": "no_data", "submitted_to_argo": False}

        # Build a trigger covering ALL agents, not just those with verdicts.
        all_agents = list(AgentType)
        trigger = self._build_trigger(all_agents, golden_result, live_result)
        await self._store_trigger_dataset(trigger)

        submitted = False
        if self.argo_api_url:
            try:
                await self.submit_optimization(trigger)
                submitted = True
            except Exception as e:
                logger.error(f"Force cycle: Argo submission failed: {e}")

        return {
            "status": "ok",
            "agents_triggered": [a.value for a in all_agents],
            "submitted_to_argo": submitted,
        }

    async def run(self):
        """Main monitoring loop. Runs golden and live evals on different cadences."""
        logger.info(
            f"Quality monitor starting for tenant={self.tenant_id} "
            f"(golden every {self.golden_eval_interval}s, "
            f"live every {self.live_eval_interval}s)"
        )

        last_golden = 0.0
        last_live = 0.0

        while True:
            now = asyncio.get_event_loop().time()

            golden_result = None
            live_result = None

            if now - last_golden >= self.golden_eval_interval:
                try:
                    golden_result = await self.evaluate_golden_set()
                    last_golden = now
                    logger.info(
                        f"Golden eval: MRR={golden_result.mean_mrr:.3f}, "
                        f"nDCG={golden_result.mean_ndcg:.3f}, "
                        f"P@5={golden_result.mean_precision_at_5:.3f}"
                    )
                except Exception as e:
                    logger.error(f"Golden eval failed: {e}")

            if now - last_live >= self.live_eval_interval:
                try:
                    live_result = await self.evaluate_live_traffic()
                    last_live = now
                    for agent, result in live_result.agent_results.items():
                        logger.info(
                            f"Live eval {agent.value}: score={result.score:.3f}, "
                            f"samples={result.sample_count}"
                        )
                except Exception as e:
                    logger.error(f"Live eval failed: {e}")

            if golden_result or live_result:
                verdicts = self.check_thresholds(golden_result, live_result)
                agents_to_optimize = [
                    agent
                    for agent, verdict in verdicts.items()
                    if verdict != Verdict.SKIP
                ]

                if agents_to_optimize:
                    trigger = self._build_trigger(
                        agents_to_optimize, golden_result, live_result
                    )
                    await self._store_trigger_dataset(trigger)

                    if self.argo_api_url:
                        await self.submit_optimization(trigger)
                    else:
                        logger.warning(
                            f"Optimization needed for {agents_to_optimize} "
                            f"but no Argo API URL configured"
                        )

            await asyncio.sleep(60)

    async def evaluate_golden_set(self) -> GoldenEvalResult:
        """Run golden queries against /search, score with IR metrics."""
        queries = self._load_golden_queries()
        client = self._get_http_client()

        per_query_scores = []
        low_scoring = []
        high_scoring = []

        for query_data in queries:
            query = query_data["query"]
            expected_videos = query_data.get("expected_videos", [])

            try:
                response = await client.post(
                    f"{self.runtime_url}/search/",
                    json={
                        "query": query,
                        "profile": "test_colpali",
                        "top_k": 10,
                    },
                )

                if response.status_code != 200:
                    logger.warning(
                        f"Search failed for '{query}': {response.status_code}"
                    )
                    continue

                results = response.json().get("results", [])
                retrieved_ids = [
                    r.get("source_id", r.get("video_id", "")) for r in results
                ]

                metrics = calculate_metrics_suite(
                    retrieved_ids, expected_videos, k_values=[1, 5, 10]
                )

                entry = {
                    "query": query,
                    "expected_videos": expected_videos,
                    "retrieved_videos": retrieved_ids[:10],
                    **metrics,
                }
                per_query_scores.append(entry)

                if metrics["mrr"] < 0.3:
                    low_scoring.append(entry)
                elif metrics["mrr"] >= 0.8:
                    high_scoring.append(entry)

            except Exception as e:
                logger.warning(f"Error evaluating golden query '{query}': {e}")

        if not per_query_scores:
            raise RuntimeError("No golden queries evaluated successfully")

        mean_mrr = sum(s["mrr"] for s in per_query_scores) / len(per_query_scores)
        mean_ndcg = sum(s["ndcg"] for s in per_query_scores) / len(per_query_scores)
        mean_p5 = sum(s["precision@5"] for s in per_query_scores) / len(
            per_query_scores
        )

        result = GoldenEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id=self.tenant_id,
            mean_mrr=mean_mrr,
            mean_ndcg=mean_ndcg,
            mean_precision_at_5=mean_p5,
            query_count=len(per_query_scores),
            per_query_scores=per_query_scores,
            low_scoring_queries=low_scoring,
            high_scoring_queries=high_scoring,
        )

        await self._store_golden_eval_result(result)
        return result

    async def evaluate_live_traffic(self) -> LiveEvalResult:
        """Sample recent spans per agent, score with LLM judge."""
        from cogniverse_evaluation.span_evaluator import SpanEvaluator

        span_evaluator = SpanEvaluator(
            tenant_id=self.tenant_id,
            project_name=f"cogniverse-{self.tenant_id}-runtime",
        )

        result = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id=self.tenant_id,
        )

        for agent_type in AgentType:
            span_name = SPAN_NAME_BY_AGENT[agent_type]
            try:
                spans_df = await span_evaluator.get_recent_spans(
                    hours=int(self.live_eval_interval / 3600) or 4,
                    operation_name=span_name,
                    limit=200,
                )

                if spans_df.empty:
                    logger.debug(f"No spans found for {agent_type.value}")
                    continue

                sampled = spans_df.sample(
                    n=min(self.live_sample_count, len(spans_df))
                )

                agent_result = await self._evaluate_agent_spans(
                    agent_type, sampled
                )
                result.agent_results[agent_type] = agent_result

            except Exception as e:
                logger.warning(f"Live eval failed for {agent_type.value}: {e}")

        await self._store_live_eval_result(result)
        return result

    async def _evaluate_agent_spans(
        self, agent_type: AgentType, spans_df: pd.DataFrame
    ) -> AgentEvalResult:
        """Score sampled spans for a specific agent using LLM judge."""
        judge = self._get_llm_judge()
        scores = []
        low_scoring = []
        high_scoring = []

        for _, span in spans_df.iterrows():
            attributes = span.get("attributes", {})
            outputs = span.get("outputs", {})
            query = attributes.get("query", "")

            if agent_type == AgentType.SEARCH:
                results = outputs.get("results", [])
                prompt = self._build_search_judge_prompt(query, results)
            elif agent_type == AgentType.SUMMARY:
                summary = outputs.get("summary", "")
                prompt = self._build_summary_judge_prompt(query, summary)
            elif agent_type == AgentType.REPORT:
                report = outputs.get("report", "")
                prompt = self._build_report_judge_prompt(query, report)
            elif agent_type == AgentType.ROUTING:
                # Routing already has its own feedback loop via OptimizationOrchestrator.
                # Still score here for unified visibility.
                routing = outputs.get("routing_decision", {})
                prompt = self._build_routing_judge_prompt(query, routing)
            else:
                continue

            try:
                response = await judge._call_llm(
                    prompt=prompt,
                    system_prompt=(
                        "You are an evaluation judge. Score the quality of AI agent "
                        "outputs on a scale of 0-10. Always include 'Score: X/10' in "
                        "your response."
                    ),
                )
                score, explanation = judge._extract_score_from_response(response)

                example = {
                    "query": query,
                    "agent": agent_type.value,
                    "score": score,
                    "explanation": explanation,
                    "span_id": span.get("span_id", ""),
                }
                scores.append(score)

                if score < 0.5:
                    example["output"] = outputs
                    low_scoring.append(example)
                elif score >= 0.8:
                    example["output"] = outputs
                    high_scoring.append(example)

            except Exception as e:
                logger.warning(f"LLM judge failed for span: {e}")

        mean_score = sum(scores) / len(scores) if scores else 0.0

        baseline_score = await self._get_agent_baseline(agent_type)
        degradation = 0.0
        if baseline_score and baseline_score > 0:
            degradation = (baseline_score - mean_score) / baseline_score

        return AgentEvalResult(
            agent=agent_type,
            score=mean_score,
            baseline_score=baseline_score,
            degradation_pct=degradation,
            sample_count=len(scores),
            low_scoring_examples=low_scoring,
            high_scoring_examples=high_scoring,
            metrics={"mean_score": mean_score, "sample_count": len(scores)},
        )

    def _build_search_judge_prompt(
        self, query: str, results: List[Dict[str, Any]]
    ) -> str:
        top_results = results[:5]
        results_text = "\n".join(
            f"  {i+1}. {r.get('video_id', r.get('source_id', 'unknown'))} "
            f"(score: {r.get('score', 'N/A')})"
            for i, r in enumerate(top_results)
        )
        return (
            f"Query: {query}\n\n"
            f"Search Results:\n{results_text}\n\n"
            f"Rate the relevance of these search results to the query. "
            f"Consider: Are the results topically relevant? Is the ranking order "
            f"sensible? Would a user find these results helpful?\n"
            f"Score: X/10"
        )

    def _build_summary_judge_prompt(self, query: str, summary: str) -> str:
        return (
            f"Original Query: {query}\n\n"
            f"Generated Summary:\n{summary}\n\n"
            f"Rate the quality of this summary. Consider: Is it accurate? "
            f"Is it concise? Does it address the query? Is it coherent?\n"
            f"Score: X/10"
        )

    def _build_report_judge_prompt(self, query: str, report: str) -> str:
        return (
            f"Original Query: {query}\n\n"
            f"Generated Report:\n{report[:2000]}\n\n"
            f"Rate the quality of this report. Consider: Is it comprehensive? "
            f"Are the findings well-supported? Are recommendations actionable? "
            f"Is the structure logical?\n"
            f"Score: X/10"
        )

    def _build_routing_judge_prompt(
        self, query: str, routing: Dict[str, Any]
    ) -> str:
        return (
            f"Query: {query}\n\n"
            f"Routing Decision: {json.dumps(routing, default=str)}\n\n"
            f"Rate the routing decision. Consider: Was the right agent selected? "
            f"Is the confidence calibrated? Does the workflow make sense for "
            f"this query type?\n"
            f"Score: X/10"
        )

    def check_thresholds(
        self,
        golden: Optional[GoldenEvalResult],
        live: Optional[LiveEvalResult],
    ) -> Dict[AgentType, Verdict]:
        """Compare eval results against baselines, then consult XGBoost
        meta-model for smarter optimization timing decisions."""
        verdicts: Dict[AgentType, Verdict] = {}

        # Golden set only covers search agent
        if golden:
            search_verdict = Verdict.SKIP
            baseline_mrr = self._last_golden_baseline_mrr
            if baseline_mrr and baseline_mrr > 0:
                drop = (baseline_mrr - golden.mean_mrr) / baseline_mrr
                if drop >= self.thresholds.golden_mrr_drop_pct:
                    search_verdict = Verdict.OPTIMIZE
                    logger.warning(
                        f"Golden MRR dropped {drop:.1%} "
                        f"({baseline_mrr:.3f} → {golden.mean_mrr:.3f})"
                    )
            verdicts[AgentType.SEARCH] = search_verdict

        # Live traffic covers all agents
        if live:
            for agent_type, agent_result in live.agent_results.items():
                if agent_result.sample_count < self.thresholds.min_samples_for_verdict:
                    continue

                verdict = Verdict.SKIP

                if agent_result.score < self.thresholds.live_score_floor:
                    verdict = Verdict.OPTIMIZE
                    logger.warning(
                        f"{agent_type.value} live score {agent_result.score:.3f} "
                        f"< floor {self.thresholds.live_score_floor}"
                    )
                elif agent_result.degradation_pct > self.thresholds.golden_mrr_drop_pct:
                    verdict = Verdict.OPTIMIZE
                    logger.warning(
                        f"{agent_type.value} degraded {agent_result.degradation_pct:.1%} "
                        f"from baseline"
                    )

                existing = verdicts.get(agent_type, Verdict.SKIP)
                verdicts[agent_type] = max(existing, verdict)

        # Consult XGBoost meta-model for smarter timing
        verdicts = self._apply_training_decision_model(verdicts, golden, live)

        return verdicts

    def _apply_training_decision_model(
        self,
        verdicts: Dict[AgentType, Verdict],
        golden: Optional[GoldenEvalResult],
        live: Optional[LiveEvalResult],
    ) -> Dict[AgentType, Verdict]:
        """Use XGBoost TrainingDecisionModel to confirm or override verdicts.

        The naive threshold check says "quality dropped." The meta-model
        adds: "is optimization likely to help given current data volume,
        model staleness, and recent performance?"
        """
        if self._telemetry_provider is None:
            return verdicts

        try:
            model = self._get_training_decision_model()

            for agent_type, verdict in list(verdicts.items()):
                context = self._build_modeling_context(agent_type, golden, live)
                if context is None:
                    continue

                should_train, expected_improvement = model.should_train(context)

                if verdict == Verdict.OPTIMIZE and not should_train:
                    logger.info(
                        f"XGBoost overrides {agent_type.value} OPTIMIZE → SKIP "
                        f"(expected improvement {expected_improvement:.3f} too low)"
                    )
                    verdicts[agent_type] = Verdict.SKIP
                elif verdict == Verdict.SKIP and should_train:
                    logger.info(
                        f"XGBoost upgrades {agent_type.value} SKIP → OPTIMIZE "
                        f"(expected improvement {expected_improvement:.3f})"
                    )
                    verdicts[agent_type] = Verdict.OPTIMIZE

        except Exception as e:
            logger.debug(f"XGBoost training decision failed (using naive verdicts): {e}")

        return verdicts

    def _get_training_decision_model(self):
        """Lazy-load XGBoost TrainingDecisionModel."""
        if self._training_decision_model is None:
            from cogniverse_agents.routing.xgboost_meta_models import (
                TrainingDecisionModel,
            )

            self._training_decision_model = TrainingDecisionModel(
                telemetry_provider=self._telemetry_provider,
                tenant_id=self.tenant_id,
            )
        return self._training_decision_model

    def _build_modeling_context(
        self, agent_type: AgentType, golden, live
    ):
        """Build ModelingContext from eval results for XGBoost."""
        try:
            from cogniverse_agents.routing.xgboost_meta_models import ModelingContext
            from cogniverse_agents.search.multi_modal_reranker import QueryModality

            modality_map = {
                AgentType.SEARCH: QueryModality.VIDEO,
                AgentType.SUMMARY: QueryModality.TEXT,
                AgentType.REPORT: QueryModality.TEXT,
                AgentType.ROUTING: QueryModality.MIXED,
            }

            score = 0.0
            sample_count = 0
            if live and agent_type in live.agent_results:
                result = live.agent_results[agent_type]
                score = result.score
                sample_count = result.sample_count

            return ModelingContext(
                modality=modality_map.get(agent_type, QueryModality.MIXED),
                real_sample_count=sample_count,
                synthetic_sample_count=0,
                success_rate=score,
                avg_confidence=score,
                days_since_last_training=7,
                current_performance_score=score,
            )
        except Exception as e:
            logger.debug(f"Failed to build ModelingContext: {e}")
            return None

    @property
    def _last_golden_baseline_mrr(self) -> Optional[float]:
        """Load the last golden baseline MRR from Phoenix dataset."""
        try:
            from phoenix.client import Client as PhoenixSyncClient

            sync_client = PhoenixSyncClient(base_url=self.phoenix_http_endpoint)
            dataset_name = f"quality-baseline-{self.tenant_id}"
            dataset = sync_client.datasets.get_dataset(dataset=dataset_name)
            df = dataset.to_dataframe()
            if df.empty:
                return None

            # Phoenix may nest values under input/output dicts
            if "mean_mrr" in df.columns:
                return float(df["mean_mrr"].iloc[-1])
            elif "output" in df.columns:
                last_row = df.iloc[-1]
                output = last_row["output"]
                if isinstance(output, dict) and "mean_mrr" in output:
                    return float(output["mean_mrr"])
            # Try flattened Phoenix column names
            for col in df.columns:
                if "mean_mrr" in str(col):
                    return float(df[col].iloc[-1])
        except Exception as e:
            logger.debug(f"Failed to read baseline: {e}")
        return None

    def _build_trigger(
        self,
        agents_to_optimize: List[AgentType],
        golden: Optional[GoldenEvalResult],
        live: Optional[LiveEvalResult],
    ) -> OptimizationTrigger:
        """Package evaluation results into an optimization trigger."""
        low_scoring: Dict[AgentType, List[Dict[str, Any]]] = {}
        high_scoring: Dict[AgentType, List[Dict[str, Any]]] = {}
        misrouted: List[Dict[str, Any]] = []

        if golden:
            low_scoring[AgentType.SEARCH] = golden.low_scoring_queries
            high_scoring[AgentType.SEARCH] = golden.high_scoring_queries

        if live:
            for agent_type, result in live.agent_results.items():
                existing_low = low_scoring.get(agent_type, [])
                existing_high = high_scoring.get(agent_type, [])
                low_scoring[agent_type] = existing_low + result.low_scoring_examples
                high_scoring[agent_type] = existing_high + result.high_scoring_examples

        return OptimizationTrigger(
            timestamp=datetime.utcnow(),
            tenant_id=self.tenant_id,
            agents_to_optimize=agents_to_optimize,
            golden_eval=golden,
            live_eval=live,
            low_scoring_examples=low_scoring,
            high_scoring_examples=high_scoring,
            misrouted_queries=misrouted,
        )

    async def _store_golden_eval_result(self, result: GoldenEvalResult):
        """Store golden eval as Phoenix dataset for baseline comparison."""
        store = self._get_dataset_store()
        dataset_name = f"quality-baseline-{self.tenant_id}"

        df = pd.DataFrame(
            [
                {
                    "timestamp": result.timestamp.isoformat(),
                    "mean_mrr": result.mean_mrr,
                    "mean_ndcg": result.mean_ndcg,
                    "mean_precision_at_5": result.mean_precision_at_5,
                    "query_count": result.query_count,
                }
            ]
        )

        try:
            await store.create_dataset(
                name=dataset_name,
                data=df,
                metadata={
                    "description": f"Golden eval baseline for tenant {self.tenant_id}",
                    "input_keys": ["timestamp"],
                    "output_keys": ["mean_mrr", "mean_ndcg", "mean_precision_at_5"],
                },
            )
            logger.info(f"Stored golden eval baseline: {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to store golden baseline: {e}")

    async def _store_live_eval_result(self, result: LiveEvalResult):
        """Store live eval results as Phoenix dataset."""
        store = self._get_dataset_store()
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        dataset_name = f"quality-live-{self.tenant_id}-{timestamp}"

        records = []
        for agent_type, agent_result in result.agent_results.items():
            records.append(
                {
                    "agent": agent_type.value,
                    "score": agent_result.score,
                    "baseline_score": agent_result.baseline_score,
                    "degradation_pct": agent_result.degradation_pct,
                    "sample_count": agent_result.sample_count,
                }
            )

        if not records:
            return

        df = pd.DataFrame(records)
        try:
            await store.create_dataset(
                name=dataset_name,
                data=df,
                metadata={
                    "description": f"Live traffic eval for tenant {self.tenant_id}",
                    "input_keys": ["agent"],
                    "output_keys": ["score", "degradation_pct"],
                },
            )
            logger.info(f"Stored live eval results: {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to store live eval: {e}")

    async def _store_trigger_dataset(self, trigger: OptimizationTrigger):
        """Store optimization trigger payload as Phoenix dataset."""
        store = self._get_dataset_store()
        timestamp = trigger.timestamp.strftime("%Y%m%d_%H%M%S")
        dataset_name = f"optimization-trigger-{self.tenant_id}-{timestamp}"

        records = []
        for agent_type in trigger.agents_to_optimize:
            low = trigger.low_scoring_examples.get(agent_type, [])
            high = trigger.high_scoring_examples.get(agent_type, [])
            for example in low:
                records.append(
                    {
                        "agent": agent_type.value,
                        "category": "low_scoring",
                        "query": example.get("query", ""),
                        "score": example.get("score", 0.0),
                        "output": json.dumps(
                            example.get("output", {}), default=str
                        ),
                    }
                )
            for example in high:
                records.append(
                    {
                        "agent": agent_type.value,
                        "category": "high_scoring",
                        "query": example.get("query", ""),
                        "score": example.get("score", 0.0),
                        "output": json.dumps(
                            example.get("output", {}), default=str
                        ),
                    }
                )

        if not records:
            return

        df = pd.DataFrame(records)
        try:
            await store.create_dataset(
                name=dataset_name,
                data=df,
                metadata={
                    "description": (
                        f"Optimization trigger for {self.tenant_id}: "
                        f"{[a.value for a in trigger.agents_to_optimize]}"
                    ),
                    "input_keys": ["agent", "category", "query"],
                    "output_keys": ["score", "output"],
                },
            )
            logger.info(
                f"Stored trigger dataset: {dataset_name} "
                f"({len(records)} examples for "
                f"{[a.value for a in trigger.agents_to_optimize]})"
            )
        except Exception as e:
            logger.error(f"Failed to store trigger dataset: {e}")

    async def _get_agent_baseline(self, agent_type: AgentType) -> Optional[float]:
        """Load the last live eval baseline for an agent from Phoenix."""
        try:
            from phoenix.client import Client as PhoenixSyncClient

            sync_client = PhoenixSyncClient(base_url=self.phoenix_http_endpoint)
            dataset_name = f"quality-baseline-{self.tenant_id}"
            dataset = sync_client.datasets.get_dataset(dataset=dataset_name)
            df = dataset.to_dataframe()

            if df.empty:
                return None

            # Filter to this agent if column exists
            if "agent" in df.columns:
                agent_df = df[df["agent"] == agent_type.value]
                if not agent_df.empty and "score" in agent_df.columns:
                    return float(agent_df["score"].iloc[-1])

            return None
        except Exception as e:
            logger.debug(f"Failed to read agent baseline for {agent_type.value}: {e}")
            return None

    async def submit_optimization(self, trigger: OptimizationTrigger):
        """Submit Argo optimization workflow via k8s API."""
        client = self._get_http_client()
        agents_csv = ",".join(a.value for a in trigger.agents_to_optimize)
        timestamp = trigger.timestamp.strftime("%Y%m%d-%H%M%S")

        workflow_manifest = {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Workflow",
            "metadata": {
                "generateName": f"quality-triggered-optimization-{timestamp}-",
                "namespace": self.argo_namespace,
                "labels": {
                    "app": "cogniverse",
                    "trigger": "quality-monitor",
                    "tenant": self.tenant_id,
                },
            },
            "spec": {
                "entrypoint": "optimize",
                "arguments": {
                    "parameters": [
                        {"name": "tenant-id", "value": self.tenant_id},
                        {"name": "agents", "value": agents_csv},
                        {
                            "name": "trigger-dataset",
                            "value": (
                                f"optimization-trigger-{self.tenant_id}-"
                                f"{trigger.timestamp.strftime('%Y%m%d_%H%M%S')}"
                            ),
                        },
                    ]
                },
                "templates": [
                    {
                        "name": "optimize",
                        "container": {
                            "image": "cogniverse-runtime:latest",
                            "command": ["python", "-m", "cogniverse_runtime.optimization_cli"],
                            "args": [
                                "--mode", "triggered",
                                "--tenant-id", "{{workflow.parameters.tenant-id}}",
                                "--agents", "{{workflow.parameters.agents}}",
                                "--trigger-dataset", "{{workflow.parameters.trigger-dataset}}",
                            ],
                        },
                    }
                ],
            },
        }

        try:
            response = await client.post(
                f"{self.argo_api_url}/api/v1/workflows/{self.argo_namespace}",
                json={"workflow": workflow_manifest},
            )
            if response.status_code in (200, 201):
                workflow_name = response.json().get("metadata", {}).get("name", "")
                logger.info(
                    f"Submitted optimization workflow: {workflow_name} "
                    f"for agents={agents_csv}"
                )
            else:
                logger.error(
                    f"Argo submit failed ({response.status_code}): "
                    f"{response.text[:500]}"
                )
        except Exception as e:
            logger.error(f"Failed to submit Argo workflow: {e}")

    async def update_baseline(
        self, golden_result: Optional[GoldenEvalResult] = None,
        live_results: Optional[Dict[AgentType, float]] = None,
    ):
        """Update baselines after successful optimization."""
        if golden_result:
            await self._store_golden_eval_result(golden_result)
            logger.info(
                f"Updated golden baseline: MRR={golden_result.mean_mrr:.3f}"
            )

        if live_results:
            store = self._get_dataset_store()
            dataset_name = f"quality-baseline-{self.tenant_id}"
            records = [
                {"agent": agent.value, "score": score}
                for agent, score in live_results.items()
            ]
            df = pd.DataFrame(records)
            try:
                await store.create_dataset(
                    name=dataset_name,
                    data=df,
                    metadata={
                        "description": "Live eval baselines per agent",
                        "input_keys": ["agent"],
                        "output_keys": ["score"],
                    },
                )
            except Exception as e:
                logger.error(f"Failed to update live baselines: {e}")

    async def grow_golden_set(self, new_queries: List[Dict[str, Any]]):
        """Add high-scoring live queries to the golden dataset."""
        if not new_queries:
            return

        path = Path(self.golden_dataset_path)
        existing = self._load_golden_queries()

        existing_query_set = {q["query"] for q in existing}
        added = 0
        for q in new_queries:
            if q.get("query") not in existing_query_set:
                existing.append(q)
                existing_query_set.add(q["query"])
                added += 1

        if added > 0:
            with open(path, "w") as f:
                json.dump(existing, f, indent=2)
            self._golden_queries = existing
            logger.info(
                f"Added {added} queries to golden set "
                f"(total: {len(existing)})"
            )

    async def close(self):
        """Clean up resources."""
        if self._http_client:
            await self._http_client.aclose()
