"""
Convert telemetry traces to instruction-tuning datasets.

Uses TelemetryProvider stores (TraceStore, AnnotationStore) to extract
agent traces with annotations and convert them to instruction formats.

Supports both:
1. Single-turn instruction extraction (TraceToInstructionConverter)
2. Multi-turn trajectory extraction (TraceToTrajectoryConverter)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from cogniverse_foundation.telemetry.providers.base import TelemetryProvider

logger = logging.getLogger(__name__)


@dataclass
class InstructionExample:
    """Single instruction-response example for fine-tuning."""

    instruction: str
    input: str
    output: str
    metadata: Dict[str, Any]


@dataclass
class InstructionDataset:
    """Dataset of instruction-response examples."""

    examples: List[InstructionExample]
    metadata: Dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(
            [
                {
                    "instruction": ex.instruction,
                    "input": ex.input,
                    "output": ex.output,
                    **ex.metadata,
                }
                for ex in self.examples
            ]
        )

    def save(self, path: str, format: Literal["jsonl", "parquet"] = "jsonl") -> None:
        """
        Save dataset to file.

        Args:
            path: Output file path
            format: Output format (jsonl or parquet)
        """
        df = self.to_dataframe()

        if format == "jsonl":
            df.to_json(path, orient="records", lines=True)
        elif format == "parquet":
            df.to_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(self.examples)} examples to {path} ({format})")


@dataclass
class ConversationTurn:
    """Single turn in a multi-turn conversation."""

    turn_id: int
    query: str
    response: str
    timestamp: datetime
    span_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTrajectory:
    """Complete conversation trajectory for multi-turn fine-tuning."""

    session_id: str
    turns: List[ConversationTurn]
    session_outcome: Optional[str] = None  # "success", "failure", "partial"
    session_score: Optional[float] = None  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for dataset creation."""
        return {
            "session_id": self.session_id,
            "num_turns": len(self.turns),
            "conversation": [
                {
                    "turn": turn.turn_id,
                    "query": turn.query,
                    "response": turn.response,
                    "timestamp": turn.timestamp.isoformat() if turn.timestamp else None,
                    "span_id": turn.span_id,
                    **turn.metadata,
                }
                for turn in self.turns
            ],
            "session_outcome": self.session_outcome,
            "session_score": self.session_score,
            **(self.metadata or {}),
        }


@dataclass
class TrajectoryDataset:
    """Dataset of conversation trajectories."""

    trajectories: List[ConversationTrajectory]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([traj.to_dict() for traj in self.trajectories])

    def save(self, path: str, format: Literal["jsonl", "parquet"] = "jsonl") -> None:
        """Save trajectory dataset."""
        df = self.to_dataframe()

        if format == "jsonl":
            df.to_json(path, orient="records", lines=True)
        elif format == "parquet":
            df.to_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        total_turns = sum(len(t.turns) for t in self.trajectories)
        logger.info(
            f"Saved {len(self.trajectories)} trajectories "
            f"({total_turns} total turns) to {path}"
        )


class TraceToInstructionConverter:
    """
    Convert telemetry traces to instruction-tuning datasets.

    Uses TelemetryProvider's TraceStore and AnnotationStore to extract
    agent execution traces with human annotations (approved/rejected)
    and convert them to instruction-response pairs for supervised fine-tuning.
    """

    def __init__(self, provider: TelemetryProvider):
        """
        Initialize converter with telemetry provider.

        Args:
            provider: Initialized TelemetryProvider (e.g., PhoenixProvider)
        """
        self.provider = provider

    async def convert(
        self,
        project: str,
        agent_type: Literal["routing", "profile_selection", "entity_extraction"],
        min_annotations: int = 20,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        annotation_filter: Optional[str] = None,
    ) -> InstructionDataset:
        """
        Convert traces to instruction dataset.

        Args:
            project: Project name (e.g., "cogniverse-tenant1")
            agent_type: Type of agent to extract traces for
            min_annotations: Minimum number of annotated traces required
            start_time: Optional start time filter
            end_time: Optional end time filter
            annotation_filter: Optional annotation name filter (e.g., "human_approval")

        Returns:
            InstructionDataset with examples extracted from traces

        Raises:
            ValueError: If insufficient annotated traces found
        """
        logger.info(
            f"Converting traces to instructions: project={project}, "
            f"agent={agent_type}, min_annotations={min_annotations}"
        )

        # Get trace store and annotation store from provider (using public properties)
        trace_store = self.provider.traces
        annotation_store = self.provider.annotations

        # 1. Query spans for the agent type
        logger.info(f"Querying spans from project={project}...")
        spans_df = await trace_store.get_spans(
            project=project,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        if spans_df.empty:
            raise ValueError(f"No spans found in project {project}")

        # Filter spans by agent type (look for agent name in span name)
        agent_spans = self._filter_agent_spans(spans_df, agent_type)
        logger.info(f"Found {len(agent_spans)} {agent_type} spans")

        # 2. Get annotations for these spans
        logger.info(f"Querying annotations for {len(agent_spans)} spans...")
        annotations_df = await annotation_store.get_annotations(
            spans_df=agent_spans,
            project=project,
            annotation_names=[annotation_filter] if annotation_filter else None,
        )

        logger.info(f"Found {len(annotations_df)} annotations")

        # 3. Filter for approved annotations only (for supervised fine-tuning)
        approved_annotations = self._filter_approved(annotations_df)
        logger.info(f"Found {len(approved_annotations)} approved annotations")

        if len(approved_annotations) < min_annotations:
            raise ValueError(
                f"Insufficient approved annotations: {len(approved_annotations)} < {min_annotations}"
            )

        # 4. Convert to instruction examples
        examples = self._create_instruction_examples(
            agent_spans, approved_annotations, agent_type
        )

        logger.info(f"Created {len(examples)} instruction examples")

        return InstructionDataset(
            examples=examples,
            metadata={
                "project": project,
                "agent_type": agent_type,
                "total_spans": len(spans_df),
                "agent_spans": len(agent_spans),
                "total_annotations": len(annotations_df),
                "approved_annotations": len(approved_annotations),
                "created_at": datetime.utcnow().isoformat(),
            },
        )

    def _filter_agent_spans(
        self, spans_df: pd.DataFrame, agent_type: str
    ) -> pd.DataFrame:
        """
        Filter spans for specific agent type.

        Args:
            spans_df: All spans from project
            agent_type: Agent type to filter for

        Returns:
            Filtered DataFrame with agent-specific spans
        """
        # Agent span naming convention: agent_type appears in span name
        # e.g., "RoutingAgent.route", "ProfileSelectionAgent.select_profiles"
        agent_keywords = {
            "routing": ["routing", "route"],
            "profile_selection": ["profile", "selection"],
            "entity_extraction": ["entity", "extraction"],
        }

        keywords = agent_keywords.get(agent_type, [agent_type])

        # Filter spans by name containing keywords (case-insensitive)
        mask = spans_df["name"].str.lower().str.contains("|".join(keywords), na=False)
        return spans_df[mask].copy()

    def _filter_approved(self, annotations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter for approved annotations.

        Args:
            annotations_df: All annotations

        Returns:
            DataFrame with approved annotations only
        """
        if annotations_df.empty:
            return annotations_df

        # Approved annotations have label="approved" or score >= 0.5
        mask = (annotations_df["result.label"] == "approved") | (
            annotations_df["result.score"] >= 0.5
        )
        return annotations_df[mask].copy()

    def _create_instruction_examples(
        self,
        spans_df: pd.DataFrame,
        annotations_df: pd.DataFrame,
        agent_type: str,
    ) -> List[InstructionExample]:
        """
        Create instruction examples from spans and annotations.

        Args:
            spans_df: Agent spans
            annotations_df: Approved annotations
            agent_type: Type of agent

        Returns:
            List of InstructionExample objects
        """
        examples = []

        # Get span IDs with approved annotations
        approved_span_ids = set(annotations_df["span_id"].unique())

        # Create example for each approved span
        for _, span_row in spans_df.iterrows():
            span_id = span_row.get("context.span_id")
            if span_id not in approved_span_ids:
                continue

            # Extract instruction and output from span attributes
            example = self._extract_example_from_span(span_row, agent_type)
            if example:
                examples.append(example)

        return examples

    def _extract_example_from_span(
        self, span_row: pd.Series, agent_type: str
    ) -> Optional[InstructionExample]:
        """
        Extract instruction example from span row.

        Args:
            span_row: Span data row
            agent_type: Type of agent

        Returns:
            InstructionExample or None if extraction fails
        """
        try:
            # Extract input/output from span attributes
            # Convention: input in attributes.input.*, output in attributes.output.*
            attributes = {
                k: v for k, v in span_row.items() if k.startswith("attributes.")
            }

            # Get input (query, request, etc.)
            input_text = self._extract_input(attributes, agent_type)
            if not input_text:
                logger.warning(
                    f"No input found in span {span_row.get('context.span_id')}"
                )
                return None

            # Get output (response, decision, etc.)
            output_text = self._extract_output(attributes, agent_type)
            if not output_text:
                logger.warning(
                    f"No output found in span {span_row.get('context.span_id')}"
                )
                return None

            # Get instruction template for agent type
            instruction = self._get_instruction_template(agent_type)

            return InstructionExample(
                instruction=instruction,
                input=input_text,
                output=output_text,
                metadata={
                    "span_id": span_row.get("context.span_id"),
                    "agent_type": agent_type,
                    "start_time": span_row.get("start_time"),
                    "end_time": span_row.get("end_time"),
                },
            )

        except Exception as e:
            logger.error(f"Failed to extract example from span: {e}")
            return None

    def _extract_input(self, attributes: Dict[str, Any], agent_type: str) -> str:
        """Extract input text from span attributes."""
        # Common input attribute names
        input_keys = [
            "attributes.input.query",
            "attributes.input.text",
            "attributes.input.request",
            "attributes.query",
            "attributes.text",
        ]

        for key in input_keys:
            if key in attributes and attributes[key]:
                return str(attributes[key])

        return ""

    def _extract_output(self, attributes: Dict[str, Any], agent_type: str) -> str:
        """Extract output text from span attributes."""
        # Common output attribute names
        output_keys = [
            "attributes.output.response",
            "attributes.output.result",
            "attributes.output.decision",
            "attributes.response",
            "attributes.result",
        ]

        for key in output_keys:
            if key in attributes and attributes[key]:
                value = attributes[key]
                # Handle dict/list outputs
                if isinstance(value, (dict, list)):
                    import json

                    return json.dumps(value)
                return str(value)

        return ""

    def _get_instruction_template(self, agent_type: str) -> str:
        """Get instruction template for agent type."""
        templates = {
            "routing": "Route the following query to the appropriate modality agent.",
            "profile_selection": "Select the optimal backend profile(s) for the following query.",
            "entity_extraction": "Extract entities and relationships from the following text.",
        }

        return templates.get(agent_type, f"Process the following {agent_type} request.")


class TraceToTrajectoryConverter:
    """
    Convert telemetry traces to conversation trajectories.

    Groups spans by session_id and orders chronologically to build
    complete conversation sequences for trajectory-level fine-tuning.
    """

    def __init__(self, provider: TelemetryProvider):
        """
        Initialize converter.

        Args:
            provider: Initialized TelemetryProvider
        """
        self.provider = provider

    async def convert(
        self,
        project: str,
        agent_type: Literal["routing", "profile_selection", "entity_extraction"],
        min_turns_per_session: int = 2,
        require_session_annotation: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> TrajectoryDataset:
        """
        Convert traces to trajectory dataset.

        Args:
            project: Project name
            agent_type: Agent type to extract
            min_turns_per_session: Minimum turns required
            require_session_annotation: Only include annotated sessions
            start_time: Optional time filter
            end_time: Optional time filter

        Returns:
            TrajectoryDataset with conversation trajectories

        Raises:
            ValueError: If no spans found with session_id
        """
        logger.info(f"Extracting trajectories for {agent_type} from {project}")

        # Get trace store from provider
        trace_store = self.provider.traces

        # Query spans
        spans_df = await trace_store.get_spans(
            project=project, start_time=start_time, end_time=end_time, limit=10000
        )

        if spans_df.empty:
            raise ValueError(f"No spans found in project {project}")

        # Filter by agent type
        agent_spans = self._filter_agent_spans(spans_df, agent_type)
        logger.info(f"Found {len(agent_spans)} {agent_type} spans")

        if agent_spans.empty:
            raise ValueError(f"No {agent_type} spans found in project {project}")

        # Check for session_id attribute
        session_id_col = None
        for col in ["attributes.session_id", "attributes.session.id", "session_id"]:
            if col in agent_spans.columns:
                session_id_col = col
                break

        if session_id_col is None:
            raise ValueError(
                "No session_id in span attributes - cannot group trajectories. "
                "Ensure spans are created with session_context()."
            )

        # Group by session_id
        sessions: Dict[str, List[pd.Series]] = {}
        for _, span in agent_spans.iterrows():
            session_id = span.get(session_id_col)
            if not session_id or session_id == "unknown":
                continue

            if session_id not in sessions:
                sessions[session_id] = []

            sessions[session_id].append(span)

        logger.info(f"Found {len(sessions)} unique sessions")

        if not sessions:
            raise ValueError("No spans with valid session_id found")

        # Get annotation store if required
        annotation_store = self.provider.annotations

        # Extract trajectories
        trajectories = []
        for session_id, session_spans in sessions.items():
            # Filter by minimum turns
            if len(session_spans) < min_turns_per_session:
                continue

            # Check for session annotation
            session_outcome = None
            session_score = None

            if require_session_annotation:
                # Query annotations for this session
                session_spans_df = pd.DataFrame(session_spans)
                annotations_df = await annotation_store.get_annotations(
                    spans_df=session_spans_df,
                    project=project,
                    annotation_names=["session_evaluation"],
                )

                if annotations_df.empty:
                    continue  # Skip unannotated

                # Get session outcome from annotation
                session_ann = annotations_df.iloc[0]
                session_outcome = session_ann.get("result.label")
                session_score = session_ann.get("result.score")

            # Sort spans chronologically
            session_spans.sort(key=lambda s: s.get("start_time", datetime.min))

            # Build turns
            turns = []
            for turn_idx, span in enumerate(session_spans, 1):
                turn = self._create_turn_from_span(span, turn_idx, agent_type)
                if turn:
                    turns.append(turn)

            if not turns:
                continue

            # Create trajectory
            trajectory = ConversationTrajectory(
                session_id=session_id,
                turns=turns,
                session_outcome=session_outcome,
                session_score=session_score,
                metadata={
                    "project": project,
                    "agent_type": agent_type,
                    "extracted_at": datetime.utcnow().isoformat(),
                },
            )
            trajectories.append(trajectory)

        total_turns = sum(len(t.turns) for t in trajectories)
        logger.info(
            f"Extracted {len(trajectories)} trajectories with {total_turns} total turns"
        )

        return TrajectoryDataset(
            trajectories=trajectories,
            metadata={
                "project": project,
                "agent_type": agent_type,
                "min_turns_per_session": min_turns_per_session,
                "total_sessions": len(trajectories),
                "total_turns": total_turns,
                "extracted_at": datetime.utcnow().isoformat(),
            },
        )

    def _filter_agent_spans(
        self, spans_df: pd.DataFrame, agent_type: str
    ) -> pd.DataFrame:
        """Filter spans for specific agent type."""
        agent_keywords = {
            "routing": ["routing", "route"],
            "profile_selection": ["profile", "selection"],
            "entity_extraction": ["entity", "extraction"],
        }

        keywords = agent_keywords.get(agent_type, [agent_type])

        # Filter spans by name containing keywords (case-insensitive)
        mask = spans_df["name"].str.lower().str.contains("|".join(keywords), na=False)
        return spans_df[mask].copy()

    def _create_turn_from_span(
        self, span: pd.Series, turn_idx: int, agent_type: str
    ) -> Optional[ConversationTurn]:
        """
        Create a ConversationTurn from a span.

        Args:
            span: Span data
            turn_idx: Turn index (1-based)
            agent_type: Agent type

        Returns:
            ConversationTurn or None if extraction fails
        """
        try:
            # Extract query/response from span attributes
            attributes = {k: v for k, v in span.items() if k.startswith("attributes.")}

            # Get query (input)
            query = self._extract_query(attributes)

            # Get response (output)
            response = self._extract_response(attributes)

            # Get span ID
            span_id = span.get("context.span_id", "")

            # Get timestamp
            timestamp = span.get("start_time")
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    timestamp = datetime.utcnow()

            return ConversationTurn(
                turn_id=turn_idx,
                query=query,
                response=response,
                timestamp=timestamp,
                span_id=span_id,
                metadata={
                    "agent_type": agent_type,
                    "latency_ms": span.get("attributes.latency_ms"),
                },
            )

        except Exception as e:
            logger.warning(f"Failed to create turn from span: {e}")
            return None

    def _extract_query(self, attributes: Dict[str, Any]) -> str:
        """Extract query from span attributes."""
        # Try various input attribute names
        input_keys = [
            "attributes.input.value",
            "attributes.input.query",
            "attributes.input.text",
            "attributes.query",
        ]

        for key in input_keys:
            if key in attributes and attributes[key]:
                value = attributes[key]
                # Handle JSON-encoded input
                if isinstance(value, str) and value.startswith("{"):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, dict):
                            return parsed.get("query", value)
                    except json.JSONDecodeError:
                        pass
                return str(value)

        return ""

    def _extract_response(self, attributes: Dict[str, Any]) -> str:
        """Extract response from span attributes."""
        # Try various output attribute names
        output_keys = [
            "attributes.output.value",
            "attributes.output.response",
            "attributes.output.result",
            "attributes.response",
            "attributes.result",
        ]

        for key in output_keys:
            if key in attributes and attributes[key]:
                value = attributes[key]
                # Handle dict/list outputs
                if isinstance(value, (dict, list)):
                    return json.dumps(value)
                return str(value)

        return ""
