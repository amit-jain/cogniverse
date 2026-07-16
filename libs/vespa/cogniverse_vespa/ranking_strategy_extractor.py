"""
Ranking strategy extractor for Vespa schemas.
Extracts ranking profile configurations from schema JSON files for use by search backends.
"""

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# A dense 1-d indexed tensor (ANN-capable attribute) — mapped/multi-dim
# tensors (multi-vector patch embeddings) never match.
_DENSE_TENSOR_TYPE_RE = re.compile(
    r"^tensor<(?P<cell>float|bfloat16|double|int8)>\(\s*\w+\[\d+\]\s*\)$"
)


def _dense_cell_type(field_type: str) -> Optional[str]:
    """Cell type of a dense 1-d indexed tensor field, else None."""
    m = _DENSE_TENSOR_TYPE_RE.match((field_type or "").strip())
    return m.group("cell") if m else None


# Memoize the per-directory extraction: the schema JSONs are static config, so
# re-globbing and re-parsing ~21 files on every /search/strategies call (and
# every backend strategy resolution) is pure waste. Keyed by (resolved dir, the
# set of (filename, mtime_ns)) so a genuine schema edit invalidates the entry
# while an unchanged dir is a cheap glob+stat instead of 21 json.load calls.
_ALL_STRATEGIES_CACHE: Dict[
    Tuple[str, frozenset], Dict[str, Dict[str, "RankingStrategyInfo"]]
] = {}
_ALL_STRATEGIES_LOCK = threading.Lock()


class SearchStrategyType(Enum):
    """Types of search strategies"""

    PURE_VISUAL = "pure_visual"
    PURE_TEXT = "pure_text"
    HYBRID = "hybrid"


@dataclass
class RankingStrategyInfo:
    """Information about a ranking strategy extracted from schema"""

    name: str
    strategy_type: SearchStrategyType
    needs_float_embeddings: bool = False
    needs_binary_embeddings: bool = False
    needs_text_query: bool = False
    use_nearestneighbor: bool = False
    nearestneighbor_field: Optional[str] = None
    nearestneighbor_tensor: Optional[str] = None
    embedding_field: Optional[str] = None
    query_tensor_name: Optional[str] = None
    timeout: float = 2.0
    description: str = ""
    inputs: Dict[str, str] = field(default_factory=dict)
    query_tensors_needed: List[str] = field(default_factory=list)
    schema_name: str = ""


class RankingStrategyExtractor:
    """Extracts ranking strategies from Vespa schema JSON files"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_from_schema(self, schema_path: Path) -> Dict[str, RankingStrategyInfo]:
        """Extract ranking strategies from a schema JSON file"""

        with open(schema_path, "r") as f:
            schema_json = json.load(f)

        schema_name = schema_json.get("schema", schema_json.get("name", ""))

        fields = {f["name"]: f for f in schema_json["document"]["fields"]}

        strategies = {}

        for profile in schema_json.get(
            "rank-profiles", schema_json.get("rank_profiles", [])
        ):
            strategy_info = self._parse_ranking_profile(profile, fields, schema_name)
            strategies[strategy_info.name] = strategy_info

        return strategies

    def _parse_ranking_profile(
        self,
        profile: Dict[str, Any],
        fields: Dict[str, Dict],
        schema_name: str,
    ) -> RankingStrategyInfo:
        """Parse a single ranking profile"""

        profile_name = profile["name"]

        # Parse inputs: strip the "query(name)" wrapper to get the bare parameter name
        inputs = {}
        for input_def in profile.get("inputs", []):
            input_name = input_def["name"]
            if "(" in input_name and ")" in input_name:
                param_name = input_name[input_name.find("(") + 1 : input_name.find(")")]
            else:
                param_name = input_name
            inputs[param_name] = input_def["type"]

        needs_float_embeddings = any("float" in t for t in inputs.values())
        needs_binary_embeddings = any("int8" in t for t in inputs.values())

        first_phase = profile.get("first-phase", profile.get("first_phase", {}))
        if isinstance(first_phase, dict):
            first_phase_expr = first_phase.get("expression", "")
        else:
            first_phase_expr = str(first_phase)

        needs_text_query = (
            "bm25" in profile_name.lower()
            or "bm25(" in first_phase_expr
            or "userInput" in first_phase_expr
            # Token match — a bare substring test classified any name merely
            # embedding the letters (e.g. "context_boost") as text-seeking.
            or "text" in profile_name.lower().split("_")
        )

        if needs_text_query and not (needs_float_embeddings or needs_binary_embeddings):
            strategy_type = SearchStrategyType.PURE_TEXT
        elif (
            needs_float_embeddings or needs_binary_embeddings
        ) and not needs_text_query:
            strategy_type = SearchStrategyType.PURE_VISUAL
        else:
            strategy_type = SearchStrategyType.HYBRID

        # nearestNeighbor (ANN) is derived structurally: the profile's FIRST
        # phase must score against a dense 1-d embedding attribute (directly
        # via closeness/attribute or through a profile function). The previous
        # profile-NAME allowlist silently dropped ANN for any profile named
        # outside it — the wiki hybrid/semantic_search profiles ranked
        # BM25-only with a dead closeness term. Multi-vector (mapped) fields
        # never match; text-first profiles (bm25 first phase with a vector
        # second phase) correctly stay off ANN.
        use_nearestneighbor = False
        nearestneighbor_field = None
        nearestneighbor_tensor = None

        if strategy_type in [
            SearchStrategyType.PURE_VISUAL,
            SearchStrategyType.HYBRID,
        ]:
            ann_field = self._first_phase_embedding_field(profile)
            cell = (
                _dense_cell_type(fields.get(ann_field, {}).get("type", ""))
                if ann_field
                else None
            )
            if cell is not None:
                want_int8 = cell == "int8"
                for input_name, input_type in inputs.items():
                    if ("int8" in input_type) == want_int8:
                        use_nearestneighbor = True
                        nearestneighbor_field = ann_field
                        nearestneighbor_tensor = input_name
                        break

        query_tensor_name = None

        if "q" in inputs:
            query_tensor_name = "q"
        elif "qt" in inputs:
            query_tensor_name = "qt"
        elif "qtb" in inputs:
            query_tensor_name = "qtb"
        else:
            for inp_name in inputs:
                query_tensor_name = inp_name
                break

        embedding_field = self._extract_embedding_field(profile, query_tensor_name)

        # If expression parsing yields nothing, fall back to convention-based names
        if not embedding_field:
            if query_tensor_name == "q":
                embedding_field = "embeddings"  # plural for chunks schema
            elif query_tensor_name == "qt":
                embedding_field = "embedding"
            elif query_tensor_name == "qtb":
                embedding_field = "embedding_binary"

        description = self._generate_description(
            profile_name, strategy_type, needs_float_embeddings, needs_binary_embeddings
        )

        query_tensors_needed = list(inputs.keys())

        return RankingStrategyInfo(
            name=profile_name,
            strategy_type=strategy_type,
            needs_float_embeddings=needs_float_embeddings,
            needs_binary_embeddings=needs_binary_embeddings,
            needs_text_query=needs_text_query,
            use_nearestneighbor=use_nearestneighbor,
            nearestneighbor_field=nearestneighbor_field,
            nearestneighbor_tensor=nearestneighbor_tensor,
            embedding_field=embedding_field,
            query_tensor_name=query_tensor_name,
            timeout=profile.get("timeout", 2.0),
            description=description,
            inputs=inputs,
            query_tensors_needed=query_tensors_needed,
            schema_name=schema_name,
        )

    def _first_phase_embedding_field(self, profile: Dict[str, Any]) -> Optional[str]:
        """Embedding attribute the FIRST phase scores against, or None.

        Resolves profile-function indirection (``first_phase: visual_sim``
        with ``visual_sim = closeness(field, embedding)``) by substituting
        function bodies, then extracts the closeness/attribute reference.
        Only the first phase matters — it drives retrieval; a second-phase
        vector rerank on top of a bm25 first phase must not switch retrieval
        to ANN.
        """
        functions = {
            f.get("name", ""): f.get("expression", "")
            for f in profile.get("functions", [])
        }
        first_phase = profile.get("first_phase", profile.get("first-phase", ""))
        if isinstance(first_phase, dict):
            expr = first_phase.get("expression", "")
        else:
            expr = str(first_phase or "")

        for _ in range(4):  # bounded function-indirection depth
            expanded = expr
            for name, body in functions.items():
                if name:
                    expanded = re.sub(rf"\b{re.escape(name)}\b", f"({body})", expanded)
            if expanded == expr:
                break
            expr = expanded

        m = re.search(r"closeness\(field,\s*(\w+)\)", expr)
        if m:
            return m.group(1)
        m = re.search(r"attribute\((\w+)\)", expr)
        if m:
            return m.group(1)
        return None

    def _extract_embedding_field(
        self, profile: Dict[str, Any], query_tensor_name: str | None
    ) -> str | None:
        """Extract the actual embedding field name from rank profile expressions.

        Parses attribute(field_name) and closeness(field, field_name) references
        from function expressions, first-phase, and second-phase to find the
        real schema field being queried — rather than assuming hardcoded names.
        """
        import re

        all_expressions = []
        for func in profile.get("functions", []):
            all_expressions.append(func.get("expression", ""))

        first_phase = profile.get("first_phase", profile.get("first-phase", ""))
        if isinstance(first_phase, dict):
            all_expressions.append(first_phase.get("expression", ""))
        elif isinstance(first_phase, str):
            all_expressions.append(first_phase)

        second_phase = profile.get("second_phase", profile.get("second-phase", {}))
        if isinstance(second_phase, dict):
            all_expressions.append(second_phase.get("expression", ""))

        # Extract attribute(field_name) and closeness(field, field_name) references
        attr_fields = []
        for expr in all_expressions:
            attr_fields.extend(re.findall(r"attribute\((\w+)\)", expr))
            attr_fields.extend(re.findall(r"closeness\(field,\s*(\w+)\)", expr))

        if not attr_fields:
            return None

        # Match the field to the query tensor type
        if query_tensor_name in ("qt", "q"):
            # Float tensor — prefer non-binary embedding field
            for f in attr_fields:
                if "binary" not in f:
                    return f
        elif query_tensor_name == "qtb":
            # Binary tensor — prefer binary embedding field
            for f in attr_fields:
                if "binary" in f:
                    return f

        # Return first field found as fallback
        return attr_fields[0] if attr_fields else None

    def _generate_description(
        self,
        profile_name: str,
        strategy_type: SearchStrategyType,
        needs_float: bool,
        needs_binary: bool,
    ) -> str:
        """Generate human-readable description"""

        descriptions = {
            "default": "Default ranking profile",
            "bm25_only": "Pure text search using BM25",
            "bm25_no_description": "BM25 text search excluding descriptions",
            "float_float": "Visual search with float embeddings",
            "binary_binary": "Fast visual search with binary embeddings",
            "float_binary": "Float query with binary document embeddings",
            "phased": "Two-phase ranking: binary first, float reranking",
            "hybrid_float_bm25": "Combined visual (float) and text search",
            "hybrid_binary_bm25": "Combined visual (binary) and text search",
            "hybrid_bm25_binary": "Text-first search with visual reranking",
            "hybrid_bm25_float": "Text-first search with visual reranking",
        }

        # Check for no_description variant
        if "no_description" in profile_name:
            base_name = profile_name.replace("_no_description", "")
            if base_name in descriptions:
                return descriptions[base_name] + " (excluding descriptions)"

        if profile_name in descriptions:
            return descriptions[profile_name]

        # Generate based on strategy type
        if strategy_type == SearchStrategyType.PURE_TEXT:
            return "Text-based search"
        elif strategy_type == SearchStrategyType.PURE_VISUAL:
            if needs_binary:
                return "Binary embedding search"
            else:
                return "Float embedding search"
        else:
            return "Hybrid text and visual search"


def extract_all_ranking_strategies(
    schema_dir: Path,
) -> Dict[str, Dict[str, RankingStrategyInfo]]:
    """Extract ranking strategies from all schemas in a directory.

    Memoized by directory content signature: repeated calls for an unchanged
    schema dir reuse the parsed result (a schema edit invalidates it).
    """
    schema_files = sorted(schema_dir.glob("*.json"))
    signature_parts = []
    for f in schema_files:
        if f.name == "ranking_strategies.json":
            continue
        try:
            signature_parts.append((f.name, f.stat().st_mtime_ns))
        except OSError:
            # File vanished between glob and stat (concurrent rewrite) —
            # treat it as absent rather than failing the whole listing.
            continue
    signature = frozenset(signature_parts)
    dir_key = str(schema_dir.resolve())
    cache_key = (dir_key, signature)

    with _ALL_STRATEGIES_LOCK:
        cached = _ALL_STRATEGIES_CACHE.get(cache_key)
    if cached is not None:
        # Shallow copy so a caller mutating the mapping can't poison the memo.
        return dict(cached)

    extractor = RankingStrategyExtractor()
    all_strategies = {}

    for schema_file in schema_files:
        # Skip ranking_strategies.json
        if schema_file.name == "ranking_strategies.json":
            continue

        try:
            schema_name = schema_file.stem.removesuffix("_schema")
            strategies = extractor.extract_from_schema(schema_file)
            all_strategies[schema_name] = strategies
            logger.info(f"Extracted {len(strategies)} strategies from {schema_name}")
        except Exception as e:
            logger.error(f"Failed to extract strategies from {schema_file}: {e}")

    with _ALL_STRATEGIES_LOCK:
        # Single entry per directory: a schema edit REPLACES the dir's memo
        # instead of accreting one dead entry per edit for the process's life.
        for stale_key in [k for k in _ALL_STRATEGIES_CACHE if k[0] == dir_key]:
            del _ALL_STRATEGIES_CACHE[stale_key]
        _ALL_STRATEGIES_CACHE[cache_key] = all_strategies
    return dict(all_strategies)


def save_ranking_strategies(
    strategies: Dict[str, Dict[str, RankingStrategyInfo]], output_path: Path
):
    """Save extracted ranking strategies to JSON file"""

    # Convert to serializable format
    output = {}
    for schema_name, schema_strategies in strategies.items():
        output[schema_name] = {}
        for strategy_name, strategy_info in schema_strategies.items():
            output[schema_name][strategy_name] = {
                "name": strategy_info.name,
                "strategy_type": strategy_info.strategy_type.value,
                "needs_float_embeddings": strategy_info.needs_float_embeddings,
                "needs_binary_embeddings": strategy_info.needs_binary_embeddings,
                "needs_text_query": strategy_info.needs_text_query,
                "use_nearestneighbor": strategy_info.use_nearestneighbor,
                "nearestneighbor_field": strategy_info.nearestneighbor_field,
                "nearestneighbor_tensor": strategy_info.nearestneighbor_tensor,
                "embedding_field": strategy_info.embedding_field,
                "query_tensor_name": strategy_info.query_tensor_name,
                "timeout": strategy_info.timeout,
                "description": strategy_info.description,
                "inputs": strategy_info.inputs,
                "query_tensors_needed": strategy_info.query_tensors_needed,
                "schema_name": strategy_info.schema_name,
            }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved ranking strategies to {output_path}")
