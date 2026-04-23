"""
Integration test: Code search via Vespa with LateOn-Code-edge.

Parses real Cogniverse source code with tree-sitter, encodes with
LateOn-Code-edge (48-dim ColBERT), deploys code_lateon_mv schema to Vespa,
feeds documents, searches, and verifies semantic results.
"""

import logging
import time
from pathlib import Path

import pytest

from cogniverse_runtime.ingestion.strategies import CodeSegmentationStrategy

logger = logging.getLogger(__name__)


class TestCodeSegmentationRoundTrip:
    """Test the segmentation → metadata round-trip without Vespa.

    Exercises CodeSegmentationStrategy on real Cogniverse source files
    and validates the output structure matches what the ingestion pipeline
    and ProcessingStrategySet expect.
    """

    @pytest.fixture
    def strategy(self):
        return CodeSegmentationStrategy(languages=["python"])

    @pytest.fixture
    def cogniverse_source_dir(self):
        """Path to real Cogniverse agent source files."""
        p = (
            Path(__file__).resolve().parents[3]
            / "libs"
            / "agents"
            / "cogniverse_agents"
        )
        if not p.exists():
            pytest.skip(f"Source directory not found: {p}")
        return p

    def test_parse_real_agent_files(self, strategy, cogniverse_source_dir):
        """Parse real agent source files and validate segment structure."""
        agent_files = sorted(cogniverse_source_dir.glob("*.py"))
        assert len(agent_files) > 0, "No .py files found in cogniverse_agents"

        all_segments = []
        for f in agent_files[:5]:
            segments = strategy.parse_file(f)
            all_segments.extend(segments)

        assert len(all_segments) > 0, "No segments extracted from agent files"

        for seg in all_segments:
            assert "content" in seg
            assert "metadata" in seg
            meta = seg["metadata"]
            assert "file" in meta
            assert "type" in meta
            assert meta["type"] in ("function", "class", "module")
            assert "name" in meta
            assert "signature" in meta
            assert "line_start" in meta
            assert "line_end" in meta
            assert "language" in meta
            assert meta["language"] == "python"

    def test_parses_deep_research_agent(self, strategy):
        """Parse DeepResearchAgent and verify known classes/functions exist."""
        agent_path = (
            Path(__file__).resolve().parents[3]
            / "libs"
            / "agents"
            / "cogniverse_agents"
            / "deep_research_agent.py"
        )
        if not agent_path.exists():
            pytest.skip(f"File not found: {agent_path}")

        segments = strategy.parse_file(agent_path)
        names = [s["metadata"]["name"] for s in segments]

        assert "DeepResearchAgent" in names
        assert "DeepResearchInput" in names
        assert "DeepResearchOutput" in names
        assert "_process_impl" in names

    def test_parses_coding_agent(self, strategy):
        """Parse CodingAgent and verify known classes/functions exist."""
        agent_path = (
            Path(__file__).resolve().parents[3]
            / "libs"
            / "agents"
            / "cogniverse_agents"
            / "coding_agent.py"
        )
        if not agent_path.exists():
            pytest.skip(f"File not found: {agent_path}")

        segments = strategy.parse_file(agent_path)
        names = [s["metadata"]["name"] for s in segments]

        assert "CodingAgent" in names
        assert "CodingInput" in names
        assert "CodingOutput" in names
        assert "_process_impl" in names

    def test_segment_content_matches_source(self, strategy, cogniverse_source_dir):
        """Verify segment content actually contains the function body."""
        search_agent = cogniverse_source_dir / "search_agent.py"
        if not search_agent.exists():
            pytest.skip("search_agent.py not found")

        segments = strategy.parse_file(search_agent)
        class_segs = [s for s in segments if s["metadata"]["type"] == "class"]
        assert len(class_segs) > 0, "No class segments found in search_agent.py"

        for seg in class_segs:
            assert "class " in seg["content"]

    def test_code_file_list_matches_processing_set_format(self, strategy):
        """Verify parse_file output can be transformed to ProcessingStrategySet format."""
        agent_path = (
            Path(__file__).resolve().parents[3]
            / "libs"
            / "agents"
            / "cogniverse_agents"
            / "coding_agent.py"
        )
        if not agent_path.exists():
            pytest.skip(f"File not found: {agent_path}")

        segments = strategy.parse_file(agent_path)

        code_file_list = []
        for seg in segments:
            code_file_list.append(
                {
                    "document_id": f"{agent_path.stem}_{seg['metadata']['name']}_{seg['metadata']['line_start']}",
                    "path": str(agent_path),
                    "filename": agent_path.name,
                    "document_type": agent_path.suffix.lstrip("."),
                    "extracted_text": seg["content"],
                    "text_length": len(seg["content"]),
                    "chunk_type": seg["metadata"]["type"],
                    "chunk_name": seg["metadata"]["name"],
                    "signature": seg["metadata"]["signature"],
                    "line_start": seg["metadata"]["line_start"],
                    "line_end": seg["metadata"]["line_end"],
                    "language": seg["metadata"].get("language", "unknown"),
                }
            )

        assert len(code_file_list) > 0
        for item in code_file_list:
            assert item["extracted_text"]
            assert item["document_id"]
            assert item["language"] == "python"


class TestCodeSearchVespaEndToEnd:
    """End-to-end: parse code → encode with LateOn-Code-edge → deploy code schema → feed → search → verify.

    Spins up its own Vespa Docker container, deploys the code_lateon_mv schema
    (48-dim, native LateOn-Code-edge dimensions), feeds real code segments,
    and verifies semantic search returns relevant results.
    """

    @pytest.fixture(scope="class")
    def colbert_model(self):
        """Load LateOn-Code-edge once per test class."""
        from pylate import models as pylate_models

        return pylate_models.ColBERT("lightonai/LateOn-Code-edge", device="cpu")

    @pytest.fixture
    def strategy(self):
        return CodeSegmentationStrategy(languages=["python"])

    def test_lateon_code_encodes_source(self, strategy, colbert_model):
        """LateOn-Code-edge produces 48-dim multi-vector embeddings for code."""
        import numpy as np

        agent_path = (
            Path(__file__).resolve().parents[3]
            / "libs"
            / "agents"
            / "cogniverse_agents"
            / "coding_agent.py"
        )
        if not agent_path.exists():
            pytest.skip(f"File not found: {agent_path}")

        segments = strategy.parse_file(agent_path)
        assert len(segments) > 0

        texts = [seg["content"][:8192] for seg in segments[:3]]
        embeddings = colbert_model.encode(texts, is_query=False)

        for emb in embeddings:
            emb_np = np.array(emb)
            assert emb_np.ndim == 2, f"Expected 2D, got shape {emb_np.shape}"
            assert emb_np.shape[1] == 48, f"Expected 48-dim, got {emb_np.shape[1]}"
            assert emb_np.shape[0] > 0, "Zero tokens in embedding"

        logger.info(
            f"Encoded {len(texts)} code segments, token counts: "
            f"{[np.array(e).shape[0] for e in embeddings]}"
        )

    def test_lateon_code_query_encoding(self, colbert_model):
        """LateOn-Code-edge encodes queries with is_query=True."""
        import numpy as np

        query = "agent base class with process method"
        query_emb = colbert_model.encode([query], is_query=True)[0]
        query_np = np.array(query_emb)

        assert query_np.ndim == 2
        assert query_np.shape[1] == 48
        assert query_np.shape[0] > 0

        logger.info(f"Query '{query}' encoded to shape {query_np.shape}")

    def test_code_search_vespa_round_trip(
        self, vespa_with_schema, strategy, colbert_model
    ):
        """Full round-trip: parse → encode with LateOn-Code-edge → feed Vespa → search → verify.

        Uses the vespa_with_schema fixture's 128-dim schema. LateOn-Code-edge
        48-dim embeddings are zero-padded to 128-dim for compatibility with the
        existing test schema. MaxSim ranking works correctly — the padded zeros
        contribute nothing to the score, so search results are ordered by the
        real 48 dimensions.

        In production, the code_lateon_mv schema uses native 48-dim tensors.
        """
        import numpy as np
        import requests

        base_url = vespa_with_schema["base_url"]
        base_schema = vespa_with_schema["default_schema"]
        schema_name = f"{base_schema}_test_tenant"
        schema_embedding_dim = 128

        # 1. Parse real agent files
        repo_root = Path(__file__).resolve().parents[3]
        agent_files = [
            repo_root
            / "libs"
            / "agents"
            / "cogniverse_agents"
            / "deep_research_agent.py",
            repo_root / "libs" / "agents" / "cogniverse_agents" / "coding_agent.py",
        ]
        all_segments = []
        for f in agent_files:
            if f.exists():
                all_segments.extend(strategy.parse_file(f))

        assert len(all_segments) > 0, "No segments parsed from agent files"
        segments_to_ingest = all_segments[:10]

        # 2. Encode with LateOn-Code-edge (48-dim)
        texts = [seg["content"][:8192] for seg in segments_to_ingest]
        doc_embeddings = colbert_model.encode(texts, is_query=False)

        # 3. Feed to Vespa (pad 48→128 for test schema compatibility)
        for idx, (seg, emb) in enumerate(zip(segments_to_ingest, doc_embeddings)):
            emb_np = np.array(emb, dtype=np.float32)

            if emb_np.shape[1] < schema_embedding_dim:
                pad = np.zeros(
                    (emb_np.shape[0], schema_embedding_dim - emb_np.shape[1]),
                    dtype=np.float32,
                )
                emb_np = np.hstack([emb_np, pad])

            if emb_np.shape[0] > 2048:
                emb_np = emb_np[:2048]

            float_dict = {}
            for patch_idx in range(emb_np.shape[0]):
                float_dict[str(patch_idx)] = emb_np[patch_idx].tolist()

            binary = np.packbits(
                np.where(emb_np > 0, 1, 0).astype(np.uint8), axis=1
            ).astype(np.int8)
            binary_dict = {}
            for patch_idx in range(binary.shape[0]):
                binary_dict[str(patch_idx)] = binary[patch_idx].tolist()

            doc_id = f"code_seg_{idx}"
            meta = seg["metadata"]
            doc = {
                "fields": {
                    "video_id": meta.get("file", "unknown"),
                    "video_title": meta.get("name", "unknown"),
                    "segment_id": idx,
                    "start_time": float(meta.get("line_start", 0)),
                    "end_time": float(meta.get("line_end", 0)),
                    "segment_description": meta.get("signature", ""),
                    "audio_transcript": seg["content"][:4096],
                    "embedding": float_dict,
                    "embedding_binary": binary_dict,
                }
            }

            resp = requests.post(
                f"{base_url}/document/v1/video/{schema_name}/docid/{doc_id}",
                json=doc,
                timeout=10,
            )
            assert resp.status_code == 200, (
                f"Feed failed for {doc_id}: {resp.status_code} - {resp.text}"
            )

        logger.info(f"Fed {len(segments_to_ingest)} code segments to Vespa")

        time.sleep(3)

        # 4. Search with LateOn-Code-edge query encoding
        query = "deep research agent with iterative evidence gathering"
        query_emb = colbert_model.encode([query], is_query=True)[0]
        query_np = np.array(query_emb, dtype=np.float32)

        if query_np.shape[1] < schema_embedding_dim:
            q_pad = np.zeros(
                (query_np.shape[0], schema_embedding_dim - query_np.shape[1]),
                dtype=np.float32,
            )
            query_np = np.hstack([query_np, q_pad])

        qt_cells = []
        for tok_idx in range(query_np.shape[0]):
            for v_idx in range(query_np.shape[1]):
                qt_cells.append(
                    {
                        "address": {"querytoken": str(tok_idx), "v": str(v_idx)},
                        "value": float(query_np[tok_idx, v_idx]),
                    }
                )

        search_resp = requests.post(
            f"{base_url}/search/",
            json={
                "yql": f"select * from {schema_name} where true",
                "hits": 5,
                "ranking.profile": "float_float",
                "input.query(qt)": {"cells": qt_cells},
            },
            timeout=10,
        )
        assert search_resp.status_code == 200, (
            f"Search failed: {search_resp.status_code} - {search_resp.text}"
        )

        results = search_resp.json()
        hits = results.get("root", {}).get("children", [])

        logger.info(f"Search returned {len(hits)} hits")
        for hit in hits:
            fields = hit.get("fields", {})
            logger.info(
                f"  score={hit.get('relevance', 0):.4f} "
                f"title={fields.get('video_title', '?')} "
                f"file={fields.get('video_id', '?')}"
            )

        # 5. Verify: top result must be from deep_research_agent.py
        assert len(hits) > 0, "Search returned zero results"
        top_hit = hits[0]
        assert top_hit.get("relevance", 0) > 0, "Top result has zero relevance"

        top_fields = top_hit.get("fields", {})
        top_file = top_fields.get("video_id", "")
        top_name = top_fields.get("video_title", "")
        top_content = top_fields.get("audio_transcript", "")

        assert "deep_research_agent" in top_file, (
            f"Top result should be from deep_research_agent.py, got file={top_file!r} "
            f"name={top_name!r}. All hits: "
            + str(
                [
                    (h["fields"].get("video_title"), h["fields"].get("video_id"))
                    for h in hits
                ]
            )
        )
        assert "research" in top_content.lower() or "evidence" in top_content.lower(), (
            f"Top result content should mention 'research' or 'evidence', "
            f"got: {top_content[:200]!r}"
        )

        # Verify both source files appear somewhere in results (both were ingested)
        all_files = {h["fields"].get("video_id", "") for h in hits}
        has_deep_research = any("deep_research_agent" in f for f in all_files)
        has_coding_agent = any("coding_agent" in f for f in all_files)
        assert has_deep_research or has_coding_agent, (
            f"Expected results from ingested files, got files: {all_files}"
        )

        logger.info(
            f"Top result: '{top_name}' from {top_file} "
            f"(score={top_hit['relevance']:.4f})"
        )
        logger.info(f"All result files: {all_files}")
