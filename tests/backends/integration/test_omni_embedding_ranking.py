"""EmbeddingGemma 2 against X-CLIP (video) and CLAP (audio), recorded once.

Video: the 13 clips of ``data/testset/evaluation/sample_videos`` cut into the
chunks the ``video_xclip_sv_chunk_6s`` segmenter produces, embedded by X-CLIP
(8 frames per chunk, as the ``video_embed`` sidecar samples), by EmbeddingGemma
2 from 1 fps frames and the chunk's audio, and by EmbeddingGemma 2 from frames
only. The queries are the 125 golden retrieval queries, each with one expected
video. A video's score is its best chunk's.

Audio: the first 200 clips of the Clotho v2.1 evaluation split by file name,
each cut into consecutive windows of at most 10 s and embedded by CLAP and by
EmbeddingGemma 2 window by window. The queries are all five captions of those
clips. A clip's score is its best window's.

The vectors are recorded from the cluster's services and committed; Clotho's
audio and captions are not (their licences do not allow it), so the manifest
names clips and caption indices, and the re-record path downloads the archive
and checks its Zenodo md5. Re-record against the e2e cluster:

    RECORD_GOLDEN=1 uv run pytest \\
        tests/backends/integration/test_omni_embedding_ranking.py

Every metric is recomputed from the recording and pinned exactly, and the
EmbeddingGemma 2 chunk profile is fed through the production ingestion path
into a test-owned Vespa, where the served search must return the same top
video as the recorded vectors for every golden query.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import pytest
import requests

from cogniverse_core.common.models.model_loaders import (
    ModelLoaderFactory,
    RemoteClapClient,
    RemoteInferenceClient,
    _extract_audio_wav_b64,
    audio_windows,
    media_duration_seconds,
)
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.registries.schema_registry import DeployedSchemaNames
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.inference_specs import get_inference_service_spec
from cogniverse_runtime.ingestion.processors.single_vector_processor import (
    SingleVectorVideoProcessor,
)
from cogniverse_sdk.document import ContentType, Document, ProcessingStatus
from cogniverse_vespa.search_backend import VespaSearchBackend
from tests.utils.vespa_test_helpers import make_config_manager

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMAS_DIR = REPO_ROOT / "configs" / "schemas"
PROFILES = json.loads((REPO_ROOT / "configs" / "config.json").read_text())["backend"][
    "profiles"
]
SAMPLE_DIR = REPO_ROOT / "data/testset/evaluation/sample_videos"
GOLDEN = json.loads(
    (
        REPO_ROOT / "data/testset/evaluation/sample_videos_retrieval_queries.json"
    ).read_text()
)
FIXTURES = Path(__file__).parent / "fixtures" / "omni_eval"
MANIFEST_PATH = FIXTURES / "manifest.json"
VECTORS_PATH = FIXTURES / "vectors.npz"
CLOTHO = json.loads((FIXTURES / "clotho.json").read_text())
RECORD_GOLDEN = os.environ.get("RECORD_GOLDEN") == "1"
CLOTHO_CLIPS = 200
# The corpus videos by name: every expected video plus three distractors. The
# files are gitignored, so the recording keeps their durations.
SAMPLE_VIDEO_STEMS = sorted(
    {row["expected_videos"][0] for row in GOLDEN}
    | {"big_buck_bunny_clip", "elephant_dream_clip", "for_bigger_blazes"}
)
OMNI_VIDEO_PROFILE = "video_embeddinggemma2_sv_chunk_6s"

# Arm -> (service, vector array of its windows, vector array of its queries).
VIDEO_ARMS = {
    "xclip": ("video_embed", "xclip_chunks", "xclip_queries"),
    "omni_video_audio": ("vllm_omni_embed", "omni_av_chunks", "omni_queries"),
    "omni_video": ("vllm_omni_embed", "omni_v_chunks", "omni_queries"),
}
AUDIO_ARMS = {
    "clap": ("clap_embed", "clap_windows", "clap_captions"),
    "omni_audio": ("vllm_omni_embed", "omni_audio_windows", "omni_captions"),
}


def _sample_videos() -> list[Path]:
    return sorted(
        path for path in SAMPLE_DIR.iterdir() if path.suffix in {".mp4", ".mkv"}
    )


def _segmenter() -> SingleVectorVideoProcessor:
    return SingleVectorVideoProcessor(
        logging.getLogger("omni-eval"),
        **PROFILES["video_xclip_sv_chunk_6s"]["strategies"]["segmentation"]["params"],
    )


def _video_durations(videos: list[Path]) -> dict[str, float]:
    segmenter = _segmenter()
    return {
        video.name: segmenter._get_video_info(video)["duration"] for video in videos
    }


def _video_windows(durations: dict[str, float]) -> list[tuple[str, float, float]]:
    """The chunks the X-CLIP profile's own segmenter cuts each video into."""
    segmenter = _segmenter()
    return [
        (name, round(start, 3), round(end, 3))
        for name in sorted(durations)
        for start, end in segmenter._calculate_segment_boundaries(durations[name])
    ]


def _clotho_audio_windows(
    durations: dict[str, float],
) -> list[tuple[str, float, float]]:
    return [
        (name, round(start, 3), round(end, 3))
        for name in sorted(durations)
        for start, end in audio_windows(durations[name])
    ]


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _clotho_inputs(cache: Path) -> tuple[Path, dict[str, list[str]]]:
    """Downloads (once) and verifies Clotho's evaluation audio and captions;
    returns the directory holding the first clips and their captions."""
    import py7zr

    cache.mkdir(parents=True, exist_ok=True)
    local = {}
    for name, entry in CLOTHO["files"].items():
        path = cache / name
        if not path.exists() or _md5(path) != entry["md5"]:
            with requests.get(entry["url"], stream=True, timeout=600) as response:
                response.raise_for_status()
                with path.open("wb") as handle:
                    for block in response.iter_content(1 << 20):
                        handle.write(block)
        assert _md5(path) == entry["md5"], f"{name} does not match its Zenodo md5"
        local[name] = path
    with local["clotho_captions_evaluation.csv"].open(newline="") as handle:
        captions = {
            row["file_name"]: [row[f"caption_{i}"] for i in range(1, 6)]
            for row in csv.DictReader(handle)
        }
    clips = sorted(captions)[:CLOTHO_CLIPS]
    audio_dir = cache / "evaluation"
    missing = [clip for clip in clips if not (audio_dir / clip).exists()]
    if missing:
        with py7zr.SevenZipFile(local["clotho_audio_evaluation.7z"]) as archive:
            archive.extract(path=cache, targets=[f"evaluation/{c}" for c in missing])
    return audio_dir, {clip: captions[clip] for clip in clips}


def _record(remote_inference) -> None:
    endpoints = {
        service: remote_inference.resolve(service)
        for service in ("video_embed", "vllm_omni_embed", "clap_embed")
    }

    def omni(modalities):
        loader = ModelLoaderFactory.create_loader(
            "google/embeddinggemma-2",
            {
                "model_loader": "embeddinggemma2",
                "remote_inference_url": endpoints["vllm_omni_embed"].base_url,
                "modalities": modalities,
            },
            _resolved_headers=endpoints["vllm_omni_embed"].headers,
        )
        return loader.load_model()[0]

    xclip = RemoteInferenceClient(
        endpoints["video_embed"].base_url,
        _resolved_headers=endpoints["video_embed"].headers,
    )
    clap = RemoteClapClient(
        endpoints["clap_embed"].base_url,
        _resolved_headers=endpoints["clap_embed"].headers,
    )
    omni_av, omni_v, omni_a = omni(["video", "audio"]), omni(["video"]), omni(["audio"])

    videos = _sample_videos()
    by_name = {video.name: video for video in videos}
    video_durations = _video_durations(videos)
    video_windows = _video_windows(video_durations)
    queries = [row["query"] for row in GOLDEN]

    audio_dir, captions = _clotho_inputs(
        Path.home() / ".cache" / "cogniverse" / "clotho_v2.1"
    )
    durations = {
        clip: round(media_duration_seconds(audio_dir / clip), 3) for clip in captions
    }
    windows = _clotho_audio_windows(durations)
    caption_refs = [(clip, i) for clip in sorted(captions) for i in range(5)]

    def clap_window(clip: str, start: float, end: float) -> np.ndarray:

        return clap._request_vector(
            "/embed/audio",
            {
                "audio_b64": _extract_audio_wav_b64(
                    audio_dir / clip, start, end, sample_rate=48000
                )
            },
        )

    vectors = {
        "xclip_chunks": [
            xclip.embed_video_segment(by_name[v], s, e) for v, s, e in video_windows
        ],
        "omni_av_chunks": [
            omni_av.process_video_segment(by_name[v], s, e)["embeddings_np"]
            for v, s, e in video_windows
        ],
        "omni_v_chunks": [
            omni_v.process_video_segment(by_name[v], s, e)["embeddings_np"]
            for v, s, e in video_windows
        ],
        "xclip_queries": [xclip.embed_text(q) for q in queries],
        "omni_queries": [omni_av.embed_text(q) for q in queries],
        "clap_windows": [clap_window(c, s, e) for c, s, e in windows],
        "omni_audio_windows": [
            omni_a.embed_audio_window(audio_dir / c, s, e) for c, s, e in windows
        ],
        "clap_captions": [
            clap.generate_acoustic_text_embedding(captions[c][i])
            for c, i in caption_refs
        ],
        "omni_captions": [omni_a.embed_text(captions[c][i]) for c, i in caption_refs],
    }
    FIXTURES.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        VECTORS_PATH,
        **{
            name: _unit(np.asarray(rows, dtype=np.float32)).astype(np.float16)
            for name, rows in vectors.items()
        },
    )
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "services": {
                    service: {
                        "model": endpoint.model_id,
                        "revision": endpoint.model_revision,
                    }
                    for service, endpoint in endpoints.items()
                },
                "video_durations": video_durations,
                "video_windows": [list(w) for w in video_windows],
                "clotho_durations": durations,
                "audio_windows": [list(w) for w in windows],
                "captions": [list(ref) for ref in caption_refs],
            },
            indent=1,
        )
        + "\n"
    )


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _load() -> tuple[dict, dict[str, np.ndarray]]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    with np.load(VECTORS_PATH) as recorded:
        vectors = {name: recorded[name].astype(np.float32) for name in recorded.files}
    return manifest, vectors


def _validate(manifest: dict, vectors: dict[str, np.ndarray]) -> None:
    """The recording's shape, derived from production code, never restated."""
    assert manifest["services"] == {
        service: {
            "model": get_inference_service_spec(service).model_id,
            "revision": get_inference_service_spec(service).model_revision,
        }
        for service in ("video_embed", "vllm_omni_embed", "clap_embed")
    }
    assert sorted(Path(n).stem for n in manifest["video_durations"]) == (
        SAMPLE_VIDEO_STEMS
    )
    assert [tuple(w) for w in manifest["video_windows"]] == _video_windows(
        manifest["video_durations"]
    )
    assert sorted(manifest["clotho_durations"]) == CLOTHO["clips"]
    assert [tuple(w) for w in manifest["audio_windows"]] == _clotho_audio_windows(
        manifest["clotho_durations"]
    )
    assert manifest["captions"] == [
        [clip, i] for clip in CLOTHO["clips"] for i in range(5)
    ]
    rows = {
        "video_windows": len(manifest["video_windows"]),
        "golden": len(GOLDEN),
        "audio_windows": len(manifest["audio_windows"]),
        "captions": len(manifest["captions"]),
    }
    expected_shapes = {
        "xclip_chunks": (rows["video_windows"], "video_embed"),
        "omni_av_chunks": (rows["video_windows"], "vllm_omni_embed"),
        "omni_v_chunks": (rows["video_windows"], "vllm_omni_embed"),
        "xclip_queries": (rows["golden"], "video_embed"),
        "omni_queries": (rows["golden"], "vllm_omni_embed"),
        "clap_windows": (rows["audio_windows"], "clap_embed"),
        "omni_audio_windows": (rows["audio_windows"], "vllm_omni_embed"),
        "clap_captions": (rows["captions"], "clap_embed"),
        "omni_captions": (rows["captions"], "vllm_omni_embed"),
    }
    assert {name: array.shape for name, array in vectors.items()} == {
        name: (count, get_inference_service_spec(service).output_dimension)
        for name, (count, service) in expected_shapes.items()
    }
    for name, array in vectors.items():
        norms = np.linalg.norm(array, axis=1)
        assert float(np.max(np.abs(norms - 1.0))) < 1e-3, name


@pytest.fixture(scope="module")
def recording(request) -> tuple[dict, dict[str, np.ndarray]]:
    if RECORD_GOLDEN:
        _record(request.getfixturevalue("remote_inference"))
    manifest, vectors = _load()
    _validate(manifest, vectors)
    return manifest, vectors


def _item_ranking(
    queries: np.ndarray, windows: np.ndarray, owners: list[str]
) -> list[list[str]]:
    """Items ranked per query by their best window's cosine."""
    names = sorted(set(owners))
    index = {name: i for i, name in enumerate(names)}
    owner_ids = np.asarray([index[o] for o in owners])
    scores = queries @ windows.T
    best = np.full((len(queries), len(names)), -np.inf)
    for column, owner in enumerate(owner_ids):
        best[:, owner] = np.maximum(best[:, owner], scores[:, column])
    order = np.argsort(-best, axis=1, kind="stable")
    return [[names[i] for i in row] for row in order]


def _metrics(rankings: list[list[str]], expected: list[str]) -> dict:
    ranks = [ranking.index(item) + 1 for ranking, item in zip(rankings, expected)]
    return {
        "recall_at_1": sum(rank == 1 for rank in ranks),
        "recall_at_10": sum(rank <= 10 for rank in ranks),
        "mrr": round(float(np.mean([1 / rank for rank in ranks])), 4),
        "correct_at_1": [i for i, rank in enumerate(ranks) if rank == 1],
    }


def _video_rankings(manifest, vectors, arm: str) -> list[list[str]]:
    _, windows, queries = VIDEO_ARMS[arm]
    owners = [Path(w[0]).stem for w in manifest["video_windows"]]
    return _item_ranking(vectors[queries], vectors[windows], owners)


def _audio_rankings(manifest, vectors, arm: str) -> list[list[str]]:
    _, windows, queries = AUDIO_ARMS[arm]
    owners = [w[0] for w in manifest["audio_windows"]]
    return _item_ranking(vectors[queries], vectors[windows], owners)


GOLDEN_EXPECTED = [row["expected_videos"][0] for row in GOLDEN]


@pytest.mark.parametrize("arm", sorted(VIDEO_ARMS))
def test_video_arm_metrics(recording, arm):
    manifest, vectors = recording
    metrics = _metrics(_video_rankings(manifest, vectors, arm), GOLDEN_EXPECTED)

    assert {k: v for k, v in metrics.items() if k != "recall_at_10"} == (
        VIDEO_METRICS[arm]
    )


@pytest.mark.parametrize("arm", sorted(AUDIO_ARMS))
def test_audio_arm_metrics(recording, arm):
    manifest, vectors = recording
    expected = [clip for clip, _ in manifest["captions"]]
    metrics = _metrics(_audio_rankings(manifest, vectors, arm), expected)

    assert {k: v for k, v in metrics.items() if k != "correct_at_1"} == (
        AUDIO_METRICS[arm]
    )


MUTATIONS = {
    "drop a chunk vector": lambda m, v: v.__setitem__(
        "omni_av_chunks", v["omni_av_chunks"][1:]
    ),
    "widen a query vector": lambda m, v: v.__setitem__(
        "omni_queries", np.pad(v["omni_queries"], ((0, 0), (0, 1)))
    ),
    "unnormalize a window": lambda m, v: v["clap_windows"].__setitem__(
        0, v["clap_windows"][0] * 2
    ),
    "stale revision": lambda m, v: m["services"]["vllm_omni_embed"].__setitem__(
        "revision", "0" * 40
    ),
    "swap a clip": lambda m, v: m["clotho_durations"].__setitem__(
        "not_a_clotho_clip.wav", 1.0
    ),
    "move a chunk boundary": lambda m, v: m["video_windows"][0].__setitem__(2, 5.5),
    "change a duration": lambda m, v: m["video_durations"].__setitem__(
        "v_-6dz6tBH77I.mp4", 9.5
    ),
}


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_each_pin_goes_red_on_a_drifted_recording(recording, mutation):
    manifest, vectors = (
        copy.deepcopy(recording[0]),
        {name: array.copy() for name, array in recording[1].items()},
    )
    MUTATIONS[mutation](manifest, vectors)

    with pytest.raises(AssertionError):
        _validate(manifest, vectors)


@pytest.fixture(scope="module")
def omni_video_index(vespa_instance, recording):
    """The EmbeddingGemma 2 chunks fed through production ingestion."""
    manifest, vectors = recording
    tenant = f"omnirank{uuid.uuid4().hex[:8]}:unit"
    config_manager = make_config_manager(vespa_instance)
    ingestion = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        tenant_id=tenant,
        config={
            "backend": {
                "url": "http://localhost",
                "config_port": vespa_instance["config_port"],
                "port": vespa_instance["http_port"],
            }
        },
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
    )
    ingestion.schema_registry.deploy_schema(
        tenant_id=tenant, base_schema_name=OMNI_VIDEO_PROFILE
    )
    documents = []
    for i, (video, start, end) in enumerate(manifest["video_windows"]):
        stem = Path(video).stem
        document = Document(
            id=f"{stem}_{i:04d}",
            content_type=ContentType.VIDEO,
            content_id=stem,
            status=ProcessingStatus.COMPLETED,
        )
        document.add_embedding(
            "embedding", vectors["omni_av_chunks"][i], {"type": "float", "raw": True}
        )
        document.add_metadata("video_id", stem)
        document.add_metadata("video_title", video)
        document.add_metadata("segment_index", i)
        document.add_metadata("start_time", float(start))
        document.add_metadata("end_time", float(end))
        documents.append(document)
    for begin in range(0, len(documents), 50):
        result = ingestion.ingest_documents(
            documents[begin : begin + 50], OMNI_VIDEO_PROFILE
        )
        assert result["failed_count"] == 0, result["failed_documents"]
    search = VespaSearchBackend(
        config={
            "url": "http://localhost",
            "port": vespa_instance["http_port"],
            "profiles": {OMNI_VIDEO_PROFILE: PROFILES[OMNI_VIDEO_PROFILE]},
        },
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(SCHEMAS_DIR),
        is_schema_deployed=DeployedSchemaNames(config_manager),
        enable_connection_pool=False,
    )
    yield search, tenant
    search.close()


def test_served_search_returns_the_recorded_top_video_for_every_query(
    omni_video_index, recording
):
    manifest, vectors = recording
    search, tenant = omni_video_index
    recorded_top = [
        ranking[0] for ranking in _video_rankings(manifest, vectors, "omni_video_audio")
    ]

    served_top = [
        Path(
            search.search(
                {
                    "query": row["query"],
                    "type": "video",
                    "profile": OMNI_VIDEO_PROFILE,
                    "strategy": "float_float",
                    "top_k": 1,
                    "tenant_id": tenant,
                    "query_embeddings": vectors["omni_queries"][i],
                }
            )[0].document.metadata["video_title"]
        ).stem
        for i, row in enumerate(GOLDEN)
    ]

    assert served_top == recorded_top


# Measured from the committed recording (see the module docstring).
VIDEO_METRICS: dict = {}
AUDIO_METRICS: dict = {}
