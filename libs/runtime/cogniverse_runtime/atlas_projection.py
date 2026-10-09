"""UMAP maps of a tenant's documents under one profile.

A map places each document's pooled embedding on a 2D UMAP layout, groups
the layout into automatic clusters named by their most distinctive terms, and
keeps the fitted layout so queries can be placed on it and compared with
every document.

Maps are cached per tenant, profile and document limit. Each tenant and
profile has a generation counter in Redis; a cached map is used only while
its generation is current, so ``invalidate`` on any replica makes every
replica rebuild on its next read. A built map is stored in Redis under its
generation, so every worker process and replica serves the one layout; one
process builds it while the others wait for it, and concurrent reads within
a process share one build. A stored map is a compressed numpy archive read
without pickle: arrays for the layout and the fitted reducer's state, and
JSON for the rest; an entry that does not read as one is discarded and the
map built again. A Redis that does not answer raises
``AtlasCacheUnavailableError``; no map is built without the cache.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from redis.exceptions import RedisError

# UMAP cannot lay out fewer documents than this.
MIN_DOCUMENTS = 4
# The fixed seed that makes a layout reproducible for a given set.
UMAP_SEED = 42
UNCLUSTERED = -1
CLUSTER_LABEL_TERMS = 3
CACHE_CAPACITY = 8
GENERATION_KEY = "cogniverse:embedding-atlas:generation:{tenant}:{profile}"
LAYOUT_KEY = "cogniverse:embedding-atlas:layout:{tenant}:{profile}:{limit}:{generation}"
BUILD_KEY = "cogniverse:embedding-atlas:build:{tenant}:{profile}:{limit}:{generation}"
# A stored map is kept this long; a newer generation retires it before then.
LAYOUT_TTL_S = 24 * 60 * 60
# How long one process holds the build of a map before another may take it.
BUILD_LEASE_S = 300
# How often a process waiting for another's build looks for the stored map.
BUILD_POLL_S = 0.25
_RELEASE_BUILD_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) end return 0"
)
# File extensions document titles carry, never a cluster's name.
FILE_EXTENSIONS = frozenset(
    "avi csv doc docx flac gif htm html jpeg jpg json m4a md mkv mov mp3 mp4 "
    "mpeg ogg pdf png ppt pptx srt txt vtt wav webm webp xls xlsx".split()
)
_FILE_EXTENSION = re.compile(
    r"\.(?:" + "|".join(sorted(FILE_EXTENSIONS)) + r")$", re.IGNORECASE
)
_WORD_JOINERS = re.compile(r"[_.\-]+")

logger = logging.getLogger(__name__)


@dataclass
class DocumentMap:
    """One built map: the documents, their unit pooled vectors, the fitted
    layout, each document's place and cluster, and the cluster names."""

    documents: List[Dict[str, Any]]
    # None for a map laid out from given places without vectors.
    vectors: Optional[np.ndarray]
    # None for a map laid out from given places.
    reducer: Any
    coords: np.ndarray
    clusters: np.ndarray
    cluster_names: Dict[int, str]
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generation: int = 0
    # JSON facts about what the caller read the documents from, kept with
    # the map.
    source: Any = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class AtlasCacheUnavailableError(RuntimeError):
    """The shared-state Redis holding the maps did not answer."""


class UnreadableLayoutError(ValueError):
    """A stored map is not an archive this module writes."""


# Fitted UMAP functions, chosen by the reducer's metric names when it is read.
_REDUCER_FUNCTIONS = (
    "_input_distance_func",
    "_inverse_distance_func",
    "_output_distance_func",
)
_REDUCER_ARRAY = "reducer/"
_REDUCER_SPARSE = ("data", "indices", "indptr")


def _reducer_state(reducer: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """A fitted reducer's attributes as arrays and JSON, its numba distance
    functions left out. An attribute of any other kind raises: a reducer
    that cannot be read back is not stored."""
    import scipy.sparse

    arrays: Dict[str, np.ndarray] = {}
    values: Dict[str, Any] = {}
    sparse: Dict[str, List[int]] = {}
    tuples: List[str] = []
    for name, value in vars(reducer).items():
        if name in _REDUCER_FUNCTIONS:
            continue
        if isinstance(value, np.ndarray):
            arrays[f"{_REDUCER_ARRAY}{name}"] = value
        elif scipy.sparse.issparse(value):
            matrix = value.tocsr()
            for part in _REDUCER_SPARSE:
                arrays[f"{_REDUCER_ARRAY}{name}.{part}"] = getattr(matrix, part)
            sparse[name] = list(matrix.shape)
        elif isinstance(value, np.generic):
            values[name] = value.item()
        else:
            if isinstance(value, tuple):
                tuples.append(name)
            try:
                json.dumps(value)
            except TypeError as exc:
                raise TypeError(
                    f"UMAP reducer attribute {name} ({type(value).__name__}) "
                    "cannot be stored"
                ) from exc
            values[name] = value
    return arrays, {"values": values, "sparse": sparse, "tuples": tuples}


def _restored_reducer(arrays: Dict[str, np.ndarray], state: Dict[str, Any]) -> Any:
    """The reducer ``_reducer_state`` stored, its distance functions chosen
    by its metric names as fitting chose them."""
    import scipy.sparse
    from umap import UMAP
    from umap import distances as umap_distances

    reducer = UMAP.__new__(UMAP)
    attributes: Dict[str, Any] = dict(state["values"])
    for name in state["tuples"]:
        attributes[name] = tuple(attributes[name])
    for name, value in arrays.items():
        if name.startswith(_REDUCER_ARRAY) and "." not in name:
            attributes[name[len(_REDUCER_ARRAY) :]] = value
    for name, shape in state["sparse"].items():
        data, indices, indptr = (
            arrays[f"{_REDUCER_ARRAY}{name}.{part}"] for part in _REDUCER_SPARSE
        )
        attributes[name] = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=tuple(shape)
        )
    metric, output_metric = attributes["metric"], attributes["output_metric"]
    if attributes["_sparse_data"]:
        raise UnreadableLayoutError("a reducer fitted on sparse data is not read")
    attributes["_input_distance_func"] = umap_distances.named_distances[metric]
    attributes["_inverse_distance_func"] = (
        umap_distances.named_distances_with_gradients.get(metric)
    )
    attributes["_output_distance_func"] = umap_distances.named_distances_with_gradients[
        output_metric
    ]
    reducer.__dict__.update(attributes)
    return reducer


def _encoded(document_map: DocumentMap) -> str:
    """``document_map`` as a base64 compressed numpy archive."""
    arrays: Dict[str, np.ndarray] = {
        "coords": document_map.coords,
        "clusters": np.asarray(document_map.clusters),
    }
    if document_map.vectors is not None:
        arrays["vectors"] = document_map.vectors
    reducer = None
    if document_map.reducer is not None:
        reducer_arrays, reducer = _reducer_state(document_map.reducer)
        arrays.update(reducer_arrays)
    meta = {
        "documents": document_map.documents,
        "cluster_names": [
            [int(cluster), name] for cluster, name in document_map.cluster_names.items()
        ],
        "computed_at": document_map.computed_at.isoformat(),
        "generation": document_map.generation,
        "source": document_map.source,
        "reducer": reducer,
    }
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return base64.b64encode(buffer.getvalue()).decode()


def _decoded(payload: str) -> DocumentMap:
    """The map ``_encoded`` wrote; ``UnreadableLayoutError`` for anything
    else. Object arrays are refused, so nothing in the payload is unpickled."""
    try:
        with np.load(
            io.BytesIO(base64.b64decode(payload, validate=True)), allow_pickle=False
        ) as archive:
            arrays = {name: archive[name] for name in archive.files}
        meta = json.loads(arrays.pop("meta").tobytes())
        reducer = (
            None
            if meta["reducer"] is None
            else _restored_reducer(arrays, meta["reducer"])
        )
        coords, clusters = arrays["coords"], arrays["clusters"]
        documents = meta["documents"]
        if coords.shape != (len(documents), 2) or clusters.shape != (len(documents),):
            raise UnreadableLayoutError(
                f"{len(documents)} documents with coords {coords.shape} and "
                f"clusters {clusters.shape}"
            )
        return DocumentMap(
            documents=documents,
            vectors=arrays.get("vectors"),
            reducer=reducer,
            coords=coords,
            clusters=clusters,
            cluster_names={
                int(cluster): name for cluster, name in meta["cluster_names"]
            },
            computed_at=datetime.fromisoformat(meta["computed_at"]),
            generation=meta["generation"],
            source=meta["source"],
        )
    except UnreadableLayoutError:
        raise
    except Exception as exc:
        raise UnreadableLayoutError(
            f"not a stored embedding atlas map: {type(exc).__name__}: {exc}"
        ) from exc


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Each row scaled to length one; an all-zero row stays zero."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def umap_layout(vectors: np.ndarray) -> Tuple[Any, np.ndarray]:
    """The fitted UMAP reducer of ``vectors`` and each row's 2D place."""
    from umap import UMAP

    reducer = UMAP(
        n_components=2,
        n_neighbors=min(15, len(vectors) - 1),
        random_state=UMAP_SEED,
        n_jobs=1,
    )
    coords = reducer.fit_transform(vectors)
    return reducer, np.asarray(coords, dtype=np.float64)


def label_text(text: str) -> str:
    """``text`` as cluster naming reads it: each word's trailing file
    extension cut off, the rest split at ``_``, ``-`` and ``.``, and the
    pieces holding a digit (ids, frame numbers) left out, so
    ``for_bigger_blazes.mp4`` reads ``for bigger blazes``, ``rivers-1.txt``
    reads ``rivers`` and ``v_-6Os86HzwCs`` reads ``v``."""
    pieces = (
        piece
        for word in text.split()
        for piece in _WORD_JOINERS.split(_FILE_EXTENSION.sub("", word))
    )
    return " ".join(p for p in pieces if p and not any(ch.isdigit() for ch in p))


def _unique_names(
    ranked_terms: Dict[int, List[str]],
) -> Dict[int, str]:
    """Each cluster's name from its ranked terms, unique within the map: a
    name an earlier cluster (by id) already has takes the cluster's next
    terms until it differs, and failing that the cluster's number."""
    names: Dict[int, str] = {}
    taken = set()
    for cluster in sorted(ranked_terms):
        terms = ranked_terms[cluster]
        name = ", ".join(terms[:CLUSTER_LABEL_TERMS]) or f"Cluster {cluster + 1}"
        extra = CLUSTER_LABEL_TERMS
        while name in taken and extra < len(terms):
            extra += 1
            name = ", ".join(terms[:extra])
        if name in taken:
            name = f"{name} (cluster {cluster + 1})"
        taken.add(name)
        names[cluster] = name
    return names


def automatic_clusters(
    coords: np.ndarray, texts: Sequence[str]
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Density clusters of the 2D places (``UNCLUSTERED`` for a document in
    none) and each cluster's name: its ``CLUSTER_LABEL_TERMS`` most
    distinctive terms by TF-IDF across the clusters' texts (read through
    ``label_text``, file extensions left out), unique within the map."""
    from sklearn.cluster import HDBSCAN
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    labels = HDBSCAN(min_cluster_size=max(2, len(coords) // 10)).fit_predict(coords)
    ids = sorted({int(label) for label in labels if label != UNCLUSTERED})
    if not ids:
        return labels, {}
    corpus = [
        " ".join(
            label_text(text) for text, label in zip(texts, labels) if label == cluster
        )
        for cluster in ids
    ]
    try:
        vectorizer = TfidfVectorizer(
            stop_words=sorted(ENGLISH_STOP_WORDS | FILE_EXTENSIONS),
            token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
        )
        weights = vectorizer.fit_transform(corpus).toarray()
    except ValueError:
        # No cluster has a term to name it by.
        return labels, {cluster: f"Cluster {cluster + 1}" for cluster in ids}
    terms = vectorizer.get_feature_names_out()
    ranked = {
        cluster: [
            str(terms[i])
            for i in sorted(
                (i for i in range(len(terms)) if row[i] > 0),
                key=lambda i: (-row[i], terms[i]),
            )
        ]
        for cluster, row in zip(ids, weights)
    }
    return labels, _unique_names(ranked)


def build_map(
    documents: List[Dict[str, Any]],
    vectors: Optional[np.ndarray],
    coords: Optional[np.ndarray] = None,
) -> DocumentMap:
    """The map of ``documents`` (each with ``title`` and ``text``) from their
    pooled ``vectors``, laid out with UMAP, or at the given ``coords`` (one
    2D place per document) when there are some; ``vectors`` may then be
    None."""
    if len(documents) < MIN_DOCUMENTS:
        raise TooFewDocumentsError(len(documents))
    reducer = None
    if coords is None:
        reducer, coords = umap_layout(vectors)
    clusters, names = automatic_clusters(
        coords,
        [" ".join(filter(None, (d.get("title"), d.get("text")))) for d in documents],
    )
    return DocumentMap(
        documents=documents,
        vectors=None if vectors is None else unit_rows(vectors),
        reducer=reducer,
        coords=np.asarray(coords, dtype=np.float64),
        clusters=clusters,
        cluster_names=names,
    )


def place_queries(document_map: DocumentMap, vectors: np.ndarray) -> np.ndarray:
    """Each query vector's place on the map's fitted layout."""
    with document_map.lock:
        return np.asarray(document_map.reducer.transform(vectors), dtype=np.float64)


def most_similar(
    document_map: DocumentMap, vector: np.ndarray, count: int = 3
) -> List[Tuple[int, float]]:
    """The ``count`` documents whose pooled vector is most similar to
    ``vector`` by cosine similarity, most similar first, as (index,
    similarity); ties keep document order."""
    query = unit_rows(vector.reshape(1, -1))[0]
    scores = document_map.vectors @ query
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:count]
    return [(i, float(scores[i])) for i in order]


class TooFewDocumentsError(ValueError):
    """A map needs at least ``MIN_DOCUMENTS`` documents with an embedding."""

    def __init__(self, count: int) -> None:
        super().__init__(f"{count} documents, fewer than {MIN_DOCUMENTS}")
        self.count = count


class ProjectionCache:
    """Built maps per (tenant, profile, limit), current while their tenant and
    profile's Redis generation is unchanged, shared through Redis by every
    process."""

    def __init__(self, redis: Any, capacity: int = CACHE_CAPACITY) -> None:
        self._redis = redis
        self._capacity = capacity
        self._maps: "OrderedDict[Tuple[str, str, int], DocumentMap]" = OrderedDict()
        self._building: Dict[Tuple[Tuple[str, str, int], int], asyncio.Task] = {}

    @staticmethod
    def _key(tenant_id: str, profile: str) -> str:
        return GENERATION_KEY.format(tenant=tenant_id, profile=profile)

    async def generation(self, tenant_id: str, profile: str) -> int:
        value = await self._redis.get(self._key(tenant_id, profile))
        return int(value or 0)

    async def invalidate(self, tenant_id: str, profile: str) -> int:
        """Retire every cached map of ``tenant_id`` and ``profile`` on every
        replica; returns the new generation."""
        return int(await self._redis.incr(self._key(tenant_id, profile)))

    async def get(
        self,
        tenant_id: str,
        profile: str,
        limit: int,
        generation: int,
        build: Callable[[], DocumentMap],
    ) -> DocumentMap:
        """The map of ``generation`` (the current one, read with
        ``generation``): this process's copy, else the one stored in Redis,
        else built with ``build`` in a worker thread and stored for every
        process. A build that fails is not kept, and every reader waiting on
        it receives its error; a Redis that stops answering raises
        ``AtlasCacheUnavailableError``."""
        key = (tenant_id, profile, limit)
        cached = self._maps.get(key)
        if cached is not None and cached.generation == generation:
            self._maps.move_to_end(key)
            return cached
        flight = (key, generation)
        task = self._building.get(flight)
        if task is None:
            task = asyncio.create_task(self._build(key, generation, build))
            self._building[flight] = task
            task.add_done_callback(lambda done: self._settled(flight, done))
        return await asyncio.shield(task)

    def _settled(self, flight, task: asyncio.Task) -> None:
        self._building.pop(flight, None)
        if not task.cancelled():
            # Every waiter has its error already; mark it retrieved for the
            # case where all of them were cancelled first.
            task.exception()

    async def _build(
        self,
        key: Tuple[str, str, int],
        generation: int,
        build: Callable[[], DocumentMap],
    ) -> DocumentMap:
        document_map = await self._shared(key, generation, build)
        document_map.generation = generation
        current = self._maps.get(key)
        if current is None or current.generation <= generation:
            self._maps[key] = document_map
            self._maps.move_to_end(key)
            while len(self._maps) > self._capacity:
                self._maps.popitem(last=False)
        return document_map

    async def _stored(self, layout_key: str) -> Optional[DocumentMap]:
        """The map stored under ``layout_key``; None when there is none or
        the entry does not read as a map, which is then deleted."""
        stored = await self._redis.get(layout_key)
        if stored is None:
            return None
        try:
            return await asyncio.to_thread(_decoded, stored)
        except UnreadableLayoutError as exc:
            logger.warning(
                "Discarding the unreadable embedding atlas map %s: %s",
                layout_key,
                exc,
            )
            await self._redis.delete(layout_key)
            return None

    async def _shared(
        self,
        key: Tuple[str, str, int],
        generation: int,
        build: Callable[[], DocumentMap],
    ) -> DocumentMap:
        """The map every process serves for ``key`` and ``generation``: the
        one stored in Redis, else one this process builds holding the
        build lease and stores, waiting while another process holds it."""
        tenant, profile, limit = key
        names = {
            "tenant": tenant,
            "profile": profile,
            "limit": limit,
            "generation": generation,
        }
        layout_key, build_key = LAYOUT_KEY.format(**names), BUILD_KEY.format(**names)
        token = uuid.uuid4().hex
        try:
            while True:
                stored = await self._stored(layout_key)
                if stored is not None:
                    return stored
                if await self._redis.set(build_key, token, nx=True, ex=BUILD_LEASE_S):
                    break
                await asyncio.sleep(BUILD_POLL_S)
        except (RedisError, OSError) as exc:
            raise AtlasCacheUnavailableError(
                f"the embedding atlas cache did not answer for {tenant}/{profile} "
                f"(limit {limit}): {type(exc).__name__}: {exc}"
            ) from exc
        try:
            document_map = await asyncio.to_thread(build)
            payload = await asyncio.to_thread(_encoded, document_map)
            try:
                if not await self._redis.set(
                    layout_key, payload, nx=True, ex=LAYOUT_TTL_S
                ):
                    # A process that took over the lease stored its map first.
                    stored = await self._stored(layout_key)
                    if stored is not None:
                        return stored
            except (RedisError, OSError) as exc:
                raise AtlasCacheUnavailableError(
                    f"the embedding atlas cache did not store the map of "
                    f"{tenant}/{profile} (limit {limit}): {type(exc).__name__}: "
                    f"{exc}"
                ) from exc
            return document_map
        finally:
            try:
                await self._redis.eval(_RELEASE_BUILD_LUA, 1, build_key, token)
            except (RedisError, OSError) as exc:
                logger.warning(
                    "Embedding atlas build lease %s not released; it lapses in "
                    "%ds: %s: %s",
                    build_key,
                    BUILD_LEASE_S,
                    type(exc).__name__,
                    exc,
                )


_cache: Optional[ProjectionCache] = None


def set_projection_cache(cache: Optional[ProjectionCache]) -> None:
    global _cache
    _cache = cache


def projection_cache() -> Optional[ProjectionCache]:
    return _cache
