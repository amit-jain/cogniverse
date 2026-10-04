"""Per-keyframe face extraction against the face-embed sidecar.

Reads the keyframes ``KeyframeProcessor`` wrote into a
``VideoIngestionPipeline`` result (``results["keyframes"]["keyframes"]``, each
``{frame_number, timestamp, filename, path}``), POSTs each frame's image to
the face-embed sidecar and accumulates ``FaceMention`` records keyed by
``(source_doc_id, segment_id, bbox)``. A keyframe's ``segment_id`` is its
index in that list, the content schema's segment id (doc
``<video_id>_seg_<index>``) and the keyframe-aligned transcript segments'.
The image is read from ``path`` and sent base64-encoded, one request at a
time; nothing is added to the result.

Output is deterministic — records are sorted by ``(segment_id, bbox)``
ascending before return so re-invocations on the same input produce
byte-equal results. The clustering consumer downstream relies on this
ordering for golden-file replay.
"""

from __future__ import annotations

import base64
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import httpx

from cogniverse_agents.graph.graph_schema import FaceMention
from cogniverse_foundation.config.inference_auth import inference_headers

_EMBED_PATH = "/embed"
_DEFAULT_TIMEOUT_S = 30.0


def _canonical_bearer_headers(headers: Mapping[str, str] | None) -> Dict[str, str]:
    if not headers:
        return {}
    if set(headers) != {"Authorization"}:
        raise ValueError("headers must contain only Authorization")
    authorization = headers["Authorization"]
    scheme, separator, token = authorization.partition(" ")
    if scheme != "Bearer" or not separator or not token or token != token.strip():
        raise ValueError("headers Authorization must be a canonical bearer value")
    return {"Authorization": authorization}


def keyframes_of(processing_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The keyframe records ``KeyframeProcessor`` wrote, in extraction order."""
    section = processing_results.get("keyframes") or {}
    items = section.get("keyframes") if isinstance(section, dict) else None
    return list(items) if isinstance(items, list) else []


def _iter_keyframes(processing_results: Dict[str, Any]):
    """Yield ``(segment_id, ts_start, path)`` for every keyframe.

    Raises ``ValueError`` naming the keyframe when one carries no ``path``
    or ``timestamp``: a keyframe that cannot be read is a broken contract,
    not an empty frame.
    """
    for index, item in enumerate(keyframes_of(processing_results)):
        if (
            not isinstance(item, dict)
            or not item.get("path")
            or (item.get("timestamp") is None)
        ):
            raise ValueError(
                f"keyframe {index} has no readable path and timestamp: "
                f"{sorted(item) if isinstance(item, dict) else type(item).__name__}"
            )
        yield str(index), float(item["timestamp"]), Path(item["path"])


def _bbox_tuple(raw_bbox) -> Tuple[int, int, int, int]:
    """Coerce a sidecar bbox response into a 4-int tuple."""
    return (int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3]))


def _vec_tuple(raw_vec) -> Tuple[float, ...]:
    """Coerce a sidecar embedding into a tuple of floats."""
    return tuple(float(v) for v in raw_vec)


def _post_one(
    client: httpx.Client, base_url: str, segment_id: str, path: Path
) -> Dict[str, Any]:
    """POST a single keyframe image to the sidecar. Raise on non-200 status."""
    url = base_url.rstrip("/") + _EMBED_PATH
    try:
        image = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"keyframe image for segment_id={segment_id!r} is unreadable at "
            f"{path}: {exc}"
        ) from exc
    payload = {"image_b64": base64.b64encode(image).decode("ascii")}
    try:
        resp = client.post(url, json=payload, timeout=_DEFAULT_TIMEOUT_S)
    except httpx.HTTPError as exc:
        raise RuntimeError(
            f"face-embed sidecar request failed for segment_id={segment_id!r}: {exc}"
        ) from exc
    if resp.status_code != 200:
        raise RuntimeError(
            f"face-embed sidecar returned HTTP {resp.status_code} for "
            f"segment_id={segment_id!r}: {resp.text[:200]}"
        )
    return resp.json()


def extract_faces_per_keyframe(
    processing_results: Dict[str, Any],
    source_doc_id: str,
    face_embed_url: str,
    *,
    headers: Mapping[str, str] | None = None,
    client: httpx.Client | None = None,
) -> List[FaceMention]:
    """Return a deterministic list of ``FaceMention`` records.

    Empty keyframes (no faces detected) contribute zero records. Multiple
    faces in one keyframe produce that many distinct records.

    Raises ``ValueError`` for a keyframe without ``path`` or ``timestamp``,
    and ``RuntimeError`` with the failing ``segment_id`` when its image is
    unreadable, the sidecar returns non-200 (status in the message) or the
    request itself fails.
    """
    explicit_headers = _canonical_bearer_headers(headers)
    configured_headers = inference_headers(face_embed_url.rstrip("/"))
    if configured_headers and explicit_headers:
        raise ValueError("headers must not be supplied for a Modal endpoint")
    resolved_headers = configured_headers or explicit_headers
    if resolved_headers and client is not None:
        raise ValueError("headers and client cannot both be supplied")
    owns_client = client is None
    if client is None:
        client = httpx.Client(headers=resolved_headers)
    records: List[FaceMention] = []
    try:
        keyframes = list(_iter_keyframes(processing_results))
        # POST keyframes concurrently — the sidecar calls are independent and
        # httpx.Client is thread-safe, so face extraction does not scale
        # linearly in keyframe count. Output stays deterministic (sorted below);
        # executor.map surfaces a sidecar failure as the first-by-order error,
        # matching the serial contract.
        if keyframes:
            with ThreadPoolExecutor(max_workers=min(8, len(keyframes))) as executor:
                responses = list(
                    executor.map(
                        lambda kf: _post_one(client, face_embed_url, kf[0], kf[2]),
                        keyframes,
                    )
                )
        else:
            responses = []

        for (segment_id, ts_start, _path), response in zip(keyframes, responses):
            for face in response.get("faces", []):
                records.append(
                    FaceMention(
                        source_doc_id=source_doc_id,
                        segment_id=segment_id,
                        ts_start=ts_start,
                        ts_end=ts_start,
                        bbox=_bbox_tuple(face["bbox"]),
                        vec=_vec_tuple(face["vec"]),
                        det_score=float(face["det_score"]),
                    )
                )
    finally:
        if owns_client:
            client.close()

    records.sort(key=lambda m: (m.segment_id, m.bbox))
    return records


def face_mention_as_jsonable(m: FaceMention) -> Dict[str, Any]:
    """Round-trip-safe dict serialisation for golden files / API payloads.

    ``dataclasses.asdict`` on a frozen ``FaceMention`` already produces
    plain-Python types but tuples come out as lists which matters for
    byte-equal JSON. Convert bbox to a 4-list and vec to a list-of-floats
    explicitly so downstream JSON serialisation is deterministic.
    """
    d = dataclasses.asdict(m)
    d["bbox"] = list(d["bbox"])
    d["vec"] = list(d["vec"])
    return d
