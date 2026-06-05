"""Per-keyframe face extraction against the face-embed sidecar.

Given a ``VideoIngestionPipeline`` result dict that contains a
``keyframes.items`` list (each item carrying ``segment_id``, ``ts_start``,
and either ``image_b64`` or ``image_url``), POST each keyframe to the
face-embed sidecar and accumulate ``FaceMention`` records keyed by
``(source_doc_id, segment_id, bbox)``.

Output is deterministic — records are sorted by ``(segment_id, bbox)``
ascending before return so re-invocations on the same input produce
byte-equal results. The clustering consumer downstream relies on this
ordering for golden-file replay.
"""

from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import httpx

from cogniverse_agents.graph.graph_schema import FaceMention

_EMBED_PATH = "/embed"
_DEFAULT_TIMEOUT_S = 30.0


def _iter_keyframes(processing_results: Dict[str, Any]):
    """Yield (segment_id, ts_start, image_payload_dict) from a pipeline result.

    Accepts both flat ``keyframes`` lists and the nested
    ``keyframes.items`` shape that ``VideoIngestionPipeline`` emits.
    Missing-keyframes input → yields nothing (zero output records).
    """
    kf = processing_results.get("keyframes") or {}
    if isinstance(kf, dict):
        items = kf.get("items") or kf.get("keyframes") or []
    elif isinstance(kf, list):
        items = kf
    else:
        items = []

    for item in items:
        segment_id = item.get("segment_id")
        ts_start = float(item.get("ts_start", item.get("timestamp", 0.0)))
        payload: Dict[str, str] = {}
        if "image_b64" in item:
            payload["image_b64"] = item["image_b64"]
        elif "image_url" in item:
            payload["image_url"] = item["image_url"]
        else:
            continue
        if segment_id is None:
            continue
        yield segment_id, ts_start, payload


def _bbox_tuple(raw_bbox) -> Tuple[int, int, int, int]:
    """Coerce a sidecar bbox response into a 4-int tuple."""
    return (int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3]))


def _vec_tuple(raw_vec) -> Tuple[float, ...]:
    """Coerce a sidecar embedding into a tuple of floats."""
    return tuple(float(v) for v in raw_vec)


def _post_one(
    client: httpx.Client, base_url: str, segment_id: str, payload: Dict[str, str]
) -> Dict[str, Any]:
    """POST a single keyframe to the sidecar. Raise on non-200 status."""
    url = base_url.rstrip("/") + _EMBED_PATH
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
    client: httpx.Client | None = None,
) -> List[FaceMention]:
    """Return a deterministic list of ``FaceMention`` records.

    Empty keyframes (no faces detected) contribute zero records. Multiple
    faces in one keyframe produce that many distinct records.

    Raises ``RuntimeError`` with the failing ``segment_id`` and the HTTP
    status code embedded in the message when the sidecar returns non-200
    OR the request itself fails.
    """
    owns_client = client is None
    if client is None:
        client = httpx.Client()
    records: List[FaceMention] = []
    try:
        keyframes = list(_iter_keyframes(processing_results))
        # POST keyframes concurrently — the sidecar calls are independent and
        # httpx.Client is thread-safe, so face extraction no longer scales
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

        for (segment_id, ts_start, _payload), response in zip(keyframes, responses):
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
