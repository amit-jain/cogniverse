"""HTTP client for the GLiNER2 inference service.

One GLiNER2 pass extracts entity spans and answers classification tasks over
the same text. :class:`RemoteGliner2Client` posts to
``cogniverse_cli.modal_inference.servers.gliner2`` and validates the response
shape; :func:`get_or_load_gliner2` caches one client per endpoint.
"""

import hashlib
import logging
import math
import threading
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TypedDict

import requests

from cogniverse_core.common.models.model_loaders import (
    _resolved_inference_headers,
    _store_bounded_cache,
)

logger = logging.getLogger(__name__)

GLINER2_REQUEST_TIMEOUT_S = 5.0
_GLINER2_CACHE_CAPACITY = 8


class Gliner2Entity(TypedDict):
    text: str
    label: str
    score: float
    start: int
    end: int


class Gliner2Choice(TypedDict):
    label: str
    score: float


class Gliner2Result(TypedDict):
    entities: List[Gliner2Entity]
    classifications: Dict[str, Gliner2Choice]


def _score(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} score must be a number")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{where} score {value!r} must be in [0, 1]")
    return score


def validate_gliner2_result(
    data: Any,
    text: str,
    entity_types: Iterable[str],
    classifications: Mapping[str, Iterable[str]],
    source: str,
) -> Gliner2Result:
    """``data`` as a :class:`Gliner2Result` for this request, or ValueError.

    Every entity must carry one of ``entity_types`` and offsets whose slice of
    ``text`` is its ``text``; ``classifications`` must answer exactly the
    requested tasks, each with one of that task's labels.
    """
    if not isinstance(data, dict) or set(data) != {
        "model",
        "entities",
        "classifications",
    }:
        raise ValueError(f"{source} must be {{model, entities, classifications}}")
    if not isinstance(data["entities"], list):
        raise ValueError(f"{source} entities must be a list")
    allowed = set(entity_types)
    entities: List[Gliner2Entity] = []
    for index, entity in enumerate(data["entities"]):
        where = f"{source} entity {index}"
        if not isinstance(entity, dict) or set(entity) != set(
            Gliner2Entity.__annotations__
        ):
            raise ValueError(
                f"{where} must have exactly {sorted(Gliner2Entity.__annotations__)}"
            )
        if entity["label"] not in allowed:
            raise ValueError(f"{where} label {entity['label']!r} was not requested")
        start, end = entity["start"], entity["end"]
        if not (isinstance(start, int) and isinstance(end, int)) or isinstance(
            start, bool
        ):
            raise ValueError(f"{where} offsets must be integers")
        if not 0 <= start < end <= len(text) or text[start:end] != entity["text"]:
            raise ValueError(
                f"{where} text {entity['text']!r} is not the query at [{start}, {end})"
            )
        entities.append(
            Gliner2Entity(
                text=entity["text"],
                label=entity["label"],
                score=_score(entity["score"], where),
                start=start,
                end=end,
            )
        )
    answered = data["classifications"]
    if not isinstance(answered, dict) or set(answered) != set(classifications):
        raise ValueError(
            f"{source} classifications must answer exactly {sorted(classifications)}"
        )
    choices: Dict[str, Gliner2Choice] = {}
    for task, task_labels in classifications.items():
        labels = list(task_labels)
        choice = answered[task]
        where = f"{source} classification {task!r}"
        if not isinstance(choice, dict) or set(choice) != {"label", "score"}:
            raise ValueError(f"{where} must be {{label, score}}")
        if choice["label"] not in labels:
            raise ValueError(
                f"{where} label {choice['label']!r} is not one of {labels}"
            )
        choices[task] = Gliner2Choice(
            label=choice["label"], score=_score(choice["score"], where)
        )
    return Gliner2Result(entities=entities, classifications=choices)


class RemoteGliner2Client:
    """Client for ``POST /extract`` on the GLiNER2 inference service.

    ``extract`` raises on any transport failure, non-2xx status or malformed
    body: an outage is never an empty extraction.
    """

    def __init__(
        self,
        url: str,
        model: str,
        *,
        timeout: float = GLINER2_REQUEST_TIMEOUT_S,
        _resolved_headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._url = url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            _resolved_headers
            if _resolved_headers is not None
            else _resolved_inference_headers(self._url, None)
        )

    def _close(self) -> None:
        self._session.close()

    def extract(
        self,
        text: str,
        entity_types: Mapping[str, str],
        classifications: Mapping[str, Mapping[str, str]],
        *,
        threshold: float,
    ) -> Gliner2Result:
        """Entities of ``entity_types`` and one label per classification task.

        ``entity_types`` maps each type to a description the model reads;
        ``classifications`` maps each task to its ``{label: description}``.
        """
        endpoint = f"{self._url}/extract"
        resp = self._session.post(
            endpoint,
            json={
                "model": self._model,
                "text": text,
                "entity_types": dict(entity_types),
                "classifications": {
                    task: dict(labels) for task, labels in classifications.items()
                },
                "threshold": threshold,
            },
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        source = f"GLiNER2 response from {endpoint}"
        if isinstance(data, dict) and data.get("model") != self._model:
            raise ValueError(
                f"{source} answered for model {data.get('model')!r}, "
                f"not {self._model!r}"
            )
        return validate_gliner2_result(
            data, text, entity_types, classifications, source
        )


_client_cache: "OrderedDict[Tuple[str, str, float, str], RemoteGliner2Client]" = (
    OrderedDict()
)
_client_lock = threading.Lock()


def get_or_load_gliner2(
    model: str,
    *,
    inference_url: str,
    timeout: float = GLINER2_REQUEST_TIMEOUT_S,
) -> RemoteGliner2Client:
    """The cached :class:`RemoteGliner2Client` for ``model`` at ``inference_url``.

    Built once per (model, url, timeout, credential); concurrent first calls
    build one client. Modal URLs authenticate with
    ``COGNIVERSE_INFERENCE_API_KEY``.
    """
    if not model:
        raise ValueError("model must name the GLiNER2 model the service serves")
    if not inference_url:
        raise ValueError("inference_url must name the GLiNER2 inference service")
    headers = _resolved_inference_headers(inference_url, None)
    fingerprint = hashlib.sha256(headers.get("Authorization", "").encode()).hexdigest()
    key = (model, inference_url, timeout, fingerprint)
    with _client_lock:
        cached = _client_cache.get(key)
        if cached is not None:
            _client_cache.move_to_end(key)
            return cached
        client = RemoteGliner2Client(
            inference_url, model, timeout=timeout, _resolved_headers=headers
        )
        _store_bounded_cache(
            _client_cache,
            key,
            client,
            capacity=_GLINER2_CACHE_CAPACITY,
            label="GLiNER2 client cache",
        )
        logger.info("GLiNER2 client ready: %s at %s", model, inference_url)
        return client
