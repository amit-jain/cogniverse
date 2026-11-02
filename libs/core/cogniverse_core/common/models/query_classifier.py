"""Loading the gateway's query classifier, in process or over HTTP.

The model and its checkpoint format live in
``cogniverse_foundation.query_classifier``. :class:`RemoteQueryClassifierClient`
calls the inference server in
``cogniverse_cli.modal_inference.servers.query_classifier`` and returns the
same ``{head: {"label", "score"}}`` shape as :meth:`QueryClassifier.classify`.
"""

import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Tuple

import requests

from cogniverse_core.common.models.model_loaders import (
    _resolved_inference_headers,
    _store_bounded_cache,
)
from cogniverse_foundation.query_classifier import (
    Predictions,
    QueryClassifier,
    validate_heads,
    validate_predictions,
)

logger = logging.getLogger(__name__)

QUERY_CLASSIFIER_REQUEST_TIMEOUT_S = 5.0
_QUERY_CLASSIFIER_CACHE_CAPACITY = 8

__all__ = [
    "Predictions",
    "QueryClassifier",
    "RemoteQueryClassifierClient",
    "get_or_load_query_classifier",
    "validate_heads",
    "validate_predictions",
]


class RemoteQueryClassifierClient:
    """HTTP client for the query classifier inference server.

    ``classify`` raises on any transport failure, non-2xx status or malformed
    body: an outage is never a prediction.
    """

    def __init__(
        self,
        url: str,
        model: str,
        *,
        timeout: float = QUERY_CLASSIFIER_REQUEST_TIMEOUT_S,
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
        self.heads: Optional[Dict[str, List[str]]] = None

    def _close(self) -> None:
        self._session.close()

    def classify(self, text: str) -> Predictions:
        endpoint = f"{self._url}/classify"
        resp = self._session.post(
            endpoint, json={"text": text, "model": self._model}, timeout=self._timeout
        )
        resp.raise_for_status()
        data = resp.json()
        source = f"Query classifier response from {endpoint}"
        if not isinstance(data, dict) or set(data) != {"model", "heads", "predictions"}:
            raise ValueError(f"{source} must be {{model, heads, predictions}}")
        if data["model"] != self._model:
            raise ValueError(
                f"{source} answered for model {data['model']!r}, not {self._model!r}"
            )
        heads = validate_heads(data["heads"])
        self.heads = heads
        return validate_predictions(data["predictions"], heads, source)


_classifier_cache: "OrderedDict[Tuple[str, str, str, float], Any]" = OrderedDict()
_classifier_lock = threading.Lock()


def get_or_load_query_classifier(
    model: str,
    *,
    inference_url: Optional[str] = None,
    device: str = "cpu",
    timeout: float = QUERY_CLASSIFIER_REQUEST_TIMEOUT_S,
) -> Any:
    """The cached classifier for ``model``, built once per (model, url, device,
    timeout).

    With ``inference_url`` it is a :class:`RemoteQueryClassifierClient` naming
    ``model`` whose requests give up after ``timeout`` seconds; otherwise
    ``model`` is a local checkpoint directory loaded with
    :meth:`QueryClassifier.load`. Concurrent first calls build it once.
    """
    if not model:
        raise ValueError("model must name a query classifier checkpoint")
    key = (model, inference_url or "_local_", device, timeout)
    with _classifier_lock:
        cached = _classifier_cache.get(key)
        if cached is not None:
            _classifier_cache.move_to_end(key)
            return cached
        if inference_url:
            instance: Any = RemoteQueryClassifierClient(
                inference_url, model, timeout=timeout
            )
        else:
            try:
                instance = QueryClassifier.load(model, device=device)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load query classifier {model!r} on {device}: {exc}"
                ) from exc
        _store_bounded_cache(
            _classifier_cache,
            key,
            instance,
            capacity=_QUERY_CLASSIFIER_CACHE_CAPACITY,
            label="query classifier cache",
        )
        logger.info(
            "Query classifier %s ready (%s)",
            model,
            f"remote {inference_url}" if inference_url else f"local {device}",
        )
        return instance
