"""Whether a similarity threshold exists that the response cache could use.

Near-match caching answers a request out of a *different* request's entry
whenever the two embed closer than the threshold. It is admissible only when
some threshold separates two populations:

  * requests that must NOT share an entry - the recorded incident pair, and the
    closest pairs in the shipped evaluation corpus that ask about different
    content;
  * requests that MUST share one - the same request differing only by
    whitespace.

The router's own embedding endpoint scores both populations here, so the answer
is measured against the model the cache would actually key on rather than
assumed. When the populations overlap there is no admissible threshold and the
shipped configuration keeps ``mode: exact`` - which is what these pin.
"""

from __future__ import annotations

import itertools
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from tests.utils.semantic_router_stack import render_router_config

pytestmark = [pytest.mark.integration]

_REPO = Path(__file__).resolve().parents[3]
_CORPUS = (
    _REPO / "data" / "testset" / "evaluation" / "sample_videos_retrieval_queries.json"
)

# The pair that produced the incident: a near match served the answer to one
# numbered topic for another.
_INCIDENT_PAIR = ("topic 5", "topic 2")

# How many of the corpus's closest different-content pairs are scored. The
# threshold has to clear ALL of them, so the count only has to be large enough
# to include the closest; twelve is well past where the curve flattens.
_NEAR_MISS_PAIRS = 12
_EQUIVALENT_QUERIES = 10


def _embed(container: str, texts: list[str]) -> list[list[float]]:
    """Embeddings from the router's own model - the one the cache keys on.

    The router's management API binds to loopback inside the container, so the
    request is issued from inside it.
    """
    script = (
        "import json,sys,urllib.request\n"
        "texts=json.load(sys.stdin)\n"
        "out=[]\n"
        "for i in range(0,len(texts),25):\n"
        "    req=urllib.request.Request('http://127.0.0.1:8080/api/v1/embeddings',\n"
        "        data=json.dumps({'texts':texts[i:i+25]}).encode(),\n"
        "        headers={'Content-Type':'application/json'})\n"
        "    out+=[e['embedding'] for e in "
        "json.loads(urllib.request.urlopen(req,timeout=300).read())['embeddings']]\n"
        "print(json.dumps(out))\n"
    )
    proc = subprocess.run(
        ["docker", "exec", "-i", container, "python3", "-c", script],
        input=json.dumps(texts),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-600:]
    return json.loads(proc.stdout)


def _cosine(a: list[float], b: list[float]) -> float:
    norm = sum(x * x for x in a) ** 0.5 * sum(y * y for y in b) ** 0.5
    return sum(x * y for x, y in zip(a, b)) / norm


@pytest.fixture(scope="module")
def scored(semantic_router_stack) -> dict:
    """Every pair scored on the router's embedding model, once."""
    rows = json.loads(_CORPUS.read_text())
    content: dict[str, set[str]] = {}
    for row in rows:
        content.setdefault(row["query"], set()).update(row["expected_videos"])
    texts = sorted(content)

    equivalent = [(q, q + " ") for q in texts[:_EQUIVALENT_QUERIES]]
    probe = texts + [t for pair in equivalent for t in pair] + list(_INCIDENT_PAIR)
    vectors = dict(zip(probe, _embed(semantic_router_stack["router_container"], probe)))

    different_content = sorted(
        (
            (_cosine(vectors[a], vectors[b]), a, b)
            for a, b in itertools.combinations(texts, 2)
            if not (content[a] & content[b])
        ),
        reverse=True,
    )[:_NEAR_MISS_PAIRS]
    incident = _cosine(vectors[_INCIDENT_PAIR[0]], vectors[_INCIDENT_PAIR[1]])
    return {
        "must_miss": different_content + [(incident, *_INCIDENT_PAIR)],
        "must_hit": [(_cosine(vectors[a], vectors[b]), a, b) for a, b in equivalent],
        "incident": incident,
        "corpus_queries": len(texts),
    }


class TestNoThresholdSeparatesTheTwoPopulations:
    def test_the_corpus_supplies_the_pairs(self, scored):
        """Derived from the shipped corpus, not from a list written here."""
        assert scored["corpus_queries"] == 95
        assert len(scored["must_miss"]) == _NEAR_MISS_PAIRS + 1
        assert len(scored["must_hit"]) == _EQUIVALENT_QUERIES

    def test_the_incident_pair_would_hit_at_the_routers_default_threshold(self, scored):
        """0.5 is the value the router initialises its cache with when no
        threshold is configured, which is how 'topic 5' answered 'topic 2'."""
        assert round(scored["incident"], 3) == 0.738
        assert scored["incident"] > 0.5

    def test_the_closest_different_content_pair_outscores_every_equivalent_pair(
        self, scored
    ):
        """The admissibility test. A usable threshold T needs
        max(must_miss) < T <= min(must_hit); this asserts the interval is
        empty, so no T exists and similarity matching cannot be enabled."""
        worst_miss = max(score for score, _, _ in scored["must_miss"])
        weakest_hit = min(score for score, _, _ in scored["must_hit"])
        assert round(worst_miss, 3) == 0.993
        assert round(weakest_hit, 3) == 0.987
        assert worst_miss >= weakest_hit

    def test_the_shipped_configuration_matches_the_derivation(self):
        """The conclusion is wired, not just recorded: every decision in every
        routing profile keys on the exact request."""
        config = yaml.safe_load(render_router_config())
        profiles = [("auto", config["routing"])] + [
            (recipe["name"], recipe["routing"]) for recipe in config["recipes"]
        ]
        modes = {
            f"{name}/{decision['name']}": [
                plugin["configuration"]["mode"]
                for plugin in decision["plugins"]
                if plugin["type"] == "response_cache"
            ]
            for name, profile in profiles
            for decision in profile["decisions"]
        }
        assert modes == {
            "auto/pro-technical-keyword": ["exact"],
            "auto/pro-technical-domain": ["exact"],
            "auto/pro-default": ["exact"],
            "auto/free-default": ["exact"],
            "auto/base-default": ["exact"],
            "classification/classification-pro": ["exact"],
            "classification/classification-free": ["exact"],
            "classification/classification-base": ["exact"],
            "vision/vision-pro": ["exact"],
            "vision/vision-free": ["exact"],
            "vision/vision-base": ["exact"],
        }
