"""Run the collected tests in reverse or shuffled order.

``-p tests.fixtures.node_order --node-order=reverse`` runs the selected node
ids last-first; ``--node-order=shuffle`` runs them in a random order drawn
from ``--node-order-seed`` (a fresh seed when omitted). The order actually
run is printed as the session's first lines, seed included, so a failing
shuffle is replayed exactly by passing the printed seed back.

A test that passes only in collection order reads state another test left
behind; running a module both ways is how that dependence shows.
"""

from __future__ import annotations

import random

import pytest

ORDERS = ("collected", "reverse", "shuffle")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("node-order")
    group.addoption(
        "--node-order",
        choices=ORDERS,
        default="collected",
        help="run the selected tests in collection, reverse or shuffled order",
    )
    group.addoption(
        "--node-order-seed",
        type=int,
        default=None,
        help="seed for --node-order=shuffle (printed when drawn)",
    )


def ordered(items: list, order: str, seed: int | None) -> list:
    """``items`` in the requested order; a shuffle is a function of the seed."""
    if order == "collected":
        return list(items)
    if order == "reverse":
        return list(reversed(items))
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session, config, items) -> None:
    order = config.getoption("--node-order")
    if order == "collected":
        return
    seed = config.getoption("--node-order-seed")
    if order == "shuffle" and seed is None:
        seed = random.SystemRandom().randrange(2**32)
    items[:] = ordered(items, order, seed)
    config.stash[_ORDER_LINES] = [
        f"node order: {order}" + (f" (seed {seed})" if order == "shuffle" else ""),
        *(f"  {item.nodeid}" for item in items),
    ]


_ORDER_LINES = pytest.StashKey[list]()


def pytest_report_collectionfinish(config, start_path, items):
    return config.stash.get(_ORDER_LINES, [])
