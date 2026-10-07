import pytest

from tests.e2e.batch_optimization import argo_phases_between


@pytest.mark.parametrize(
    "before,after,expected",
    [
        ("Pending", "Pending", ("Pending",)),
        (None, "", ("Pending",)),
        ("Running", "Running", ("Running",)),
        ("Succeeded", "Succeeded", ("Succeeded",)),
        ("Pending", "Running", ("Pending", "Running")),
        ("", "Running", ("Pending", "Running")),
        ("Pending", "Succeeded", ("Pending", "Running", "Succeeded")),
        (None, "Failed", ("Pending", "Running", "Failed")),
        ("Running", "Succeeded", ("Running", "Succeeded")),
        ("Running", "Error", ("Running", "Error")),
    ],
)
def test_the_phases_run_from_before_through_after_in_argo_order(
    before, after, expected
):
    assert argo_phases_between(before, after) == expected


@pytest.mark.parametrize(
    "before,after",
    [
        ("Running", "Pending"),
        ("Succeeded", "Running"),
        ("Succeeded", "Failed"),
        ("Failed", None),
        ("Pending", "Skipped"),
    ],
)
def test_a_pair_argo_cannot_produce_raises(before, after):
    with pytest.raises(ValueError, match=r"Argo|not an Argo workflow phase"):
        argo_phases_between(before, after)
