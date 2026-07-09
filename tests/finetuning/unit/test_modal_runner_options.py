"""Remote Modal training must apply the caller's resource config.

The deployed Modal functions declare fixed defaults (A10G / 4 CPU / 16 GB /
1 h). The runner read self.config.gpu only into a log line and called
.remote() without .with_options(), so a request for an A100/H100 or more
memory silently ran on the defaults while the log claimed the requested GPU.
"""

from __future__ import annotations

from cogniverse_finetuning.training.modal_runner import (
    ModalJobConfig,
    ModalTrainingRunner,
)


class _FakeModalFn:
    """Records with_options calls; returns itself so .remote() stays callable."""

    def __init__(self):
        self.options = None

    def with_options(self, **kwargs):
        self.options = kwargs
        return self


def test_gpu_spec_single_gpu():
    runner = ModalTrainingRunner(ModalJobConfig(gpu="A100-80GB", gpu_count=1))
    assert runner._gpu_spec() == "A100-80GB"


def test_gpu_spec_multi_gpu_encodes_count():
    runner = ModalTrainingRunner(ModalJobConfig(gpu="H100", gpu_count=4))
    assert runner._gpu_spec() == "H100:4"


def test_apply_options_forwards_all_resources():
    config = ModalJobConfig(
        gpu="A100-80GB", gpu_count=2, cpu=16, memory=80000, timeout=7200
    )
    runner = ModalTrainingRunner(config)
    fn = _FakeModalFn()

    returned = runner._apply_options(fn)

    assert returned is fn
    assert fn.options == {
        "gpu": "A100-80GB:2",
        "cpu": 16,
        "memory": 80000,
        "timeout": 7200,
    }


def test_apply_options_uses_defaults_when_unset():
    runner = ModalTrainingRunner(ModalJobConfig())
    fn = _FakeModalFn()
    runner._apply_options(fn)
    assert fn.options == {"gpu": "A10G", "cpu": 4, "memory": 16384, "timeout": 3600}
