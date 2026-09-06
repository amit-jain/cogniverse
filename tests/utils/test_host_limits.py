"""Contract for the host kernel limits the K3s-backed fixtures depend on."""

from tests.utils.host_limits import (
    K3S_INOTIFY_INSTANCES,
    inotify_in_use,
    inotify_preflight_message,
    inotify_shortfall,
)


class TestInotifyShortfall:
    """Headroom arithmetic: how many instances a new K3s still needs."""

    def test_ample_headroom_reports_no_shortfall(self):
        assert inotify_shortfall(cap=512, in_use=41) == 0

    def test_exactly_enough_headroom_reports_no_shortfall(self):
        assert inotify_shortfall(cap=128, in_use=128 - K3S_INOTIFY_INSTANCES) == 0

    def test_one_short_reports_exactly_one(self):
        assert inotify_shortfall(cap=128, in_use=128 - K3S_INOTIFY_INSTANCES + 1) == 1

    def test_the_default_cap_beside_a_k3d_node_and_desktop_is_short(self):
        # 128 default, one k3d node (~35) plus a loaded desktop (~60) leaves 33.
        assert inotify_shortfall(cap=128, in_use=95) == 2

    def test_usage_above_the_cap_adds_the_overshoot_to_the_requirement(self):
        # 2 over the cap, so 2 must be freed before the 35 a K3s needs.
        assert inotify_shortfall(cap=128, in_use=130) == K3S_INOTIFY_INSTANCES + 2


class TestInotifyPreflightMessage:
    """The message an engineer reads instead of 'too many open files'."""

    def test_no_message_when_a_k3s_fits(self):
        assert inotify_preflight_message(cap=512, in_use=41) is None

    def test_message_names_the_sysctl_the_numbers_and_the_shortfall(self):
        message = inotify_preflight_message(cap=128, in_use=110)
        assert message == (
            "fs.inotify.max_user_instances is 128 with 110 in use, leaving 18 "
            "for a K3s that needs 35. Raise it: "
            "sudo sysctl -w fs.inotify.max_user_instances=512"
        )


class TestInotifyInUseCountsRealInstances:
    """The live counter, driven against real kernel objects.

    The arithmetic above is exact but injected. A counter that always returns
    zero satisfies every one of those tests while reporting infinite headroom,
    so it is pinned here against instances this test creates itself.
    """

    def test_opening_one_inotify_instance_raises_the_count_by_exactly_one(self):
        import ctypes
        import os

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        before = inotify_in_use()
        fd = libc.inotify_init1(0)
        assert fd >= 0, os.strerror(ctypes.get_errno())
        try:
            during = inotify_in_use()
        finally:
            os.close(fd)
        after = inotify_in_use()

        assert (during - before, after - before) == (1, 0)
