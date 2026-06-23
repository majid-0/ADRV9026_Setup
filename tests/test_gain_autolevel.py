"""Unit tests for the guarded ORx auto-leveler (hardware-free, callback-driven)."""

from __future__ import annotations

import numpy as np

from adrvtrx.gain import ORX_GAIN_MAX, ORX_GAIN_MIN, autolevel_orx


class FakeOrx:
    """Linear gain->level model with a hard clamp window, like the real ORx table.

    dBFS = slope * (gain - gain_ref); clamped so indices outside the valid window
    read the rail level (mirrors the bench: below ~190 clamps to max gain).
    """

    def __init__(self, slope=0.45, gain_ref=255, top_dbfs=-17.0):
        self.slope = slope
        self.gain_ref = gain_ref
        self.top_dbfs = top_dbfs
        self.gain = 220

    def set_gain(self, g):
        self.gain = int(np.clip(g, ORX_GAIN_MIN, ORX_GAIN_MAX))

    def measure(self):
        return self.top_dbfs - self.slope * (self.gain_ref - self.gain)


def test_converges_to_reachable_target():
    orx = FakeOrx()
    res = autolevel_orx(orx.set_gain, orx.measure, target_dbfs=-25.0, tolerance_db=1.0)
    assert res.converged, res.reason
    assert abs(res.final_dbfs - (-25.0)) <= 1.0
    assert ORX_GAIN_MIN <= res.final_gain_index <= ORX_GAIN_MAX


def test_fails_loud_when_target_above_max_reachable():
    # Top reachable is -17 dBFS; -10 is impossible -> pin at max gain, report it.
    orx = FakeOrx(top_dbfs=-17.0)
    res = autolevel_orx(orx.set_gain, orx.measure, target_dbfs=-10.0, tolerance_db=1.0)
    assert not res.converged
    assert res.final_gain_index == ORX_GAIN_MAX
    assert "max" in res.reason and "TX power" in res.reason


def test_fails_loud_when_target_below_min_reachable():
    # Bottom reachable ~ -17 - 0.45*(255-190) ~ -46 dBFS; -60 is impossible -> pin at min.
    orx = FakeOrx()
    res = autolevel_orx(orx.set_gain, orx.measure, target_dbfs=-60.0, tolerance_db=1.0)
    assert not res.converged
    assert res.final_gain_index == ORX_GAIN_MIN
    assert "min" in res.reason


def test_never_leaves_valid_window():
    orx = FakeOrx()
    seen = []
    autolevel_orx(lambda g: (seen.append(g), orx.set_gain(g))[1], orx.measure, target_dbfs=-30.0)
    assert all(ORX_GAIN_MIN <= g <= ORX_GAIN_MAX for g in seen)
