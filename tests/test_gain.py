from __future__ import annotations

import numpy as np

from adrvtrx._enums import RxChannel
from adrvtrx.gain import clip_report, level_orx, peak_window


def test_clip_report_detects_railing():
    i = np.array([32767, 100, -32767, 5], dtype=np.int32)
    q = np.array([0, 200, 0, 5], dtype=np.int32)
    rep = clip_report(i, q, 16)
    assert rep.is_clipping
    assert rep.railed_samples == 2
    assert np.isclose(rep.peak_dbfs, 0.0, atol=1e-3)


def test_clip_report_clean_signal():
    i = np.array([1000, -2000, 500], dtype=np.int32)
    q = np.array([0, 0, 0], dtype=np.int32)
    rep = clip_report(i, q, 16)
    assert not rep.is_clipping
    assert rep.peak_dbfs < 0
    assert rep.peak_index == 1


def test_clip_report_empty():
    rep = clip_report(np.array([]), np.array([]), 16)
    assert rep.n_samples == 0
    assert rep.railed_samples == 0


def test_peak_window_centers_on_peak():
    n = 1000
    i = np.zeros(n, dtype=np.int32)
    q = np.zeros(n, dtype=np.int32)
    i[700] = 30000  # peak well inside
    wi, wq = peak_window(i, q, 100)
    assert len(wi) == 100
    assert wi.max() == 30000  # the peak survived the windowing


def test_peak_window_larger_than_signal_returns_all():
    i = np.arange(10, dtype=np.int32)
    q = np.zeros(10, dtype=np.int32)
    wi, wq = peak_window(i, q, 50)
    assert len(wi) == 10


class _FakeLeveler:
    """Models a monotonic level vs gain index for the leveling loop."""

    def __init__(self, start_gain=180, dbfs_at_start=-20.0, db_per_index=0.5):
        self._gain = start_gain
        self._start = start_gain
        self._dbfs0 = dbfs_at_start
        self._slope = db_per_index

    def get_rx_gain(self, ch):
        return self._gain

    def set_rx_gain(self, ch, g):
        self._gain = g

    def rx_dec_power_dbfs(self, ch):
        return self._dbfs0 + (self._gain - self._start) * self._slope


def test_level_orx_converges_up():
    lev = _FakeLeveler(start_gain=180, dbfs_at_start=-20.0)
    res = level_orx(lev, RxChannel.ORX1, target_dbfs=-12.0, tolerance_db=1.0)
    assert res.converged
    assert abs(res.final_dbfs - (-12.0)) <= 1.0
    assert res.final_gain_index > 180


def test_level_orx_stops_at_gain_rail():
    lev = _FakeLeveler(start_gain=250, dbfs_at_start=-40.0)
    res = level_orx(lev, RxChannel.ORX1, target_dbfs=-12.0, gain_max=255, max_iterations=20)
    assert res.final_gain_index <= 255
