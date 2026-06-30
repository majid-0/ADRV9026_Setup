from __future__ import annotations

import numpy as np

from adrvtrx.gain import clip_report, peak_window


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
