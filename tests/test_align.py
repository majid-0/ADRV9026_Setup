"""Unit tests for the time-alignment / delay-estimation helpers (no hardware)."""

from __future__ import annotations

import numpy as np

from adrvtrx.align import apply_delay, estimate_and_align, estimate_delay


def _ref(n=2048, seed=0):
    """A structured complex signal with good autocorrelation."""
    t = np.arange(n)
    rng = np.random.default_rng(seed)
    base = np.exp(1j * 2 * np.pi * 0.07 * t) * (1 + 0.5 * np.cos(2 * np.pi * 0.005 * t))
    return base + 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def test_finds_reference_inside_2x_capture():
    # The whole point: a 1-period ref placed at an arbitrary offset in a 2x capture.
    x = _ref(2048)
    off = 900
    y = np.zeros(2 * len(x), dtype=complex)
    y[off : off + len(x)] = x
    d = estimate_delay(x, y, fs=1.0)
    assert abs(d - off) < 0.5, d


def test_trimming_bug_would_fail_without_fix():
    # Sanity: with a large offset, the correct delay is recoverable (the fixed path).
    x = _ref(4096)
    off = 3000  # > len(x)//2; the old trim-to-min path could not see this
    y = np.zeros(2 * len(x), dtype=complex)
    y[off : off + len(x)] = x
    assert abs(estimate_delay(x, y, fs=1.0) - off) < 0.5


def test_estimate_and_align_recovers_clean_copy():
    x = _ref(2048)
    off = 700
    y = np.zeros(2 * len(x), dtype=complex)
    y[off : off + len(x)] = x
    xa, ya, d = estimate_and_align(x, y, fs=1.0)
    m = min(len(xa), len(ya))
    cc = np.abs(np.vdot(xa[:m], ya[:m])) / (np.linalg.norm(xa[:m]) * np.linalg.norm(ya[:m]))
    assert cc > 0.99, cc
    assert abs(d - off) < 0.5


def test_fractional_delay_roundtrip():
    # Inject a known fractional delay (equal length), confirm the estimate recovers it.
    x = _ref(8192)
    _, y = apply_delay(x, x, 3.4, upsample_factor=20)
    d = estimate_delay(x, y, fs=1.0)
    assert abs(d - 3.4) < 0.1, d


def test_shorter_capture_than_reference_does_not_crash():
    x = _ref(4096)
    y = x[:1000].copy()  # capture shorter than ref -> trimmed path
    d = estimate_delay(x, y, fs=1.0)
    assert np.isfinite(d)
