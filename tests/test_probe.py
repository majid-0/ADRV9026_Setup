"""The probe tool (task6_fast_lock.md, Task 6b): peak events, shared windows, the build and its
check, and the fidelity of a probe on full-signal captures (the real bench's when present)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from adrvtrx.metrics import PEAK_REGION, gain_compression_db
from adrvtrx.probe import (
    ProbeSettings,
    build_probe,
    peak_events,
    probe_fidelity,
    select_windows,
)
from adrvtrx.waveform import prepare_tx


def cfr_signal(seed: int, n: int = 65536, bw: float = 0.15, clip: float = 2.9) -> np.ndarray:
    """A band-limited signal with its crest factor reduced (as a test-model signal is), unit
    peak: a dozen peak events at or above 0.975 of its maximum."""
    rng = np.random.default_rng(seed)
    lp = np.abs(np.fft.fftfreq(n)) < bw
    x = np.fft.ifft(np.fft.fft(rng.normal(size=n) + 1j * rng.normal(size=n)) * lp)
    x /= np.sqrt(np.mean(np.abs(x) ** 2))
    for _ in range(6):
        a = np.abs(x)
        x = np.where(a > clip, x * clip / np.maximum(a, 1e-12), x)
        x = np.fft.ifft(np.fft.fft(x) * lp)
    return x / np.abs(x).max()


def codes(x) -> np.ndarray:
    i, q = prepare_tx(np.asarray(x), 12)
    return i + 1j * q


def spikes(n: int, at: dict[int, float], scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """Low noise (|x| about 0.05) with single-sample peaks of the given heights."""
    rng = np.random.default_rng(seed)
    x = 0.03 * (rng.normal(size=n) + 1j * rng.normal(size=n))
    for i, h in at.items():
        x[i] = h
    return x * scale


# -- peaks and windows --------------------------------------------------------------------------


def test_peak_events_are_the_local_maxima_near_the_top_highest_first():
    x = spikes(1000, {100: 1.0, 300: 0.99, 500: 0.97, 700: 0.976})
    assert list(peak_events(x)) == [100, 300, 700]  # 0.97 is below 0.975 of the maximum
    assert list(peak_events(x, 0.9)) == [100, 300, 700, 500]
    assert PEAK_REGION == 0.975  # the region gain_compression_db averages over


def test_the_bands_peaks_are_ranked_together_each_against_its_own_maximum():
    a = spikes(8000, {1000: 1.0, 5000: 0.98})
    b = spikes(8000, {3000: 1.0, 7000: 0.976}, scale=0.3, seed=1)  # a band 10 dB lower
    ranges = select_windows([a, b], window=200)
    assert ranges == [(900, 1100), (2900, 3100), (4900, 5100), (6900, 7100)]  # time order
    # with a budget for two windows, the two maxima (one per band) come first
    assert select_windows([a, b], window=200, budget=400) == [(900, 1100), (2900, 3100)]


def test_a_covered_peak_takes_no_window_and_overlapping_windows_merge():
    a = spikes(8000, {1000: 1.0, 1200: 0.99, 1100: 0.98, 4000: 0.98, 4150: 0.976})
    assert select_windows([a], window=200) == [(900, 1300), (3900, 4250)]
    # 1100's window is all inside 1000's and 1200's: skipped; 4150's merges with 4000's
    assert select_windows([a], window=200, budget=600) == [(900, 1300), (3900, 4100)]
    # with a budget of three windows, 1100 (skipped) costs nothing: 4000 still fits, 4150 not


def test_a_window_across_the_period_end_is_two_ranges():
    a = spikes(4000, {20: 1.0, 2000: 0.99})
    assert select_windows([a], window=200) == [(0, 120), (1900, 2100), (3920, 4000)]


def test_each_band_gives_at_most_max_peaks_its_highest_and_its_maximum_always():
    a = spikes(20000, {1000: 1.0, 5000: 0.99, 9000: 0.985, 13000: 0.98})
    b = spikes(20000, {3000: 1.0, 7000: 0.976}, scale=0.3, seed=1)
    every = [(900, 1100), (2900, 3100), (4900, 5100), (6900, 7100), (8900, 9100), (12900, 13100)]
    assert select_windows([a, b], window=200, max_peaks=None) == every
    assert select_windows([a, b], window=200, max_peaks=10) == every  # a cap above the count
    # two per band: a's 1000 and 5000, b's 3000 and 7000, still ranked and in time order
    assert select_windows([a, b], window=200, max_peaks=2) == [
        (900, 1100),
        (2900, 3100),
        (4900, 5100),
        (6900, 7100),
    ]
    assert select_windows([a, b], window=200, max_peaks=1) == [(900, 1100), (2900, 3100)]


def test_a_capped_probe_reports_its_bias_and_is_not_refused():
    sig = [codes(cfr_signal(1)), codes(cfr_signal(2))]
    full = build_probe(sig, window=300, crossfade=8, max_peaks=None)
    capped = build_probe(sig, window=300, crossfade=8, max_peaks=3)
    assert all(not st["capped"] for st in full.stats)
    assert full.ok and capped.n < full.n and capped.settings["max_peaks"] == 3
    for st, s in zip(capped.stats, sig):
        assert st["capped"] and st["peaks_covered"] < st["peaks_total"]
        assert int(np.argmax(np.abs(s))) in capped.positions  # the band's maximum is always in
    assert "peaks covered" not in capped.reason
    default = build_probe(sig, window=300, crossfade=8)  # 25: more than these signals have
    assert default.n == full.n and default.ranges == full.ranges


# -- the build ----------------------------------------------------------------------------------


@pytest.mark.parametrize("seeds, window", [((1,), 400), ((1, 2), 300), ((1, 2, 5), 300)])
def test_every_band_is_cut_at_the_same_indices_and_keeps_its_codes(seeds, window):
    sig = [codes(cfr_signal(seed)) for seed in seeds]
    bands = len(seeds)
    p = build_probe(sig, window=window, crossfade=8)
    assert p.ok, p.reason
    assert len(p.codes) == bands and {len(c) for c in p.codes} == {p.n}
    real = p.positions >= 0
    for c, s in zip(p.codes, sig):
        np.testing.assert_array_equal(c[real], s[p.positions[real]])  # unchanged outside fades
        assert np.abs(c).max() == np.abs(s).max()  # the peak drive is the full signal's
    assert p.n % 16 == 0 and p.n * 4 <= len(sig[0])
    peaks = np.concatenate([peak_events(s) for s in sig])
    assert np.isin(peaks, p.positions).all()  # every peak at or above the threshold is in it
    for st, pk in zip(p.stats, (peak_events(s) for s in sig)):
        assert st["peaks_covered"] == st["peaks_total"] == len(pk)
        assert abs(st["mean_power_db"]) <= 0.2
        assert st["papr_db"] - st["papr_full_db"] == pytest.approx(-st["mean_power_db"], abs=0.002)


def test_every_peak_keeps_half_a_window_of_real_signal_and_no_crossfade_comes_near():
    sig = [codes(cfr_signal(1)), codes(cfr_signal(2))]
    w, cf = 300, 8
    p = build_probe(sig, window=w, crossfade=cf)
    where = {int(q): k for k, q in enumerate(p.positions) if q >= 0}
    fades = [j["at"] for j in p.joins]
    n_full = len(sig[0])
    for s in sig:
        for q in peak_events(s):
            k = where[int(q)]
            around = [p.positions[(k + d) % p.n] for d in range(-w // 2, w // 2)]
            assert around == [(q + d) % n_full for d in range(-w // 2, w // 2)]  # real, in order
            for at in fades:  # the crossfade [at, at + cf) never closer than w/2 - cf
                dist = min(abs(k - at), abs(k - (at + cf - 1)), p.n - abs(k - at))
                assert dist >= w // 2 - cf, (q, at)
    assert all(j["jump_after"] <= j["jump_before"] + 1.0 for j in p.joins)
    assert p.joins and all(p.positions[j["at"] : j["at"] + cf].max() == -1 for j in p.joins)


@pytest.mark.parametrize("block", [1, 7, 64])
def test_the_length_is_a_multiple_of_16_and_of_the_tx_block(block):
    p = build_probe([codes(cfr_signal(1)), codes(cfr_signal(2))], window=300, crossfade=8,
                    block=block)  # fmt: skip
    assert p.n % math.lcm(16, block) == 0 and p.ok, (p.n, p.reason)


def test_a_budget_keeps_the_highest_peaks_and_only_reports_the_rest():
    sig = [codes(cfr_signal(1)), codes(cfr_signal(2))]
    p = build_probe(sig, window=300, crossfade=8, budget=1500)
    covered = [st["peaks_covered"] for st in p.stats]
    assert sum(covered) < sum(st["peaks_total"] for st in p.stats)
    assert "peaks covered" not in p.reason  # with a budget the bias is reported, not refused
    for s in sig:  # the band's maximum is always in
        assert int(np.argmax(np.abs(s))) in p.positions or np.abs(s).max() < max(
            np.abs(t).max() for t in sig
        )


def test_the_build_check_refuses_with_its_reasons():
    sig = [codes(cfr_signal(1)), codes(cfr_signal(2))]
    short = build_probe(sig, window=100, crossfade=8)  # windows too short to be typical
    assert not short.ok and "mean power" in short.reason and "from the full signal" in short.reason
    long = build_probe(sig, window=2000, crossfade=8)
    assert not long.ok and "more than a quarter of the signal" in long.reason


@pytest.mark.parametrize(
    "bad",
    [{"window": 40}, {"window": 30000}, {"crossfade": 200}, {"budget": 100}, {"block": 0},
     {"block": 5000}, {"max_peaks": 0}, {"max_peaks": 1001}],
)  # fmt: skip
def test_the_settings_have_their_ranges(bad):
    with pytest.raises(ValueError):
        ProbeSettings(**bad)
    s = ProbeSettings()
    assert (s.window, s.crossfade, s.budget, s.block, s.multiple, s.max_peaks) == (
        600,
        16,
        None,
        1,
        16,
        25,
    )


def test_the_fidelity_of_a_probe_on_a_memoryless_pa():
    from sim_bench import rapp

    x = [cfr_signal(1), cfr_signal(2)]
    p = build_probe([codes(v) for v in x], window=300, crossfade=8)
    y = [rapp(v * 1.6) for v in x]
    f = probe_fidelity(x, y, p)
    for b, xb, yb in zip(f["bands"], x, y):
        assert b["full_db"] == pytest.approx(gain_compression_db(xb, yb)[0], abs=1e-4)
        assert abs(b["offset_db"]) < 0.03
    one = probe_fidelity(x[0], y[0], p)  # one band's arrays
    assert one["bands"][0] == f["bands"][0]


# -- on the real bench's captures (skipped where they are not) ------------------------------------

BENCH = (
    Path.home()
    / "OneDrive - aus.edu"
    / "Scalability_Final_Rev"
    / "Polynomials"
    / "multiband_setup"
    / "captures"
    / "20261020_1901_S20-100_real"
    / "TX2-TX3_1440-1860MHz_100-100BW_3-3GC"
)


def _load(path: Path) -> np.ndarray:
    a = np.loadtxt(path, delimiter="\t")
    return a[:, 0] + 1j * a[:, 1]


@pytest.mark.skipif(not BENCH.is_dir(), reason="the S20-100 bench captures are not on this machine")
@pytest.mark.parametrize("max_peaks, bound", [(25, 0.1), (None, 0.03)])
def test_the_default_probe_holds_the_bench_compression(max_peaks, bound):
    """S20-100, TX2 + TX3, the full signals' 0..3 dB backoff captures of 2026-10-20: compression on
    the probe's (600 / 16) sample positions against the full period. Every peak (42 / 27): within
    0.03 dB; the default cap of 25 peaks per band (TX2 loses 17): within 0.1 dB (0.091 measured)."""
    stem = BENCH.name
    x = [_load(BENCH / f"{stem}_{b}_in.txt") for b in ("TX2", "TX3")]
    p = build_probe([codes(v) for v in x], max_peaks=max_peaks)
    covered = [st["peaks_covered"] for st in p.stats]
    total = [st["peaks_total"] for st in p.stats]
    assert (covered == total) == (max_peaks is None) and p.ok, (covered, total, p.reason)
    assert [st["capped"] for st in p.stats] == [
        max_peaks is not None and t > max_peaks for t in total
    ]
    for k in range(4):
        y = [_load(BENCH / f"{stem}_{k}dB_{b}_out.txt") for b in ("TX2", "TX3")]
        offsets = [b["offset_db"] for b in probe_fidelity(x, y, p)["bands"]]
        assert all(abs(o) <= bound for o in offsets), (k, offsets)
