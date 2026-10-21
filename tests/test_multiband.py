"""Several bands at once: the search on a synthetic two-band PA, then a case on the fake PA."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel, multiband
from adrvtrx.compression import search_tx_compression
from adrvtrx.config import Config, DllConfig
from adrvtrx.fake import FakeRadio
from adrvtrx.metrics import PA_CLIP_SLOPE
from adrvtrx.multiband import (
    JOINT_KEYS,
    METHODS,
    SLOPE_PRIOR,
    BandCapture,
    BandLock,
    BandSpec,
    CaseWriter,
    MultibandSearch,
    add_joint_metrics,
    capture_case,
    case_name,
    find_multiband_operating_point,
    lock_bands,
    play_bands,
    prior_slopes,
    search_multiband_compression,
    update_slopes,
)
from adrvtrx.transmit import transmit_bands


class TwoBandPa:
    """Gain compression of each band: its own Rapp-like curve plus ``cross`` dB per dB of the other
    band's
    attenuation below 20 dB. ORx peak follows attenuation and gain index (0.5 dB per index).
    ``orx_peak`` pins a band's ORx peak reading (``{0: 0.5}``: band 1 always reads +0.5 dBFS,
    railed): an ORx that never levels. ``comp_nan`` makes every n-th reading of a band NaN
    (``{1: 3}``: every third of band 2's; ``{1: 1}``: all of them), as a failed alignment does.
    ``railed_first`` rails a band's first readings with the peak still in band (``{0: 1}``)."""

    def __init__(
        self,
        locks=(14.0, 15.0),
        cross=(0.3, 0.3),
        p=1.4,
        noise=0.02,
        seed=0,
        clip_below=None,
        orx_peak=None,
        comp_nan=None,
        railed_first=None,
    ):
        self.locks, self.cross, self.p, self.noise = locks, cross, p, noise
        self.rng = np.random.default_rng(seed)
        self.clip_below = clip_below or {}
        self.orx_peak = orx_peak or {}
        self.comp_nan = comp_nan or {}
        self.railed_first = railed_first or {}
        self.band_reads = [0, 0]
        self.a = [20.0, 20.0]
        self.g = [210, 210]
        self.reads = 0

    def comp(self, i: int) -> float:
        own = self.a[i] - self.locks[i]  # 0 at this band's own lock (other band at 20 dB)
        k = 10 ** (self.p * 4.0 / 10) - 1  # 4 dB at own = 0
        c = 10 / self.p * np.log10(1 + k * 10 ** (-self.p * own / 10))
        return float(c + self.cross[i] * (20.0 - min(self.a[1 - i], 20.0)))

    def measure(self, i: int):
        self.reads += 1
        # the output peak follows the drive minus the compression at the peak
        # (-1 dBFS at 20 dB, gain 210)
        a_now = self.a[i]
        self.a[i] = 20.0
        c_start = self.comp(i)
        self.a[i] = a_now
        peak = -1.0 + (20.0 - a_now) - (self.comp(i) - c_start) + 0.5 * (self.g[i] - 210)
        peak = self.orx_peak.get(i, peak)
        railed = 5 if peak > 0 else 0
        slope = 0.05 if self.a[i] < self.clip_below.get(i, -1.0) else 0.4
        comp = self.comp(i) + float(self.rng.normal(0, self.noise))
        if self.band_reads[i] < self.railed_first.get(i, 0):
            railed = 4  # a few samples at the rail; the peak reads in band
        self.band_reads[i] += 1
        if i in self.comp_nan and self.band_reads[i] % self.comp_nan[i] == 0:
            comp = float("nan")
        return min(peak, 0.0), railed, comp, {"papr_compression_db": comp - 1.0, "top_slope": slope}

    def search(self, **kw):
        defaults = {
            "targets_db": [4.0, 4.0],
            "start_attens_db": [20.0, 20.0],
            "atten_mins_db": [10.0, 10.0],
            "orx_gains": [210, 210],
            "names": ["TX1", "TX4"],
        }
        defaults.update(kw)
        return search_multiband_compression(
            lambda i, db: self.a.__setitem__(i, db),
            lambda i, g: self.g.__setitem__(i, g),
            self.snapshot,
            **defaults,
        )

    def snapshot(self, idx):
        return [self.measure(i) for i in idx]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cross", [(0.0, 0.0), (0.3, 0.3), (0.0, 0.4)])
def test_both_methods_lock_two_coupled_bands(method, cross):
    pa = TwoBandPa(cross=cross)
    res = pa.search(method=method)
    assert res.converged, res.reason
    for i, bl in enumerate(res.bands):
        assert bl.status == "converged"
        assert abs(pa.comp(i) - 4.0) <= 0.1 + 0.06  # tolerance plus the reading noise
    assert pa.a == [bl.atten_db for bl in res.bands]  # the radio is left at the lock
    assert res.captures == res.snapshots >= res.steps  # one snapshot reads both bands


def test_a_band_that_needs_more_drive_than_its_limit_stops_there():
    pa = TwoBandPa(locks=(14.0, 9.0), cross=(0.0, 0.0))  # band 2 would need 9 dB, its limit is 12
    res = pa.search(atten_mins_db=[10.0, 12.0])
    assert not res.converged and "TX4" in res.reason
    assert res.bands[0].status == "converged"
    assert res.bands[1].status == "at limit" and res.bands[1].atten_db == 12.0


def test_clip_guard_per_band():
    pa = TwoBandPa(
        locks=(12.0, 14.0), cross=(0.0, 0.0), clip_below={0: 13.0}
    )  # band 1 clips below 13 dB
    res = pa.search()
    assert res.bands[0].status == "clip-limited" and res.bands[0].atten_db >= 13.0
    assert res.bands[1].status == "converged"
    assert any("clip guard" in h["action"] for h in res.history if h["band"] == "TX1")


def test_an_unleveled_orx_moves_its_gain_not_the_attenuation():
    pa = TwoBandPa()
    res = pa.search(orx_gains=[240, 180])  # band 1 rails, band 2 far too low
    assert res.converged
    relevels = [h for h in res.history if h["action"].startswith("ORx gain")]
    assert {h["band"] for h in relevels} == {"TX1", "TX4"}
    assert all(h["move_db"] == 0.0 for h in relevels)
    assert res.snapshots > res.steps


def test_a_reading_outside_the_agc_window_counts_and_its_gain_is_corrected():
    """F5: no railed sample and a peak between target - 3 and 0 dBFS: it counts (the band moves)
    and the next step's ORx gain brings it back toward the target."""
    pa = TwoBandPa(noise=0.0, cross=(0.0, 0.0))
    res = pa.search(orx_gains=[206, 210])  # band 1 reads -3 dBFS: outside the window, usable
    first = res.history[0]
    assert first["band"] == "TX1" and -3.1 < first["peak_dbfs"] < -2.9
    assert first["counts"] and not first["leveled"] and first["move_db"] < 0  # it moved
    assert not [h for h in res.history if h["action"].startswith("ORx gain")]  # no re-level
    second = next(h for h in res.history if h["band"] == "TX1" and h["step"] == 1)
    assert -1.6 <= second["peak_dbfs"] <= -0.7  # leveled again by the next step's gain
    assert res.converged


def test_a_band_whose_orx_never_levels_keeps_its_attenuation():
    pa = TwoBandPa(orx_peak={0: 0.5})  # band 1's ORx always reads +0.5 dBFS (railed)
    res = pa.search(max_steps=4, gain_min=100)  # never at the ORx gain floor
    tx1 = [h for h in res.history if h["band"] == "TX1"]
    assert len(tx1) == 4 * 7  # max_relevel + 1 captures per step, every one re-leveling
    assert {h["atten_db"] for h in tx1} == {20.0} and all(h["move_db"] == 0.0 for h in tx1)
    assert all(h["action"].startswith("ORx gain -") for h in tx1)
    assert sum("atten held" in h["action"] for h in tx1) == 4
    assert pa.a[0] == 20.0 and res.bands[0].status == "not converged"
    tx4 = [h for h in res.history if h["band"] == "TX4"]
    assert tx4[0]["action"] == "joint" and tx4[-1]["atten_db"] < 20.0  # the other band goes on
    assert res.snapshots == 4 * 7  # the other band's reading rode the first snapshot of a step


@pytest.mark.parametrize("orx", [{"orx_gains": [240, 180]}, {"orx_peak": {1: 0.5}}])
def test_every_capture_is_one_history_row_with_its_own_action(orx):
    pa = TwoBandPa(**{k: v for k, v in orx.items() if k == "orx_peak"})
    res = pa.search(max_steps=6, **{k: v for k, v in orx.items() if k != "orx_peak"})
    assert res.fatal == ("orx_peak" in orx)  # a band that rails at the ORx gain floor stops it
    assert len(res.history) == pa.reads
    assert len({id(h) for h in res.history}) == len(res.history)  # no row recorded twice
    for h in res.history:
        assert h["action"]
        if h["action"].startswith("ORx gain"):
            assert h["move_db"] == 0.0 and h["slopes"] == ""


def test_a_nan_compression_is_measured_again_and_never_counts():
    pa = TwoBandPa(comp_nan={1: 3})  # every third reading of band 2 fails to align
    res = pa.search()
    assert res.converged, res.reason
    nan_rows = [h for h in res.history if not np.isfinite(h["gain_compression_db"])]
    assert nan_rows and all(h["band"] == "TX4" for h in nan_rows)
    for h in nan_rows:  # measured again at the same gain and attenuation: no move
        assert h["action"] == "measure again (compression not finite)" and h["move_db"] == 0.0
        assert not h["leveled"] and not h["counts"]
    assert all(np.isfinite(bl.compression_db) for bl in res.bands)
    assert all("nan" not in h["slopes"].lower() for h in res.history)


def test_a_band_that_never_aligns_holds_and_the_search_does_not_crash():
    pa = TwoBandPa(comp_nan={1: 1})
    res = pa.search(max_steps=5)
    assert not res.converged
    tx4 = [h for h in res.history if h["band"] == "TX4"]
    assert {h["atten_db"] for h in tx4} == {20.0} and {h["orx_gain"] for h in tx4} == {210}
    assert all(h["action"].startswith("measure again") for h in tx4)
    assert res.bands[1].status == "not converged" and pa.a[1] == 20.0
    assert "closest" not in res.reason  # no step counted, so none was kept as the best


def _counted_steps(res, names=("TX1", "TX4")) -> dict[int, dict[str, dict]]:
    """Per step, each band's reading that counted (the last row of the band in that step)."""
    rows: dict[int, dict[str, dict]] = {}
    for h in res.history:
        if h["counts"]:
            rows.setdefault(h["step"], {})[h["band"]] = h
    return {s: r for s, r in rows.items() if set(r) == set(names)}


@pytest.mark.parametrize(
    "pa_kw, search_kw",
    [
        ({"orx_peak": {1: 0.5}}, {"max_steps": 3, "gain_min": 100}),  # never a counted step
        ({"cross": (0.3, 0.3)}, {"max_steps": 2}),
        ({"cross": (0.6, 0.6)}, {"max_steps": 8}),
    ],
)
def test_at_max_steps_the_radio_is_left_where_it_was_measured(pa_kw, search_kw):
    pa = TwoBandPa(**pa_kw)
    res = pa.search(**search_kw)
    assert not res.converged and res.steps == search_kw["max_steps"]
    assert pa.a == [bl.atten_db for bl in res.bands] and pa.g == [bl.orx_gain for bl in res.bands]
    for bl in res.bands:  # every band's lock is one of its captures: atten, gain and reading
        assert any(
            (h["atten_db"], h["orx_gain"], h["gain_compression_db"])
            == (bl.atten_db, bl.orx_gain, bl.compression_db)
            for h in res.history
            if h["band"] == bl.name
        ), bl
    last = [h for h in res.history if h["step"] == res.steps - 1]
    assert all(h["move_db"] == 0.0 for h in last)  # no move after the last step


def test_at_max_steps_the_step_closest_to_the_targets_is_restored():
    pa = TwoBandPa(cross=(0.6, 0.6))
    res = pa.search(max_steps=8)
    steps = _counted_steps(res)
    worst = {s: max(abs(h["error_db"]) for h in rows.values()) for s, rows in steps.items()}
    best = min(worst, key=worst.get)
    assert best != max(steps)  # the last step is not the best one here
    assert "closest" in res.reason
    assert pa.a == [steps[best][name]["atten_db"] for name in ("TX1", "TX4")]
    assert [bl.compression_db for bl in res.bands] == [
        steps[best][name]["gain_compression_db"] for name in ("TX1", "TX4")
    ]


def test_without_a_counted_step_the_last_measured_step_is_kept():
    pa = TwoBandPa(orx_peak={1: 0.5})
    res = pa.search(max_steps=3, gain_min=100)
    tx1 = [h for h in res.history if h["band"] == "TX1" and h["counts"]]
    assert tx1[0]["atten_db"] == 20.0 and tx1[1]["atten_db"] < 20.0 and len(tx1) == 3
    assert pa.a[0] == tx1[-1]["atten_db"] and pa.g[0] == tx1[-1]["orx_gain"]
    assert res.bands[0].compression_db == tx1[-1]["gain_compression_db"]
    assert "last step measured" in res.reason


class LinearPa:
    """Two bands whose compression is linear in the attenuations, noise-free:
    ``c = 4 + S @ (locks - a)`` (``S[i][j]``: dB of band ``i``'s compression per dB less
    attenuation on band ``j``). The ORx peak follows the drive minus the compression exactly,
    -1 dBFS at the start, ``db_per_index`` dB per gain index from index 500."""

    def __init__(self, slopes, locks=(15.0, 16.0), start=20.0, db_per_index=0.5):
        self.S, self.locks = np.asarray(slopes, dtype=float), np.asarray(locks, dtype=float)
        self.start, self.dbpi = start, db_per_index
        self.a, self.g = np.array([start, start]), [500, 500]

    def comp(self, a=None) -> np.ndarray:
        return 4.0 + self.S @ (self.locks - (self.a if a is None else a))

    def measure(self, i: int):
        c, c0 = self.comp(), self.comp(np.full(2, self.start))
        peak = -1.0 + (self.start - self.a[i]) - (c[i] - c0[i]) + self.dbpi * (self.g[i] - 500)
        railed = 5 if peak > 0 else 0
        return min(peak, 0.0), railed, float(c[i]), {"top_slope": 0.5}

    def search(self, **kw):
        args = {
            "targets_db": [4.0, 4.0],
            "start_attens_db": [self.start, self.start],
            "atten_mins_db": [5.0, 5.0],
            "orx_gains": [500, 500],
            "names": ["TX1", "TX4"],
            "db_per_index": self.dbpi,
            "gain_min": 0,
            "gain_max": 1000,
        }
        args.update(kw)
        return search_multiband_compression(
            lambda i, db: self.a.__setitem__(i, db),
            lambda i, g: self.g.__setitem__(i, g),
            lambda idx: [self.measure(i) for i in idx],
            **args,
        )


#: The first bench lock's PA (S20-100, TX2 + TX3): own 0.3 dB/dB, cross 0.06 dB/dB.
BENCH_SLOPES = [[0.3, 0.06], [0.06, 0.3]]
#: Locks 7 dB below the 20 dB start (and around it).
SEVEN_DB_AWAY = [(13.0, 13.0), (13.0, 15.0), (14.5, 13.0), (16.0, 14.0)]


@pytest.mark.parametrize("locks", SEVEN_DB_AWAY)
def test_joint_locks_the_bench_pa_from_7_db_away_in_at_most_5_snapshots(locks):
    pa = LinearPa(BENCH_SLOPES, locks=locks)
    res = pa.search(method="joint")
    assert res.converged, res.reason
    np.testing.assert_allclose(pa.comp(), 4.0, atol=0.1)  # the truth, not the reading
    assert res.snapshots <= 5 and res.captures == res.snapshots, res.snapshots


@pytest.mark.parametrize("locks", SEVEN_DB_AWAY)
def test_pingpong_locks_the_bench_pa_from_7_db_away_in_at_most_12_snapshots(locks):
    pa = LinearPa(BENCH_SLOPES, locks=locks)
    res = pa.search(method="pingpong")
    assert res.converged, res.reason
    np.testing.assert_allclose(pa.comp(), 4.0, atol=0.1)
    assert res.snapshots <= 12, res.snapshots
    moved = [[h["band"] for h in res.history if h["step"] == s and h["move_db"]]
             for s in range(res.steps)]  # fmt: skip
    assert all(len(m) <= 1 for m in moved)  # one band at a time


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "slopes, prior",
    [
        ([[1.0, 0.6], [0.6, 1.0]], SLOPE_PRIOR),  # the old fake board's strong coupling
        (BENCH_SLOPES, (0.1, 0.02)),  # the prior 3x too shallow
        (BENCH_SLOPES, (0.93, 0.18)),  # ...and 3x too steep
        ([[0.9, 0.3], [0.3, 0.9]], (0.1, 0.0)),
    ],
)
@pytest.mark.parametrize("locks", [(13.0, 13.0), (13.0, 17.5), (17.5, 14.0)])
def test_hard_cases_converge_or_stop_with_a_reason_never_wander(method, slopes, prior, locks):
    """Strong coupling and a prior 3x off either way: the trust region keeps every step near
    what was measured, so the search converges (here: always) and never wanders off."""
    pa = LinearPa(slopes, locks=locks)
    visited = []
    res = pa.search(method=method, slope_prior=prior, max_steps=16,
                    on_step=lambda rows: visited.append([r["atten_db"] for r in rows]))  # fmt: skip
    assert res.converged, (res.reason, res.steps)
    np.testing.assert_allclose(pa.comp(), 4.0, atol=0.1)
    assert res.snapshots <= (8 if method == "joint" else 16), res.snapshots
    lo, hi = min(min(locks), 20.0) - 4.0, 20.0 + 4.0  # never further than one cap past the box
    assert all(lo <= a <= hi for step in visited for a in step), visited


@pytest.mark.parametrize("method", METHODS)
def test_a_band_starting_at_its_floor_is_held_and_the_other_solved(method):
    """TX1 starts at its 14 dB floor and needs 12 dB: held there ("at limit"); TX4 is solved
    with it fixed and converges; the search stops with the reason."""
    pa = LinearPa(BENCH_SLOPES, locks=(12.0, 15.0), start=14.0)
    res = pa.search(method=method, start_attens_db=[14.0, 20.0], atten_mins_db=[14.0, 5.0])
    assert not res.converged and "TX1 at the attenuation limit" in res.reason
    assert res.bands[0].status == "at limit" and res.bands[0].atten_db == 14.0
    assert res.bands[1].status == "converged"
    assert {h["atten_db"] for h in res.history if h["band"] == "TX1"} == {14.0}
    assert res.snapshots <= 8


def test_the_trust_region_undoes_a_step_that_made_it_worse_and_halves_the_cap():
    """A prior ten times too shallow on a steep PA: the first move (4 dB, the cap) overshoots
    the target and the error grows from 1 to 3 dB; the next step goes back halfway and the cap
    halves."""
    pa = LinearPa([[1.0, 0.0], [0.0, 1.0]], locks=(19.0, 19.0))
    res = pa.search(slope_prior=(0.1, 0.0), max_move_db=4.0)
    tx1 = [h for h in res.history if h["band"] == "TX1" and h["counts"]]
    assert [h["error_db"] for h in tx1[:2]] == pytest.approx([1.0, -3.0])  # it grew
    assert tx1[1]["action"] == "undo half (error grew; cap 2 dB)"
    assert tx1[1]["move_db"] == pytest.approx(-0.5 * tx1[0]["move_db"])  # back halfway
    assert abs(tx1[2]["move_db"]) <= 2.0 and res.converged


def test_the_slope_update_is_what_locks_strongly_coupled_bands_in_few_steps(monkeypatch):
    """0.6 dB/dB from the other band next to 1 dB/dB of its own, from the bench prior (0.31 /
    0.06): only the update learns it."""
    coupled = [[1.0, 0.6], [0.6, 1.0]]
    grid = [(l1, l2) for l1 in np.arange(13.0, 17.6, 1.5) for l2 in np.arange(13.0, 17.6, 1.5)]

    def snapshots() -> float:
        return float(
            np.mean([LinearPa(coupled, locks=locks).search(max_steps=12).snapshots
                     for locks in grid])
        )  # fmt: skip

    with_update = snapshots()
    monkeypatch.setattr(multiband, "update_slopes", lambda jac, da, dc, **_k: jac)
    prior_only = snapshots()
    assert with_update < prior_only, (with_update, prior_only)


@pytest.mark.parametrize("locks", [(15.0, 16.0), (13.0, 17.5), (16.5, 14.0)])
def test_the_orx_gain_follows_the_output_peak_so_no_band_relevels(locks):
    pa = LinearPa([[0.5, 0.0], [0.0, 0.5]], locks=locks)  # the typical own slope, exact
    res = pa.search()
    assert res.converged, res.reason
    assert [h for h in res.history if h["action"].startswith("ORx gain")] == []
    assert all(h["counts"] for h in res.history)  # every reading usable: none re-captured


def test_a_railed_capture_does_not_count_even_with_the_peak_in_band():
    pa = TwoBandPa(railed_first={0: 1})
    res = pa.search()
    tx1 = [h for h in res.history if h["band"] == "TX1"]
    assert -1.6 <= tx1[0]["peak_dbfs"] <= -0.7 and tx1[0]["railed"] == 4
    assert not tx1[0]["counts"] and tx1[0]["action"] == "ORx gain -1"
    assert tx1[0]["move_db"] == 0.0
    assert tx1[1]["step"] == 0 and tx1[1]["counts"] and tx1[1]["action"] == "joint"
    assert tx1[1]["orx_gain"] == tx1[0]["orx_gain"] - 1
    assert res.converged


def test_one_band_ends_where_the_single_band_search_ends():
    pa = TwoBandPa(cross=(0.0, 0.0), noise=0.0)
    one = search_multiband_compression(
        lambda i, db: pa.a.__setitem__(0, db),
        lambda i, g: pa.g.__setitem__(0, g),
        lambda idx: [pa.measure(0)],
        targets_db=[4.0],
        start_attens_db=[20.0],
        atten_mins_db=[10.0],
        orx_gains=[210],
    )
    pa2 = TwoBandPa(cross=(0.0, 0.0), noise=0.0)
    single = search_tx_compression(
        lambda db: pa2.a.__setitem__(0, db),
        lambda g: pa2.g.__setitem__(0, g),
        lambda: pa2.measure(0),
        target_compression_db=4.0,
        start_atten_db=20.0,
        atten_min_db=10.0,
        orx_gain=210,
        coarse_step_db=2.5,
        fine_step_db=0.2,
        comp_tol_db=0.1,
    )
    assert one.converged and single.converged
    assert one.bands[0].atten_db == pytest.approx(single.final_atten_db, abs=0.2)


@pytest.mark.parametrize(
    "bad, match",
    [({"max_steps": 0}, "max_steps"), ({"comp_tol_db": 0.0}, "comp_tol_db"),
     ({"comp_tol_db": -0.1}, "comp_tol_db"), ({"max_move_db": 0.0}, "max_move_db"),
     ({"slope_prior": (0.05, 0.0)}, "slope_prior"), ({"slope_prior": (0.3, 0.2)}, "slope_prior"),
     ({"method": "walk"}, "walk was removed"), ({"method": "slopes"}, "replaced by 'joint'"),
     ({"method": "newton"}, "method must be"),
     ({"coarse_step_db": 2.5}, "coarse_step_db was removed: max_move_db"),
     ({"fine_step_db": 0.2}, "fine_step_db was removed")],
)  # fmt: skip
def test_the_search_rejects_settings_it_cannot_work_with(bad, match):
    pa = TwoBandPa()
    with pytest.raises(ValueError, match=match):
        pa.search(**bad)
    assert pa.reads == 0


def test_a_negative_backoff_is_rejected_before_the_radio_is_touched():
    radio = MagicMock(name="radio")
    band = BandSpec(TxChannel.TX1, RxChannel.ORX1, "LO1", 1_440_000_000, 80, np.ones(64))
    with pytest.raises(ValueError, match="backoffs must be >= 0"):
        find_multiband_operating_point(
            radio, [band], tx_bits=12, rx_bits=12, fs=1e6, backoffs=(0, -1)
        )
    assert radio.mock_calls == []


# -- the slope matrix ----------------------------------------------------------------------


def test_the_prior_has_the_own_slope_on_the_diagonal_and_the_coupling_off_it():
    np.testing.assert_allclose(prior_slopes(2), [[-0.31, -0.06], [-0.06, -0.31]])
    np.testing.assert_allclose(prior_slopes(3, (0.5, 0.1))[2], [-0.1, -0.1, -0.5])


def test_the_slope_update_explains_the_step_it_learns_from():
    jac = np.diag([-0.7, -0.8])
    da, dc = np.array([-1.0, -0.5]), np.array([1.0, 0.75])  # what the step did
    new = update_slopes(jac, da, dc)
    np.testing.assert_allclose(new @ da, dc)  # the secant equation
    v = np.array([0.5, -1.0])  # orthogonal to da: J is unchanged along it
    np.testing.assert_allclose(new @ v, jac @ v)
    assert new[0, 1] < 0 and new[1, 0] < 0  # more drive on one band compresses the other


def test_a_one_band_step_learns_that_bands_column_exactly():
    new = update_slopes(prior_slopes(2), np.array([-2.0, 0.0]), np.array([0.6, 0.12]))
    np.testing.assert_allclose(new[:, 0], [-0.3, -0.06])  # own and cross, as measured
    np.testing.assert_allclose(new[:, 1], prior_slopes(2)[:, 1])


def test_a_cross_slope_is_never_more_than_half_the_own_slope():
    jac = np.diag([-0.5, -0.5])
    new = update_slopes(jac, np.array([0.0, -1.0]), np.array([2.0, 0.5]))  # a noisy step
    assert new[0, 1] == pytest.approx(-0.25)  # the step alone says -2.0: bound to |J[0, 0]| / 2
    assert abs(new[1, 0]) <= abs(new[1, 1]) / 2
    held = update_slopes(np.diag([-1.4, -0.3]), np.array([-1.0, 0.0]), np.array([3.0, 0.0]))
    assert held[0, 0] == pytest.approx(-1.5)  # own slopes stay in 0.1 .. 1.5
    low = update_slopes(np.diag([-0.3, -0.3]), np.array([-1.0, 0.0]), np.array([0.0, 0.0]))
    assert low[0, 0] == pytest.approx(-0.1)


def test_the_slope_update_ignores_a_step_it_cannot_learn_from():
    jac = np.array([[-0.7, -0.1], [-0.2, -0.8]])
    np.testing.assert_array_equal(update_slopes(jac, np.zeros(2), np.array([1.0, 1.0])), jac)
    nan_step = update_slopes(jac, np.array([-1.0, 0.0]), np.array([np.nan, 0.2]))
    np.testing.assert_array_equal(nan_step, jac)


def test_case_names_carry_paths_carriers_bandwidths_and_targets():
    def b(tx, f, bw, gc):
        return BandSpec(tx, RxChannel.ORX1, "LO1", f, bw, None, target_db=gc)

    assert case_name(
        [b(TxChannel.TX1, 1_440_000_000, 80, 4.0), b(TxChannel.TX4, 1_860_000_000, 80, 4.0)]
    ) == ("TX1-TX4_1440-1860MHz_80-80BW_4-4GC")
    assert case_name([b(TxChannel.TX1, 758_500_000, 40, 3.5)]) == "TX1_758p5MHz_40BW_3p5GC"


# -- a whole case on the fake board with a shared PA ------------------------------------------

N = 16384
PA = {
    "shared": [["TX1", "TX4"]],
    "coupling": 4.0,
    "noise_codes": 0.5,
    "bands": {"TX1": {"drive": 6.8, "smoothness": 1.45}, "TX4": {"drive": 8.4, "smoothness": 1.75}},
}


def _signal(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.fft.ifft(
        np.fft.fft(rng.normal(size=N) + 1j * rng.normal(size=N))
        * (np.abs(np.fft.fftfreq(N)) < 0.15)
    )
    return x / np.abs(x).max()


def _case_radio() -> FakeRadio:
    radio = FakeRadio(Config(dll=DllConfig(install_dir=Path("C:/nonexistent"))), seed=2, pa=PA)
    radio._safe_hooks_installed = True
    radio.connect()
    radio.program()
    return radio


def _case_bands() -> list[BandSpec]:
    return [
        BandSpec(
            TxChannel.TX1,
            RxChannel.ORX1,
            "LO1",
            1_440_000_000,
            2.0,
            _signal(1),
            signal_name="Signal1.txt",
        ),
        BandSpec(
            TxChannel.TX4,
            RxChannel.ORX4,
            "LO2",
            1_860_000_000,
            2.0,
            _signal(2),
            signal_name="Signal2.txt",
        ),
    ]


def test_a_whole_case_on_the_shared_pa(tmp_path):
    radio = _case_radio()
    fs = radio.board_model.rate_hz
    bands = _case_bands()
    failed_once = []

    def check(caps):  # TX4's first capture of each backoff "fails" once: it is taken again
        out = {}
        for c in caps:
            bad = c.name == "TX4" and c.tries == 1
            failed_once.append(bad)
            out[c.name] = {"ok": not bad, "why": "test" if bad else ""}
        return out

    writer = CaseWriter(
        tmp_path,
        bands,
        tx_bits=12,
        rx_bits=12,
        fs=fs,
        meta={"case": "S20-80", "version": "v1", "profile": "fake"},
    )
    res = find_multiband_operating_point(
        radio, bands, tx_bits=12, rx_bits=12, fs=fs, backoffs=(0, 2), check=check, writer=writer
    )
    assert res.search.converged, res.search.reason
    assert all(abs(bl.compression_db - 4.0) <= 0.1 for bl in res.search.bands)
    assert radio.rx_tx_enable_get()[1] == 0  # TX off at the end
    locked = {bl.name: bl.atten_db for bl in res.search.bands}
    for k, caps in res.captures.items():
        for c in caps:
            assert c.atten_db == pytest.approx(locked[c.name] + k)
            assert c.tries == (2 if c.name == "TX4" else 1) and c.check["ok"]
            assert c.ref is res.refs[[b.name for b in bands].index(c.name)]
    gc = {
        k: {c.name: c.metrics["gain_compression_db"] for c in caps}
        for k, caps in res.captures.items()
    }
    assert all(
        gc[2][b] < gc[0][b] - 0.5 for b in ("TX1", "TX4")
    )  # backing off lowers the compression

    d = tmp_path / "TX1-TX4_1440-1860MHz_2-2GC".replace("2-2GC", "2-2BW_4-4GC")
    assert d == res.case_dir and d.is_dir()
    names = sorted(p.name for p in d.iterdir())
    stem = d.name
    expect = {f"{stem}_TX1_in.txt", f"{stem}_TX4_in.txt", "conditions.csv", "lock.csv"}
    expect |= {f"{stem}_{k}dB_{tx}_out.txt" for k in (0, 2) for tx in ("TX1", "TX4")}
    assert set(names) == expect
    rows = list(csv.DictReader(open(d / "conditions.csv", encoding="utf-8")))
    assert len(rows) == 4 and {r["case"] for r in rows} == {"S20-80"}
    assert {r["freqs_mhz"] for r in rows} == {"1440/1860"}
    assert all(r["check_ok"] == "True" for r in rows)
    lock = list(csv.DictReader(open(d / "lock.csv", encoding="utf-8")))
    assert {r["band"] for r in lock} == {"TX1", "TX4"}
    writer.write_case_json({"lock": [bl.status for bl in res.search.bands]})
    assert json.loads((d / "case.json").read_text())["case"] == "S20-80"

    with pytest.raises(FileExistsError, match="written once"):
        writer.write_output(0, "TX1", res.captures[0][0].y_aligned)


def test_every_conditions_row_carries_the_joint_compression_of_its_state(tmp_path):
    radio = _case_radio()
    fs = radio.board_model.rate_hz
    bands = _case_bands()
    writer = CaseWriter(tmp_path, bands, tx_bits=12, rx_bits=12, fs=fs)
    res = find_multiband_operating_point(
        radio, bands, tx_bits=12, rx_bits=12, fs=fs, backoffs=(0, 3), writer=writer
    )
    rows = list(csv.DictReader(open(res.case_dir / "conditions.csv", encoding="utf-8")))
    assert len(rows) == 4 and all(k in rows[0] for k in JOINT_KEYS)
    for r in rows:
        for k in JOINT_KEYS:
            if k != "joint_levels_db":
                assert np.isfinite(float(r[k])), (k, r[k])
        levels = [float(v) for v in r["joint_levels_db"].split("/")]
        attens = [float(x["tx_atten_db"]) for x in rows if x["backoff_db"] == r["backoff_db"]]
        assert levels == pytest.approx([min(attens) - a for a in attens], abs=0.01)
    by_k = {}
    for r in rows:  # one joint figure per state, the same on every band's row
        by_k.setdefault(r["backoff_db"], set()).add(r["joint_gain_compression_db"])
    assert all(len(v) == 1 for v in by_k.values())
    j0, j3 = (float(next(iter(by_k[k]))) for k in ("0", "3"))
    assert 0 < j3 < j0 - 0.5  # backing off lowers it
    for c in res.captures[0]:
        assert c.metrics["joint_gain_compression_db"] == pytest.approx(j0)


def test_a_state_the_joint_figure_cannot_read_gets_nan_and_keeps_its_captures():
    caps = [
        BandCapture("TX1", 6.0, 0, None, np.ones(8, complex), 0.0, 0, {"x": 1.0}, ref=np.ones(8)),
        BandCapture("TX4", 7.0, 0, None, np.ones(7, complex), 0.0, 0, {}, ref=np.ones(8)),
    ]  # lengths differ
    assert add_joint_metrics(_case_bands(), caps, 1e6) == {}
    assert caps[0].metrics["x"] == 1.0 and np.isnan(caps[0].metrics["joint_gain_compression_db"])
    assert caps[1].metrics["joint_levels_db"] == ""


def test_a_fatal_search_captures_and_writes_nothing(tmp_path, monkeypatch):
    radio, bands = _case_radio(), _case_bands()
    nan = float("nan")

    def fatal_search(*_a, **_k):  # band 1's ORx clipped at its minimum gain; band 2 unread
        locks = [
            BandLock("TX1", 20.05, 185, 1.2, 0.2, 0.4, 0.0, 9, 10.0, "not converged"),
            BandLock("TX4", nan, 210, nan, nan, nan, nan, 0, 10.0, "not converged"),
        ]
        return MultibandSearch(False, "TX1: ORx still clips at minimum gain", "joint", locks,
                               1, 3, fatal=True)  # fmt: skip

    monkeypatch.setattr(multiband, "search_multiband_compression", fatal_search)
    captured = []
    monkeypatch.setattr(multiband, "capture_case", lambda *a, **k: captured.append(a))
    writer = CaseWriter(tmp_path, bands, tx_bits=12, rx_bits=12, fs=radio.board_model.rate_hz)
    res = find_multiband_operating_point(
        radio, bands, tx_bits=12, rx_bits=12, fs=radio.board_model.rate_hz, writer=writer
    )
    assert res.search.fatal and res.captures == {} and res.case_dir is None
    assert captured == [] and not writer.dir.exists()  # not even the folder
    assert radio.rx_tx_enable_get()[1] == 0  # TX off


def test_a_lock_at_the_limit_is_still_captured(monkeypatch):
    radio = _case_radio()
    bands = [replace(b, atten_min_db=17.0) for b in _case_bands()]  # both need more drive
    fs = radio.board_model.rate_hz
    res = find_multiband_operating_point(
        radio, bands, tx_bits=12, rx_bits=12, fs=fs, backoffs=(0, 1)
    )
    assert not res.search.converged and not res.search.fatal
    assert {bl.status for bl in res.search.bands} == {"at limit"}
    assert sorted(res.captures) == [0, 1]
    assert [c.atten_db for c in res.captures[1]] == [bl.atten_db + 1 for bl in res.search.bands]


def test_a_used_case_folder_is_refused_before_the_radio_is_touched(tmp_path):
    bands = _case_bands()
    writer = CaseWriter(tmp_path, bands, tx_bits=12, rx_bits=12, fs=10e6)
    writer.dir.mkdir()
    (writer.dir / "conditions.csv").write_text("from an earlier run")
    radio = MagicMock(name="radio")
    with pytest.raises(FileExistsError, match="not empty"):
        find_multiband_operating_point(radio, bands, tx_bits=12, rx_bits=12, fs=10e6, writer=writer)
    assert radio.mock_calls == []


def test_a_failed_write_keeps_the_captures_in_the_error(tmp_path, monkeypatch):
    radio, bands = _case_radio(), _case_bands()
    fs = radio.board_model.rate_hz
    writer = CaseWriter(tmp_path, bands, tx_bits=12, rx_bits=12, fs=fs)

    def disk_full(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(writer, "write_conditions", disk_full)
    with pytest.raises(OSError, match="disk full") as err:
        find_multiband_operating_point(
            radio, bands, tx_bits=12, rx_bits=12, fs=fs, backoffs=(0, 2), writer=writer
        )
    kept = err.value.case_result
    assert sorted(kept.captures) == [0, 2] and kept.search.converged
    assert all(len(caps) == 2 for caps in kept.captures.values())
    assert radio.rx_tx_enable_get()[1] == 0


def _playing_case():
    radio, bands = _case_radio(), _case_bands()
    for b in bands:
        radio.retune_lo(b.lo, b.freq_hz)
    refs = play_bands(radio, bands, tx_bits=12)
    return radio, bands, refs, radio.board_model.rate_hz


def test_capture_case_without_agc_records_the_gains_it_set_never_a_readback(monkeypatch):
    radio, bands, refs, fs = _playing_case()
    monkeypatch.setattr(radio, "get_rx_gain", lambda _ch: 0)  # what the real chip returns
    with pytest.raises(ValueError, match="gains"):
        capture_case(radio, bands, refs, [16.0, 16.0], rx_bits=12, fs=fs, agc=False)
    caps = capture_case(
        radio,
        bands,
        refs,
        [16.0, 16.0],
        rx_bits=12,
        fs=fs,
        agc=False,
        gains=[205, 207],
        check=lambda cs: {c.name: {"ok": c.tries > 1} for c in cs},  # every band retaken once
    )
    assert [c.orx_gain for c in caps] == [205, 207] and [c.tries for c in caps] == [2, 2]
    state = radio.board_model.state["rx_gain"]
    assert (state["ORX1"], state["ORX4"]) == (205, 207)


def test_capture_case_relevels_a_railed_band_and_records_the_new_gain(monkeypatch):
    radio, bands, refs, fs = _playing_case()
    levels = []
    real_agc, real_measure = multiband.autolevel_for, multiband.measure_band

    def agc(radio_, orx, *a, **k):
        result = real_agc(radio_, orx, *a, **k)
        levels.append((orx, result.final_gain_index))
        return result

    def measure(radio_, orx, *a, **k):
        peak, railed, comp, extra = real_measure(radio_, orx, *a, **k)
        first_tx1 = orx == RxChannel.ORX1 and sum(o == orx for o, _g in levels) == 1
        return peak, 3 if first_tx1 else railed, comp, extra  # TX1's first capture railed

    monkeypatch.setattr(multiband, "autolevel_for", agc)
    monkeypatch.setattr(multiband, "measure_band", measure)
    caps = capture_case(
        radio,
        bands,
        refs,
        [16.0, 16.0],
        rx_bits=12,
        fs=fs,
        check=lambda cs: {c.name: {"ok": c.railed == 0} for c in cs},
    )
    assert [o for o, _g in levels] == [RxChannel.ORX1, RxChannel.ORX4, RxChannel.ORX1]
    assert caps[0].tries == 2 and caps[0].railed == 0 and caps[0].orx_gain == levels[-1][1]
    assert caps[1].tries == 1 and caps[1].orx_gain == levels[1][1]


@pytest.mark.parametrize(
    "second, match",
    [
        ({"tx": TxChannel.TX2, "lo": "LO1", "freq_hz": 1_860_000_000}, "shared by bands"),
        ({"lo": "LO1", "freq_hz": 1_440_000_000}, "routes TX4 to 'LO2'"),
        ({"tx": TxChannel.TX2, "lo": "LO2"}, "routes TX2 to 'LO1'"),
    ],
)
@pytest.mark.parametrize("call", ["find", "lock"])
def test_bands_must_use_the_lo_their_tx_is_routed_to_at_one_frequency(second, match, call):
    radio = MagicMock(name="radio")
    radio.config = Config(dll=DllConfig(install_dir=Path("C:/nonexistent")))  # TX4 on LO2
    bands = [_case_bands()[0], replace(_case_bands()[1], **second)]
    run = find_multiband_operating_point if call == "find" else lock_bands
    with pytest.raises(ValueError, match=match):
        run(radio, bands, tx_bits=12, rx_bits=12, fs=10e6)
    assert radio.mock_calls == []


def test_two_bands_on_one_lo_at_one_frequency_are_fine():
    radio = _case_radio()
    tx2 = BandSpec(TxChannel.TX2, RxChannel.ORX2, "LO1", 1_440_000_000, 2.0, _signal(3))
    bands = [_case_bands()[0], tx2]
    multiband._check_los(bands, radio.config.clocks)  # TX1 and TX2 share LO1
    play_bands(radio, bands, tx_bits=12)
    assert radio.board_model.state["tx_mask"] == int(TxChannel.TX1) | int(TxChannel.TX2)


def test_case_json_carries_the_lock(tmp_path):
    bands = _case_bands()
    res = TwoBandPa().search()
    writer = CaseWriter(tmp_path, bands, tx_bits=12, rx_bits=12, fs=10e6, meta={"case": "C1"})
    writer.write_lock(res)
    doc = json.loads(writer.write_case_json({"note": "after the run"}).read_text())
    assert doc["lock"] == multiband.lock_summary(res) and doc["case"] == "C1"
    assert [row["band"] for row in doc["lock"]] == ["TX1", "TX4"]
    assert doc["lock"][0]["atten_db"] == res.bands[0].atten_db
    other = CaseWriter(tmp_path / "b", bands, tx_bits=12, rx_bits=12, fs=10e6)
    assert "lock" not in json.loads(other.write_case_json().read_text())  # no lock written yet
    assert "lock_summary" in multiband.__all__


def test_the_search_aligns_each_band_in_full_once_then_with_its_fraction(monkeypatch):
    calls = {"full": 0, "fraction": 0}
    from adrvtrx import capture as capture_module

    real_full, real_fraction = capture_module.estimate_and_align, capture_module.align_with_fraction

    def full(*a, **k):
        calls["full"] += 1
        return real_full(*a, **k)

    def fraction(*a, **k):
        calls["fraction"] += 1
        return real_fraction(*a, **k)

    monkeypatch.setattr(capture_module, "estimate_and_align", full)
    monkeypatch.setattr(capture_module, "align_with_fraction", fraction)
    radio, bands = _case_radio(), _case_bands()
    search, _refs = lock_bands(radio, bands, tx_bits=12, rx_bits=12, fs=radio.board_model.rate_hz)
    assert search.converged
    assert calls["full"] == 2  # each band's first reading came leveled from the AGC
    assert calls["fraction"] == len(search.history) - 2


@pytest.mark.parametrize("bench, tol_db", [("sim", 0.005), ("rich", 0.03)])
def test_a_cached_fraction_reads_the_compression_the_full_alignment_reads(bench, tol_db):
    """Within 0.005 dB on the memoryless sim bench (path delay 0.37 samples); within 0.03 dB on the
    rich fake PA, whose AM/PM and memory move its apparent delay by 0.03 sample from 0.1 to
    5.7 dB of compression (a quarter of the search's 0.1 dB tolerance)."""
    from adrvtrx.align import align_with_fraction, estimate_and_align
    from adrvtrx.capture import capture
    from adrvtrx.metrics import gain_compression_db

    if bench == "sim":
        from sim_bench import FS, SimRadio, make_signal, reference_codes

        radio, fs, x = SimRadio(seed=3), FS, make_signal()
        ref, orx, attens = reference_codes(x), RxChannel.ORX1, (14, 10, 8, 6, 4)
        radio.set_rx_gain(orx, 200)
        transmit_bands(radio, {TxChannel.TX1: x}, 12)
    else:
        radio = FakeRadio(
            Config(dll=DllConfig(install_dir=Path("C:/nonexistent"))), seed=2, pa="rich"
        )
        radio._safe_hooks_installed = True
        radio.connect()
        radio.program()
        radio.set_rx_gain(RxChannel.ORX1, 205)
        band = BandSpec(TxChannel.TX1, RxChannel.ORX1, "LO1", 2_200_000_000, 2.0, _signal(1))
        radio.retune_lo("LO1", band.freq_hz)
        radio.set_tx_atten(TxChannel.TX1, 20.0)
        ref, orx, attens = (
            play_bands(radio, [band], tx_bits=12)[0],
            RxChannel.ORX1,
            (20, 16, 13, 11, 9),
        )
        fs = radio.board_model.rate_hz
    frac, compressions = None, []
    for atten in attens:  # the fraction of the first, least compressed capture serves them all
        radio.set_tx_atten(TxChannel.TX1, atten)
        y = capture(radio, int(orx), 2 * len(ref) / fs * 1e3, bits=12).channels[orx].iq
        xa, ya, d = estimate_and_align(ref, y, fs)
        frac = d % 1.0 if frac is None else frac
        xb, yb, _d = align_with_fraction(ref, y, frac)
        full, cheap = gain_compression_db(xa, ya)[0], gain_compression_db(xb, yb)[0]
        compressions.append(full)
        assert cheap == pytest.approx(full, abs=tol_db), (atten, full, cheap)
    assert compressions[-1] > 3.5  # the range covers the lock targets


def test_each_backoff_levels_from_the_gain_predicted_from_the_lock(monkeypatch):
    starts, real_capture_case = {}, multiband.capture_case

    def capture_case_spy(*a, **k):
        starts[len(starts)] = k["gains"]
        return real_capture_case(*a, **k)

    monkeypatch.setattr(multiband, "capture_case", capture_case_spy)
    radio, bands = _case_radio(), _case_bands()
    floor_starts = []
    real_set = radio.set_rx_gain
    radio.set_rx_gain = lambda ch, g: (floor_starts.append(g == 185), real_set(ch, g))[1]
    res = find_multiband_operating_point(
        radio, bands, tx_bits=12, rx_bits=12, fs=radio.board_model.rate_hz, backoffs=(0, 2, 4)
    )
    lock = [bl.orx_gain for bl in res.search.bands]
    # 0.5 dB of output per dB of backoff, 0.5 dB per index: k indices up
    assert starts == {0: lock, 1: [g + 2 for g in lock], 2: [g + 4 for g in lock]}
    assert sum(floor_starts) == 2  # only the lock's own AGC (one per band) started at the floor
    for caps in res.captures.values():
        assert all(c.railed == 0 and -1.6 <= c.peak_dbfs <= -0.7 for c in caps)


def test_capture_case_agc_starts_from_the_given_gains():
    radio, bands, refs, fs = _playing_case()
    set_gains = []
    real_set = radio.set_rx_gain
    radio.set_rx_gain = lambda ch, g: (set_gains.append((ch, g)), real_set(ch, g))[1]
    caps = capture_case(radio, bands, refs, [16.0, 16.0], rx_bits=12, fs=fs, gains=[219, 221])
    assert set_gains[0] == (RxChannel.ORX1, 219)
    assert (RxChannel.ORX4, 221) in set_gains and (RxChannel.ORX1, 185) not in set_gains
    assert all(c.railed == 0 for c in caps)


def test_a_band_describes_itself_for_case_json_and_conditions(tmp_path):
    from adrvtrx._enums import channel_name
    from adrvtrx.compression import _quantize_atten, quantize_atten

    band = _case_bands()[1]
    assert band.describe() == {
        "name": "TX4",
        "tx": "TX4",
        "orx": "ORX4",
        "lo": "LO2",
        "freq_hz": 1_860_000_000,
        "bw_mhz": 2.0,
        "signal": "Signal2.txt",
        "target_db": 4.0,
        "atten_start_db": 20.0,
        "atten_min_db": 10.0,
    }
    writer = CaseWriter(tmp_path, _case_bands(), tx_bits=12, rx_bits=12, fs=10e6)
    doc = json.loads(writer.write_case_json().read_text())
    assert doc["bands"] == [b.describe() for b in _case_bands()]
    assert channel_name(TxChannel.TX4) == "TX4" and channel_name("TXA") == "TXA"
    assert _quantize_atten is quantize_atten and quantize_atten(12.37) == 12.35


def test_a_case_writer_creates_its_folder_with_its_first_file(tmp_path):
    writer = CaseWriter(tmp_path, _case_bands(), tx_bits=12, rx_bits=12, fs=10e6)
    writer.require_empty()  # nothing there yet: fine, and still nothing created
    assert not writer.dir.exists() and list(tmp_path.iterdir()) == []
    writer.write_text("notes.md", "first")
    assert [p.name for p in writer.dir.iterdir()] == ["notes.md"]
    other = CaseWriter(tmp_path / "b", _case_bands(), tx_bits=12, rx_bits=12, fs=10e6)
    target = other.path("TX1_extra.txt")  # naming a file to write makes room for it
    target.write_text("x")
    assert target.parent == other.dir and other.dir.is_dir()


@pytest.mark.parametrize("method", METHODS)
def test_a_band_that_can_never_level_stops_the_search_when_nothing_can_change(method):
    pa = TwoBandPa(noise=0.0, comp_nan={1: 1})  # TX4 never aligns: measured again, same gain
    res = pa.search(method=method)  # max_steps 20 would take 161 captures
    assert not res.converged and res.steps < 10 and res.captures < 70
    assert res.reason.startswith("no band could move for 2 steps (TX4 not leveled")
    assert "TX1 within tolerance" in res.reason and res.bands[0].status == "converged"
    last_two = [h for h in res.history if h["step"] >= res.steps - 2]
    assert all(h["move_db"] == 0.0 for h in last_two)
    assert pa.a == [bl.atten_db for bl in res.bands]  # left where it was measured


# -- a convergence oracle, independent of the pinned golden file ------------------------------

_LOCK_GRID = [(l1, l2) for l1 in (13.0, 14.5, 16.0, 17.5) for l2 in (13.0, 14.5, 16.0, 17.5)]


def _oracle_cases(method: str):
    """Linear PAs (``c = 4 + S (locks - a)``, noise-free) with own slopes from the bench's 0.3 to
    1 dB/dB and up to 0.4 dB/dB from the other band, from the bench prior (0.31 / 0.06)."""
    for own in (0.3, 0.7, 1.0):
        for cross in (0.0, 0.1, 0.2, 0.3, 0.4):
            if cross > own:
                continue
            for symmetric in (True, False):
                yield [[own, cross], [cross if symmetric else 0.0, own]]


@pytest.mark.parametrize("method, budget, mean_max", [("joint", 8, 4.5), ("pingpong", 16, 9.0)])
def test_both_methods_lock_linear_coupled_bands_within_a_step_budget(method, budget, mean_max):
    steps = []
    for slopes in _oracle_cases(method):
        for locks in _LOCK_GRID:
            pa = LinearPa(slopes, locks=locks)
            res = pa.search(method=method, max_steps=budget)
            assert res.converged, (method, slopes, locks, res.reason)
            np.testing.assert_allclose(pa.comp(), 4.0, atol=0.1)  # the truth, not the reading
            steps.append(res.steps)
    assert np.mean(steps) <= mean_max, np.mean(steps)


# -- behaviour the second review's injected bugs found unguarded --------------------------------


@pytest.mark.parametrize("method, max_steps", [("joint", 5)])
def test_a_clipped_last_step_is_never_the_one_kept(method, max_steps):
    """The last step clips (top slope under the guard) and is the closest to the targets; the
    step kept at max_steps must be the best one that did not clip."""
    pa = TwoBandPa(locks=(12.0, 14.0), cross=(0.0, 0.0), clip_below={0: 13.0}, noise=0.0)
    res = pa.search(method=method, max_steps=max_steps)
    steps = _counted_steps(res)
    last = steps[max(steps)]
    assert last["TX1"]["top_slope"] < PA_CLIP_SLOPE  # the scenario: the last step clipped
    worst = {s: max(abs(h["error_db"]) for h in rows.values()) for s, rows in steps.items()}
    unclipped = [
        s
        for s, rows in steps.items()
        if all(h["top_slope"] >= PA_CLIP_SLOPE for h in rows.values())
    ]
    best = min(unclipped, key=worst.get)
    assert worst[max(steps)] < worst[best]  # ...and it was closer to the targets
    assert "closest" in res.reason
    assert pa.a == [steps[best][name]["atten_db"] for name in ("TX1", "TX4")]
    assert all(bl.top_slope >= PA_CLIP_SLOPE for bl in res.bands)


def test_a_band_at_its_limit_is_held_and_left_out_of_the_solve():
    """TX4 needs more drive than its 12 dB limit allows; once it is there, the joint method
    solves for TX1 alone (TX4's move is fixed at 0), although J has cross terms."""
    pa = TwoBandPa(locks=(14.0, 9.0), cross=(0.3, 0.3), noise=0.0)
    res = pa.search(method="joint", atten_mins_db=[10.0, 12.0])
    steps = _counted_steps(res)
    checked = 0
    for rows in steps.values():
        t1, t4 = rows["TX1"], rows["TX4"]
        if t4["action"] != "at limit":
            continue
        assert t4["atten_db"] == 12.0 and t4["move_db"] == 0.0
        assert t1["action"] in ("joint", "within tolerance")
        jac = np.array(json.loads(t1["slopes"]))
        assert abs(jac[0, 1]) > 0.05  # a cross term the 2x2 solve would have used
        if t1["action"] == "within tolerance":
            continue
        move = float(np.clip(t1["error_db"] / jac[0, 0], -4.0, 4.0))  # TX1's own row only
        expected = multiband.quantize_atten(t1["atten_db"] + move) - t1["atten_db"]
        assert t1["move_db"] == pytest.approx(expected, abs=1e-9)
        checked += 1
    assert checked >= 1 and res.bands[1].status == "at limit"


def test_the_fatal_band_ends_one_clip_step_above_where_it_clipped():
    pa = TwoBandPa(noise=0.0, orx_peak={1: 0.5})  # TX4's ORx rails at every gain
    res = pa.search()
    assert res.fatal and "TX4" in res.reason
    fatal_row = res.history[-1]
    assert fatal_row["band"] == "TX4" and fatal_row["action"].startswith("ORx clips")
    assert fatal_row["atten_db"] == 20.0  # held at its start while it re-leveled
    assert res.bands[1].atten_db == pa.a[1] == pytest.approx(20.2)  # one clip step up
    tx1 = [h for h in res.history if h["band"] == "TX1"]
    assert pa.a[0] == tx1[-1]["atten_db"]  # the other band stays where it was measured


def test_a_railed_retake_levels_from_the_floor_not_from_the_predicted_gain(monkeypatch):
    radio, bands, refs, fs = _playing_case()
    starts = []
    real_agc, real_measure = multiband.autolevel_for, multiband.measure_band

    def agc(radio_, orx, *a, **k):
        starts.append((orx, k.get("start_gain")))
        return real_agc(radio_, orx, *a, **k)

    def measure(radio_, orx, *a, **k):
        peak, railed, comp, extra = real_measure(radio_, orx, *a, **k)
        first_tx1 = orx == RxChannel.ORX1 and sum(o == orx for o, _g in starts) == 1
        return peak, 3 if first_tx1 else railed, comp, extra  # TX1's first capture railed

    monkeypatch.setattr(multiband, "autolevel_for", agc)
    monkeypatch.setattr(multiband, "measure_band", measure)
    caps = capture_case(
        radio,
        bands,
        refs,
        [16.0, 16.0],
        rx_bits=12,
        fs=fs,
        gains=[219, 221],
        check=lambda cs: {c.name: {"ok": c.railed == 0} for c in cs},
    )
    assert starts == [(RxChannel.ORX1, 219), (RxChannel.ORX4, 221), (RxChannel.ORX1, None)]
    assert caps[0].tries == 2 and caps[0].railed == 0


def test_the_orx_settings_reach_the_lock_and_every_backoff_capture():
    """The ORx level, window and gain range given to find_multiband_operating_point level the ORx
    in the lock and in every backoff capture alike (not only in the lock)."""
    radio = _case_radio()
    fs = radio.board_model.rate_hz
    bands = _case_bands()
    orx = {
        "target_dbfs": -3.0,
        "tol_up_db": 0.3,
        "tol_down_db": 0.6,
        "gain_min": 190,
        "gain_max": 250,
    }
    res = find_multiband_operating_point(
        radio, bands, tx_bits=12, rx_bits=12, fs=fs, backoffs=(0, 2), **orx
    )
    assert res.search.converged, res.search.reason
    for k, caps in res.captures.items():
        for c in caps:
            assert -3.6 <= c.peak_dbfs <= -2.7, (
                k,
                c.name,
                c.peak_dbfs,
            )  # leveled to -3 dBFS, not -1
            assert 190 <= c.orx_gain <= 250, (k, c.name, c.orx_gain)
    assert all(190 <= h["orx_gain"] <= 250 for h in res.search.history)
