"""Several bands at once: the search on a synthetic two-band PA, then a case on the fake PA."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.compression import search_tx_compression
from adrvtrx.config import Config, DllConfig
from adrvtrx.fake import FakeRadio
from adrvtrx.multiband import (
    BandSpec,
    CaseWriter,
    case_name,
    find_multiband_operating_point,
    search_multiband_compression,
)


class TwoBandPa:
    """Gain compression of each band: its own Rapp-like curve plus ``cross`` dB per dB of the other
    band's
    attenuation below 20 dB. ORx peak follows attenuation and gain index (0.5 dB per index).
    ``orx_peak`` pins a band's ORx peak reading (``{0: 0.5}``: band 1 always reads +0.5 dBFS,
    railed): an ORx that never levels. ``comp_nan`` makes every n-th reading of a band NaN
    (``{1: 3}``: every third of band 2's; ``{1: 1}``: all of them), as a failed alignment does."""

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
    ):
        self.locks, self.cross, self.p, self.noise = locks, cross, p, noise
        self.rng = np.random.default_rng(seed)
        self.clip_below = clip_below or {}
        self.orx_peak = orx_peak or {}
        self.comp_nan = comp_nan or {}
        self.band_reads = [0, 0]
        self.a = [20.0, 20.0]
        self.g = [210, 210]
        self.reads = 0

    def comp(self, i: int) -> float:
        own = self.a[i] - self.locks[i]  # 0 at this band's own lock (other band at 20 dB)
        k = 10 ** (self.p * 4.0 / 10) - 1  # 4 dB at own = 0
        c = 10 / self.p * np.log10(1 + k * 10 ** (-self.p * own / 10))
        return float(c + self.cross[i] * (20.0 - self.a[1 - i]))

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
            self.measure,
            **defaults,
        )


@pytest.mark.parametrize("method", ["slopes", "walk"])
@pytest.mark.parametrize("cross", [(0.0, 0.0), (0.3, 0.3), (0.0, 0.4)])
def test_both_methods_lock_two_coupled_bands(method, cross):
    pa = TwoBandPa(cross=cross)
    res = pa.search(method=method)
    assert res.converged, res.reason
    for i, bl in enumerate(res.bands):
        assert bl.status == "converged"
        assert abs(pa.comp(i) - 4.0) <= 0.1 + 0.06  # tolerance plus the reading noise
    assert pa.a == [bl.atten_db for bl in res.bands]  # the radio is left at the lock
    assert res.captures >= 2 * res.steps


def test_measured_slopes_need_no_more_captures_than_the_walk():
    totals = {}
    for method in ("slopes", "walk"):
        totals[method] = sum(
            TwoBandPa(locks=locks, cross=cross, seed=s).search(method=method).captures
            for s, (locks, cross) in enumerate(
                [
                    ((14, 15), (0.3, 0.3)),
                    ((13, 16.5), (0.0, 0.4)),
                    ((15.5, 14), (0.2, 0.0)),
                    ((12.5, 13), (0.1, 0.1)),
                ]
            )
        )
    assert totals["slopes"] <= totals["walk"], totals


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
    assert res.captures > 2 * res.steps


@pytest.mark.parametrize("method", ["slopes", "walk"])
def test_a_band_whose_orx_never_levels_keeps_its_attenuation(method):
    pa = TwoBandPa(orx_peak={0: 0.5})  # band 1's ORx always reads +0.5 dBFS (railed)
    res = pa.search(method=method, max_steps=4, gain_min=100)  # never at the ORx gain floor
    tx1 = [h for h in res.history if h["band"] == "TX1"]
    assert len(tx1) == 4 * 7  # max_relevel + 1 captures per step, every one re-leveling
    assert {h["atten_db"] for h in tx1} == {20.0} and all(h["move_db"] == 0.0 for h in tx1)
    assert all(h["action"].startswith("ORx gain -") for h in tx1)
    assert sum("atten held" in h["action"] for h in tx1) == 4
    assert pa.a[0] == 20.0 and res.bands[0].status == "not converged"
    tx4 = [h for h in res.history if h["band"] == "TX4"]
    assert tx4[0]["action"] == "coarse" and tx4[-1]["atten_db"] < 20.0  # the other band goes on


@pytest.mark.parametrize("method", ["slopes", "walk"])
@pytest.mark.parametrize("orx", [{"orx_gains": [240, 180]}, {"orx_peak": {1: 0.5}}])
def test_every_capture_is_one_history_row_with_its_own_action(method, orx):
    pa = TwoBandPa(**{k: v for k, v in orx.items() if k == "orx_peak"})
    res = pa.search(method=method, max_steps=6, **{k: v for k, v in orx.items() if k != "orx_peak"})
    assert res.fatal == ("orx_peak" in orx)  # a band that rails at the ORx gain floor stops it
    assert res.captures == len(res.history) == pa.reads
    assert len({id(h) for h in res.history}) == len(res.history)  # no row recorded twice
    for h in res.history:
        assert h["action"]
        if h["action"].startswith("ORx gain"):
            assert h["move_db"] == 0.0 and h["slopes"] == ""


@pytest.mark.parametrize("method", ["slopes", "walk"])
def test_a_nan_compression_is_measured_again_and_never_counts(method):
    pa = TwoBandPa(comp_nan={1: 3})  # every third reading of band 2 fails to align
    res = pa.search(method=method)
    assert res.converged, res.reason
    nan_rows = [h for h in res.history if not np.isfinite(h["gain_compression_db"])]
    assert nan_rows and all(h["band"] == "TX4" for h in nan_rows)
    for h in nan_rows:  # measured again at the same gain and attenuation: no move
        assert h["action"] == "measure again (compression not finite)" and h["move_db"] == 0.0
        assert not h["leveled"]
    assert all(np.isfinite(bl.compression_db) for bl in res.bands)
    assert all("nan" not in h["slopes"].lower() for h in res.history)


@pytest.mark.parametrize("method", ["slopes", "walk"])
def test_a_band_that_never_aligns_holds_and_the_search_does_not_crash(method):
    pa = TwoBandPa(comp_nan={1: 1})
    res = pa.search(method=method, max_steps=5)
    assert not res.converged
    tx4 = [h for h in res.history if h["band"] == "TX4"]
    assert {h["atten_db"] for h in tx4} == {20.0} and {h["orx_gain"] for h in tx4} == {210}
    assert all(h["action"].startswith("measure again") for h in tx4)
    assert res.bands[1].status == "not converged" and pa.a[1] == 20.0
    assert "closest" not in res.reason  # no step counted, so none was kept as the best


def test_a_band_first_leveled_late_still_takes_its_coarse_step():
    pa = TwoBandPa(cross=(0.0, 0.0), noise=0.0)
    res = pa.search(method="walk", orx_gains=[240, 210], max_relevel=1)  # band 1: 2 steps hot
    tx1 = [h for h in res.history if h["band"] == "TX1" and not h["action"].startswith("ORx")]
    assert tx1[0]["action"] == "coarse" and tx1[0]["move_db"] == -2.5
    assert res.converged


def test_one_band_ends_where_the_single_band_walk_ends():
    pa = TwoBandPa(cross=(0.0, 0.0), noise=0.0)
    one = search_multiband_compression(
        lambda i, db: pa.a.__setitem__(0, db),
        lambda i, g: pa.g.__setitem__(0, g),
        lambda i: pa.measure(0),
        targets_db=[4.0],
        start_attens_db=[20.0],
        atten_mins_db=[10.0],
        orx_gains=[210],
        method="walk",
        coarse_step_db=2.5,
        fine_step_db=0.2,
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
    assert one.bands[0].atten_db == pytest.approx(single.final_atten_db, abs=0.1)


@pytest.mark.parametrize(
    "bad, match",
    [({"max_steps": 0}, "max_steps"), ({"comp_tol_db": 0.0}, "comp_tol_db"),
     ({"comp_tol_db": -0.1}, "comp_tol_db")],
)  # fmt: skip
def test_the_search_rejects_a_budget_or_tolerance_it_cannot_work_with(bad, match):
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


def test_a_whole_case_on_the_shared_pa(tmp_path):
    radio = FakeRadio(Config(dll=DllConfig(install_dir=Path("C:/nonexistent"))), seed=2, pa=PA)
    radio._safe_hooks_installed = True
    radio.connect()
    radio.program()
    fs = radio.board_model.rate_hz
    bands = [
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
