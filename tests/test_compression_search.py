from __future__ import annotations

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.compression import find_compression_point, search_tx_compression
from adrvtrx.metrics import gain_compression_db, window_compression_db
from adrvtrx.transmit import transmit_bands
from sim_bench import BITS, FS, SimRadio, make_signal, rapp, reference_codes


class Bench:
    """Memoryless Rapp PA + ORx gain, driven through the search callbacks."""

    def __init__(self, atten=20.0, gain=230):
        rng = np.random.default_rng(1)
        x = (rng.normal(size=8000) + 1j * rng.normal(size=8000)) / np.sqrt(2)
        x[1000] *= 6.0
        self.x = x / np.max(np.abs(x))
        self.atten = atten
        self.gain = gain
        self.atten_calls: list[float] = []
        self.gain_calls: list[int] = []

    def set_tx_atten(self, db):
        self.atten = float(db)
        self.atten_calls.append(self.atten)

    def set_orx_gain(self, g):
        self.gain = int(g)
        self.gain_calls.append(self.gain)

    def measure(self):
        y = rapp(self.x * 4.0 * 10 ** (-self.atten / 20.0))
        z = y * 1500 * 10 ** ((self.gain - 210) * 0.5 / 20.0)
        per = np.maximum(np.abs(z.real), np.abs(z.imag))
        railed = int(np.count_nonzero(per >= 2047))
        peak = 20 * np.log10(min(per.max(), 2047) / 2048)
        comp, _, _ = window_compression_db(self.x, y, fs=1.0, window_s=400)
        return peak, railed, comp


class GainBench(Bench):
    """The same bench, reporting gain compression (and the extra readings)."""

    def __init__(self, *a, pa=rapp, **kw):
        super().__init__(*a, **kw)
        self.pa = pa

    def measure(self):
        y = self.pa(self.x * 4.0 * 10 ** (-self.atten / 20.0))
        z = y * 1500 * 10 ** ((self.gain - 210) * 0.5 / 20.0)
        per = np.maximum(np.abs(z.real), np.abs(z.imag))
        railed = int(np.count_nonzero(per >= 2047))
        peak = 20 * np.log10(min(per.max(), 2047) / 2048)
        papr, _, _ = window_compression_db(self.x, y, fs=1.0, window_s=400)
        gain, slope = gain_compression_db(self.x, y)
        extra = {"papr_compression_db": papr, "gain_compression_db": gain, "top_slope": slope}
        return peak, railed, gain, extra


def _hard_pa(v):
    return rapp(v, p=20.0)


def _run(bench, **kw):
    args = {
        "target_compression_db": 3.0,
        "start_atten_db": 20.0,
        "atten_min_db": 2.0,
        "orx_gain": bench.gain,
        "coarse_step_db": 4.0,
        "comp_tol_db": 0.3,
    }
    args.update(kw)
    return search_tx_compression(bench.set_tx_atten, bench.set_orx_gain, bench.measure, **args)


def test_converges_within_tolerance_and_leaves_radio_there():
    bench = Bench()
    res = _run(bench)
    assert res.converged
    assert abs(res.compression_db - 3.0) <= 0.3
    assert bench.atten == res.final_atten_db
    assert bench.gain == res.final_orx_gain
    assert res.history[-1]["action"] == "converged"
    assert res.history[-1]["orx_ok"]


def test_orx_gain_follows_each_atten_step():
    res = _run(Bench())
    for prev, cur in zip(res.history, res.history[1:]):
        if prev["action"].startswith("TX atten"):
            delta = cur["atten_db"] - prev["atten_db"]
            assert cur["orx_gain"] - prev["orx_gain"] == int(round(delta / 0.5))


def test_first_step_is_coarse():
    res = _run(Bench(), coarse_step_db=4.0)
    first = next(h for h in res.history if h["action"].startswith("TX atten"))
    assert first["action"] == "TX atten -4.00 dB"


def test_attenuation_is_quantized_to_0p05():
    bench = Bench()
    res = _run(bench, start_atten_db=20.03)
    assert res.history[0]["atten_db"] == 20.05
    for h in res.history:
        assert h["atten_db"] * 20 == pytest.approx(round(h["atten_db"] * 20))


def test_floor_reached_before_target():
    res = _run(Bench(), target_compression_db=20.0, atten_min_db=10.0)
    assert not res.converged
    assert res.at_atten_floor
    assert res.final_atten_db == 10.0
    assert min(h["atten_db"] for h in res.history) >= 10.0


def test_ceiling_reached():
    calls = []

    def measure():
        return -1.0, 0, 5.0  # always too much compression, always leveled

    res = search_tx_compression(
        lambda db: calls.append(db),
        lambda g: None,
        measure,
        target_compression_db=3.0,
        start_atten_db=41.9,
        atten_min_db=10.0,
        orx_gain=220,
    )
    assert not res.converged
    assert res.reason.startswith("attenuation ceiling")
    assert res.final_atten_db == 41.95


def test_rails_at_gain_floor_is_fatal():
    def measure():
        return 0.0, 50, 1.0

    res = search_tx_compression(
        lambda db: None,
        lambda g: None,
        measure,
        target_compression_db=3.0,
        start_atten_db=20.0,
        atten_min_db=5.0,
        orx_gain=185,
        fine_step_db=0.25,
    )
    assert res.fatal
    assert not res.converged
    assert res.final_atten_db == 20.25


def test_untrusted_reading_only_moves_orx_gain():
    readings = iter([(-10.0, 0, 9.9), (-1.0, 0, 3.0)])

    res = search_tx_compression(
        lambda db: None,
        lambda g: None,
        lambda: next(readings),
        target_compression_db=3.0,
        start_atten_db=20.0,
        atten_min_db=5.0,
        orx_gain=200,
    )
    assert res.converged
    assert res.history[0]["action"] == "ORx gain +18"
    assert res.history[1]["atten_db"] == 20.0


def test_max_iterations_returns_closest():
    def measure():
        return -1.0, 0, 2.0 if state["atten"] > 15.0 else 4.0

    state = {"atten": 20.0}

    res = search_tx_compression(
        lambda db: state.update(atten=db),
        lambda g: None,
        measure,
        target_compression_db=3.0,
        start_atten_db=20.0,
        atten_min_db=5.0,
        orx_gain=220,
        coarse_step_db=5.0,
        max_iterations=8,
    )
    assert not res.converged
    assert res.reason.startswith("max iterations")
    assert abs(res.compression_db - 3.0) == pytest.approx(1.0)
    assert state["atten"] == res.final_atten_db


def test_bad_arguments():
    with pytest.raises(ValueError):
        _run(Bench(), atten_min_db=25.0)
    with pytest.raises(ValueError):
        _run(Bench(), comp_tol_db=0.0)


def test_find_compression_point_on_sim_bench():
    radio = SimRadio()
    x = make_signal()
    transmit_bands(radio, {TxChannel.TX1: x}, BITS)
    res = find_compression_point(
        radio,
        TxChannel.TX1,
        RxChannel.ORX1,
        reference_codes(x),
        rx_bits=BITS,
        fs=FS,
        target_compression_db=3.0,
        start_atten_db=20.0,
        atten_min_db=5.0,
        coarse_step_db=5.0,
        fine_step_db=0.2,
        comp_tol_db=0.1,
    )
    assert res.converged
    assert abs(res.compression_db - 3.0) <= 0.1
    assert radio.atten == res.final_atten_db
    assert radio.gain == res.final_orx_gain
    assert 5.0 <= res.final_atten_db < 20.0


def test_gain_lock_converges_and_records_every_reading():
    bench = GainBench()
    res = _run(bench, target_compression_db=4.0, comp_tol_db=0.2)
    assert res.converged
    assert abs(res.compression_db - 4.0) <= 0.2
    assert res.gain_compression_db == res.compression_db
    assert np.isfinite(res.papr_compression_db) and np.isfinite(res.top_slope)
    for h in res.history:
        assert {"papr_compression_db", "gain_compression_db", "top_slope"} <= set(h)
    assert "pa_clipped" not in res.history[0]  # guard off by default


def test_clip_guard_stops_at_the_last_unclipped_attenuation():
    """A hard-limiting PA reaches 4 dB of gain compression only with a flat top."""
    bench = GainBench(pa=_hard_pa)
    res = _run(bench, target_compression_db=4.0, comp_tol_db=0.2, min_top_slope=0.08)
    assert not res.converged
    assert res.clip_limited
    assert res.top_slope >= 0.08
    assert bench.atten == res.final_atten_db
    clipped = [h for h in res.history if h.get("pa_clipped")]
    assert clipped, "the guard never tripped"
    first = res.history.index(clipped[0])
    assert all(h["atten_db"] > clipped[0]["atten_db"] for h in res.history[first + 1 :])


def test_clip_guard_off_lets_the_search_clip():
    res = _run(GainBench(pa=_hard_pa), target_compression_db=4.0, comp_tol_db=0.2)
    assert res.converged
    assert res.top_slope < 0.08
    assert not res.clip_limited


def test_find_compression_point_locks_on_gain_on_sim_bench():
    radio = SimRadio()
    x = make_signal()
    transmit_bands(radio, {TxChannel.TX1: x}, BITS)
    res = find_compression_point(
        radio,
        TxChannel.TX1,
        RxChannel.ORX1,
        reference_codes(x),
        rx_bits=BITS,
        fs=FS,
        target_compression_db=4.0,
        start_atten_db=20.0,
        atten_min_db=5.0,
        lock_on="gain",
        min_top_slope=0.08,
        coarse_step_db=5.0,
        fine_step_db=0.2,
        comp_tol_db=0.2,
    )
    assert res.converged
    assert res.lock_on == "gain"
    assert abs(res.gain_compression_db - 4.0) <= 0.2
    assert res.compression_db == res.gain_compression_db
    assert np.isfinite(res.papr_compression_db)
    assert radio.atten == res.final_atten_db


def test_find_compression_point_rejects_unknown_lock():
    with pytest.raises(ValueError, match="lock_on"):
        find_compression_point(
            SimRadio(),
            TxChannel.TX1,
            RxChannel.ORX1,
            reference_codes(make_signal()),
            rx_bits=BITS,
            fs=FS,
            target_compression_db=4.0,
            start_atten_db=20.0,
            atten_min_db=5.0,
            lock_on="rms",
        )
