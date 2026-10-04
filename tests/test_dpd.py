"""ILA DPD blocks: peak units, normalization, the peak limit, and the loop on two offline DUTs.

ACLR acceptance (both DUTs): the worst adjacent channel of the last iteration is
at least ``ACLR_GAIN_DB`` better than iteration 0, and no iteration is worse than
the one before by more than ``ACLR_SLACK_DB``. Both DUTs improve by about 30 dB
over the four iterations, so 20 dB leaves room without hiding a regression.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.dpd import (
    FULL_SCALE_DBM,
    IlaStep,
    iteration_table,
    limit_peak,
    normalize_pair,
    peak_dbm,
)
from adrvtrx.gmp import GMP, peak_block
from adrvtrx.linearize import linearize
from adrvtrx.metrics import aclr_db, nmse_db, papr_db
from sim_bench import BITS, BW, FS, SimRadio, make_signal
from test_linearize import _condition

FIXTURE = Path(__file__).parent / "data" / "tx1_2400mhz_100bw_excerpt.npz"
PEAK_LIMIT_DBM = 9.9
ACLR_GAIN_DB = 20.0
ACLR_SLACK_DB = 0.3
NMSE_GAIN_DB = 20.0


def _worst(lower_upper) -> float:
    return max(lower_upper)


def _check_aclr(worst: list[float]) -> None:
    assert worst[-1] <= worst[0] - ACLR_GAIN_DB, f"ACLR per iteration {worst}"
    for prev, cur in zip(worst, worst[1:]):
        assert cur <= prev + ACLR_SLACK_DB, f"ACLR got worse: {worst}"


# -- peak_dbm / normalize_pair --------------------------------------------------


def test_peak_dbm_is_10_dbm_at_full_scale():
    assert peak_dbm(np.array([0.3, -1.0j, 0.5])) == pytest.approx(FULL_SCALE_DBM)
    assert peak_dbm(np.array([0.5])) == pytest.approx(10.0 + 20 * math.log10(0.5))
    assert peak_dbm(np.zeros(4)) == -math.inf
    assert peak_dbm(np.array([])) == -math.inf
    assert math.isnan(peak_dbm(np.array([1.0, np.nan])))


def test_normalize_pair_removes_gain_and_phase():
    x = make_signal()
    y = 3.7 * np.exp(1j * 0.9) * x
    xn, yn = normalize_pair(x * 0.4, y)
    assert np.abs(xn).max() == pytest.approx(1.0)
    np.testing.assert_allclose(yn, xn, atol=1e-12)
    _, kept = normalize_pair(x, y, rotate=False)
    np.testing.assert_allclose(kept, np.exp(1j * 0.9) * x / np.abs(x).max(), atol=1e-12)


def test_normalize_pair_rejects_bad_input():
    with pytest.raises(ValueError):
        normalize_pair(np.ones(4), np.ones(5))
    with pytest.raises(ValueError):
        normalize_pair(np.ones(4), np.zeros(4))


# -- limit_peak -------------------------------------------------------------------


def _expander(v):
    """An expansive 'DPD': peak a * (1 + 0.5 a^2) for an input peak a."""
    return v * (1 + 0.5 * np.abs(v) ** 2)


def test_limit_peak_finds_the_smallest_margin():
    x = make_signal()
    res = limit_peak(_expander, x, peak_limit_dbm=PEAK_LIMIT_DBM)
    # exact margin: a (1 + a^2 / 2) = 10**((9.9 - 10) / 20)
    target = 10 ** ((PEAK_LIMIT_DBM - 10) / 20)
    a = np.roots([0.5, 0, 1, -target])
    a = float(a[np.isreal(a)].real[0])
    exact = -20 * math.log10(a)
    assert res.ok
    assert res.peak_dbm <= PEAK_LIMIT_DBM
    assert exact <= res.margin_db <= exact + 0.01 + 1e-9
    assert peak_dbm(res.waveform) == pytest.approx(res.peak_dbm)
    np.testing.assert_allclose(res.waveform, _expander(x * 10 ** (-res.margin_db / 20)))


def test_limit_peak_keeps_the_start_margin_when_it_is_enough():
    x = make_signal()
    res = limit_peak(lambda v: v, x, peak_limit_dbm=PEAK_LIMIT_DBM, start_margin_db=0.2)
    assert res.ok and res.margin_db == 0.2
    assert res.peak_dbm == pytest.approx(9.8)
    assert len(res.tries) == 1


def test_limit_peak_flags_an_unreachable_limit():
    x = make_signal()
    res = limit_peak(lambda v: 2.0 * v, x, peak_limit_dbm=PEAK_LIMIT_DBM, max_margin_db=4.0)
    assert not res.ok
    assert res.margin_db == 4.0
    assert res.peak_dbm == pytest.approx(10 + 20 * math.log10(2) - 4)
    nan_res = limit_peak(lambda v: v * np.nan, x)
    assert not nan_res.ok


def test_limit_peak_does_not_assume_monotonic_peaks():
    """Bad below 0.7 dB, good in [0.7, 0.9), bad again in [0.9, 2.0), good above."""

    def dpd(v):
        m = -20 * math.log10(np.abs(v).max())
        ok = 0.7 <= m + 1e-12 < 0.9 or m >= 2.0
        return v / np.abs(v).max() * 10 ** (((9.5 if ok else 11.0) - 10) / 20)

    res = limit_peak(dpd, make_signal(), peak_limit_dbm=PEAK_LIMIT_DBM, tol_db=0.01)
    assert res.ok
    assert 0.7 - 1e-9 <= res.margin_db <= 0.71
    assert res.peak_dbm <= PEAK_LIMIT_DBM


def test_limit_peak_rejects_bad_settings():
    with pytest.raises(ValueError):
        limit_peak(lambda v: v, make_signal(), start_margin_db=5.0, max_margin_db=4.0)
    with pytest.raises(ValueError):
        limit_peak(lambda v: v, make_signal(), tol_db=0.0)


# -- IlaStep ----------------------------------------------------------------------


class _Spy:
    """A model that records what it was fitted on and predicts ``gain * input``."""

    seen: dict = {}

    def __init__(self, gain=1.0):
        self.gain = gain

    def fit(self, u, target, block):
        _Spy.seen = {"u": np.array(u), "target": np.array(target), "block": block}
        return self

    def predict(self, u):
        return self.gain * np.asarray(u)


def test_step_fits_normalized_z_to_the_transmitted_u_as_is():
    x = make_signal()
    u = 0.5 * x  # transmitted 6 dB below full scale
    z = 123.0 * np.exp(-1j * 0.4) * np.tanh(np.abs(u)) * np.exp(1j * np.angle(u))
    step = IlaStep(lambda: _Spy(0.9), n_train=1024)
    nxt = step(x, u, z, 0)
    seen = _Spy.seen
    np.testing.assert_allclose(seen["target"], u)  # absolute DAC units
    assert np.abs(seen["u"]).max() == pytest.approx(1.0)
    assert abs(np.angle(np.vdot(x, seen["u"]))) < 1e-9
    assert seen["block"] == peak_block(seen["u"], 1024)
    # 0.9 * x at the 0.2 dB start margin
    np.testing.assert_allclose(nxt, 0.9 * x / np.abs(x).max() * 10 ** (-0.2 / 20))
    (row,) = step.history
    assert row["iteration"] == 1 and row["ok"]
    assert row["margin_db"] == 0.2
    assert row["dpd_peak_dbm"] == pytest.approx(peak_dbm(nxt))
    assert row["papr_expansion_db"] == pytest.approx(0.0, abs=1e-9)


def test_step_returns_none_when_the_limit_cannot_be_met():
    x = make_signal()
    step = IlaStep(lambda: _Spy(3.0), max_margin_db=4.0)
    assert step(x, x, x, 0) is None
    assert not step.history[-1]["ok"]
    assert "limit" in step.reason


def _sim_loop(n_iter=4, seed=1):
    radio = SimRadio(seed=seed)
    step = IlaStep(lambda: GMP(5, 5, 2), n_train=8192, peak_limit_dbm=PEAK_LIMIT_DBM)
    res = linearize(
        radio,
        _condition(),
        make_signal(),
        step,
        tx=TxChannel.TX1,
        orx=RxChannel.ORX1,
        tx_bits=BITS,
        rx_bits=BITS,
        fs=FS,
        n_iter=n_iter,
    )
    return radio, step, res


def test_ila_gmp_linearizes_the_sim_bench():
    assert _condition().bw_mhz * 1e6 == BW  # ACLR channels of the sim signal
    radio, step, res = _sim_loop()
    assert res.reason == "n_iter" and len(res.records) == 4
    assert len(step.history) == 3
    nmse = [r.nmse_db for r in res.records]
    worst = [max(r.aclr_lower_dbc, r.aclr_upper_dbc) for r in res.records]
    assert nmse[-1] <= nmse[0] - NMSE_GAIN_DB, nmse
    _check_aclr(worst)
    for h in step.history:
        assert h["ok"] and h["dpd_peak_dbm"] <= PEAK_LIMIT_DBM
    for rec in res.records:
        assert rec.tx_clipped == 0 and rec.railed == 0
    assert not radio.tx_on

    table = iteration_table(res.records, step.history)
    assert [row["iteration"] for row in table] == [0, 1, 2, 3]
    assert math.isnan(table[0]["margin_db"]) and math.isnan(table[0]["dpd_peak_dbm"])
    assert table[2]["dpd_peak_dbm"] == step.history[1]["dpd_peak_dbm"]
    assert [row["aclr_worst_dbc"] for row in table] == worst
    assert table[1]["papr_expansion_db"] == res.records[1].papr_expansion_db


def test_ila_stops_before_sending_a_waveform_over_the_limit():
    radio = SimRadio()
    step = IlaStep(lambda: _Spy(3.0))
    res = linearize(
        radio,
        _condition(),
        make_signal(),
        step,
        tx=TxChannel.TX1,
        orx=RxChannel.ORX1,
        tx_bits=BITS,
        rx_bits=BITS,
        fs=FS,
        n_iter=4,
    )
    assert res.reason == "step returned None"
    assert len(res.records) == 1 and len(radio.transmitted) == 1
    assert "limit" in step.reason
    assert not radio.tx_on


# -- real-signal fixture ------------------------------------------------------------


def _fixture():
    data = np.load(FIXTURE)
    return (
        data["x"].astype(np.complex128),
        data["y"].astype(np.complex128),
        float(data["fs_hz"]),
        float(data["bw_hz"]),
    )


def test_fixture_is_small_and_peaks_at_full_scale():
    x, y, fs, bw = _fixture()
    assert FIXTURE.stat().st_size < 2_000_000
    assert len(x) == len(y) == 65536
    assert np.abs(x).max() == pytest.approx(1.0, abs=1e-3)
    assert (fs, bw) == (491.52e6, 100e6)
    assert nmse_db(x, y) == pytest.approx(-17.6, abs=0.3)


def test_ila_gmp_linearizes_a_forward_model_of_the_real_pa():
    """DUT: a GMP fitted x -> y on the 2.4 GHz / 100 MHz excerpt, input clamped at its peak."""
    x, y, fs, bw = _fixture()
    xn, yn = normalize_pair(x, y)
    pa = GMP(5, 5, 2).fit(xn, yn)

    def dut(u):
        a = np.abs(u)
        return pa.predict(np.where(a > 1.0, u / np.maximum(a, 1e-30), u))

    step = IlaStep(lambda: GMP(5, 5, 2), n_train=8192, peak_limit_dbm=PEAK_LIMIT_DBM)
    u = xn
    nmse, worst, peaks = [], [], [peak_dbm(u)]
    for it in range(4):
        z = dut(u)
        nmse.append(nmse_db(xn, z))
        worst.append(_worst(aclr_db(z, fs, bw)))
        if it == 3:
            break
        u = step(xn, u, z, it)
        assert u is not None, step.reason
        peaks.append(peak_dbm(u))

    assert nmse[0] == pytest.approx(-17.6, abs=0.5)
    assert nmse[-1] <= nmse[0] - NMSE_GAIN_DB, nmse
    _check_aclr(worst)
    assert all(p <= PEAK_LIMIT_DBM for p in peaks[1:]), peaks
    assert [h["dpd_peak_dbm"] for h in step.history] == peaks[1:]
    assert all(h["papr_expansion_db"] > 0 for h in step.history)
    assert papr_db(u) > papr_db(xn)
