"""ILA DPD blocks: peak units, normalization, the peak limit, and the loop on two offline DUTs.

ACLR acceptance (both DUTs): the worst adjacent channel of the last iteration is
at least ``ACLR_GAIN_DB`` better than iteration 0, and no iteration is worse than
the one before by more than ``ACLR_SLACK_DB``. Both DUTs improve by about 30 dB,
so 20 dB leaves room without hiding a regression.

Output level: the step anchors every capture on the iteration-0 output peak, so
the output peak must stay within ``OUT_TOL_DB`` of ``-target_backoff_db`` on every
DPD iteration.

The sim bench PA is a memoryless Rapp, so its loop tests use a memoryless
GMP(9, 1, 0), which inverts it well enough to hit the target and hold ACLR. The
default GMP(5, 5, 2) cannot invert the hard Rapp top completely: on the sim its
output stays flat but about 0.1 dB short of the target, and at that fixed target
its worst ACLR is best after the second pass and then gets worse (-60.7 then
-60.3 dBc; noise-free 0.6-2 dB per pass after the third). That run is checked
separately for what it does hold (``test_gmp552_on_the_sim_bench``) and the drift
is reported in ``docs/dpd_pass.md``; the real-PA fixture uses GMP(5, 5, 2) with
the full ACLR rule. With the DPD output band-limited to the ACLR span
(``tx_cutoff = 1.5 * BW / FS``) GMP(5, 5, 2) holds its ACLR on the sim
(``test_tx_cutoff_holds_gmp552_on_the_sim_bench``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.dpd import (
    FULL_SCALE_DBM,
    TARGET_BACKOFF_DB,
    IlaStep,
    band_limit,
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
OUT_TOL_DB = 0.05


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
    step = IlaStep(lambda: _Spy(0.9), n_train=1024, target_backoff_db=0.2)
    nxt = step(x, u, z, 0)
    seen = _Spy.seen
    np.testing.assert_allclose(seen["target"], u)  # absolute DAC units
    assert np.abs(seen["u"]).max() == pytest.approx(1.0)
    assert abs(np.angle(np.vdot(x, seen["u"]))) < 1e-9
    assert seen["block"] == peak_block(seen["u"], 1024)
    # 0.9 * x at the 0.2 dB target backoff
    np.testing.assert_allclose(nxt, 0.9 * x / np.abs(x).max() * 10 ** (-0.2 / 20))
    (row,) = step.history
    assert row["iteration"] == 1 and row["ok"]
    assert row["target_backoff_db"] == 0.2 and row["guard_db"] == 0.0
    assert row["margin_db"] == 0.2
    assert row["z_peak_db"] == 0.0
    assert row["dpd_peak_dbm"] == pytest.approx(peak_dbm(nxt))
    assert row["papr_expansion_db"] == pytest.approx(0.0, abs=1e-9)
    assert step.anchor_peak == pytest.approx(np.abs(z).max())


def test_step_anchors_every_capture_on_the_first():
    x = make_signal()
    z0 = 50.0 * x
    step = IlaStep(lambda: _Spy(0.5), target_backoff_db=0.1)
    step(x, x, z0, 0)
    step(x, x, 0.8 * np.exp(1j * 2.0) * z0, 1)  # output 1.94 dB lower, rotated
    seen = _Spy.seen["u"]
    assert np.abs(seen).max() == pytest.approx(0.8)  # divided by the first peak
    assert abs(np.angle(np.vdot(x, seen))) < 1e-9  # still rotated onto x
    assert step.history[1]["z_peak_db"] == pytest.approx(20 * math.log10(0.8))
    assert step.anchor_peak == pytest.approx(np.abs(z0).max())

    each = IlaStep(lambda: _Spy(0.5), target_backoff_db=0.1, anchor="each")
    each(x, x, z0, 0)
    each(x, x, 0.8 * z0, 1)
    assert np.abs(_Spy.seen["u"]).max() == pytest.approx(1.0)  # its own peak
    assert each.history[1]["z_peak_db"] == pytest.approx(20 * math.log10(0.8))

    step(x, x, 0.8 * z0, 0)  # iteration 0 starts a new run
    assert len(step.history) == 1
    assert step.anchor_peak == pytest.approx(0.8 * np.abs(z0).max())


def test_step_guard_adds_only_what_the_limit_needs():
    x = make_signal()
    # the identity post-inverse: DPD peak = 10 dBm - backoff
    tight = IlaStep(lambda: _Spy(1.0), target_backoff_db=0.0, peak_limit_dbm=PEAK_LIMIT_DBM)
    tight(x, x, x, 0)
    row = tight.history[0]
    assert row["ok"] and row["target_backoff_db"] == 0.0
    assert 0.1 - 1e-9 <= row["guard_db"] <= 0.11 + 1e-9
    assert row["margin_db"] == pytest.approx(row["target_backoff_db"] + row["guard_db"])
    assert row["dpd_peak_dbm"] <= PEAK_LIMIT_DBM

    loose = IlaStep(lambda: _Spy(1.0), target_backoff_db=0.3, peak_limit_dbm=PEAK_LIMIT_DBM)
    loose(x, x, x, 0)
    assert loose.history[0]["guard_db"] == 0.0
    assert loose.history[0]["dpd_peak_dbm"] == pytest.approx(9.7)


def test_step_rejects_bad_settings():
    with pytest.raises(ValueError, match="anchor"):
        IlaStep(lambda: _Spy(), anchor="last")
    with pytest.raises(ValueError, match="target_backoff_db"):
        IlaStep(lambda: _Spy(), target_backoff_db=5.0, max_margin_db=4.0)
    with pytest.raises(ValueError, match="target_backoff_db"):
        IlaStep(lambda: _Spy(), target_backoff_db=-0.1)
    with pytest.raises(ValueError):
        IlaStep(lambda: _Spy())(make_signal(), make_signal(), np.zeros(4096), 0)


def test_default_target_backoff():
    assert IlaStep(lambda: _Spy()).target_backoff_db == TARGET_BACKOFF_DB == 0.15


def test_step_returns_none_when_the_limit_cannot_be_met():
    x = make_signal()
    step = IlaStep(lambda: _Spy(3.0), max_margin_db=4.0)
    assert step(x, x, x, 0) is None
    assert not step.history[-1]["ok"]
    assert "limit" in step.reason


SIM_MODEL = (9, 1, 0)  # the sim PA is memoryless


def _sim_loop(spec=SIM_MODEL, n_iter=5, seed=1, **step_kw):
    radio = SimRadio(seed=seed)
    step = IlaStep(lambda: GMP(*spec), n_train=8192, **step_kw)
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
    return radio, step, res, iteration_table(res.records, step.history)


def _check_loop(table, step, limit=PEAK_LIMIT_DBM) -> None:
    nmse = [row["nmse_db"] for row in table]
    assert nmse[-1] <= nmse[0] - NMSE_GAIN_DB, nmse
    _check_aclr([row["aclr_worst_dbc"] for row in table])
    for h in step.history:
        assert h["ok"] and h["dpd_peak_dbm"] <= limit
    for row in table:
        assert row["tx_clipped"] == 0 and row["railed"] == 0


def test_ila_linearizes_the_sim_bench_and_holds_the_output():
    assert _condition().bw_mhz * 1e6 == BW  # ACLR channels of the sim signal
    radio, step, res, table = _sim_loop()
    assert res.reason == "n_iter" and len(res.records) == 5
    assert len(step.history) == 4
    _check_loop(table, step)
    assert not radio.tx_on
    for row in table[1:]:
        assert abs(row["output_peak_db"] + TARGET_BACKOFF_DB) <= OUT_TOL_DB, table
        assert row["guard_db"] == 0.0

    assert [row["iteration"] for row in table] == [0, 1, 2, 3, 4]
    assert table[0]["output_peak_db"] == 0.0
    assert math.isnan(table[0]["margin_db"]) and math.isnan(table[0]["dpd_peak_dbm"])
    assert table[2]["dpd_peak_dbm"] == step.history[1]["dpd_peak_dbm"]
    assert table[2]["target_backoff_db"] == TARGET_BACKOFF_DB
    assert table[1]["papr_expansion_db"] == res.records[1].papr_expansion_db
    rec = res.records[2]
    assert table[2]["pa_papr_compression_db"] == rec.papr_dpd_db - rec.papr_out_db
    assert table[0]["papr_compression_db"] > 3.0 > table[3]["papr_compression_db"]


@pytest.mark.parametrize("target", [0.0, 0.1, 0.2])
def test_output_peak_holds_the_target_on_the_sim_bench(target):
    _, step, _, table = _sim_loop(target_backoff_db=target)
    _check_loop(table, step)
    for row, h in zip(table[1:], step.history):
        # a guard that engages lowers the target by what it added
        assert abs(row["output_peak_db"] + h["margin_db"]) <= OUT_TOL_DB, table
        assert h["guard_db"] <= 0.05


def test_peak_guard_engages_when_the_limit_is_tight():
    limit = 8.5
    _, step, _, table = _sim_loop(peak_limit_dbm=limit)
    _check_loop(table, step, limit=limit)
    for h in step.history:
        assert h["guard_db"] > 0.1
        assert h["margin_db"] == pytest.approx(TARGET_BACKOFF_DB + h["guard_db"])
        assert limit - 0.05 <= h["dpd_peak_dbm"] <= limit
    out = [row["output_peak_db"] for row in table[1:]]
    assert max(out) - min(out) <= OUT_TOL_DB, out  # the guard does not accumulate
    assert max(out) < -TARGET_BACKOFF_DB


def test_gmp552_on_the_sim_bench():
    """The default model on the hard Rapp top: big ACLR gain, flat output, about 0.1 dB short.

    Its ACLR after the second pass drifts (see the module docstring), so the
    per-iteration ACLR rule is not asserted here.
    """
    _, step, _, table = _sim_loop(spec=(5, 5, 2), n_iter=4)
    worst = [row["aclr_worst_dbc"] for row in table]
    assert min(worst[1:]) <= worst[0] - 25.0, worst
    assert worst[-1] <= worst[0] - ACLR_GAIN_DB, worst
    assert all(h["ok"] and h["dpd_peak_dbm"] <= PEAK_LIMIT_DBM for h in step.history)
    out = [row["output_peak_db"] for row in table[1:]]
    assert max(out) - min(out) <= OUT_TOL_DB, out
    assert all(-TARGET_BACKOFF_DB - 0.15 <= o <= -TARGET_BACKOFF_DB for o in out), out


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


def _fixture_loop(n_iter=5, **step_kw):
    """DUT: a GMP fitted x -> y on the 2.4 GHz / 100 MHz excerpt, input clamped at its peak."""
    x, y, fs, bw = _fixture()
    xn, yn = normalize_pair(x, y)
    pa = GMP(5, 5, 2).fit(xn, yn)

    def dut(u):
        a = np.abs(u)
        return pa.predict(np.where(a > 1.0, u / np.maximum(a, 1e-30), u))

    step = IlaStep(lambda: GMP(5, 5, 2), n_train=8192, **step_kw)
    u = xn
    rows, peaks = [], [peak_dbm(u)]
    z0_peak = None
    for it in range(n_iter):
        z = dut(u)
        z0_peak = np.abs(z).max() if z0_peak is None else z0_peak
        rows.append(
            {
                "nmse_db": nmse_db(xn, z),
                "aclr_worst_dbc": _worst(aclr_db(z, fs, bw)),
                "output_peak_db": 20 * math.log10(np.abs(z).max() / z0_peak),
            }
        )
        if it == n_iter - 1:
            break
        u = step(xn, u, z, it)
        assert u is not None, step.reason
        peaks.append(peak_dbm(u))
    return xn, u, step, rows, peaks


def test_ila_gmp_linearizes_a_forward_model_of_the_real_pa():
    xn, u, step, rows, peaks = _fixture_loop()
    nmse = [r["nmse_db"] for r in rows]
    assert nmse[0] == pytest.approx(-17.6, abs=0.5)
    assert nmse[-1] <= nmse[0] - NMSE_GAIN_DB, nmse
    _check_aclr([r["aclr_worst_dbc"] for r in rows])
    assert all(p <= PEAK_LIMIT_DBM for p in peaks[1:]), peaks
    assert [h["dpd_peak_dbm"] for h in step.history] == peaks[1:]
    for r in rows[1:]:
        assert abs(r["output_peak_db"] + TARGET_BACKOFF_DB) <= OUT_TOL_DB, rows
    for h in step.history:
        assert h["guard_db"] == 0.0 and h["papr_expansion_db"] > 2.0
    assert papr_db(u) > papr_db(xn)


def test_peak_guard_on_the_real_pa_model():
    limit = 8.5
    _, _, step, rows, peaks = _fixture_loop(n_iter=4, peak_limit_dbm=limit)
    _check_aclr([r["aclr_worst_dbc"] for r in rows])
    assert all(limit - 0.05 <= p <= limit for p in peaks[1:]), peaks
    assert all(h["guard_db"] > 0.1 for h in step.history), step.history
    out = [r["output_peak_db"] for r in rows[1:]]
    assert max(out) - min(out) <= OUT_TOL_DB, out


# -- band limit (tx_cutoff) ---------------------------------------------------------


def test_band_limit_keeps_the_band_and_zeroes_the_rest():
    n = 1024
    t = np.arange(n)
    inside = np.exp(2j * np.pi * 10 / n * t)
    outside = 0.3 * np.exp(-2j * np.pi * 300 / n * t)
    np.testing.assert_allclose(band_limit(inside + outside, 0.1), inside, atol=1e-12)
    for bad in (0.0, -0.1, 0.6, float("nan")):
        with pytest.raises(ValueError):
            band_limit(inside, bad)


class _Cubic(_Spy):
    """A post-inverse whose output is wider than its input (a cubic term)."""

    def predict(self, u):
        u = np.asarray(u)
        return self.gain * (u + 0.3 * u * np.abs(u) ** 2)


def test_step_tx_cutoff_band_limits_the_waveform():
    x = make_signal()  # occupies +-0.45 BW; the cubic spreads it to +-1.35 BW
    z = np.tanh(np.abs(x)) * np.exp(1j * np.angle(x))
    cut = 0.75 * BW / FS
    f = np.abs(np.fft.fftfreq(len(x)))
    wide = IlaStep(lambda: _Cubic(0.7), n_train=1024)(x, x, z, 0)
    spec = np.abs(np.fft.fft(wide)) ** 2
    assert spec[f > cut].sum() > 1e-6 * spec.sum()
    step = IlaStep(lambda: _Cubic(0.7), n_train=1024, tx_cutoff=cut)
    nxt = step(x, x, z, 0)
    spec = np.abs(np.fft.fft(nxt)) ** 2
    assert spec[f > cut].sum() <= 1e-20 * spec.sum()
    (row,) = step.history
    assert row["ok"] and row["dpd_peak_dbm"] == pytest.approx(peak_dbm(nxt))
    assert row["dpd_peak_dbm"] <= PEAK_LIMIT_DBM
    for bad in (0.0, 0.7):
        with pytest.raises(ValueError):
            IlaStep(lambda: _Cubic(), tx_cutoff=bad)


def test_tx_cutoff_holds_gmp552_on_the_sim_bench():
    """GMP(5, 5, 2) drifts after its best pass on the sim (module docstring).

    Band-limited to the ACLR span it holds."""
    _, _, _, plain = _sim_loop(spec=(5, 5, 2), n_iter=8)
    _, step, _, table = _sim_loop(spec=(5, 5, 2), n_iter=8, tx_cutoff=1.5 * BW / FS)
    w0 = [row["aclr_worst_dbc"] for row in plain]
    w = [row["aclr_worst_dbc"] for row in table]
    assert w0[-1] - min(w0[1:]) > 1.0, w0  # the drift this option is for
    assert w[-1] - min(w[1:]) <= ACLR_SLACK_DB, w
    assert w[-1] < w0[-1], (w, w0)
    assert min(w[1:]) <= w[0] - 25.0, w
    assert all(h["ok"] and h["dpd_peak_dbm"] <= PEAK_LIMIT_DBM for h in step.history)
