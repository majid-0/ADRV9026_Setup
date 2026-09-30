from __future__ import annotations

import csv

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.conditions import OperatingCondition
from adrvtrx.linearize import linearize
from sim_bench import BITS, BW, FS, SimRadio, drive_for, inverse_rapp, make_signal

TX, ORX = TxChannel.TX1, RxChannel.ORX1
ATTEN = 12.0


def _condition(orx_gain=213) -> OperatingCondition:
    nan = float("nan")
    return OperatingCondition(
        name="TX1_0dB_2000MHz_2BW",
        signal="sig.txt",
        freq_hz=2_000_000_000,
        bw_mhz=int(BW / 1e6),
        backoff_db=0,
        locked_atten_db=ATTEN,
        tx_atten_db=ATTEN,
        orx_gain=orx_gain,
        compression_db=3.0,
        converged=True,
        delay_samples=nan,
        delay_ns=nan,
        corr=nan,
        nmse_db=nan,
        papr_in_db=nan,
        papr_out_db=nan,
        papr_compression_db=nan,
        aclr_lower_dbc=nan,
        aclr_upper_dbc=nan,
        peak_dbfs=nan,
        rms_dbfs=nan,
        railed=0,
        in_file="sig_in.txt",
        out_file="out.txt",
    )


def ideal_step(x, u, z, it):
    """Exact inverse of the simulated PA: the output peak lands at 0.9 of saturation."""
    return inverse_rapp(0.9 * x) / drive_for(ATTEN)


def _run(radio, x, step, **kw):
    return linearize(
        radio,
        _condition(**kw.pop("cond", {})),
        x,
        step,
        tx=TX,
        orx=ORX,
        tx_bits=BITS,
        rx_bits=BITS,
        fs=FS,
        **kw,
    )


def test_ideal_step_linearizes():
    radio = SimRadio()
    x = make_signal()
    res = _run(radio, x, ideal_step, n_iter=2)
    assert res.reason == "n_iter"
    assert len(res.records) == 2
    before, after = res.records
    assert after.nmse_db < before.nmse_db - 10
    assert after.aclr_upper_dbc < before.aclr_upper_dbc - 10
    assert abs(after.papr_compression_db) < 0.5 < before.papr_compression_db
    assert after.papr_expansion_db > 1.0
    assert radio.retunes == [("LO1", 2_000_000_000)]
    assert radio.gain_sets == [213]
    assert radio.atten == ATTEN
    assert not radio.tx_on


def test_step_sees_aligned_normalized_signals():
    radio = SimRadio()
    x = make_signal()
    seen = {}

    def step(x_in, u, z, it):
        seen.update(x=x_in, u=u, z=z, it=it)
        return None

    res = _run(radio, x, step, n_iter=3)
    assert res.reason == "step returned None"
    assert len(res.records) == 1
    assert seen["it"] == 0
    np.testing.assert_allclose(seen["x"], x)
    assert np.max(np.abs(seen["u"])) == pytest.approx(1.0, abs=1e-3)
    assert len(seen["z"]) == len(x)
    assert np.max(np.abs(seen["z"])) <= 1.0
    radio.atten = ATTEN
    truth = radio.pa_period(seen["u"] * 2047)
    g = np.vdot(truth, seen["z"]) / np.vdot(truth, truth)
    assert np.linalg.norm(seen["z"] - g * truth) / np.linalg.norm(seen["z"]) < 0.02


def test_stops_on_rail():
    radio = SimRadio()
    res = _run(radio, make_signal(), ideal_step, n_iter=4, cond={"orx_gain": 255})
    assert res.reason == "railed"
    assert len(res.records) == 1 and res.records[0].railed > 0


def test_rail_ignored_when_disabled():
    radio = SimRadio()
    res = _run(
        radio, make_signal(), ideal_step, n_iter=2, cond={"orx_gain": 255}, stop_on_rail=False
    )
    assert res.reason == "n_iter" and len(res.records) == 2


def test_saves_files_and_csv(tmp_path):
    radio = SimRadio()
    res = _run(radio, make_signal(), ideal_step, n_iter=2, save_dir=tmp_path, label="ila")
    with open(tmp_path / "ila.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["iteration"] for r in rows] == ["0", "1"]
    for rec in res.records:
        assert (tmp_path / rec.in_file).is_file()
        assert (tmp_path / rec.out_file).is_file()


def test_tx_disabled_when_step_raises():
    radio = SimRadio()

    def step(*_):
        raise RuntimeError("model blew up")

    with pytest.raises(RuntimeError):
        _run(radio, make_signal(), step, n_iter=3)
    assert not radio.tx_on


def test_wrong_length_from_step_raises():
    radio = SimRadio()
    with pytest.raises(ValueError, match="samples"):
        _run(radio, make_signal(), lambda *a: np.ones(10), n_iter=3)
    assert not radio.tx_on


def test_n_iter_must_be_positive():
    with pytest.raises(ValueError):
        _run(SimRadio(), make_signal(), ideal_step, n_iter=0)
