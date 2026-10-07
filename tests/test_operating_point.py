from __future__ import annotations

import numpy as np
import pytest

import adrvtrx.operating_point as op_mod
from adrvtrx import RxChannel, TxChannel
from adrvtrx.conditions import ConditionLog, load_conditions
from adrvtrx.operating_point import find_operating_point
from sim_bench import BITS, BW, FS, SimRadio, make_signal, reference_codes

FREQ = 2_000_000_000
BW_MHZ = int(BW / 1e6)


def _find(radio, signal, **kw):
    args = {
        "tx_bits": BITS,
        "rx_bits": BITS,
        "fs": FS,
        "freq_hz": FREQ,
        "bw_mhz": BW_MHZ,
        "target_compression_db": 3.0,
        "start_atten_db": 20.0,
        "atten_min_db": 5.0,
        "signal_name": "sig.txt",
        "coarse_step_db": 5.0,
        "fine_step_db": 0.2,
        "comp_tol_db": 0.1,
    }
    args.update(kw)
    return find_operating_point(radio, TxChannel.TX1, RxChannel.ORX1, signal, **args)


def test_lock_without_backoff():
    radio = SimRadio()
    x = make_signal()
    op = _find(radio, 2.5 * x)
    c = op.condition
    assert op.search.converged
    assert abs(c.compression_db - 3.0) <= 0.1
    assert c.name == f"TX1_0dB_2000MHz_{BW_MHZ}BW"
    assert c.backoff_db == 0 and c.tx_atten_db == c.locked_atten_db == op.search.final_atten_db
    assert c.orx_gain == op.search.final_orx_gain
    assert op.agc is None
    assert c.lock_on == "papr" and c.converged
    assert c.signal == "sig.txt" and c.in_file == "sig_in.txt"
    assert c.out_file == f"{c.name}_out.txt"
    assert c.railed == 0 and np.isfinite(c.nmse_db) and c.corr > 0.99
    np.testing.assert_array_equal(op.ref, reference_codes(x))
    np.testing.assert_allclose(op.x, op.ref / 2047)
    assert len(op.point.y_aligned) == len(x)
    assert radio.retunes == [("LO1", FREQ)]
    assert radio.atten == c.tx_atten_db and radio.gain == c.orx_gain
    assert not radio.tx_on
    assert op.csv_path is None


def test_backoff_runs_the_agc_at_lock_plus_backoff():
    radio = SimRadio()
    op = _find(radio, make_signal(), backoff_db=3)
    c = op.condition
    assert c.name == f"TX1_3dB_2000MHz_{BW_MHZ}BW"
    assert c.tx_atten_db == pytest.approx(c.locked_atten_db + 3.0)
    assert op.agc is not None and c.orx_gain == op.agc.final_gain_index
    assert c.orx_gain > op.search.final_orx_gain  # less signal, more ORx gain
    assert c.papr_compression_db < op.search.compression_db
    assert c.railed == 0
    assert radio.atten == c.tx_atten_db
    assert not radio.tx_on


def test_gain_lock_with_clip_guard():
    op = _find(
        SimRadio(),
        make_signal(),
        lock_on="gain",
        target_compression_db=4.0,
        comp_tol_db=0.2,
        min_top_slope=0.08,
    )
    assert op.condition.lock_on == "gain"
    assert abs(op.condition.compression_db - 4.0) <= 0.2
    assert op.search.compression_db == op.search.gain_compression_db


def test_saves_files_and_appends_rows(tmp_path):
    radio = SimRadio()
    first = _find(radio, make_signal(), save_dir=tmp_path)
    second = _find(radio, make_signal(), save_dir=tmp_path, backoff_db=2)
    csv_path = tmp_path / "TX1_conditions.csv"
    assert first.csv_path == second.csv_path == csv_path
    rows = load_conditions(csv_path)
    assert [r.name for r in rows] == [first.condition.name, second.condition.name]
    assert rows[1].orx_gain == second.condition.orx_gain
    assert rows[0].tx_atten_db == pytest.approx(first.condition.tx_atten_db)
    for r in rows:
        assert (tmp_path / r.in_file).is_file() and (tmp_path / r.out_file).is_file()


def test_append_refuses_a_csv_with_other_columns(tmp_path):
    path = tmp_path / "TX1_conditions.csv"
    path.write_text("name,signal\nA,b\n")
    with pytest.raises(ValueError, match="columns"):
        ConditionLog(path, append=True)
    with ConditionLog(path) as log:  # without append the file is replaced
        assert log.fields[0] == "name"
    assert path.read_text().startswith("name,signal,freq_hz")


def test_tx_disabled_when_the_search_fails(monkeypatch):
    radio = SimRadio()

    def boom(*a, **k):
        assert radio.tx_on
        raise RuntimeError("search failed")

    monkeypatch.setattr(op_mod, "find_compression_point", boom)
    with pytest.raises(RuntimeError):
        _find(radio, make_signal())
    assert not radio.tx_on


def test_negative_backoff_rejected():
    with pytest.raises(ValueError):
        _find(SimRadio(), make_signal(), backoff_db=-1)
