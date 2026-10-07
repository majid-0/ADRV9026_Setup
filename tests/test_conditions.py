from __future__ import annotations

import math

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.conditions import (
    CSV_FIELDS,
    DUT_FIELDS,
    ConditionLog,
    DutRecord,
    OperatingCondition,
    capture_point,
    condition_keys,
    condition_name,
    load_aligned_iq,
    load_conditions,
    load_dut_records,
    save_aligned_iq,
)
from adrvtrx.metrics import nmse_db
from adrvtrx.transmit import transmit_bands
from sim_bench import BITS, BW, DELAY, FS, SimRadio, make_signal, reference_codes

LEGACY = (
    "name,signal,freq_hz,bw_mhz,backoff_db,locked_atten_db,tx_atten_db,orx_gain,"
    "compression_db,converged,delay_samples,delay_ns,corr,nmse_db,papr_in_db,papr_out_db,"
    "papr_compression_db,aclr_lower_dbc,aclr_upper_dbc,peak_dbfs,rms_dbfs,railed,in_file,"
    "out_file\n"
    "TX1_0dB_2000MHz_40BW,new40MHz256QAM_CFRed.txt,2000000000,40,0,13.40,13.40,199,3.03,"
    "true,218016.50,443555.7,0.9972,-20.54,8.91,5.89,3.02,-29.09,-28.41,-0.88,-6.58,0,"
    "new40MHz256QAM_CFRed_in.txt,TX1_0dB_2000MHz_40BW_out.txt\n"
)


def _condition(**kw) -> OperatingCondition:
    base = {
        "name": "TX1_0dB_2000MHz_40BW",
        "signal": "sig.txt",
        "freq_hz": 2_000_000_000,
        "bw_mhz": 40,
        "backoff_db": 0,
        "locked_atten_db": 13.4,
        "tx_atten_db": 13.4,
        "orx_gain": 199,
        "compression_db": 3.03,
        "converged": True,
        "delay_samples": 218016.5,
        "delay_ns": 443555.7,
        "corr": 0.99721,
        "nmse_db": -20.54,
        "papr_in_db": 8.91,
        "papr_out_db": 5.89,
        "papr_compression_db": 3.02,
        "aclr_lower_dbc": -29.09,
        "aclr_upper_dbc": -28.41,
        "peak_dbfs": -0.88,
        "rms_dbfs": -6.58,
        "railed": 0,
        "in_file": "sig_in.txt",
        "out_file": "TX1_0dB_2000MHz_40BW_out.txt",
    }
    base.update(kw)
    return OperatingCondition(**base)


NEW_COLUMNS = ("lock_on", "gain_compression_db", "amam_top_slope", "pa_clipped")


def test_capture_columns_keep_the_legacy_order_and_add_the_new_ones_last():
    assert CSV_FIELDS == tuple(LEGACY.splitlines()[0].split(",")) + NEW_COLUMNS


def test_dut_columns_extend_capture_columns():
    extra = set(DUT_FIELDS) - set(CSV_FIELDS)
    assert extra == {"papr_dpd_db", "papr_expansion_db", "tx_peak_dbfs", "tx_clipped", "ref_file"}
    assert set(CSV_FIELDS) <= set(DUT_FIELDS)


def test_legacy_csv_loads(tmp_path):
    path = tmp_path / "TX1_conditions.csv"
    path.write_text(LEGACY)
    (row,) = load_conditions(path)
    assert row == _condition(
        signal="new40MHz256QAM_CFRed.txt",
        corr=0.9972,
        in_file="new40MHz256QAM_CFRed_in.txt",
    )


def test_legacy_csv_gets_defaults_for_the_new_columns(tmp_path):
    path = tmp_path / "TX1_conditions.csv"
    path.write_text(LEGACY)
    (row,) = load_conditions(path)
    assert row.lock_on == "papr"
    assert math.isnan(row.gain_compression_db) and math.isnan(row.amam_top_slope)
    assert row.pa_clipped is False


def test_new_columns_round_trip(tmp_path):
    cond = _condition(
        lock_on="gain", gain_compression_db=4.02, amam_top_slope=0.1534, pa_clipped=True
    )
    with ConditionLog(tmp_path / "c.csv") as log:
        log.append(cond)
    text = (tmp_path / "c.csv").read_text().splitlines()[1]
    assert text.endswith(",gain,4.02,0.153,true")
    (back,) = load_conditions(tmp_path / "c.csv")
    assert back.lock_on == "gain" and back.pa_clipped is True
    assert back.gain_compression_db == 4.02 and back.amam_top_slope == 0.153


def test_condition_keys_carry_the_lock_metric():
    keys = condition_keys(_condition(lock_on="gain"))
    assert keys["lock_on"] == "gain"
    assert "gain_compression_db" not in keys  # measured per capture, not a condition


def test_round_trip(tmp_path):
    cond = _condition()
    with ConditionLog(tmp_path / "c.csv") as log:
        log.append(cond)
        log.append(_condition(name="b", converged=False, railed=3))
    rows = load_conditions(tmp_path / "c.csv")
    assert rows[0].name == cond.name and rows[0].corr == 0.9972
    assert rows[1].converged is False and rows[1].railed == 3


def test_nan_round_trips(tmp_path):
    with ConditionLog(tmp_path / "c.csv") as log:
        log.append(_condition(nmse_db=float("nan")))
    assert math.isnan(load_conditions(tmp_path / "c.csv")[0].nmse_db)


def test_dut_round_trip_with_extra_column(tmp_path):
    rec = DutRecord(
        **{k: getattr(_condition(), k) for k in CSV_FIELDS},
        papr_dpd_db=11.2,
        papr_expansion_db=2.3,
        tx_peak_dbfs=-0.5,
        tx_clipped=4,
        ref_file="sig_in.txt",
    )
    with ConditionLog(tmp_path / "d.csv", DUT_FIELDS + ("iteration",)) as log:
        log.append(rec, iteration=2)
    (back,) = load_dut_records(tmp_path / "d.csv")
    assert back.tx_clipped == 4 and back.papr_expansion_db == 2.3
    assert (tmp_path / "d.csv").read_text().splitlines()[1].endswith(",2")


def test_log_rejects_record_without_columns(tmp_path):
    with ConditionLog(tmp_path / "d.csv", DUT_FIELDS) as log:
        with pytest.raises(ValueError):
            log.append(_condition())


def test_missing_columns_raise(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("name,signal\na,b\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_conditions(path)


def test_iq_file_round_trip_scale(tmp_path):
    codes = np.array([2047 + 0j, -1024 + 512j, 0j])
    save_aligned_iq(codes, tmp_path / "a.txt", 12)
    back = load_aligned_iq(tmp_path / "a.txt")
    np.testing.assert_allclose(back, codes / 2047)


def test_condition_name():
    assert condition_name(TxChannel.TX1, 3, 2_200_000_000, 100) == "TX1_3dB_2200MHz_100BW"


def test_capture_point_aligns_and_scores():
    radio = SimRadio()
    x = make_signal()
    ref = reference_codes(x)
    transmit_bands(radio, {TxChannel.TX1: x}, BITS)
    radio.set_tx_atten(TxChannel.TX1, 25.0)
    radio.set_rx_gain(RxChannel.ORX1, 230)
    pc = capture_point(radio, RxChannel.ORX1, ref, rx_bits=BITS, fs=FS, bw_hz=BW)
    assert len(pc.y_aligned) == len(ref)
    frac = pc.delay_samples - np.floor(pc.delay_samples)
    assert frac == pytest.approx(DELAY, abs=0.02)
    assert pc.metrics["corr"] > 0.99
    assert np.isfinite(pc.metrics["gain_compression_db"])
    assert np.isfinite(pc.metrics["amam_top_slope"])
    truth = radio.pa_period(ref)
    assert nmse_db(truth, pc.y_aligned) < -35
    assert pc.clip.railed_samples == 0


def test_capture_point_metric_ref_length_checked():
    radio = SimRadio()
    x = make_signal()
    transmit_bands(radio, {TxChannel.TX1: x}, BITS)
    with pytest.raises(ValueError):
        capture_point(
            radio,
            RxChannel.ORX1,
            reference_codes(x),
            rx_bits=BITS,
            fs=FS,
            bw_hz=BW,
            metric_ref=np.ones(10),
        )
