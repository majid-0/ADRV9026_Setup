from __future__ import annotations

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.conditions import (
    ConditionLog,
    OperatingCondition,
    capture_point,
    condition_name,
    load_aligned_iq,
    load_conditions,
    load_dut_records,
    save_aligned_iq,
)
from adrvtrx.metrics import nmse_db, papr_db
from adrvtrx.replay import replay_conditions, stored_tx
from adrvtrx.transmit import transmit_bands
from sim_bench import BITS, BW, FS, SimRadio, make_signal, reference_codes

TX, ORX = TxChannel.TX1, RxChannel.ORX1
FREQS = [2_000_000_000, 2_200_000_000]
BACKOFFS = [0, 2]


def build_captures(tmp_path, radio, x):
    """A small capture CSV the way the sweep notebook writes it."""
    ref = reference_codes(x)
    in_file = "sig_in.txt"
    save_aligned_iq(ref, tmp_path / in_file, BITS)
    with ConditionLog(tmp_path / "TX1_conditions.csv") as log:
        for k, freq in enumerate(FREQS):
            radio.retune_lo("LO1", freq)
            transmit_bands(radio, {TX: x}, BITS)
            for backoff in BACKOFFS:
                atten = 13.0 + backoff
                gain = 214 + 2 * backoff + k
                radio.set_tx_atten(TX, atten)
                radio.set_rx_gain(ORX, gain)
                pc = capture_point(radio, ORX, ref, rx_bits=BITS, fs=FS, bw_hz=BW)
                name = condition_name(TX, backoff, freq, int(BW / 1e6))
                save_aligned_iq(pc.y_aligned, tmp_path / f"{name}_out.txt", BITS)
                log.append(
                    OperatingCondition(
                        name=name,
                        signal="sig.txt",
                        freq_hz=freq,
                        bw_mhz=int(BW / 1e6),
                        backoff_db=backoff,
                        locked_atten_db=13.0,
                        tx_atten_db=atten,
                        orx_gain=gain,
                        compression_db=3.0,
                        converged=True,
                        peak_dbfs=pc.clip.peak_dbfs,
                        railed=pc.clip.railed_samples,
                        in_file=in_file,
                        out_file=f"{name}_out.txt",
                        **pc.metrics,
                    )
                )
            radio.disable_tx()
    return tmp_path / "TX1_conditions.csv"


def _replay(radio, csv_path, waveform_for, out_dir, **kw):
    return replay_conditions(
        radio,
        csv_path,
        waveform_for,
        tx=TX,
        orx=ORX,
        tx_bits=BITS,
        rx_bits=BITS,
        fs=FS,
        out_dir=out_dir,
        label="dpd_test",
        **kw,
    )


@pytest.fixture
def bench(tmp_path):
    radio = SimRadio()
    x = make_signal()
    csv_path = build_captures(tmp_path, radio, x)
    radio.retunes.clear()
    radio.gain_sets.clear()
    radio.transmitted.clear()
    return radio, x, csv_path


def _write_dpd(tmp_path, rows, scale):
    dpd_dir = tmp_path / "dpd"
    dpd_dir.mkdir()
    x = load_aligned_iq(tmp_path / rows[0].in_file)
    for row in rows:
        save_aligned_iq(x * scale * 2047, dpd_dir / f"{row.name}_dpd.txt", BITS)
    return dpd_dir


def test_replay_applies_saved_condition_and_sends_u_as_stored(bench, tmp_path):
    radio, x, csv_path = bench
    rows = load_conditions(csv_path)
    dpd_dir = _write_dpd(tmp_path, rows, 0.8)
    out = _replay(radio, csv_path, lambda r: dpd_dir / f"{r.name}_dpd.txt", tmp_path / "out")

    records = load_dut_records(out)
    ordered = sorted(rows, key=lambda r: (r.freq_hz, r.bw_mhz, r.backoff_db))
    assert [r.name for r in records] == [r.name for r in ordered]
    assert radio.retunes == [("LO1", f) for f in FREQS]
    assert radio.gain_sets == [r.orx_gain for r in ordered]
    assert not radio.tx_on

    u = stored_tx(load_aligned_iq(dpd_dir / f"{rows[0].name}_dpd.txt"), BITS)
    np.testing.assert_array_equal(radio.transmitted[0], u.codes)

    rec = records[0]
    assert rec.tx_atten_db == ordered[0].tx_atten_db
    assert rec.orx_gain == ordered[0].orx_gain
    assert rec.ref_file == "sig_in.txt"
    assert rec.in_file == f"{ordered[0].name}_dpd.txt"
    assert rec.tx_clipped == 0


def test_replay_scores_against_x_and_aligns_to_u(bench, tmp_path):
    radio, x, csv_path = bench
    rows = load_conditions(csv_path)
    dpd_dir = _write_dpd(tmp_path, rows, 0.8)
    out = _replay(radio, csv_path, lambda r: dpd_dir / f"{r.name}_dpd.txt", tmp_path / "out")
    rec = load_dut_records(out)[0]
    x_codes = load_aligned_iq(csv_path.parent / "sig_in.txt") * 2047
    z = load_aligned_iq(out.parent / rec.out_file) * 2047
    u = stored_tx(x_codes / 2047 * 0.8, BITS)

    assert rec.nmse_db == pytest.approx(nmse_db(x_codes, z), abs=0.01)
    assert rec.papr_in_db == pytest.approx(papr_db(x_codes), abs=0.01)
    assert rec.papr_out_db == pytest.approx(papr_db(z), abs=0.01)
    assert rec.papr_compression_db == pytest.approx(rec.papr_in_db - rec.papr_out_db, abs=0.02)
    assert rec.papr_dpd_db == pytest.approx(papr_db(u.codes), abs=0.01)
    radio.atten = rec.tx_atten_db
    truth = radio.pa_period(u.codes)
    assert nmse_db(truth, z) < -35
    assert rec.corr > 0.99


def test_replay_accepts_arrays_and_counts_tx_clipping(bench, tmp_path):
    radio, x, csv_path = bench
    out = _replay(radio, csv_path, lambda r: 1.5 * x, tmp_path / "out")
    records = load_dut_records(out)
    assert all(r.tx_clipped > 0 for r in records)
    assert all(r.tx_peak_dbfs > 3.0 for r in records)
    assert (out.parent / records[0].in_file).is_file()


def test_replay_file_scale_for_code_files(bench, tmp_path):
    radio, x, csv_path = bench
    rows = load_conditions(csv_path)
    dpd_dir = tmp_path / "legacy"
    dpd_dir.mkdir()
    x_norm = load_aligned_iq(tmp_path / rows[0].in_file)
    for row in rows:
        np.savetxt(
            dpd_dir / f"{row.name}.txt",
            np.column_stack((0.8 * x_norm.real * 2048, 0.8 * x_norm.imag * 2048)),
            delimiter="\t",
        )
    _replay(radio, csv_path, lambda r: dpd_dir / f"{r.name}.txt", tmp_path / "o", file_scale=2048)
    np.testing.assert_array_equal(radio.transmitted[0], stored_tx(0.8 * x_norm, BITS).codes)


def test_replay_preflight_lists_missing_files_before_transmitting(bench, tmp_path):
    radio, x, csv_path = bench
    with pytest.raises(FileNotFoundError) as err:
        _replay(radio, csv_path, lambda r: tmp_path / "nope" / f"{r.name}.txt", tmp_path / "o")
    assert all(r.name in str(err.value) for r in load_conditions(csv_path))
    assert radio.transmitted == []


def test_replay_disables_tx_when_a_row_fails(bench, tmp_path):
    radio, x, csv_path = bench
    calls = {"n": 0}

    def waveform_for(row):
        calls["n"] += 1
        return x if row.backoff_db == 0 else x[:100]

    with pytest.raises(ValueError, match="samples"):
        _replay(radio, csv_path, waveform_for, tmp_path / "out")
    assert not radio.tx_on
    assert len(radio.transmitted) == 1
    assert len(load_dut_records(tmp_path / "out" / "dpd_test.csv")) == 1
