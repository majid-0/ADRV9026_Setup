"""``adrvtrx replay`` end to end on the fake board, through an in-process hardware server."""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from adrvtrx.cli import SUMMARY_FIELDS, tool_main
from adrvtrx.client import hardware
from adrvtrx.conditions import (
    ConditionLog,
    OperatingCondition,
    condition_name,
    load_dut_records,
    save_aligned_iq,
)
from adrvtrx.config import load_config
from adrvtrx.waveform import prepare_tx
from hw_helpers import running_server, write_config

FS = 491.52e6
FREQS = (2_000_000_000, 2_400_000_000)


def _signal(n: int = 4096, bw: float = 100e6, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n, 1 / FS)
    spec = (rng.normal(size=n) + 1j * rng.normal(size=n)) * (np.abs(f) < 0.45 * bw)
    x = np.fft.ifft(spec)
    return x / np.abs(x).max()


def _row(freq: int, atten: float) -> OperatingCondition:
    nan = float("nan")
    return OperatingCondition(
        name=condition_name("TX1", 0, freq, 100),
        signal="sig.txt",
        freq_hz=freq,
        bw_mhz=100,
        backoff_db=0,
        locked_atten_db=atten,
        tx_atten_db=atten,
        orx_gain=212,
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
        out_file="unused_out.txt",
    )


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """Server (fake board at 491.52 MSPS), a two-state conditions CSV and DPD files."""
    profile = tmp_path / "uc98.profile"
    profile.write_text(
        json.dumps(
            {
                "framer": [{"jesd204Np": 12, "rxOutputRate_kHz": 491520}],
                "deframer": [{"jesd204Np": 12, "txInputRate_kHz": 491520}],
            }
        )
    )
    cfg_path = write_config(tmp_path, profile=profile)
    cfg = load_config(cfg_path)
    caps = tmp_path / "caps"
    caps.mkdir()
    i, q = prepare_tx(_signal(), 12)
    ref = i + 1j * q
    save_aligned_iq(ref, caps / "sig_in.txt", 12)
    rows = [_row(FREQS[0], 12.0), _row(FREQS[1], 11.0)]
    with ConditionLog(caps / "TX1_conditions.csv") as log:
        for row in rows:
            log.append(row)
    dpd = tmp_path / "DPD" / "gmp"
    dpd.mkdir(parents=True)
    for row in rows:
        save_aligned_iq(0.9 * ref, dpd / f"{row.name}.txt", 12)
    monkeypatch.chdir(tmp_path)  # templates are relative to the current directory
    with running_server(cfg) as srv:
        yield srv, cfg_path, caps / "TX1_conditions.csv", [r.name for r in rows]


def _replay(cfg_path, csv_path, out, *extra):
    return tool_main(
        [
            "replay",
            "--conditions",
            str(csv_path),
            "--dpd",
            "input=input",
            "--dpd",
            "gmp=DPD/gmp/{name}.txt",
            "--out",
            str(out),
            "--config",
            str(cfg_path),
            *extra,
        ]
    )


def _summary(out) -> list[dict]:
    with open(out / "summary.csv", newline="") as fh:
        return list(csv.DictReader(fh))


def _events(srv) -> list[str]:
    files = srv.settings.log_path.glob("*.jsonl")
    return [json.loads(line)["event"] for f in files for line in f.read_text().splitlines()]


def test_replay_plays_every_label_per_state_and_writes_a_summary(bench, tmp_path):
    srv, cfg_path, csv_path, names = bench
    assert _replay(cfg_path, csv_path, tmp_path / "out") == 0
    out = tmp_path / "out"
    rows = _summary(out)
    assert tuple(rows[0]) == SUMMARY_FIELDS
    assert [(r["state"], r["label"]) for r in rows] == [
        (names[0], "input"),
        (names[0], "gmp"),
        (names[1], "input"),
        (names[1], "gmp"),
    ]
    for r in rows:
        for key in ("nmse_db", "aclr_lower_dbc", "aclr_upper_dbc", "rms_dbfs", "peak_dbfs"):
            assert math.isfinite(float(r[key])), (key, r)
    assert rows[1]["file"].replace("\\", "/") == f"DPD/gmp/{names[0]}.txt"
    for label in ("input", "gmp"):
        records = load_dut_records(out / f"{label}.csv")
        assert [rec.name for rec in records] == names
        for rec in records:
            assert (out / rec.out_file).is_file()
            assert rec.corr > 0.9
    # LO retuned once per state, not per label; TX off and the job released at the end
    ops = list(srv.radio.board_model.calls)
    assert ops.count("PllFrequencySet") == 2
    assert srv.radio.board_model.state["tx_mask"] == 0
    assert srv.status(live=False)["owner"] is None


def test_states_selects_rows(bench, tmp_path):
    _srv, cfg_path, csv_path, names = bench
    assert _replay(cfg_path, csv_path, tmp_path / "one", "--states", names[1]) == 0
    assert [r["state"] for r in _summary(tmp_path / "one")] == [names[1], names[1]]


def test_missing_files_are_reported_before_the_board_is_taken(bench, tmp_path, capsys):
    srv, cfg_path, csv_path, names = bench
    (tmp_path / "DPD" / "gmp" / f"{names[1]}.txt").unlink()
    code = _replay(cfg_path, csv_path, tmp_path / "out", "--states", names[0], names[1], "nope")
    err = capsys.readouterr().err
    assert code == 2
    assert f"{names[1]} [gmp]" in err and "'nope'" in err
    assert "acquire" not in _events(srv)
    assert not (tmp_path / "out").exists()


def test_no_wait_fails_fast_when_the_board_is_busy(bench, tmp_path, capsys):
    srv, cfg_path, csv_path, _names = bench
    with hardware("holder", config=srv.config):
        code = _replay(cfg_path, csv_path, tmp_path / "out", "--no-wait")
    assert code == 1 and '"holder"' in capsys.readouterr().err


def test_bad_dpd_spec_is_rejected(bench, tmp_path, capsys):
    _srv, cfg_path, csv_path, _names = bench
    code = tool_main(
        ["replay", "--conditions", str(csv_path), "--dpd", "noequals", "--out", str(tmp_path)]
    )
    assert code == 2 and "LABEL=TEMPLATE" in capsys.readouterr().err
