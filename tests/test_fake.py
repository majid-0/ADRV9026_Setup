"""The fake board backend: the real Radio code runs on it, state persists across processes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import capture
from adrvtrx.config import Config, DllConfig
from adrvtrx.fake import FakeRadio
from adrvtrx.radio import MAX_TX_ATTEN_DB
from adrvtrx.transmit import transmit_bands


def _config() -> Config:
    return Config(dll=DllConfig(install_dir=Path("C:/nonexistent")))


def _radio(**kw) -> FakeRadio:
    radio = FakeRadio(_config(), seed=1, **kw)
    radio._safe_hooks_installed = True  # keep pytest's own SIGINT handler
    return radio


DATA = Path(__file__).parent / "data"


def test_program_with_a_profile_changes_the_sample_rate_and_clears_tx_ram():
    radio = _radio()
    radio.connect()
    radio.program(profile=str(DATA / "stub_uc98.profile"))
    board = radio.board_model
    assert (board.rate_hz, board.tx_bits, board.rx_bits) == (491.52e6, 12, 12)
    radio.set_tx_atten(TxChannel.TX1, 15.0)
    transmit_bands(radio, {TxChannel.TX1: 0.5 * np.ones(1024, complex)}, 12)
    assert "TX1" in board.waves

    radio.program(profile=str(DATA / "stub_uc61.profile"))
    assert radio.config.profile_path == DATA / "stub_uc61.profile"
    assert (board.rate_hz, board.tx_bits, board.rx_bits) == (368.64e6, 12, 12)
    assert not board.waves  # programming clears TX RAM
    cap = capture(radio, int(RxChannel.ORX1), 1.0, bits=12).channels[RxChannel.ORX1]
    assert len(cap.i) == 368_640  # 1 ms at the new rate

    radio.program()  # no profile: keeps the current one
    assert radio.config.profile_path == DATA / "stub_uc61.profile"


def test_a_profile_that_fails_to_program_does_not_become_the_configs():
    radio = _radio()
    radio.connect()
    radio.program(profile=str(DATA / "stub_uc98.profile"))
    board = radio.board_model
    loads = board.calls.count("ConfigFileLoad")

    with pytest.raises(FileNotFoundError, match="not found"):  # checked before the board
        radio.program(profile=str(DATA / "no_such.profile"))
    assert board.calls.count("ConfigFileLoad") == loads
    assert radio.config.profile_path == DATA / "stub_uc98.profile"

    def program_fails():
        raise RuntimeError("Program failed")

    radio.board.Program = program_fails  # the load succeeds, programming fails
    with pytest.raises(RuntimeError, match="Program failed"):
        radio.program(profile=str(DATA / "stub_uc61.profile"))
    assert radio.config.profile_path == DATA / "stub_uc98.profile"
    assert board.rate_hz == 491.52e6


def test_program_and_status_run_the_real_radio_code():
    radio = _radio()
    radio.connect()
    radio.force_safe()  # unprogrammed: TxAttenSet is rejected, force_safe swallows it
    radio.program()
    st = radio.status()
    assert st["connected"] and st["pll_lock"].startswith("0xF")
    assert st["lo1_hz"] == radio.config.lo.lo1_hz
    assert st["tx_atten_db"]["TX1"] == radio.config.levels.tx_atten_for("tx1")
    radio.safe_state()
    assert radio.get_tx_atten(TxChannel.TX3) == MAX_TX_ATTEN_DB
    assert radio.rx_tx_enable_get()[1] == 0


def test_retune_waits_for_every_pll_and_raises_if_one_stays_unlocked(monkeypatch):
    radio = _radio()
    radio.connect()
    radio.program()
    assert radio.retune_lo("LO2", 1_860_000_000) == 0xF
    assert radio.get_lo("LO2") == 1_860_000_000

    reads = iter([0x0, 0x9, 0xF])  # unlocked, partly locked (the old check passed here), locked
    monkeypatch.setattr(radio, "pll_lock_status", lambda: next(reads))
    assert radio.retune_lo("LO1", 1_440_000_000, poll_interval_s=0) == 0xF

    monkeypatch.setattr(radio, "pll_lock_status", lambda: 0xB)  # one PLL never locks
    with pytest.raises(RuntimeError, match="not all locked"):
        radio.retune_lo("LO1", 1_440_000_000, settle_poll=5, poll_interval_s=0)


def test_run_bands_captures_with_the_immediate_trigger_by_default(tmp_path):
    from adrvtrx._enums import RxTrigSource
    from adrvtrx.bands import make_bands, run_bands

    radio = _radio()
    radio.connect()
    radio.program()
    radio.set_tx_atten(TxChannel.TX1, 15.0)
    x = 0.5 * np.exp(2j * np.pi * 0.01 * np.arange(4096))
    run = run_bands(
        radio,
        make_bands([("b1", TxChannel.TX1, x, RxChannel.ORX1, 0.05)]),
        tx_bits=12,
        rx_bits=12,
        out_dir=tmp_path,
    )
    assert run.captures.trig == RxTrigSource.IMMEDIATE
    assert [p.name for p in run.files] == ["b1.txt"]


def test_tx_atten_is_rejected_before_programming():
    radio = _radio()
    radio.connect()
    with pytest.raises(RuntimeError, match="not programmed"):
        radio.set_tx_atten(TxChannel.TX1, 20.0)


def test_loopback_capture_follows_the_transmitted_tx():
    radio = _radio()
    radio.connect()
    radio.program()
    radio.set_tx_atten(TxChannel.TX1, 15.0)
    radio.set_rx_gain(RxChannel.ORX1, 214)
    x = 0.5 * np.exp(2j * np.pi * 0.01 * np.arange(4096))
    transmit_bands(radio, {TxChannel.TX1: x}, 12)
    on = capture(radio, int(RxChannel.ORX1), 0.8192, bits=12).channels[RxChannel.ORX1]
    assert len(on.i) == 8192 and on.clip().peak_dbfs > -10
    radio.disable_tx()
    off = capture(radio, int(RxChannel.ORX1), 0.8192, bits=12).channels[RxChannel.ORX1]
    assert off.clip().peak_dbfs < -40


def test_perform_tx_must_get_converted_buffers():
    radio = _radio()
    radio.connect()
    radio.program()
    with pytest.raises(TypeError, match="ArrayList"):
        radio.board.PerformTx(1, [np.zeros(4, dtype=np.int32)] * 8, 1, 1)
    radio.perform_tx([np.zeros(4, dtype=np.int32)] * 8, int(TxChannel.TX1))  # Radio converts
    assert radio.board_model.state["tx_len"] == 4


def test_state_persists_across_instances_and_logs_events(tmp_path):
    state = tmp_path / "board.json"
    first = _radio(state_path=state)
    first.connect()
    first.program()
    first.enable_tx(int(TxChannel.TX2))
    second = _radio(state_path=state)  # e.g. a fresh process after a kill
    second.connect()
    assert second.rx_tx_enable_get()[1] == int(TxChannel.TX2)
    second.force_safe()
    data = json.loads(state.read_text())
    assert data["tx_mask"] == 0
    assert set(data["tx_atten_mdb"].values()) == {round(MAX_TX_ATTEN_DB * 1000)}
    ops = [json.loads(line)["op"] for line in state.with_suffix(".events.jsonl").open()]
    assert ops[0] == "Connect" and ops[-1] == "RxTxEnableSet"


def test_refuse_file_makes_connect_fail(tmp_path):
    state = tmp_path / "board.json"
    state.with_suffix(".refuse").touch()
    with pytest.raises(ConnectionError):
        _radio(state_path=state).connect()


def test_delays_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("ADRVTRX_FAKE_DELAYS", '{"Program": 0.25}')
    radio = _radio()
    assert radio.board_model.delays == {"Program": 0.25}


def test_the_device_clock_and_the_ad9528_follow_the_programmed_profile():
    """[clocks] device_clock_khz = 0 (the default): each profile's own deviceClock_kHz, AD9528 =
    (device, 122.88 MHz VCXO, 122.88 MHz ref A, device); a profile without one keeps 245.76 MHz."""
    radio = _radio()
    radio.connect()
    radio.program(profile=str(DATA / "stub_uc83c.profile"))
    board = radio.board_model
    assert board.clock_args == (184320, 122880, 122880, 184320)
    assert (board.rate_hz, board.tx_bits) == (368.64e6, 16)
    radio.program(profile=str(DATA / "stub_uc98.profile"))
    assert board.clock_args == (245760, 122880, 122880, 245760)
    radio.config.clocks.device_clock_khz = 200000  # a fixed clock overrides the profile's
    radio.program(profile=str(DATA / "stub_uc83c.profile"))
    assert board.clock_args == (200000, 122880, 122880, 200000)
