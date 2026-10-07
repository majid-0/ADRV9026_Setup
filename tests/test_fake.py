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
