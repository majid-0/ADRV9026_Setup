"""Loaded-signal tracking and skipped identical reloads in the hardware server (fake board)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import capture
from adrvtrx.client import hardware, server_status, tx_signal_ids
from adrvtrx.config import load_config
from adrvtrx.fake import FakeRadio
from adrvtrx.transmit import transmit_bands
from hw_helpers import running_server, write_config

TX1, TX2, ORX1 = TxChannel.TX1, TxChannel.TX2, RxChannel.ORX1


def wave(n: int = 4096, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=n) + 1j * rng.normal(size=n)


def loads(srv) -> int:
    return list(srv.radio.board_model.calls).count("PerformTx")


@pytest.fixture
def cfg(tmp_path):
    return load_config(write_config(tmp_path))


def test_the_loaded_signal_is_tracked_and_survives_release_and_safe_state(cfg):
    x = wave()
    with running_server(cfg):
        with hardware("loader", config=cfg) as radio:
            transmit_bands(radio, {TX1: x}, 12)
            ram = radio.loaded_signals()
        tx1, tx2 = ram["channels"]["TX1"], ram["channels"]["TX2"]
        assert tx1["hash"] == tx_signal_ids({TX1: x}, 12)["TX1"]
        assert tx1["n"] == 4096 and tx1["in_mask"] and not tx1["zeros"]
        assert tx1["job"] == "loader" and tx1["continuous"] is True and tx1["load_s"] >= 0
        assert tx2["zeros"] and not tx2["in_mask"]
        assert (ram["loads"], ram["skipped"], ram["skip_identical"]) == (1, 0, True)
        # the release ran safe_state; the next job still sees the same RAM
        with hardware("next", config=cfg) as radio:
            assert radio.loaded_signals()["channels"] == ram["channels"]
        assert server_status(cfg)["tx_ram"]["channels"]["TX1"]["hash"] == tx1["hash"]


def test_identical_reload_only_reenables_tx(cfg):
    x = wave()
    with running_server(cfg) as srv:
        with hardware("first", config=cfg) as radio:
            radio.set_rx_gain(ORX1, 212)
            transmit_bands(radio, {TX1: x}, 12)
            radio.disable_tx()
            radio.retune_lo("LO2", 2_400_000_000)  # TX1 uses LO2 in the default clocks
            radio.set_tx_atten(TX1, 18.0)
            transmit_bands(radio, {TX1: x}, 12)
            assert loads(srv) == 1
            assert radio.rx_tx_enable_get()[1] == int(TX1)  # playing again
            cap = capture(radio, int(ORX1), 0.05, bits=12).channels[ORX1]
            assert cap.clip().peak_dbfs > -30  # the stored waveform plays, not silence
        with hardware("second", config=cfg) as radio:  # across a release (safe_state)
            transmit_bands(radio, {TX1: x}, 12)
            assert loads(srv) == 1 and radio.rx_tx_enable_get()[1] == int(TX1)
            ram = radio.loaded_signals()
            assert (ram["loads"], ram["skipped"]) == (1, 2)
            assert ram["channels"]["TX1"]["job"] == "first"  # still the first load

            transmit_bands(radio, {TX1: wave(seed=1)}, 12)  # another signal: reload
            assert loads(srv) == 2
            transmit_bands(radio, {TX1: wave(seed=1)}, 12, continuous=False)  # other mode
            assert loads(srv) == 3
            transmit_bands(radio, {TX1: wave(seed=1), TX2: x}, 12)  # TX2 not loaded yet
            assert loads(srv) == 4
            transmit_bands(radio, {TX2: x}, 12)  # TX2 alone: already in its RAM
            assert loads(srv) == 4 and radio.rx_tx_enable_get()[1] == int(TX2)
    records = [json.loads(line) for f in cfg.server.log_path.glob("*.jsonl") for line in f.open()]
    events = [r["event"] for r in records]
    assert events.count("tx_load") == 4 and events.count("tx_load_skipped") == 3


def test_program_and_a_new_server_invalidate_the_ram(cfg):
    x = wave()
    with running_server(cfg) as srv:
        with hardware("job", config=cfg) as radio:
            transmit_bands(radio, {TX1: x}, 12)
            radio.program()
            assert radio.loaded_signals()["channels"] == {}
            transmit_bands(radio, {TX1: x}, 12)
            assert loads(srv) == 2
    with running_server(cfg) as srv:  # restart: the RAM content is unknown
        assert server_status(cfg)["tx_ram"]["channels"] == {}
        with hardware("job", config=cfg) as radio:
            transmit_bands(radio, {TX1: x}, 12)
            assert loads(srv) == 1


def test_the_skip_can_be_switched_off(tmp_path):
    cfg = load_config(write_config(tmp_path, skip_identical_tx_load=False))
    x = wave()
    with running_server(cfg) as srv:
        with hardware("job", config=cfg) as radio:
            transmit_bands(radio, {TX1: x}, 12)
            transmit_bands(radio, {TX1: x}, 12)
            ram = radio.loaded_signals()
        assert loads(srv) == 2
        assert (ram["skip_identical"], ram["loads"], ram["skipped"]) == (False, 2, 0)


def test_fake_load_time_grows_with_the_buffer_length(cfg):
    def backend(c):
        return FakeRadio(c, seed=1, delays={"PerformTx_s_per_msample": 25.0})

    with running_server(cfg, backend):
        with hardware("timing", config=cfg) as radio:
            transmit_bands(radio, {TX1: wave(2048)}, 12)
            short = radio.loaded_signals()["last_load_s"]
            transmit_bands(radio, {TX1: wave(16384)}, 12)
            long = radio.loaded_signals()["last_load_s"]
    assert short >= 0.04 and long >= 0.38 and long > 3 * short  # 0.05 s and 0.41 s
