"""Hardware server in this process (fake board): forwarding, numpy, queue, stop paths."""

from __future__ import annotations

import inspect
import json
import os
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx._hwlink import open_connection, public_methods, request
from adrvtrx.capture import capture_arrays
from adrvtrx.client import (
    BoardBusy,
    LeaseRevoked,
    RemoteRadio,
    hardware,
    server_kick,
    server_safe,
    server_status,
)
from adrvtrx.config import load_config
from adrvtrx.fake import FakeRadio
from adrvtrx.radio import MAX_TX_ATTEN_DB, Radio
from adrvtrx.server import HardwareServer, PortInUse
from hw_helpers import LineReader, running_server, subprocess_env, wait_until, write_config

MAX_MDB = round(MAX_TX_ATTEN_DB * 1000)


@pytest.fixture
def cfg(tmp_path):
    return load_config(write_config(tmp_path))


@pytest.fixture
def server(cfg):
    with running_server(cfg) as srv:
        yield srv


def board(srv):
    return srv.radio.board_model.state


def tx_buffers(n: int = 64) -> list[np.ndarray]:
    return [np.arange(n, dtype=np.int32) * (k + 1) - 7 * k for k in range(8)]


# -- forwarding -------------------------------------------------------------------------


def test_remote_radio_has_every_public_radio_method_with_its_signature():
    names = public_methods()
    assert "perform_tx" in names and "set_rx_enable" in names and "bridge" not in names
    for name in names:
        assert callable(getattr(RemoteRadio, name)), name
        assert inspect.signature(getattr(RemoteRadio, name)) == inspect.signature(
            getattr(Radio, name)
        ), name


# Every public Radio method, in an order that is valid on a programmed board.
# connect / disconnect / print_status are checked separately below.
CALLS = [
    ("set_tx_atten", (TxChannel.TX1, 20.0), {}),
    ("get_tx_atten", (TxChannel.TX1,), {}),
    ("set_rx_gain", (RxChannel.ORX1, 214), {}),
    ("get_rx_gain", (RxChannel.ORX1,), {}),
    ("rx_dec_power_dbfs", (RxChannel.ORX1,), {}),
    ("set_lo", ("LO1", 2_400_000_000), {}),
    ("get_lo", ("LO1",), {}),
    ("retune_lo", ("LO2", 900_000_000), {"settle_poll": 3}),
    ("pll_lock_status", (), {}),
    ("enable_rx", (0x10,), {}),
    ("set_rx_enable", (0x1F,), {}),
    ("perform_tx", (tx_buffers(), int(TxChannel.TX1)), {"continuous": True}),
    ("enable_tx", (int(TxChannel.TX2),), {}),
    ("rx_tx_enable_get", (), {}),
    ("rx_tx_enable", (0x1F, int(TxChannel.TX1)), {}),
    ("perform_rx", (0x3FF, 0.0064), {}),
    ("status", (), {}),
    ("disable_tx", (), {}),
    ("safe_state", (), {}),
    ("force_safe", (), {}),
    ("program", (), {}),
]


def test_every_method_returns_what_an_in_process_radio_returns(server, cfg):
    covered = {name for name, _a, _k in CALLS} | {"connect", "disconnect", "print_status"}
    assert covered == set(public_methods()), "add new Radio methods to CALLS"

    local = FakeRadio(cfg, seed=1)  # the server's backend is FakeRadio(cfg, seed=1) too
    local._safe_hooks_installed = True
    local.connect()
    local.force_safe()
    local.program()
    with hardware("forwarding", config=cfg) as radio:
        for name, args, kwargs in CALLS:
            remote = getattr(radio, name)(*args, **kwargs)
            expected = getattr(local, name)(*args, **kwargs)
            if name == "perform_rx":
                assert all(a.dtype == np.int32 for a in remote)
                expected = capture_arrays(expected)
                assert len(remote) == len(expected) == 20
                for got, want in zip(remote, expected):
                    np.testing.assert_array_equal(got, want)
            else:
                assert remote == expected, name
        assert board(server) == {**local.board_model.state, "pid": board(server)["pid"]}


def test_disconnect_only_forces_safe_and_connect_is_a_no_op(server, cfg):
    with hardware("lifecycle", config=cfg) as radio:
        radio.perform_tx(tx_buffers(), int(TxChannel.TX1))
        radio.disconnect()
        assert server.radio._connected and board(server)["tx_mask"] == 0
        assert board(server)["tx_atten_mdb"]["TX1"] == MAX_MDB
        radio.connect()
        assert radio.get_lo("LO1") == cfg.lo.lo1_hz  # still connected, still usable


def test_print_status_prints_here(server, cfg, capsys):
    with hardware("print", config=cfg) as radio:
        st = radio.print_status()
    out = capsys.readouterr().out
    assert "Radio status:" in out and "TX OFF" in out and st["connected"] is True


def test_numpy_round_trip(server, cfg):
    bufs = tx_buffers(128)
    with hardware("numpy", config=cfg) as radio:
        radio.set_tx_atten(TxChannel.TX1, 15.0)
        radio.set_rx_gain(RxChannel.ORX1, 214)
        radio.set_rx_enable(int(RxChannel.ORX1))
        radio.perform_tx(bufs, int(TxChannel.TX1))
        # Radio converted the numpy buffers to the bridge's .NET stand-ins
        wave = server.radio.board_model.waves["TX1"]
        np.testing.assert_array_equal(wave, bufs[0] + 1j * bufs[1])
        raw = radio.perform_rx(0x3FF, 0.0128)
    assert isinstance(raw, list) and len(raw) == 20
    assert all(isinstance(a, np.ndarray) and a.dtype == np.int32 and len(a) == 128 for a in raw)
    assert np.abs(raw[8]).max() > 100  # ORx1 slot carries the loopback


def test_errors_keep_their_type_and_bad_arguments_never_leave(server, cfg):
    with hardware("errors", config=cfg) as radio:
        with pytest.raises(KeyError, match="adrvtrx-server"):
            radio.set_lo("LO9", 1)  # raised inside the server
        with pytest.raises(TypeError):
            radio.get_lo()  # rejected here by Radio's signature
        assert radio.get_lo("LO2") == cfg.lo.lo2_hz  # the job is still fine


# -- queue / lease -------------------------------------------------------------------------


def test_fifo_queue_order(server, cfg):
    order: list[str] = []

    def job(name: str) -> None:
        with hardware(name, config=cfg) as radio:
            order.append(name)
            radio.get_lo("LO1")

    first = hardware("first", config=cfg)
    threads = []
    for k, name in enumerate(["second", "third", "fourth"], start=1):
        t = threading.Thread(target=job, args=(name,))
        t.start()
        threads.append(t)
        wait_until(lambda k=k: len(server.status(live=False)["queue"]) == k, what=f"{k} queued")
    queued = [w["name"] for w in server.status(live=False)["queue"]]
    assert queued == ["second", "third", "fourth"]
    first.release()
    for t in threads:
        t.join(20)
    assert order == ["second", "third", "fourth"]


def test_no_wait_fails_fast_with_the_owner(server, cfg):
    with hardware("holder", config=cfg):
        t0 = time.monotonic()
        with pytest.raises(BoardBusy) as err:
            hardware("impatient", wait=False, config=cfg)
        assert time.monotonic() - t0 < 5
    msg = str(err.value)
    assert '"holder"' in msg and f"pid {os.getpid()}" in msg
    assert socket.gethostname() in msg and "since" in msg


def test_queue_timeout(server, cfg):
    with hardware("holder", config=cfg):
        t0 = time.monotonic()
        with pytest.raises(BoardBusy, match="timed out"):
            hardware("late", timeout=0.5, config=cfg)
        assert 0.4 < time.monotonic() - t0 < 5
    assert server.status(live=False)["queue"] == []


def test_a_waiting_client_that_disconnects_leaves_the_queue(server, cfg):
    with hardware("holder", config=cfg):
        conn = open_connection(cfg.server)
        request(conn, {"op": "hello", "name": "quitter", "pid": 1, "host": "h"})
        conn.send({"op": "acquire", "wait": True, "timeout": None})
        wait_until(lambda: len(server.status(live=False)["queue"]) == 1, what="queued")
        conn.close()
        wait_until(lambda: server.status(live=False)["queue"] == [], what="queue emptied")


def test_release_forces_safe_before_the_next_job(server, cfg):
    with hardware("first", config=cfg) as radio:
        radio.set_tx_atten(TxChannel.TX1, 10.0)
        radio.perform_tx(tx_buffers(), int(TxChannel.TX1))
        assert board(server)["tx_mask"] == int(TxChannel.TX1)
    with hardware("second", config=cfg) as radio:
        assert radio.rx_tx_enable_get()[1] == 0
        assert radio.get_tx_atten(TxChannel.TX1) == MAX_TX_ATTEN_DB


def test_job_raising_still_releases_and_forces_safe(server, cfg):
    with pytest.raises(ZeroDivisionError):
        with hardware("crashy", config=cfg) as radio:
            radio.perform_tx(tx_buffers(), int(TxChannel.TX1))
            1 / 0  # noqa: B018
    wait_until(lambda: server.status(live=False)["owner"] is None, what="released")
    assert board(server)["tx_mask"] == 0


# -- stop paths --------------------------------------------------------------------------------


@pytest.fixture
def short_heartbeat(tmp_path):
    return load_config(write_config(tmp_path, heartbeat_timeout_s=0.6))


def test_silent_client_with_tx_live_is_revoked(short_heartbeat, monkeypatch):
    cfg = short_heartbeat
    monkeypatch.setattr(RemoteRadio, "_start_heartbeat", lambda self: None)  # a frozen client
    with running_server(cfg) as srv:
        radio = hardware("frozen", config=cfg)
        radio.perform_tx(tx_buffers(), int(TxChannel.TX1))
        wait_until(lambda: srv.status(live=False)["owner"] is None, 5, what="revoked")
        assert board(srv)["tx_mask"] == 0
        assert set(board(srv)["tx_atten_mdb"].values()) == {MAX_MDB}
        with pytest.raises(LeaseRevoked, match="no heartbeat"):
            radio.get_lo("LO1")
        radio.release()


def test_silent_client_without_tx_keeps_the_board(short_heartbeat, monkeypatch):
    cfg = short_heartbeat
    monkeypatch.setattr(RemoteRadio, "_start_heartbeat", lambda self: None)
    with running_server(cfg) as srv:
        with hardware("quiet", config=cfg) as radio:
            time.sleep(1.5)
            assert srv.status(live=False)["owner"]["name"] == "quiet"
            radio.get_lo("LO1")


def test_heartbeat_keeps_an_idle_job_alive(short_heartbeat):
    cfg = short_heartbeat
    with running_server(cfg) as srv:
        with hardware("alive", config=cfg) as radio:
            radio.perform_tx(tx_buffers(), int(TxChannel.TX1))
            time.sleep(2.0)  # > 3x the timeout, heartbeats every 0.15 s
            assert srv.status(live=False)["owner"]["name"] == "alive"
            assert radio.rx_tx_enable_get()[1] == int(TxChannel.TX1)


def test_kick_revokes_the_job_and_forces_safe(server, cfg):
    with hardware("victim", config=cfg) as radio:
        radio.perform_tx(tx_buffers(), int(TxChannel.TX1))
        res = server_kick(cfg)
        assert res["kicked"]["name"] == "victim" and res["done"]
        assert board(server)["tx_mask"] == 0
        with pytest.raises(LeaseRevoked, match="kicked"):
            radio.perform_tx(tx_buffers(), int(TxChannel.TX1))
        assert board(server)["tx_mask"] == 0


def test_emergency_safe_skips_the_queue_during_a_job(tmp_path):
    cfg = load_config(write_config(tmp_path))
    backend = lambda c: FakeRadio(c, seed=1, delays={"PerformRx": 1.0})  # noqa: E731
    with running_server(cfg, backend) as srv:
        holder = hardware("holder", config=cfg)
        holder.perform_tx(tx_buffers(), int(TxChannel.TX1))
        waiting = threading.Thread(target=lambda: hardware("next", config=cfg).release())
        waiting.start()
        wait_until(lambda: len(srv.status(live=False)["queue"]) == 1, what="queued")
        busy = threading.Thread(target=lambda: holder.perform_rx(0x3FF, 0.0064))
        busy.start()
        wait_until(lambda: srv.hw.busy_info() is not None, what="hardware busy")
        t0 = time.monotonic()
        res = server_safe(cfg)  # no lease, no queue: runs right after the call in progress
        assert res["done"] and res["revoked"]["name"] == "holder"
        assert time.monotonic() - t0 < 5
        assert board(srv)["tx_mask"] == 0
        busy.join(10)
        with pytest.raises(LeaseRevoked, match="forced safe"):
            holder.enable_tx(int(TxChannel.TX1))
        waiting.join(10)
        holder.release()


def test_status_is_cached_while_the_hardware_is_busy(tmp_path):
    cfg = load_config(write_config(tmp_path))
    backend = lambda c: FakeRadio(c, seed=1, delays={"PerformRx": 1.5})  # noqa: E731
    with running_server(cfg, backend) as srv:
        with hardware("slow", config=cfg) as radio:
            radio.set_tx_atten(TxChannel.TX2, 12.5)
            busy = threading.Thread(target=lambda: radio.perform_rx(0x3FF, 0.0064))
            busy.start()
            wait_until(lambda: srv.hw.busy_info() is not None, what="hardware busy")
            t0 = time.monotonic()
            st = server_status(cfg)
            assert time.monotonic() - t0 < 1.0
            assert st["source"] == "cached" and st["hardware"]["method"] == "perform_rx"
            assert st["owner"]["name"] == "slow" and st["board"]["tx_atten_db"]["TX2"] == 12.5
            busy.join(10)
        st = server_status(cfg)
        assert st["source"] == "live" and st["owner"] is None and st["board"]["tx_live"] is False


def test_client_process_killed_releases_and_next_job_starts_safe(server, cfg, tmp_path):
    script = tmp_path / "victim.py"
    script.write_text(
        "import sys, time\n"
        "import numpy as np\n"
        "from adrvtrx import TxChannel\n"
        "from adrvtrx.client import hardware\n"
        "radio = hardware('victim', config=sys.argv[1])\n"
        "radio.set_tx_atten(TxChannel.TX1, 10.0)\n"
        "radio.perform_tx([np.ones(64, dtype=np.int32)] * 8, int(TxChannel.TX1))\n"
        "print('TX-LIVE', flush=True)\n"
        "time.sleep(120)\n"
    )
    cfg_path = tmp_path / "server.toml"
    proc = subprocess.Popen(
        [sys.executable, str(script), str(cfg_path)],
        env=subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        LineReader(proc).wait_for("TX-LIVE", 60)
        assert board(server)["tx_mask"] == int(TxChannel.TX1)
        seen = {}

        def next_job():
            with hardware("next", config=cfg) as radio:
                seen["tx_mask"] = radio.rx_tx_enable_get()[1]
                seen["atten"] = radio.get_tx_atten(TxChannel.TX1)

        t = threading.Thread(target=next_job)
        t.start()
        wait_until(lambda: len(server.status(live=False)["queue"]) == 1, what="next queued")
        proc.kill()  # our own client process
        t.join(20)
        assert seen == {"tx_mask": 0, "atten": MAX_TX_ATTEN_DB}
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(10)


def test_a_second_server_on_the_port_is_refused_before_touching_the_board(server, cfg):
    built = []
    second = HardwareServer(cfg, lambda c: built.append(c), backend_name="second")
    with pytest.raises(PortInUse):
        second.start()
    assert built == []
    assert server.status(live=False)["state"] == "ready"


def test_every_call_is_logged(server, cfg):
    with hardware("logged", config=cfg) as radio:
        radio.set_tx_atten(TxChannel.TX1, 20.0)
    path = cfg.server.log_path
    records = [
        json.loads(line) for f in path.glob("*.jsonl") for line in f.read_text().splitlines()
    ]
    calls = [r for r in records if r["event"] == "call" and r["method"] == "set_tx_atten"]
    assert calls and calls[-1]["args"] == ["TX1", 20.0] and calls[-1]["ok"] is True
    assert calls[-1]["client"]["name"] == "logged" and "dur_s" in calls[-1]
    events = [r["event"] for r in records]
    assert "acquire" in events and "release" in events
