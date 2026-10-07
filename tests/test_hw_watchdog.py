"""The supervised server as real processes (fake board): watchdog, single instance, CLI.

Each test starts ``adrvtrx-server run --backend fake`` with its own port, state
directory and fake-board state file, and only ever kills processes it started.
"""

from __future__ import annotations

import subprocess
import sys
import time

import numpy as np
import pytest

from adrvtrx import TxChannel
from adrvtrx.client import ServerConnectionLost, hardware
from adrvtrx.config import load_config
from adrvtrx.fake import FakeRadio
from adrvtrx.radio import MAX_TX_ATTEN_DB
from hw_helpers import (
    Supervised,
    fake_events,
    kill_pid,
    running_server,
    server_cli,
    subprocess_env,
    wait_until,
    write_config,
)

MAX_MDB = round(MAX_TX_ATTEN_DB * 1000)
BUFS = [np.ones(64, dtype=np.int32)] * 8


@pytest.fixture
def bench(tmp_path):
    """(config path, env, fake state file); the env points the fake at the state file."""
    state = tmp_path / "board.json"
    return tmp_path, state, subprocess_env(ADRVTRX_FAKE_STATE=str(state))


def _start(tmp_path, env, **server) -> Supervised:
    cfg_path = write_config(tmp_path, **server)
    return Supervised(cfg_path, env, tmp_path / "supervisor.log")


def _safe_from(events, pids_excluded, after):
    """pids that forced TX safe (max atten on all + TX mask cleared) after ``after``."""
    atten = {
        e["pid"]
        for e in events
        if e["op"] == "TxAttenSet" and e["mask"] == 0xF and e["mdb"] == MAX_MDB and e["t"] > after
    }
    cleared = {
        e["pid"] for e in events if e["op"] == "RxTxEnableSet" and e["tx"] == 0 and e["t"] > after
    }
    return (atten & cleared) - set(pids_excluded)


def test_killed_server_is_forced_safe_from_a_fresh_process_and_restarted(bench):
    tmp_path, state, env = bench
    sup = _start(tmp_path, env)
    try:
        first = sup.wait_ready()
        old_pid = first["server"]["pid"]
        radio = hardware("job", config=sup.config)
        radio.set_tx_atten(TxChannel.TX1, MAX_TX_ATTEN_DB)
        radio.perform_tx(BUFS, int(TxChannel.TX1))
        assert fake_events(state)[-1]["op"] == "RxTxEnableSet"  # TX enabled
        killed_at = time.time()
        kill_pid(old_pid)  # the server child of our own supervisor
        with pytest.raises(ServerConnectionLost):
            radio.get_lo("LO1")
        radio.release()
        second = sup.wait_ready(60, pid_not=old_pid)
        assert second["server"]["restarts"] == 1
        assert "crashed" in second["server"]["last_error"]
        # the restart re-programmed the board: a client can tell from the identity
        before, after = first["programming"], second["programming"]
        assert after["program_count"] == before["program_count"] + 1
        assert after["program_id"] != before["program_id"]
        assert after["programmed_at"] >= before["programmed_at"]
        new_pid = second["server"]["pid"]
        assert _safe_from(fake_events(state), [old_pid, new_pid], killed_at)  # fresh process
        assert "TX forced safe" in sup.output()
        with hardware("after", config=sup.config) as radio:
            assert radio.rx_tx_enable_get()[1] == 0
            assert radio.session_info()["programming"] == after
    finally:
        sup.close()
    assert sup.proc.returncode == 0


def test_stuck_call_is_killed_forced_safe_and_restarted(bench):
    tmp_path, state, env = bench
    env["ADRVTRX_FAKE_DELAYS"] = '{"PerformRx": 120}'
    sup = _start(tmp_path, env, call_timeout_s={"perform_rx": 1.0})
    try:
        old_pid = sup.wait_ready()["server"]["pid"]
        radio = hardware("job", config=sup.config)
        radio.perform_tx(BUFS, int(TxChannel.TX1))
        t0 = time.time()
        with pytest.raises(ServerConnectionLost):
            radio.perform_rx(0x3FF, 0.01)  # hangs in the "DLL" until the watchdog kills it
        assert time.time() - t0 < 30
        radio.release()
        second = sup.wait_ready(60, pid_not=old_pid)
        assert second["server"]["restarts"] == 1 and "stuck" in second["server"]["last_error"]
        assert _safe_from(fake_events(state), [old_pid, second["server"]["pid"]], t0)
        assert "perform_rx running" in sup.output()
    finally:
        sup.close()


def test_second_run_is_refused_without_touching_the_board(bench):
    tmp_path, state, env = bench
    sup = _start(tmp_path, env)
    try:
        child = sup.wait_ready()["server"]["pid"]
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "adrvtrx.server",
                "run",
                "--backend",
                "fake",
                "--config",
                str(sup.cfg_path),
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert out.returncode == 3, out.stdout + out.stderr
        assert "refusing to start" in out.stdout + out.stderr
        assert {e["pid"] for e in fake_events(state) if e["op"] == "Connect"} == {child}
        assert sup.status()["server"]["pid"] == child
    finally:
        sup.close()


def test_stop_leaves_tx_safe_and_the_supervisor_exits(bench):
    tmp_path, state, env = bench
    sup = _start(tmp_path, env)
    try:
        sup.wait_ready()
        radio = hardware("job", config=sup.config)
        radio.perform_tx(BUFS, int(TxChannel.TX1))
        out = server_cli("stop", "--config", str(sup.cfg_path), env=env)
        assert out.returncode == 0, out.stderr
        sup.proc.wait(30)
        assert sup.proc.returncode == 0
        events = fake_events(state)
        assert events[-1]["op"] == "Disconnect"
        assert events[-2] == {**events[-2], "op": "RxTxEnableSet", "tx": 0}
        radio.release()
    finally:
        sup.close()


def test_status_safe_kick_cli(bench):
    tmp_path, state, env = bench
    sup = _start(tmp_path, env)
    try:
        sup.wait_ready()
        cfg = str(sup.cfg_path)
        radio = hardware("cli job", config=sup.config)
        radio.perform_tx(BUFS, int(TxChannel.TX1))
        out = server_cli("status", "--config", cfg, env=env)
        assert out.returncode == 0 and '"cli job"' in out.stdout and "live: TX1" in out.stdout
        out = server_cli("kick", "--config", cfg, env=env)
        assert out.returncode == 0 and 'kicked "cli job"' in out.stdout
        radio.release()
        out = server_cli("safe", "--config", cfg, env=env)
        assert out.returncode == 0 and "TX forced safe" in out.stdout
        assert "TX       off" in server_cli("status", "--config", cfg, env=env).stdout
    finally:
        sup.close()


def test_failed_force_safe_is_loud_and_restarts_are_limited(bench):
    tmp_path, state, env = bench
    sup = _start(tmp_path, env, restart_limit=1)
    try:
        old_pid = sup.wait_ready()["server"]["pid"]
        state.with_suffix(".refuse").touch()  # the ADS9 now refuses new connections
        kill_pid(old_pid)
        sup.proc.wait(90)
        out = sup.output()
        assert "TX STATE UNKNOWN - switch off the PA supply" in out
        assert "giving up" in out
        assert sup.proc.returncode == 1
    finally:
        sup.close()


def test_safe_direct(bench):
    tmp_path, state, env = bench
    cfg_path = write_config(tmp_path)
    live = FakeRadio(load_config(cfg_path), state_path=state)  # leave the "board" with TX on
    live._safe_hooks_installed = True
    live.connect()
    live.program()
    live.perform_tx(BUFS, int(TxChannel.TX1))
    live.disconnect()

    out = server_cli("safe", "--direct", "--backend", "fake", "--config", str(cfg_path), env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "TX forced safe" in out.stdout
    events = fake_events(state)
    assert events[-1]["op"] == "Disconnect" and events[-2]["tx"] == 0


def test_safe_direct_is_refused_while_a_server_answers(bench):
    tmp_path, state, env = bench
    cfg_path = write_config(tmp_path)
    cfg = load_config(cfg_path)
    with running_server(cfg, lambda c: FakeRadio(c, state_path=state)) as srv:
        pid = srv.status(live=False)["server"]["pid"]
        before = len(fake_events(state))
        args = ("safe", "--direct", "--backend", "fake", "--config", str(cfg_path))
        out = server_cli(*args, env=env)
        assert out.returncode == 1 and f"pid {pid} answers" in out.stderr
        assert len(fake_events(state)) == before  # board untouched
        out = server_cli(*args, "--force", env=env)
        assert out.returncode == 0, out.stdout + out.stderr


def test_real_backend_is_refused_in_tests(bench):
    tmp_path, _state, env = bench
    cfg_path = write_config(tmp_path)
    out = server_cli("run", "--config", str(cfg_path), env=env)  # default backend: real
    assert out.returncode == 4 and "ADRVTRX_FORBID_HARDWARE" in out.stderr
    out = server_cli("safe", "--direct", "--force", "--config", str(cfg_path), env=env)
    assert out.returncode == 4 and "ADRVTRX_FORBID_HARDWARE" in out.stderr


def test_supervisor_gone_stops_the_server_safely(bench):
    tmp_path, state, env = bench
    sup = _start(tmp_path, env)
    try:
        child = sup.wait_ready()["server"]["pid"]
        radio = hardware("job", config=sup.config)
        radio.perform_tx(BUFS, int(TxChannel.TX1))
        sup.proc.kill()  # our own supervisor; its child must stop by itself
        sup.proc.wait(10)
        wait_until(lambda: sup.status() is None, 30, what="server gone")
        events = [e for e in fake_events(state) if e["pid"] == child]
        assert events[-1]["op"] == "Disconnect" and events[-2]["tx"] == 0
        radio.release()
    finally:
        sup.close()
