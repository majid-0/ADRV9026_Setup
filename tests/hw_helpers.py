"""Helpers for the hardware-server tests: configs, an in-process server, subprocesses.

Every server here runs the fake backend (``adrvtrx.fake``) or the simulated
bench (``sim_bench``); ``ADRVTRX_FORBID_HARDWARE`` (set in conftest) makes the
real backend impossible. Every wait is bounded.
"""

from __future__ import annotations

import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import adrvtrx
from adrvtrx.config import Config, load_config
from adrvtrx.fake import FakeRadio
from adrvtrx.server import HardwareServer

SRC = Path(adrvtrx.__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def toml_path(p: Path) -> str:
    return str(p).replace("\\", "/")


def write_config(tmp_path: Path, *, profile: Path | None = None, **server) -> Path:
    """A TOML config for a test server on a free port, state in ``tmp_path``."""
    settings = {
        "port": free_port(),
        "state_dir": toml_path(tmp_path / "state"),
        "heartbeat_timeout_s": 10.0,
        "ping_interval_s": 0.25,
        "ping_timeout_s": 5.0,
        "start_timeout_s": 30.0,
        "restart_limit": 3,
        "restart_backoff_s": 0.2,
        "force_safe_timeout_s": 30.0,
        "connect_retries": 2,
        "connect_retry_delay_s": 0.1,
    }
    settings.update(server)
    lines = ["[dll]", 'install_dir = "C:/nonexistent"', "", "[server]"]
    for key, value in settings.items():
        if isinstance(value, dict):
            inner = ", ".join(f"{k} = {v}" for k, v in value.items())
            lines.append(f"{key} = {{ {inner} }}")
        elif isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        elif isinstance(value, str):
            lines.append(f'{key} = "{value}"')
        else:
            lines.append(f"{key} = {value}")
    if profile is not None:
        lines += ["", "[profile]", f'name = "{toml_path(profile)}"']
    path = tmp_path / "server.toml"
    path.write_text("\n".join(lines) + "\n")
    return path


def wait_until(predicate, timeout: float = 10.0, interval: float = 0.05, what: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout} s waiting for {what}")


@contextmanager
def running_server(config: Config, backend=None, **kw):
    """A HardwareServer serving from a thread of this process."""
    if backend is None:

        def backend(cfg):
            return FakeRadio(cfg, seed=1)

    server = HardwareServer(config, backend, backend_name="test", echo=False, **kw)
    server.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        wait_until(lambda: server.state == "ready", 10, what="server ready")
        yield server
    finally:
        server.request_stop("test teardown")
        thread.join(20)


def subprocess_env(**extra: str) -> dict[str, str]:
    """This process's env, with this checkout's ``src`` first on PYTHONPATH."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(SRC), env.get("PYTHONPATH")) if p)
    env["ADRVTRX_FORBID_HARDWARE"] = "1"
    env.update(extra)
    return env


def server_cli(*args: str, env: dict[str, str], timeout: float = 60.0):
    return subprocess.run(
        [sys.executable, "-m", "adrvtrx.server", *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class LineReader:
    """Collect a subprocess's stdout lines on a thread (reads never block a test)."""

    def __init__(self, proc: subprocess.Popen):
        self.lines: queue.Queue[str] = queue.Queue()
        self.seen: list[str] = []
        threading.Thread(target=self._pump, args=(proc,), daemon=True).start()

    def _pump(self, proc):
        for line in proc.stdout:
            self.lines.put(line.rstrip("\n"))

    def wait_for(self, text: str, timeout: float = 30.0) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=max(0.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            self.seen.append(line)
            if text in line:
                return line
        raise AssertionError(f"no line with {text!r} in {timeout} s; got {self.seen}")


class Supervised:
    """``adrvtrx-server run --backend fake`` as a subprocess (a supervisor and its child)."""

    def __init__(self, cfg_path: Path, env: dict[str, str], log: Path):
        self.cfg_path = cfg_path
        self.config = load_config(cfg_path)
        self.log = log
        self._out = open(log, "w")
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "adrvtrx.server",
                "run",
                "--backend",
                "fake",
                "--config",
                str(cfg_path),
            ],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._out,
            stderr=subprocess.STDOUT,
        )

    def status(self):
        from adrvtrx.client import server_status

        try:
            return server_status(self.config)
        except Exception:  # noqa: BLE001 - not up (yet)
            return None

    def wait_ready(self, timeout: float = 60.0, *, pid_not: int | None = None) -> dict:
        def ready():
            st = self.status()
            ok = st is not None and st["state"] == "ready"
            return st if ok and st["server"]["pid"] != pid_not else None

        return wait_until(ready, timeout, 0.1, what=f"server ready ({self.output()})")

    def output(self) -> str:
        self._out.flush()
        return self.log.read_text(errors="replace")

    def close(self) -> None:
        from adrvtrx.client import server_stop

        if self.proc.poll() is None:
            try:
                server_stop(self.config)
                self.proc.wait(30)
            except Exception:  # noqa: BLE001 - fall through to kill
                pass
        if self.proc.poll() is None:
            # Our own supervisor: its child stops itself when the supervisor's pipe closes.
            self.proc.kill()
            self.proc.wait(10)
        self._out.close()


def fake_events(state: Path) -> list[dict]:
    path = state.with_suffix(".events.jsonl")
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def kill_pid(pid: int) -> None:
    """Hard-kill a process this test started (directly or through its supervisor)."""
    import signal

    os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
