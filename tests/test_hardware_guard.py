"""The hardware tests never run by accident (they program the real board).

A default run deselects them, ``-m hardware`` without the opt-in skips them before
anything connects, and the opt-in still refuses while a hardware server answers.
Every subprocess here points ``ADRVTRX_CONFIG`` at a board on 127.0.0.1 (a free
port, nothing listening), so even a broken guard cannot reach the real board.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from adrvtrx.config import load_config
from hw_helpers import free_port, subprocess_env, toml_path
from test_hardware import ALLOW_ENV, hardware_refusal

ROOT = Path(__file__).resolve().parents[1]


def _safe_config(tmp_path: Path, server_port: int | None = None) -> Path:
    """A config whose board and server are local ports nobody listens on."""
    path = tmp_path / "guard.toml"
    path.write_text(
        "\n".join(
            [
                "[dll]",
                'install_dir = "C:/nonexistent"',
                "[board]",
                'ip = "127.0.0.1"',
                f"port = {free_port()}",
                "[server]",
                f"port = {server_port or free_port()}",
                f'state_dir = "{toml_path(tmp_path / "state")}"',
                "",
            ]
        )
    )
    return path


def _pytest(tmp_path: Path, *args: str) -> str:
    env = subprocess_env(ADRVTRX_CONFIG=str(_safe_config(tmp_path)))
    env.pop(ALLOW_ENV, None)
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_hardware.py", "-p", "no:cacheprovider", *args],
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return out.stdout + out.stderr


def test_a_default_run_selects_no_hardware_test(tmp_path):
    out = _pytest(tmp_path, "--collect-only", "-q")  # collecting runs no fixture
    assert "test_hardware.py::" not in out, out
    deselected = re.search(r"(\d+) deselected", out)
    assert deselected and int(deselected.group(1)) >= 10, out


def test_selected_without_the_opt_in_every_hardware_test_skips(tmp_path):
    out = _pytest(tmp_path, "-m", "hardware", "-q", "-rs")
    assert re.search(r"\b(\d+) skipped\b", out) and " passed" not in out, out
    assert " failed" not in out and " error" not in out, out
    assert f"set {ALLOW_ENV}=1" in out, out


def test_without_the_opt_in_the_guard_refuses_before_any_connection(tmp_path, monkeypatch):
    def no_connection(*_a, **_k):
        raise AssertionError("the guard opened a connection without the opt-in")

    monkeypatch.setattr(socket, "create_connection", no_connection)
    cfg = load_config(_safe_config(tmp_path))
    for environ in ({}, {ALLOW_ENV: "0"}, {ALLOW_ENV: "yes"}):
        assert ALLOW_ENV in hardware_refusal(cfg, environ)


def test_with_the_opt_in_a_running_hardware_server_is_refused(tmp_path):
    with socket.socket() as listener:  # stands in for adrvtrx-server on its port
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        cfg = load_config(_safe_config(tmp_path, server_port=listener.getsockname()[1]))
        reason = hardware_refusal(cfg, {ALLOW_ENV: "1"})
    assert reason is not None and "hardware server answers" in reason


def test_with_the_opt_in_and_no_board_the_tests_skip(tmp_path):
    cfg = load_config(_safe_config(tmp_path))
    assert "not reachable" in hardware_refusal(cfg, {ALLOW_ENV: "1"})


def test_this_run_keeps_the_real_dll_forbidden():
    assert os.environ.get("ADRVTRX_FORBID_HARDWARE") == "1"


LIFTS_FLAG = re.compile(
    r"""(environ\.pop\(|del\s+os\.environ\[|delenv\(|unsetenv\()\s*["']ADRVTRX_FORBID_HARDWARE"""
)


@pytest.mark.parametrize("path", sorted((ROOT / "tests").glob("*.py")), ids=lambda p: p.name)
def test_only_the_hardware_module_lifts_the_forbid_flag(path):
    lifts = LIFTS_FLAG.search(path.read_text(encoding="utf-8"))
    assert lifts is None or path.name == "test_hardware.py", path.name
