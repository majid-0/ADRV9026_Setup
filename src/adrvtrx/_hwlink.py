"""Pieces shared by the hardware server and its clients (docs/hw_server.md).

Transport is ``multiprocessing.connection`` over TCP on 127.0.0.1 with an auth
key (HMAC challenge). Every message is a dict; every reply is
``{"ok": True, "result": ...}`` or ``{"ok": False, "error": {...}}``. While a
client waits in the queue the server may send ``{"type": "info", ...}`` first.
"""

from __future__ import annotations

import builtins
import enum
import hashlib
import inspect
import json
import os
import secrets
import threading
import time
import traceback
from datetime import date, datetime
from multiprocessing.connection import AuthenticationError, Client, Connection
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import Config, ServerConfig, load_config

HOST = "127.0.0.1"
PROTOCOL = 1
#: How often a blocking wait returns to Python (keeps Ctrl+C responsive on Windows).
POLL_S = 0.2

# Exit codes of the server process (docs/hw_server.md section 6).
EXIT_OK = 0
EXIT_CRASH = 1
EXIT_PORT_IN_USE = 3
EXIT_BAD_CONFIG = 4
EXIT_STARTUP_FAILED = 5


# -- errors ----------------------------------------------------------------------


class ServerUnavailable(ConnectionError):
    """No hardware server answers on the configured port (or no key file)."""


class ServerConnectionLost(ConnectionError):
    """The connection to the hardware server dropped (server stopped, restarted or killed)."""


class BoardBusy(RuntimeError):
    """Another job holds the board (``wait=False``) or the queue timeout passed."""


class LeaseRevoked(RuntimeError):
    """This job no longer holds the board (kick, safe, heartbeat timeout, server stop)."""


class RemoteError(RuntimeError):
    """A hardware call raised inside the server. The remote traceback is in the message."""

    def __init__(self, type_name: str, message: str, remote_traceback: str = ""):
        self.type_name = type_name
        self.remote_message = message
        self.remote_traceback = remote_traceback
        super().__init__(f"{type_name}: {message}\n--- in adrvtrx-server ---\n{remote_traceback}")


def error_payload(exc: BaseException, kind: str = "remote") -> dict[str, str]:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {"kind": kind, "type": type(exc).__name__, "message": str(exc), "traceback": tb}


def raise_error(err: dict[str, Any]) -> None:
    """Raise the client-side exception for an error reply."""
    kind = err.get("kind", "remote")
    message = str(err.get("message", ""))
    if kind == "busy":
        raise BoardBusy(message)
    if kind == "revoked":
        raise LeaseRevoked(message)
    if kind == "remote":
        type_name = str(err.get("type", "Exception"))
        tb = str(err.get("traceback", ""))
        builtin = getattr(builtins, type_name, None)
        if isinstance(builtin, type) and issubclass(builtin, Exception):
            try:
                exc = builtin(f"{message}\n--- in adrvtrx-server ---\n{tb}")
            except Exception:  # noqa: BLE001 - exotic constructor: fall back below
                exc = None
            if exc is not None:
                raise exc
        raise RemoteError(type_name, message, tb)
    raise RuntimeError(f"adrvtrx-server: {message}")


# -- the forwarded interface -------------------------------------------------------


def public_methods() -> tuple[str, ...]:
    """Every public method of :class:`~adrvtrx.radio.Radio` (the forwarded interface)."""
    from .radio import Radio

    names = []
    for name in dir(Radio):
        if name.startswith("_"):
            continue
        attr = inspect.getattr_static(Radio, name)
        if isinstance(attr, (property, staticmethod, classmethod)) or not callable(attr):
            continue
        names.append(name)
    return tuple(sorted(names))


# -- TX buffer identity ----------------------------------------------------------------


def tx_buffer_ids(tx_data) -> list[dict[str, Any]] | None:
    """Identity of each TX channel's buffer in a PerformTx payload, or None if unknown.

    ``tx_data`` is the eight arrays ``[Tx1_I, Tx1_Q, ..., Tx4_I, Tx4_Q]``. Returns
    one ``{"hash", "n", "zeros"}`` per TX channel; ``hash`` is over the int32 I
    and Q samples, so a client can compute it for buffers it has not sent.
    """
    try:
        arrays = [np.asarray(a) for a in tx_data]
    except TypeError:
        return None
    if len(arrays) != 8 or any(a.ndim != 1 for a in arrays):
        return None
    ids = []
    for k in range(4):
        i = arrays[2 * k].astype(np.int32)
        q = arrays[2 * k + 1].astype(np.int32)
        digest = hashlib.sha1(i.tobytes() + b"|" + q.tobytes()).hexdigest()[:16]
        ids.append({"hash": digest, "n": int(len(i)), "zeros": not (i.any() or q.any())})
    return ids


# -- config, key, connection ----------------------------------------------------------


def resolve_config(config: Config | str | os.PathLike[str] | None) -> Config:
    if isinstance(config, Config):
        return config
    return load_config(config)


def read_authkey(settings: ServerConfig) -> bytes:
    path = settings.authkey_path
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise ServerUnavailable(
            f"no server key at {path}: start the server first (`adrvtrx-server run`)"
        ) from None


def ensure_authkey(settings: ServerConfig) -> bytes:
    """Read the auth key, creating it (32 random bytes, owner-only on POSIX) if missing."""
    path = settings.authkey_path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_bytes()
    with os.fdopen(fd, "wb") as fh:
        fh.write(secrets.token_bytes(32))
    return path.read_bytes()


def open_connection(settings: ServerConfig, *, timeout: float = 10.0) -> Connection:
    """Connect and authenticate. ``timeout`` bounds a server that accepts but never answers."""
    key = read_authkey(settings)
    address = (HOST, settings.port)
    box: dict[str, Any] = {}

    def attempt() -> None:
        try:
            box["conn"] = Client(address, family="AF_INET", authkey=key)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            box["exc"] = exc

    worker = threading.Thread(target=attempt, name="adrvtrx-connect", daemon=True)
    worker.start()
    deadline = time.monotonic() + timeout
    while worker.is_alive() and time.monotonic() < deadline:
        worker.join(POLL_S)
    if worker.is_alive():
        raise ServerUnavailable(
            f"adrvtrx-server on {HOST}:{settings.port} accepted the connection but did not "
            f"answer within {timeout:g} s (stuck?)"
        )
    exc = box.get("exc")
    if isinstance(exc, AuthenticationError):
        raise ServerUnavailable(
            f"adrvtrx-server on {HOST}:{settings.port} rejected the key {settings.authkey_path}"
        ) from exc
    if isinstance(exc, OSError):
        raise ServerUnavailable(
            f"no adrvtrx-server on {HOST}:{settings.port} ({exc.__class__.__name__}); "
            "start it with `adrvtrx-server run`"
        ) from exc
    if exc is not None:
        raise exc
    return box["conn"]


def receive(conn: Connection, timeout: float | None = None) -> Any:
    """Next message, polling so Ctrl+C stays responsive. ``TimeoutError`` past ``timeout``."""
    deadline = None if timeout is None else time.monotonic() + timeout
    while True:
        try:
            if conn.poll(POLL_S):
                return conn.recv()
        except (EOFError, OSError) as exc:
            raise ServerConnectionLost(
                "connection to adrvtrx-server lost (the server stopped, restarted or was killed)"
            ) from exc
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"adrvtrx-server did not answer within {timeout:g} s")


def request(
    conn: Connection,
    message: dict[str, Any],
    *,
    timeout: float | None = None,
    on_info: Callable[[dict[str, Any]], None] | None = None,
) -> Any:
    """Send one request and return its result (or raise its error)."""
    try:
        conn.send(message)
    except (EOFError, OSError) as exc:
        raise ServerConnectionLost("connection to adrvtrx-server lost") from exc
    while True:
        reply = receive(conn, timeout)
        if reply.get("type") == "info":
            if on_info is not None:
                on_info(reply)
            continue
        if reply.get("ok"):
            return reply.get("result")
        raise_error(reply.get("error") or {})


# -- logging ---------------------------------------------------------------------------


def summarize(value: Any, depth: int = 0) -> Any:
    """Short, JSON-friendly description of call arguments and results for the log."""
    if isinstance(value, np.ndarray):
        return f"ndarray{list(value.shape)} {value.dtype}"
    if isinstance(value, enum.Enum):
        return value.name or int(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str):
        return value if len(value) <= 200 else value[:200] + "..."
    if isinstance(value, (list, tuple)):
        if len(value) > 8 or depth >= 2:
            head = f" of {summarize(value[0], 2)}" if len(value) else ""
            return f"{type(value).__name__}[{len(value)}]{head}"
        return [summarize(v, depth + 1) for v in value]
    if isinstance(value, dict):
        if len(value) > 24 or depth >= 2:
            return f"dict[{len(value)}]"
        return {str(k): summarize(v, depth + 1) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    text = repr(value)
    return text if len(text) <= 200 else text[:200] + "..."


class JsonlLog:
    """Append-only JSON-lines log, one file per day; never raises."""

    def __init__(self, directory: str | os.PathLike[str], source: str):
        self.directory = Path(directory)
        self.source = source
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self.directory / f"server-{date.today().isoformat()}.jsonl"

    def write(self, event: str, **fields: Any) -> None:
        record = {
            "t": datetime.now().isoformat(timespec="milliseconds"),
            "src": self.source,
            "pid": os.getpid(),
            "event": event,
            **fields,
        }
        line = json.dumps(record, default=str)
        with self._lock:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                pass  # logging must never take the server down


def describe_client(client: dict[str, Any] | None) -> str:
    if not client:
        return "unknown client"
    return (
        f'"{client.get("name", "?")}" (pid {client.get("pid", "?")} on {client.get("host", "?")})'
    )


def iso(epoch: float | None) -> str | None:
    return None if epoch is None else datetime.fromtimestamp(epoch).isoformat(timespec="seconds")
