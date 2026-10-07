"""Use the board through the hardware server (docs/hw_server.md).

::

    from adrvtrx.client import hardware

    with hardware("TX1 sweep") as radio:     # waits in the FIFO queue for the board
        transmit_bands(radio, {TxChannel.TX1: x}, info.tx_bits)
        res = capture(radio, int(RxChannel.ORX1), 0.5, bits=info.rx_bits)
    # leaving the block releases the board; the server forces TX safe

``radio`` is a :class:`RemoteRadio`: every public :class:`~adrvtrx.radio.Radio`
method with the same signature, run in the server's process. Existing modules
(``transmit``, ``capture``, ``compression``, ``replay``, ``operating_point``,
``linearize``, ``sweep``) take it in place of a ``Radio``.
"""

from __future__ import annotations

import functools
import inspect
import os
import socket
import sys
import threading
from typing import Any

import numpy as np

from ._hwlink import (
    BoardBusy,
    LeaseRevoked,
    RemoteError,
    ServerConnectionLost,
    ServerUnavailable,
    open_connection,
    public_methods,
    raise_error,
    receive,
    request,
    resolve_config,
)
from .config import Config
from .radio import Radio

__all__ = [
    "hardware",
    "RemoteRadio",
    "NumpyBridge",
    "BoardBusy",
    "LeaseRevoked",
    "RemoteError",
    "ServerConnectionLost",
    "ServerUnavailable",
    "server_status",
    "server_safe",
    "server_kick",
    "server_stop",
    "server_config",
]


class NumpyBridge:
    """``transmit.build_tx_data`` builds numpy buffers with this instead of .NET arrays.

    ``Radio.perform_tx`` in the server converts them to .NET.
    """

    def int_array(self, values) -> np.ndarray:
        return np.asarray(values).astype(np.int32)

    def array_list(self, items=()) -> list:
        return list(items)


def _hello(name: str) -> dict[str, Any]:
    return {"op": "hello", "name": name, "pid": os.getpid(), "host": socket.gethostname()}


class RemoteRadio:
    """A ``Radio`` that lives in the hardware server, held by this job until released.

    Public ``Radio`` methods are generated from the ``Radio`` class and checked
    against its signatures before anything is sent. ``disconnect()`` only forces
    TX safe (the server keeps its board connection); ``connect()`` is a no-op
    while the server is connected.
    """

    def __init__(self, conn, grant: dict[str, Any], name: str):
        self._conn = conn
        self._lock = threading.RLock()
        self.name = name
        self.lease_id: str = grant["lease"]
        #: The server's config: the one the board was programmed with.
        self.config: Config = grant["config"]
        self.bridge = NumpyBridge()
        self._heartbeat_s = max(0.05, float(grant["heartbeat_timeout_s"]) / 4.0)
        self._release_timeout_s = float(grant.get("release_timeout_s", 60.0))
        self._end: str | None = None  # why this job no longer holds the board
        self._revoked = False
        self._stop_heartbeat = threading.Event()
        self._heartbeat: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------------

    def _start_heartbeat(self) -> None:
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, name=f"adrvtrx-heartbeat-{self.lease_id}", daemon=True
        )
        self._heartbeat.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(self._heartbeat_s):
            if not self._lock.acquire(timeout=0.1):
                continue  # a call is in flight: the server knows this job is alive
            try:
                if self._end is not None:
                    return
                try:
                    request(self._conn, {"op": "heartbeat"}, timeout=self._heartbeat_s * 4)
                except LeaseRevoked as exc:
                    self._mark_revoked(str(exc))
                    return
                except Exception as exc:  # noqa: BLE001 - next call reports it
                    self._mark_lost(f"heartbeat failed: {exc}")
                    return
            finally:
                self._lock.release()

    @property
    def active(self) -> bool:
        """True while this job holds the board."""
        return self._end is None

    def release(self) -> None:
        """End the job: the server forces TX safe and hands the board to the next job."""
        self._stop_heartbeat.set()
        with self._lock:
            if self._end is None:
                try:
                    request(self._conn, {"op": "release"}, timeout=self._release_timeout_s)
                except Exception:  # noqa: BLE001 - closing the socket releases it too
                    pass
                self._end = "released"
            self._close()

    close = release

    def __enter__(self) -> RemoteRadio:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    def __repr__(self) -> str:
        state = "active" if self._end is None else self._end
        return f"<RemoteRadio {self.name!r} lease {self.lease_id} ({state})>"

    def _close(self) -> None:
        try:
            self._conn.close()
        except OSError:
            pass

    def _mark_revoked(self, reason: str) -> None:
        self._end, self._revoked = reason, True

    def _mark_lost(self, reason: str) -> None:
        if self._end is None:
            self._end = reason
        self._close()

    # -- calls -----------------------------------------------------------------------

    def _call(self, method: str, args: tuple, kwargs: dict) -> Any:
        with self._lock:
            if self._end is not None:
                if self._revoked:
                    raise LeaseRevoked(f"job {self.name!r} no longer holds the board: {self._end}")
                raise RuntimeError(f"job {self.name!r} no longer holds the board: {self._end}")
            message = {"op": "call", "method": method, "args": args, "kwargs": kwargs}
            try:
                self._conn.send(message)
                reply = receive(self._conn)
            except (ServerConnectionLost, EOFError, OSError) as exc:
                self._stop_heartbeat.set()
                self._mark_lost("connection to the server lost")
                raise ServerConnectionLost(
                    f"{method}: connection to adrvtrx-server lost (server stopped, restarted "
                    f"or killed); job {self.name!r} must be run again"
                ) from exc
            except BaseException:
                # Ctrl+C mid-request leaves the reply unread: drop the connection;
                # the server then ends the lease and forces TX safe.
                self._stop_heartbeat.set()
                self._mark_lost(f"interrupted during {method}")
                raise
        if reply.get("ok"):
            return reply.get("result")
        error = reply.get("error") or {}
        if error.get("kind") == "revoked":
            self._mark_revoked(str(error.get("message")))
        raise_error(error)

    def print_status(self) -> dict:
        """Print :meth:`status` here (the same report as ``Radio.print_status``)."""
        return Radio.print_status(self)


def _forwarder(name: str):
    target = getattr(Radio, name)
    signature = inspect.signature(target)

    @functools.wraps(target)
    def forward(self: RemoteRadio, *args, **kwargs):
        signature.bind(self, *args, **kwargs)  # a bad call fails here, before it is sent
        return self._call(name, args, kwargs)

    return forward


for _name in public_methods():
    if _name not in RemoteRadio.__dict__:
        setattr(RemoteRadio, _name, _forwarder(_name))
del _name


def _print_queue(info: dict[str, Any]) -> None:
    print(
        f"adrvtrx: {info.get('message')}; waiting in the queue (position {info.get('position')})",
        file=sys.stderr,
        flush=True,
    )


def hardware(
    name: str,
    wait: bool = True,
    timeout: float | None = None,
    *,
    config: Config | str | os.PathLike[str] | None = None,
) -> RemoteRadio:
    """Hold the board for one job: ``with hardware("my job") as radio: ...``.

    ``wait=False`` raises :class:`BoardBusy` at once if another job holds the
    board ("owned by NAME, pid, host, since"); ``timeout`` (seconds) bounds the
    wait in the FIFO queue. ``config`` (path or ``Config``) only supplies
    ``[server]`` (port, key); ``radio.config`` is the server's config. Without
    ``with``, call ``radio.release()`` when done (a notebook can keep the board
    across cells that way).
    """
    settings = resolve_config(config).server
    conn = open_connection(settings)
    try:
        request(conn, _hello(name))
        grant = request(
            conn,
            {"op": "acquire", "wait": bool(wait), "timeout": timeout},
            on_info=_print_queue,
        )
    except BaseException:
        conn.close()
        raise
    radio = RemoteRadio(conn, grant, name)
    radio._start_heartbeat()
    return radio


# -- control (no lease, never queued) -------------------------------------------------------


def _control(op: str, config, *, timeout: float = 30.0, **fields: Any) -> Any:
    settings = resolve_config(config).server
    conn = open_connection(settings, timeout=min(timeout, 10.0))
    try:
        hello = request(conn, _hello(f"adrvtrx-server {op}"), timeout=timeout)
        try:
            return request(conn, {"op": op, **fields}, timeout=timeout)
        except ServerConnectionLost:
            if op != "stop":
                raise
            # The server may exit before its reply arrives: that is the stop.
            return {"stopping": True, "pid": hello.get("server_pid")}
    finally:
        conn.close()


def server_status(config=None, *, live: bool = True) -> dict[str, Any]:
    """Owner, queue, TX state, LO, attenuation, gains, PLL, uptime, restarts, last error."""
    return _control("status", config, live=live)


def server_safe(config=None, *, wait_s: float = 10.0) -> dict[str, Any]:
    """Force TX safe now (ahead of queued calls) and end the current job."""
    return _control("safe", config, timeout=wait_s + 10.0, wait_s=wait_s)


def server_kick(config=None, *, wait_s: float = 10.0) -> dict[str, Any]:
    """End the current job (its next call raises :class:`LeaseRevoked`); TX safe."""
    return _control("kick", config, timeout=wait_s + 10.0, wait_s=wait_s)


def server_stop(config=None) -> dict[str, Any]:
    """Stop the server: leases end, TX safe, disconnect."""
    return _control("stop", config)


def server_config(config=None) -> Config:
    """The server's config (what the board was programmed with)."""
    return _control("config", config)
