"""The hardware server: one process owns the board, every other process queues for it.

Specification: docs/hw_server.md. In short:

* :class:`HardwareServer` owns one ``Radio`` (or any backend with its
  interface). On start it binds 127.0.0.1 (a second server stops here, before
  touching the board), then connects, forces TX safe and programs.
* Every public ``Radio`` method is forwarded. All hardware calls run on one
  hardware thread, in order. ``status``, ``safe``, ``kick``, ``stop`` and
  ``ping`` are answered by their own connection thread and never wait behind
  queued calls.
* A job holds the board through a lease; other jobs wait FIFO. Every lease end
  (release, kick, safe, client gone, heartbeat timeout, stop) runs
  ``safe_state()`` before the next job starts.
* ``adrvtrx-server run`` is a :class:`Supervisor` (no DLL) that spawns the
  server as a child, pings it, and on a crash or a stuck call kills it, forces
  TX safe from a fresh process and restarts it.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import itertools
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from multiprocessing.connection import Listener, answer_challenge, deliver_challenge
from pathlib import Path
from typing import Any, Callable

from ._enums import RX_SINGLE, TX_SINGLE
from ._hwlink import (
    EXIT_BAD_CONFIG,
    EXIT_CRASH,
    EXIT_OK,
    EXIT_PORT_IN_USE,
    EXIT_STARTUP_FAILED,
    HOST,
    POLL_S,
    PROTOCOL,
    JsonlLog,
    LeaseRevoked,
    ServerConnectionLost,
    ServerUnavailable,
    describe_client,
    ensure_authkey,
    error_payload,
    iso,
    open_connection,
    public_methods,
    request,
    summarize,
    tx_buffer_ids,
)
from .config import Config, ServerConfig, load_config
from .radio import FORBID_HARDWARE_ENV, MAX_TX_ATTEN_DB

__all__ = [
    "BACKENDS",
    "HardwareServer",
    "Supervisor",
    "resolve_backend",
    "force_safe_fresh",
    "format_status",
    "main",
]

#: ``--backend`` short names.
BACKENDS = {"real": "adrvtrx.radio:Radio", "fake": "adrvtrx.fake:FakeRadio"}

URGENT, NORMAL, LAST = 0, 1, 2  # hardware-thread priorities


class PortInUse(OSError):
    """Another process listens on the server port."""


class BackendError(RuntimeError):
    """The backend could not be resolved or built (before touching the board)."""


class _OpError(Exception):
    """An error reply of a given kind (``busy``, ``revoked``, ``protocol``)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.payload = {"kind": kind, "type": kind, "message": message, "traceback": ""}


class _ClientGone(Exception):
    """The client disconnected while it waited in the queue."""


def resolve_backend(spec: str) -> tuple[Callable[[Config], Any], str]:
    """``real`` / ``fake`` / ``module:attr`` -> (factory, ``module:attr``)."""
    target = BACKENDS.get(spec, spec)
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise BackendError(f"backend must be real, fake or module:attr, got {spec!r}")
    try:
        factory = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise BackendError(f"cannot load backend {target}: {exc}") from exc
    if os.environ.get(FORBID_HARDWARE_ENV):
        from .radio import Radio

        if factory is Radio:
            raise BackendError(f"{FORBID_HARDWARE_ENV} is set: the real backend is not allowed")
    return factory, target


# -- hardware thread -------------------------------------------------------------------


class _HwJob:
    def __init__(self, name: str, fn: Callable[[], Any], check: Callable[[], None] | None):
        self.name = name
        self.fn = fn
        self.check = check
        self.done = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None
        self.duration_s = 0.0

    def wait(self, timeout: float | None = None) -> bool:
        return self.done.wait(timeout)


class HardwareThread:
    """Runs every hardware call on one thread: highest priority first, then in order."""

    def __init__(self, timeout_for: Callable[[str], float]):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = itertools.count()
        self._timeout_for = timeout_for
        #: ``(name, started monotonic, timeout_s)`` of the running call, or None.
        self.current: tuple[str, float, float] | None = None
        self._thread = threading.Thread(target=self._loop, name="adrvtrx-hardware", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        priority: int = NORMAL,
        check: Callable[[], None] | None = None,
    ) -> _HwJob:
        job = _HwJob(name, fn, check)
        self._queue.put((priority, next(self._seq), job))
        return job

    def idle(self) -> bool:
        return self.current is None and self._queue.empty()

    def busy_info(self) -> dict[str, Any] | None:
        cur = self.current
        if cur is None:
            return None
        return {"method": cur[0], "busy_s": time.monotonic() - cur[1], "timeout_s": cur[2]}

    def stop(self, timeout: float = 5.0) -> None:
        self._queue.put((LAST, next(self._seq), None))
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _loop(self) -> None:
        while True:
            _priority, _seq, job = self._queue.get()
            if job is None:
                return
            t0 = time.monotonic()
            try:
                if job.check is not None:
                    job.check()
                self.current = (job.name, t0, self._timeout_for(job.name))
                job.result = job.fn()
            except BaseException as exc:  # noqa: BLE001 - delivered to the waiting caller
                job.error = exc
            finally:
                self.current = None
                job.duration_s = time.monotonic() - t0
                job.done.set()


# -- leases ---------------------------------------------------------------------------


@dataclass
class _Lease:
    id: str
    client: dict[str, Any]
    since: float
    last_seen: float  # any message (calls and heartbeats)
    last_call: float = 0.0  # hardware calls only (the idle limit)
    in_flight: int = 0
    ended: str | None = None
    revoked: bool = False

    def info(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            **self.client,
            "lease": self.id,
            "since": iso(self.since),
            "held_s": round(time.time() - self.since, 1),
            "last_seen_s": round(now - self.last_seen, 1),
            "idle_s": round(now - self.last_call, 1),
        }


@dataclass
class _Waiter:
    client: dict[str, Any]
    since: float = field(default_factory=time.time)


@dataclass
class _Session:
    id: int
    conn: Any
    client: dict[str, Any] = field(default_factory=dict)
    lease: _Lease | None = None


# -- board cache ------------------------------------------------------------------------


class _BoardCache:
    """Last commanded board state, refreshed from live reads when the hardware is idle."""

    def __init__(self):
        self.lock = threading.Lock()
        self.connected = False
        self.programmed = False
        self.tx_mask = 0
        self.rx_mask = 0
        self.lo_hz: dict[str, int] = {}
        self.tx_atten_db: dict[str, float] = {}
        self.rx_gain: dict[str, int] = {}
        self.pll_lock: int | str | None = None
        self.readback: dict[str, Any] | None = None
        self.updated: float | None = None

    @property
    def tx_live(self) -> bool:
        return self.tx_mask != 0

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "connected": self.connected,
                "programmed": self.programmed,
                "tx_live": self.tx_live,
                "tx_mask": self.tx_mask,
                "rx_mask": self.rx_mask,
                "tx_enabled": [c.name for c in TX_SINGLE if self.tx_mask & int(c)],
                "rx_enabled": [c.name for c in RX_SINGLE if self.rx_mask & int(c)],
                "lo_hz": dict(self.lo_hz),
                "tx_atten_db": dict(self.tx_atten_db),
                "rx_gain": dict(self.rx_gain),
                "pll_lock": self.pll_lock,
                "updated": iso(self.updated),
            }


_TX_NAMES = [c.name for c in TX_SINGLE]


def _names(channel: Any, singles) -> list[str]:
    return [c.name for c in singles if int(channel) & int(c)]


_SIGNATURES: dict[str, inspect.Signature] = {}


def _bound_arguments(method: str, args: tuple, kwargs: dict) -> dict[str, Any]:
    from .radio import Radio

    sig = _SIGNATURES.get(method)
    if sig is None:
        sig = _SIGNATURES[method] = inspect.signature(getattr(Radio, method))
    try:
        return dict(sig.bind(None, *args, **kwargs).arguments)
    except TypeError:
        return {}


# -- the server -------------------------------------------------------------------------


class HardwareServer:
    """Owns the board; serves leases and forwarded ``Radio`` calls (docs/hw_server.md)."""

    def __init__(
        self,
        config: Config,
        backend: Callable[[Config], Any],
        *,
        backend_name: str = "",
        program: bool = True,
        supervised: bool = False,
        restarts: int = 0,
        last_error: str | None = None,
        log: JsonlLog | None = None,
        echo: bool = True,
        token: str = "",
    ):
        self.config = config
        self.token = token  # set by the supervisor; echoed in every ping
        self.echo = echo  # print ready / stopped lines for the console
        self.settings: ServerConfig = config.server
        self._backend = backend
        self.backend_name = backend_name or getattr(backend, "__qualname__", str(backend))
        self._program = program
        self.supervised = supervised
        self.restarts = restarts
        self.last_error = last_error
        #: Identity of the board's current programming (see :meth:`_note_programmed`).
        self.programming: dict[str, Any] | None = None
        #: What each TX channel's playback RAM holds (see :meth:`_perform_tx`).
        self._ram_lock = threading.Lock()
        self.tx_ram: dict[str, dict[str, Any]] = {}
        self.tx_loads = {"loads": 0, "skipped": 0, "last_load_s": None, "load_s_total": 0.0}
        self.log = log or JsonlLog(self.settings.log_path, "server")
        self.radio: Any = None
        self.hw = HardwareThread(self.settings.timeout_for)
        self.cache = _BoardCache()
        self.methods = frozenset(public_methods())
        self.state = "starting"  # starting | ready | stopping
        self.exit_code = EXIT_OK
        self.port: int | None = None
        self.started = time.time()
        self._cond = threading.Condition()
        self._owner: _Lease | None = None
        self._ending = False  # the owner's end-of-lease safe_state has not run yet
        self._waiters: deque[_Waiter] = deque()
        #: Set by ``safe``: no queued job gets the board until ``resume``.
        self._held: dict[str, Any] | None = None
        self._stop = threading.Event()
        #: Set by a signal handler (a plain assignment: Event.set could deadlock there).
        self.signal_stop: str | None = None
        self._sessions = itertools.count(1)
        self._listener: Listener | None = None
        self._authkey = b""
        self._ops: dict[str, Callable[[_Session, dict[str, Any]], Any]] = {
            "hello": self._op_hello,
            "acquire": self._op_acquire,
            "call": self._op_call,
            "heartbeat": self._op_heartbeat,
            "release": self._op_release,
            "status": self._op_status,
            "ping": self._op_ping,
            "safe": self._op_safe,
            "kick": self._op_kick,
            "stop": self._op_stop,
            "resume": self._op_resume,
            "config": lambda _s, _m: self.config,
            "session": self._op_session,
            "loaded": lambda _s, _m: self.tx_ram_status(),
        }

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        """Bind, build the backend, then connect / force safe / program on the hardware thread.

        Raises :class:`PortInUse` (another server) or :class:`BackendError`
        before anything touches the board.
        """
        try:
            self._listener = Listener((HOST, self.settings.port), family="AF_INET", backlog=16)
        except OSError as exc:
            raise PortInUse(f"{HOST}:{self.settings.port} is in use ({exc})") from exc
        self.port = self._listener.address[1]
        try:
            self._authkey = ensure_authkey(self.settings)
            self.radio = self._backend(self.config)
            getattr(self.radio, "bridge", None)  # load the DLL now: a bad install fails here
        except Exception as exc:
            self._listener.close()
            raise BackendError(f"backend {self.backend_name} failed: {exc!r}") from exc
        self.log.write(
            "start",
            port=self.port,
            backend=self.backend_name,
            program=self._program,
            supervised=self.supervised,
            restarts=self.restarts,
            board=f"{self.config.board.ip}:{self.config.board.port}",
        )
        self.hw.start()
        self.hw.submit("startup", self._startup, priority=URGENT)
        threading.Thread(target=self._monitor, name="adrvtrx-monitor", daemon=True).start()

    def _startup(self) -> None:
        try:
            self.radio.connect()
            self.radio.force_safe()  # never trust the state a previous process left
            self._refresh("force_safe")
            if self._program:
                self.radio.program()
                self._refresh("program")
                self._clear_tx_ram("program")
                self._note_programmed()
            else:
                self.programming = self._read_programming()  # the board keeps its last one
        except Exception as exc:
            self.last_error = f"startup failed: {exc!r}"
            self.log.write("startup_failed", error=repr(exc))
            self._echo(f"adrvtrx-server: start-up failed: {exc!r}")
            self.request_stop("start-up failed", exit_code=EXIT_STARTUP_FAILED)
            return
        with self._cond:
            if self.state == "starting":
                self.state = "ready"
            self._cond.notify_all()
        self.log.write("ready")
        self._echo(
            f"adrvtrx-server: ready on {HOST}:{self.port} (pid {os.getpid()}, "
            f"board {self.config.board.ip}:{self.config.board.port})"
        )

    def serve_forever(self) -> int:
        """Accept clients until :meth:`request_stop`, then stop safely. Returns the exit code."""
        accept = threading.Thread(target=self._accept_loop, name="adrvtrx-accept", daemon=True)
        accept.start()
        try:
            while not self._stop.wait(POLL_S):  # short waits keep Ctrl+C responsive
                if self.signal_stop is not None:
                    self.request_stop(self.signal_stop)
        finally:
            self._shutdown()
        return self.exit_code

    def request_stop(self, reason: str, *, exit_code: int = EXIT_OK) -> None:
        """Ask the server to stop: leases end, TX safe, disconnect, exit."""
        with self._cond:
            if self._stop.is_set():
                return
            if exit_code != EXIT_OK:
                self.exit_code = exit_code
            self.state = "stopping"
            self._cond.notify_all()
        self.log.write("stop_requested", reason=reason)
        self._stop.set()
        if self.port is not None:  # wake the accept loop
            try:
                socket.create_connection((HOST, self.port), timeout=1).close()
            except OSError:
                pass

    def _shutdown(self) -> None:
        with self._cond:
            self.state = "stopping"
            owner = self._owner
            if owner is not None and owner.ended is None:
                owner.ended, owner.revoked = "server stopping", True
            self._owner = None
            self._ending = False
            self._cond.notify_all()
        job = self.hw.submit("shutdown", self._safe_and_disconnect, priority=URGENT)
        if not job.wait(self.settings.timeout_for("shutdown")) or job.error is not None:
            # safe_state did not run (stuck call) or failed: a non-zero exit makes
            # the supervisor force TX safe from a fresh process.
            error = "timed out" if job.error is None else repr(job.error)
            self.last_error = f"shutdown safe_state failed: {error}"
            self.log.write("shutdown_failed", error=error)
            if self.exit_code == EXIT_OK:
                self.exit_code = EXIT_CRASH
        self.hw.stop()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
        self.log.write("stopped", exit_code=self.exit_code)
        self._echo(f"adrvtrx-server: stopped, exit {self.exit_code}")

    def _echo(self, text: str) -> None:
        if self.echo:
            print(text, flush=True)

    def _safe_and_disconnect(self) -> None:
        if self.radio is None:
            return
        try:
            self.radio.safe_state()
            self._refresh("safe_state")
        finally:
            self.radio.disconnect()
            with self.cache.lock:
                self.cache.connected = False

    def _safe_only(self) -> None:
        if self.radio is not None:
            self.radio.safe_state()
            self._refresh("safe_state")

    # -- connections -----------------------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn = self._listener.accept()
            except OSError:
                if self._stop.is_set():
                    return
                time.sleep(POLL_S)
                continue
            if self._stop.is_set():
                conn.close()
                return
            session = _Session(next(self._sessions), conn)
            threading.Thread(
                target=self._serve_connection, args=(session,), name="adrvtrx-conn", daemon=True
            ).start()

    def _serve_connection(self, session: _Session) -> None:
        conn = session.conn
        try:
            deliver_challenge(conn, self._authkey)
            answer_challenge(conn, self._authkey)
        except Exception:  # noqa: BLE001 - wrong key, port scan, or the wake-up connection
            conn.close()
            return
        try:
            while True:
                try:
                    message = conn.recv()
                except (EOFError, OSError):
                    raise
                except Exception as exc:  # noqa: BLE001 - unreadable message
                    conn.send({"ok": False, "error": error_payload(exc, "protocol")})
                    continue
                reply = self._dispatch(session, message)
                try:
                    conn.send(reply)
                except (EOFError, OSError):
                    raise
                except Exception as exc:  # noqa: BLE001 - result could not be pickled
                    conn.send({"ok": False, "error": error_payload(exc, "remote")})
        except (EOFError, OSError, _ClientGone):
            pass
        finally:
            lease = session.lease
            if lease is not None and lease.ended is None:
                self._end_lease(lease, "client connection lost", revoked=False)
            try:
                conn.close()
            except OSError:
                pass

    def _dispatch(self, session: _Session, message: Any) -> dict[str, Any]:
        if not isinstance(message, dict) or message.get("op") not in self._ops:
            op = message.get("op") if isinstance(message, dict) else type(message).__name__
            return {"ok": False, "error": _OpError("protocol", f"unknown op {op!r}").payload}
        try:
            return {"ok": True, "result": self._ops[message["op"]](session, message)}
        except _ClientGone:
            raise
        except _OpError as exc:
            return {"ok": False, "error": exc.payload}
        except Exception as exc:  # noqa: BLE001 - report it to the client
            return {"ok": False, "error": error_payload(exc)}

    # -- ops ---------------------------------------------------------------------------

    def _op_hello(self, session: _Session, message: dict[str, Any]) -> dict[str, Any]:
        session.client = {
            "name": str(message.get("name", "?")),
            "pid": message.get("pid"),
            "host": str(message.get("host", "?")),
        }
        return {
            "protocol": PROTOCOL,
            "server_pid": os.getpid(),
            "methods": sorted(self.methods),
            "heartbeat_timeout_s": self.settings.heartbeat_timeout_s,
        }

    def _busy_message(self) -> str:
        owner = self._owner
        if self.state == "starting":
            return "adrvtrx-server is starting (programming the board)"
        if self._held is not None and owner is None:
            return (
                f"board held since {self._held['since']} by `adrvtrx-server safe` "
                f"({len(self._waiters)} waiting; `adrvtrx-server resume` releases it)"
            )
        if owner is None:
            return f"board busy ({len(self._waiters)} job(s) waiting)"
        c = owner.client
        return (
            f'board owned by "{c.get("name")}", pid {c.get("pid")}, host {c.get("host")}, '
            f"since {iso(owner.since)} ({len(self._waiters)} waiting)"
        )

    def _op_acquire(self, session: _Session, message: dict[str, Any]) -> dict[str, Any]:
        if session.lease is not None and session.lease.ended is None:
            raise _OpError("protocol", "this connection already holds the board")
        wait = bool(message.get("wait", True))
        timeout = message.get("timeout")
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        me = _Waiter(session.client)
        with self._cond:
            if self.state == "stopping":
                raise _OpError("busy", "adrvtrx-server is stopping")
            free = (
                self.state == "ready"
                and self._owner is None
                and not self._ending
                and self._held is None
            )
            if not wait and not (free and not self._waiters):
                raise _OpError("busy", self._busy_message())
            self._waiters.append(me)
        told = False
        try:
            while True:
                info = None
                with self._cond:
                    if self.state == "stopping":
                        raise _OpError("busy", "adrvtrx-server is stopping")
                    if (
                        self._waiters[0] is me
                        and self.state == "ready"
                        and self._owner is None
                        and not self._ending
                        and self._held is None
                    ):
                        self._waiters.popleft()
                        now = time.monotonic()
                        lease = _Lease(uuid.uuid4().hex[:12], session.client, time.time(), now, now)
                        self._owner = session.lease = lease
                        break
                    if not told:
                        info = {
                            "type": "info",
                            "position": self._waiters.index(me) + 1,
                            "message": self._busy_message(),
                        }
                    self._cond.wait(POLL_S)
                if info is not None:
                    session.conn.send(info)
                    told = True
                    self.log.write("queued", client=session.client, position=info["position"])
                if deadline is not None and time.monotonic() >= deadline:
                    with self._cond:
                        msg = self._busy_message()
                    raise _OpError("busy", f"timed out after {float(timeout):g} s: {msg}")
                if session.conn.poll(0):  # a waiting client only speaks to give up
                    try:
                        session.conn.recv()
                    except (EOFError, OSError):
                        pass
                    raise _ClientGone()
        except BaseException:
            with self._cond:
                if me in self._waiters:
                    self._waiters.remove(me)
                self._cond.notify_all()
            raise
        self.log.write("acquire", client=session.client, lease=lease.id)
        return {
            "lease": lease.id,
            "config": self.config,
            "heartbeat_timeout_s": self.settings.heartbeat_timeout_s,
            "release_timeout_s": self.settings.timeout_for("safe_state") + 5.0,
            "programming": self.programming,
        }

    def _active_lease(self, session: _Session) -> _Lease:
        lease = session.lease
        if lease is None:
            raise _OpError("protocol", "this connection does not hold the board (acquire first)")
        if lease.ended is not None:
            raise _OpError("revoked" if lease.revoked else "protocol", lease.ended)
        return lease

    def _op_call(self, session: _Session, message: dict[str, Any]) -> Any:
        lease = self._active_lease(session)
        method = message.get("method")
        if method not in self.methods:
            raise AttributeError(f"{method!r} is not a public Radio method")
        args = tuple(message.get("args", ()))
        kwargs = dict(message.get("kwargs", {}))

        def check() -> None:
            if lease.ended is not None:
                raise LeaseRevoked(lease.ended)

        with self._cond:
            lease.in_flight += 1
            lease.last_seen = lease.last_call = time.monotonic()
        try:
            job = self.hw.submit(
                method, lambda: self._invoke(method, args, kwargs, session.client), check=check
            )
            job.wait()
        finally:
            with self._cond:
                lease.in_flight -= 1
                lease.last_seen = lease.last_call = time.monotonic()
        error = job.error
        self.log.write(
            "call",
            client=session.client,
            lease=lease.id,
            method=method,
            args=summarize(args),
            kwargs=summarize(kwargs),
            dur_s=round(job.duration_s, 4),
            ok=error is None,
            error=None if error is None else repr(error),
            result=None if error is not None else summarize(job.result),
        )
        if isinstance(error, LeaseRevoked):
            raise _OpError("revoked", lease.ended or str(error))
        if error is not None:
            self.last_error = f"{method}: {error!r}"
            raise error
        return job.result

    def _invoke(
        self, method: str, args: tuple, kwargs: dict, client: dict[str, Any] | None = None
    ) -> Any:
        """Run one forwarded call on the hardware thread."""
        radio = self.radio
        if method == "perform_tx":
            result = self._perform_tx(args, kwargs, client or {})
        elif method == "disconnect":
            # The server keeps its board connection for the next job; a client's
            # disconnect only leaves TX safe.
            radio.safe_state()
            result = None
            method = "safe_state"
        elif method == "connect":
            is_connected = getattr(radio, "is_connected", None)
            if is_connected is not None and not is_connected():
                self._clear_tx_ram("reconnect")
                radio.connect()
                radio.force_safe()
                method = "force_safe"
            result = None
        else:
            result = getattr(radio, method)(*args, **kwargs)
        self._refresh(method, args, kwargs, result)
        if method == "program":
            self._clear_tx_ram("program")
            self._note_programmed()
        if method == "perform_rx":
            from .capture import capture_arrays

            result = capture_arrays(result)
        return result

    # -- TX playback RAM ----------------------------------------------------------------

    def _perform_tx(self, args: tuple, kwargs: dict, client: dict[str, Any]) -> None:
        """``perform_tx``, skipping the load when the same buffers are already in TX RAM.

        A skipped load only restarts playback (TX disabled, then enabled for the
        mask), which assumes the RAM survives TX disable (bench-checked, docs
        section 10). It needs the same buffers on every channel of the mask, the
        same trigger and continuous mode, and nothing in between that cleared the
        RAM (program, reconnect, server restart).
        """
        from ._enums import TxTrigSource

        a = _bound_arguments("perform_tx", args, kwargs)
        mask = int(a.get("channel_mask", 0))
        mode = {
            "trig": int(a.get("trig", TxTrigSource.IMMEDIATE)),
            "continuous": bool(a.get("continuous", True)),
        }
        ids = tx_buffer_ids(a.get("tx_data", ()))
        names = [c.name for c in TX_SINGLE if mask & int(c)]
        if self.settings.skip_identical_tx_load and ids is not None and names:
            same = True
            with self._ram_lock:
                for k, ch in enumerate(TX_SINGLE):
                    rec = self.tx_ram.get(ch.name)
                    if ch.name in names and (
                        rec is None
                        or (rec["hash"], rec["n"]) != (ids[k]["hash"], ids[k]["n"])
                        or (rec["trig"], rec["continuous"]) != (mode["trig"], mode["continuous"])
                    ):
                        same = False
            if same:
                self.radio.disable_tx()
                self.radio.enable_tx(mask)
                if hasattr(self.radio, "_tx_live"):
                    self.radio._tx_live = True
                with self._ram_lock:
                    self.tx_loads["skipped"] += 1
                self.log.write("tx_load_skipped", client=client, channels=names, mask=mask)
                return None
        t0 = time.perf_counter()
        try:
            self.radio.perform_tx(*args, **kwargs)
        except Exception:
            self._clear_tx_ram("load failed")
            raise
        load_s = time.perf_counter() - t0
        record = {
            "loaded_at": iso(time.time()),
            "job": client.get("name"),
            "pid": client.get("pid"),
            "host": client.get("host"),
            "load_s": round(load_s, 4),
            **mode,
        }
        with self._ram_lock:
            if ids is None:
                self.tx_ram.clear()
            else:
                for k, ch in enumerate(TX_SINGLE):
                    self.tx_ram[ch.name] = {**ids[k], "in_mask": bool(mask & int(ch)), **record}
            self.tx_loads["loads"] += 1
            self.tx_loads["last_load_s"] = round(load_s, 4)
            self.tx_loads["load_s_total"] = round(self.tx_loads["load_s_total"] + load_s, 4)
        self.log.write(
            "tx_load",
            client=client,
            channels=names,
            mask=mask,
            load_s=round(load_s, 4),
            hashes=None if ids is None else {n: ids[k]["hash"] for k, n in enumerate(_TX_NAMES)},
        )
        return None

    def _clear_tx_ram(self, reason: str) -> None:
        with self._ram_lock:
            had = bool(self.tx_ram)
            self.tx_ram.clear()
        if had:
            self.log.write("tx_ram_cleared", reason=reason)

    def tx_ram_status(self) -> dict[str, Any]:
        """What each TX channel's RAM holds, and load / skipped-load counts."""
        with self._ram_lock:
            return {
                "skip_identical": bool(self.settings.skip_identical_tx_load),
                **dict(self.tx_loads),
                "channels": {name: dict(rec) for name, rec in self.tx_ram.items()},
            }

    # -- programming identity ------------------------------------------------------------

    @property
    def _programming_path(self) -> Path:
        board = f"{self.config.board.ip}_{self.config.board.port}".replace(":", "_")
        return self.settings.state_path / f"programming-{board}.json"

    def _read_programming(self) -> dict[str, Any] | None:
        try:
            return json.loads(self._programming_path.read_text())
        except (OSError, ValueError):
            return None

    def _note_programmed(self) -> None:
        """Record a (re)programming of the board: calibrations ran again.

        ``program_count`` is kept per board in the state directory, so it keeps
        counting across server restarts (watchdog included); ``program_id`` is new
        on every programming. Clients record these with their measurements to
        tell whether the board was re-programmed between two jobs.
        """
        previous = self._read_programming() or {}
        info = {
            "program_count": int(previous.get("program_count", 0)) + 1,
            "program_id": uuid.uuid4().hex[:12],
            "programmed_at": iso(time.time()),
            "profile": Path(self.config.profile_name).name,
            "board": f"{self.config.board.ip}:{self.config.board.port}",
            "server_pid": os.getpid(),
        }
        path = self._programming_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(info))
            os.replace(tmp, path)
        except OSError as exc:
            self.log.write("programming_not_saved", error=repr(exc))
        self.programming = info
        self.log.write("programmed", **info)

    def _op_session(self, session: _Session, _message: dict[str, Any]) -> dict[str, Any]:
        lease = session.lease
        return {
            "job": session.client.get("name"),
            "lease": None if lease is None else lease.id,
            "active": lease is not None and lease.ended is None,
            "server_pid": os.getpid(),
            "server_started": iso(self.started),
            "restarts": self.restarts,
            "programming": self.programming,
        }

    def _op_heartbeat(self, session: _Session, _message: dict[str, Any]) -> dict[str, Any]:
        lease = self._active_lease(session)
        with self._cond:
            lease.last_seen = time.monotonic()
        return {"tx_live": self.cache.tx_live}

    def _op_release(self, session: _Session, _message: dict[str, Any]) -> dict[str, Any]:
        lease = session.lease
        if lease is None or lease.ended is not None:
            return {"released": False}
        job = self._end_lease(lease, "released", revoked=False)
        done = job.wait(self.settings.timeout_for("safe_state")) if job is not None else True
        return {"released": True, "safe": bool(done and (job is None or job.error is None))}

    def _end_lease(self, lease: _Lease, reason: str, *, revoked: bool) -> _HwJob | None:
        """End ``lease``; if it owns the board, queue safe_state, then free the board."""
        with self._cond:
            if lease.ended is not None:
                return None
            lease.ended, lease.revoked = reason, revoked
            if self._owner is not lease:
                return None
            self._ending = True
        self.log.write(
            "revoke" if revoked else "release", client=lease.client, lease=lease.id, reason=reason
        )

        def safe_then_free() -> None:
            try:
                self._safe_only()
            finally:
                with self._cond:
                    if self._owner is lease:
                        self._owner = None
                        self._ending = False
                    self._cond.notify_all()

        return self.hw.submit("safe_state", safe_then_free, priority=URGENT)

    def _current_owner(self) -> _Lease | None:
        with self._cond:
            owner = self._owner
            return owner if owner is not None and owner.ended is None else None

    def _op_safe(self, session: _Session, message: dict[str, Any]) -> dict[str, Any]:
        who = describe_client(session.client)
        with self._cond:  # hold the queue first: no waiting job may slip in
            if self._held is None:
                self._held = {"since": iso(time.time()), "by": "safe", "client": session.client}
            held = dict(self._held)
        owner = self._current_owner()
        job = None
        if owner is not None:
            job = self._end_lease(
                owner, f"TX forced safe by `adrvtrx-server safe` from {who}", revoked=True
            )
        if job is None:
            job = self.hw.submit("safe_state", self._safe_only, priority=URGENT)
        done = job.wait(float(message.get("wait_s", 10.0)))
        result = {
            "done": done and job.error is None,
            "error": None if job.error is None else repr(job.error),
            "revoked": None if owner is None else owner.info(),
            "busy_with": self.hw.busy_info(),
            "held": held,
        }
        self.log.write("safe", client=session.client, **summarize(result))
        return result

    def _op_resume(self, session: _Session, _message: dict[str, Any]) -> dict[str, Any]:
        with self._cond:
            held, self._held = self._held, None
            waiting = len(self._waiters)
            self._cond.notify_all()
        result = {"resumed": held is not None, "held": held, "waiting": waiting}
        self.log.write("resume", client=session.client, **summarize(result))
        return result

    def _op_kick(self, session: _Session, message: dict[str, Any]) -> dict[str, Any]:
        owner = self._current_owner()
        if owner is None:
            raise _OpError("protocol", "no job holds the board")
        who = describe_client(session.client)
        job = self._end_lease(owner, f"kicked by `adrvtrx-server kick` from {who}", revoked=True)
        done = job.wait(float(message.get("wait_s", 10.0))) if job is not None else True
        result = {"kicked": owner.info(), "done": done, "busy_with": self.hw.busy_info()}
        self.log.write("kick", client=session.client, **summarize(result))
        return result

    def _op_stop(self, session: _Session, _message: dict[str, Any]) -> dict[str, Any]:
        self.request_stop(f"stop from {describe_client(session.client)}")
        return {"stopping": True, "pid": os.getpid()}

    def _op_ping(self, _session: _Session, _message: dict[str, Any]) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "token": self.token,
            "state": self.state,
            "hw": self.hw.busy_info(),
        }

    def _op_status(self, session: _Session, message: dict[str, Any]) -> dict[str, Any]:
        st = self.status(live=bool(message.get("live", True)))
        self.log.write("status", client=session.client, source=st["source"])
        return st

    # -- status / cache ------------------------------------------------------------------

    def status(self, *, live: bool = True) -> dict[str, Any]:
        source = "cached"
        if live and self.state == "ready" and self.hw.idle():
            job = self.hw.submit("status", self._live_status)
            if job.wait(5.0) and job.error is None:
                source = "live"
        with self._cond:
            held = None if self._held is None else dict(self._held)
            owner = self._owner.info() if self._owner is not None else None
            waiting = [
                {**w.client, "waiting_s": round(time.time() - w.since, 1)} for w in self._waiters
            ]
        with self.cache.lock:
            readback = self.cache.readback
        return {
            "server": {
                "pid": os.getpid(),
                "host": HOST,
                "port": self.port,
                "started": iso(self.started),
                "uptime_s": round(time.time() - self.started, 1),
                "restarts": self.restarts,
                "last_error": self.last_error,
                "supervised": self.supervised,
                "backend": self.backend_name,
            },
            "state": self.state,
            "held": held,
            "owner": owner,
            "queue": waiting,
            "hardware": self.hw.busy_info(),
            "board": self.cache.snapshot(),
            "programming": self.programming,
            "tx_ram": self.tx_ram_status(),
            "source": source,
            "readback": readback,
        }

    def _live_status(self) -> None:
        result = self.radio.status()
        self._refresh("status", result=result)

    def _refresh(self, method: str, args: tuple = (), kwargs: dict | None = None, result=None):
        """Update the cache after a hardware call (runs on the hardware thread)."""
        radio, c, cfg = self.radio, self.cache, self.config
        a = _bound_arguments(method, args, kwargs or {}) if (args or kwargs) else {}
        with c.lock:
            c.tx_mask = int(getattr(radio, "_en_tx", c.tx_mask))
            c.rx_mask = int(getattr(radio, "_en_rx", c.rx_mask))
            c.connected = bool(getattr(radio, "_connected", True))
            if method in ("safe_state", "force_safe"):
                for ch in TX_SINGLE:
                    c.tx_atten_db[ch.name] = MAX_TX_ATTEN_DB
            elif method == "program":
                c.programmed = True
                c.lo_hz.update({"LO1": cfg.lo.lo1_hz, "LO2": cfg.lo.lo2_hz})
                for ch in TX_SINGLE:
                    if cfg.channels.tx_init_mask & int(ch):
                        c.tx_atten_db[ch.name] = cfg.levels.tx_atten_for(ch.name.lower())
                for ch in RX_SINGLE:
                    if cfg.channels.rx_init_mask & int(ch):
                        c.rx_gain[ch.name] = cfg.levels.rx_gain_for(ch.name.lower())
            elif method == "set_tx_atten" and a:
                for name in _names(a["channel"], TX_SINGLE):
                    c.tx_atten_db[name] = round(float(a["atten_db"]), 2)
            elif method == "set_rx_gain" and a:
                for name in _names(a["channel"], RX_SINGLE):
                    c.rx_gain[name] = int(a["gain_index"])
            elif method in ("set_lo", "retune_lo") and a:
                c.lo_hz[str(a["pll"])] = int(a["freq_hz"])
                if method == "retune_lo" and result is not None:
                    c.pll_lock = int(result)
            elif method == "get_lo" and a and result is not None:
                c.lo_hz[str(a["pll"])] = int(result)
            elif method == "pll_lock_status" and result is not None:
                c.pll_lock = int(result)
            elif method == "status" and isinstance(result, dict):
                c.readback = result
                for key, pll in (("lo1_hz", "LO1"), ("lo2_hz", "LO2")):
                    if result.get(key) is not None:
                        c.lo_hz[pll] = int(result[key])
                c.tx_atten_db.update(result.get("tx_atten_db") or {})
                c.pll_lock = result.get("pll_lock", c.pll_lock)
                c.connected = bool(result.get("connected", c.connected))
            c.updated = time.time()

    # -- heartbeat monitor ------------------------------------------------------------------

    def _monitor(self) -> None:
        timeout = float(self.settings.heartbeat_timeout_s)
        idle_limit = float(self.settings.idle_timeout_s)
        while not self._stop.wait(min(0.25, timeout / 4)):
            with self._cond:
                lease = self._owner
                if lease is None or lease.ended is not None or lease.in_flight:
                    continue
                now = time.monotonic()
                silent, idle = now - lease.last_seen, now - lease.last_call
            if silent > timeout and self.cache.tx_live:
                self._end_lease(
                    lease, f"no heartbeat for {silent:.1f} s while TX was enabled", revoked=True
                )
            elif idle_limit > 0 and idle > idle_limit:
                reason = f"idle: no hardware call for {idle:.0f} s (idle_timeout_s {idle_limit:g})"
                self.log.write("idle_release", client=lease.client, lease=lease.id, idle_s=idle)
                self._end_lease(lease, reason, revoked=True)


# -- fresh-process force safe ---------------------------------------------------------------


def force_safe_fresh(
    config: Config, backend: Callable[[Config], Any], *, retries: int, delay_s: float
) -> dict[str, Any]:
    """Connect with a new backend instance, force TX safe, read the TX mask back.

    This is what the watchdog runs in a fresh process after a crash, and what
    ``adrvtrx-server safe --direct`` runs. Never raises.
    """
    last = "no attempt"
    for attempt in range(1, max(1, int(retries)) + 1):
        radio = None
        try:
            radio = backend(config)
            radio.connect()
            radio.force_safe()
            tx_mask: int | None = None
            atten: dict[str, float] = {}
            try:
                tx_mask = int(radio.rx_tx_enable_get()[1])
                atten = {c.name: round(radio.get_tx_atten(c), 2) for c in TX_SINGLE}
            except Exception:  # noqa: BLE001 - readback is a bonus
                pass
            return {
                "ok": tx_mask in (None, 0),
                "verified": tx_mask == 0,
                "tx_mask": tx_mask,
                "tx_atten_db": atten,
                "attempt": attempt,
            }
        except Exception as exc:  # noqa: BLE001 - retry, then report
            last = repr(exc)
            if attempt < retries:
                time.sleep(delay_s)
        finally:
            if radio is not None:
                try:
                    radio.disconnect()
                except Exception:  # noqa: BLE001
                    pass
    return {"ok": False, "verified": False, "error": last, "attempt": retries}


# -- supervisor ----------------------------------------------------------------------------


class _InstanceLock:
    """An OS file lock, released by the OS when the holder dies."""

    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            return False
        self._fh = fh
        return True

    def release(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


UNKNOWN_TX = "TX STATE UNKNOWN - switch off the PA supply"


class Supervisor:
    """``adrvtrx-server run``: spawn the server, watch it, force safe and restart on failure."""

    def __init__(
        self,
        config: Config,
        *,
        config_path: str | None,
        backend: str,
        program: bool = True,
    ):
        self.config = config
        self.settings = config.server
        self.config_path = config_path
        self.backend = backend
        self.program = program
        self.log = JsonlLog(self.settings.log_path, "supervisor")
        self.child: subprocess.Popen | None = None

    def _cmd(self, sub: str, *extra: str) -> list[str]:
        cmd = [sys.executable, "-m", "adrvtrx.server", sub, "--backend", self.backend]
        if self.config_path:
            cmd += ["--config", self.config_path]
        return cmd + list(extra)

    def _say(self, text: str) -> None:
        print(f"adrvtrx-server[supervisor]: {text}", flush=True)

    def run(self) -> int:
        lock = _InstanceLock(self.settings.state_path / f"server-{self.settings.port}.lock")
        if not lock.acquire():
            self._say(
                f"refusing to start: another `adrvtrx-server run` holds port "
                f"{self.settings.port} (see `adrvtrx-server status`)"
            )
            return EXIT_PORT_IN_USE
        try:
            if _answers_ping(self.settings):
                self._say(
                    f"refusing to start: a server already answers on port {self.settings.port}"
                )
                return EXIT_PORT_IN_USE
            ensure_authkey(self.settings)
            return self._loop()
        finally:
            lock.release()

    def _loop(self) -> int:
        restarts: deque[float] = deque()
        last_error: str | None = None
        ever_ready = False
        while True:
            # The pid of a venv's python.exe on Windows is a launcher's, not the
            # server's: pings are matched to this spawn by a token instead.
            token = uuid.uuid4().hex
            extra = ["--supervised", "--restarts", str(len(restarts)), "--token", token]
            if not self.program:
                extra.append("--no-program")
            if last_error:
                extra += ["--last-error", last_error]
            self.child = subprocess.Popen(self._cmd("_child", *extra), stdin=subprocess.PIPE)
            self.log.write("spawn", child=self.child.pid, restarts=len(restarts))
            outcome, detail, ready = self._watch(self.child, token)
            ever_ready = ever_ready or ready
            code = self.child.returncode
            self.log.write("child_end", outcome=outcome, detail=detail, exit_code=code)
            if outcome == "stopped":
                self._say("server stopped")
                return EXIT_OK
            if outcome == "refused" and not restarts:
                self._say("server refused to start (port in use)")
                return EXIT_PORT_IN_USE
            if outcome == "bad_config":
                self._say("server could not start: bad config or backend (board untouched)")
                return EXIT_BAD_CONFIG
            self._say(f"server {outcome}: {detail}")
            if outcome == "stuck":
                self._kill_child()
            safe = self._force_safe()
            if outcome == "interrupted":
                return EXIT_OK if safe.get("ok") else EXIT_CRASH
            if not ever_ready:
                self._say("server never became ready; not restarting")
                return EXIT_STARTUP_FAILED
            now = time.monotonic()
            while restarts and now - restarts[0] > self.settings.restart_window_s:
                restarts.popleft()
            if len(restarts) >= self.settings.restart_limit:
                self._say(
                    f"giving up: {len(restarts)} restarts within "
                    f"{self.settings.restart_window_s:g} s"
                    + ("" if safe.get("ok") else f". {UNKNOWN_TX}")
                )
                self.log.write("give_up", restarts=len(restarts))
                return EXIT_CRASH
            backoff = self.settings.restart_backoff_s * (2 ** len(restarts))
            restarts.append(now)
            last_error = f"{outcome}: {detail}"
            self._say(f"restarting in {backoff:g} s (restart {len(restarts)})")
            self.log.write("restart", restarts=len(restarts), backoff_s=backoff)
            if not _sleep(backoff):
                return EXIT_OK

    def _watch(self, child: subprocess.Popen, token: str) -> tuple[str, str, bool]:
        s = self.settings
        spawned = time.monotonic()
        last_ok: float | None = None
        ready = False
        conn = None
        try:
            while True:
                code = child.poll()
                if code is not None:
                    return _classify_exit(code) + (ready,)
                reply = None
                try:
                    if conn is None:
                        conn = open_connection(s, timeout=max(2.0, s.ping_interval_s * 2))
                    reply = request(conn, {"op": "ping"}, timeout=max(2.0, s.ping_interval_s * 2))
                except Exception:  # noqa: BLE001 - not listening yet, or not answering
                    if conn is not None:
                        _close(conn)
                    conn = None
                now = time.monotonic()
                if reply is not None and reply.get("token") == token:
                    last_ok = now
                    ready = ready or reply.get("state") == "ready"
                    hw = reply.get("hw")
                    if hw and hw["busy_s"] > hw["timeout_s"]:
                        return (
                            "stuck",
                            f"{hw['method']} running {hw['busy_s']:.0f} s "
                            f"(limit {hw['timeout_s']:g} s)",
                            ready,
                        )
                elif last_ok is None and now - spawned > s.start_timeout_s:
                    return "stuck", f"no ping answer {s.start_timeout_s:g} s after start", ready
                elif last_ok is not None and now - last_ok > s.ping_timeout_s:
                    return "stuck", f"no ping answer for {s.ping_timeout_s:g} s", ready
                if not _sleep(s.ping_interval_s):
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            return self._interrupted(child) + (ready,)
        finally:
            if conn is not None:
                _close(conn)

    def _interrupted(self, child: subprocess.Popen) -> tuple[str, str]:
        """Ctrl+C: let the server stop safely (its stdin closes), kill it if it hangs."""
        self._say("Ctrl+C: stopping the server")
        try:
            if child.stdin is not None:
                child.stdin.close()
            child.wait(self.settings.timeout_for("shutdown") + 5.0)
        except (KeyboardInterrupt, subprocess.TimeoutExpired):
            self._kill_child()
        except OSError:
            pass
        if child.poll() is None:
            self._kill_child()
        code = child.returncode
        if code == EXIT_OK:
            return "stopped", "Ctrl+C"
        return "interrupted", f"Ctrl+C, server exit code {code}"

    def _kill_child(self) -> None:
        child = self.child
        if child is None or child.poll() is not None:
            return
        self.log.write("kill", child=child.pid)
        self._say(f"killing server process {child.pid}")
        child.kill()
        try:  # a server orphaned behind a launcher sees EOF and stops itself safely
            if child.stdin is not None:
                child.stdin.close()
        except OSError:
            pass
        try:
            child.wait(10)
        except subprocess.TimeoutExpired:
            pass

    def _force_safe(self) -> dict[str, Any]:
        """Run force_safe in a fresh process with a fresh board connection."""
        self._say("forcing TX safe from a fresh process")
        cmd = self._cmd("_force-safe")
        try:
            out = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self.settings.force_safe_timeout_s,
            )
            lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
            result = json.loads(lines[-1]) if lines else {"ok": False, "error": out.stderr[-500:]}
        except subprocess.TimeoutExpired:
            result = {
                "ok": False,
                "error": f"timed out after {self.settings.force_safe_timeout_s} s",
            }
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": repr(exc)}
        self.log.write("force_safe", **summarize(result))
        if result.get("ok"):
            self._say(
                f"TX forced safe (TX mask {result.get('tx_mask')}, attempt {result.get('attempt')})"
            )
        else:
            self._say(f"force_safe FAILED: {result.get('error')}. {UNKNOWN_TX}")
        return result


def _classify_exit(code: int) -> tuple[str, str]:
    if code == EXIT_OK:
        return "stopped", "exit 0"
    if code == EXIT_PORT_IN_USE:
        return "refused", "port in use"
    if code == EXIT_BAD_CONFIG:
        return "bad_config", "bad config or backend"
    if code == EXIT_STARTUP_FAILED:
        return "crashed", "start-up (connect / program) failed"
    return "crashed", f"exit code {code}"


def _sleep(seconds: float) -> bool:
    """Sleep in short steps; False if interrupted by Ctrl+C."""
    end = time.monotonic() + seconds
    try:
        while time.monotonic() < end:
            time.sleep(min(POLL_S, max(0.0, end - time.monotonic())))
    except KeyboardInterrupt:
        return False
    return True


def _close(conn) -> None:
    try:
        conn.close()
    except OSError:
        pass


def _answers_ping(settings: ServerConfig, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        conn = open_connection(settings, timeout=timeout)
    except ServerUnavailable:
        return None
    try:
        return request(conn, {"op": "ping"}, timeout=timeout)
    except Exception:  # noqa: BLE001
        return None
    finally:
        _close(conn)


# -- status formatting -----------------------------------------------------------------------


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def format_status(st: dict[str, Any]) -> str:
    """Human-readable ``adrvtrx-server status``."""
    srv, board = st["server"], st["board"]
    lines = [
        f"server   pid {srv['pid']}, {srv['host']}:{srv['port']}, up {_duration(srv['uptime_s'])}, "
        f"restarts {srv['restarts']}, {'supervised' if srv['supervised'] else 'NOT supervised'}",
        f"state    {st['state']} (backend {srv['backend']})",
    ]
    held = st.get("held")
    if held:
        lines.append(
            f"held     since {held['since']}, by {held['by']} "
            f"(queue waits for `adrvtrx-server resume`)"
        )
    owner = st["owner"]
    if owner:
        lines.append(
            f"owner    \"{owner['name']}\" pid {owner['pid']} on {owner['host']} since "
            f"{owner['since']} ({_duration(owner['held_s'])}), last heard "
            f"{owner['last_seen_s']:.1f} s ago"
        )
    else:
        lines.append("owner    -")
    waiting = st["queue"]
    if waiting:
        names = ", ".join(f"\"{w['name']}\" pid {w['pid']}" for w in waiting)
        lines.append(f"queue    {len(waiting)} waiting: {names}")
    else:
        lines.append("queue    empty")
    hw = st["hardware"]
    lines.append(
        "hardware idle"
        if not hw
        else f"hardware busy: {hw['method']} for {hw['busy_s']:.1f} s (limit {hw['timeout_s']:g} s)"
    )
    tx = board["tx_enabled"]
    lines.append(
        f"TX       live: {', '.join(tx)} (mask 0x{board['tx_mask']:X})" if tx else "TX       off"
    )
    lo = ", ".join(f"{k} {v} Hz" for k, v in sorted(board["lo_hz"].items())) or "-"
    lines.append(f"LO       {lo}")
    atten = ", ".join(f"{k} {v:.2f}" for k, v in sorted(board["tx_atten_db"].items()))
    lines.append(f"atten    {atten} dB" if atten else "atten    -")
    gains = ", ".join(f"{k} {v}" for k, v in sorted(board["rx_gain"].items())) or "-"
    lines.append(f"gains    {gains}")
    pll = board["pll_lock"]
    pll_text = f"0x{pll:X}" if isinstance(pll, int) else (pll or "-")
    lines.append(f"PLL      {pll_text} [{st['source']}]")
    ram = st.get("tx_ram") or {}
    held = [
        f"{name} {rec['hash'][:8]} n={rec['n']} by \"{rec['job']}\" at {rec['loaded_at']}"
        for name, rec in sorted((ram.get("channels") or {}).items())
        if rec.get("in_mask")
    ]
    lines.append(
        f"TX RAM   {'; '.join(held) or '-'} (loads {ram.get('loads', 0)}, skipped "
        f"{ram.get('skipped', 0)}, skip identical {'on' if ram.get('skip_identical') else 'off'})"
    )
    prog = st.get("programming")
    lines.append(
        f"program  #{prog['program_count']} at {prog['programmed_at']} ({prog['profile']}), "
        f"id {prog['program_id']}"
        if prog
        else "program  -"
    )
    lines.append(f"error    {srv['last_error'] or '-'}")
    return "\n".join(lines)


# -- CLI ---------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adrvtrx-server",
        description="One process owns the ADRV9026; other processes queue for it "
        "(docs/hw_server.md).",
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="TOML config (default: $ADRVTRX_CONFIG or bundled)")
    backend = argparse.ArgumentParser(add_help=False)
    backend.add_argument(
        "--backend",
        default="real",
        help="real (the board, default), fake (simulated, no DLL) or module:attr",
    )
    sub = parser.add_subparsers(dest="cmd", metavar="{run,status,safe,resume,kick,stop}")
    sub.required = True
    p = sub.add_parser("run", parents=[common, backend], help="start the supervised server")
    p.add_argument("--no-program", action="store_true", help="connect + force safe only")
    p = sub.add_parser("status", parents=[common], help="owner, queue, TX, LO, atten, gains")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p = sub.add_parser("safe", parents=[common, backend], help="force TX safe now (skips queue)")
    p.add_argument("--direct", action="store_true", help="server unreachable: connect directly")
    p.add_argument("--force", action="store_true", help="--direct even if a server answers")
    sub.add_parser("kick", parents=[common], help="end the current job (TX safe)")
    sub.add_parser("stop", parents=[common], help="stop the server (TX safe, disconnect)")
    sub.add_parser("resume", parents=[common], help="let queued jobs run again after `safe`")
    p = sub.add_parser("_child", parents=[common, backend])
    p.add_argument("--no-program", action="store_true")
    p.add_argument("--supervised", action="store_true")
    p.add_argument("--restarts", type=int, default=0)
    p.add_argument("--last-error", default=None)
    p.add_argument("--token", default="")
    p = sub.add_parser("_force-safe", parents=[common, backend])
    p.add_argument("--retries", type=int, default=None)
    return parser


def _abs(path: str | None) -> str | None:
    return None if path is None else str(Path(path).resolve())


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except Exception as exc:  # noqa: BLE001 - report and exit
        print(f"adrvtrx-server: cannot load config: {exc}", file=sys.stderr)
        return EXIT_BAD_CONFIG
    if args.cmd == "run":
        try:
            resolve_backend(args.backend)
        except BackendError as exc:
            print(f"adrvtrx-server: {exc}", file=sys.stderr)
            return EXIT_BAD_CONFIG
        sup = Supervisor(
            cfg, config_path=_abs(args.config), backend=args.backend, program=not args.no_program
        )
        return sup.run()
    if args.cmd == "_child":
        code = _child_main(cfg, args)
        # The server already stopped safely (TX safe, disconnected). Skip interpreter
        # finalization: daemon threads may sit in socket or DLL calls.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    if args.cmd == "_force-safe":
        return _force_safe_main(cfg, args)
    if args.cmd == "safe" and args.direct:
        return _safe_direct(cfg, args)
    return _control_main(cfg, args)


def _control_main(cfg: Config, args: argparse.Namespace) -> int:
    from .client import server_kick, server_resume, server_safe, server_status, server_stop

    try:
        if args.cmd == "status":
            st = server_status(cfg)
            print(json.dumps(st, indent=2, default=str) if args.json else format_status(st))
        elif args.cmd == "safe":
            res = server_safe(cfg)
            revoked = res["revoked"]
            print(
                ("TX forced safe" if res["done"] else "safe_state QUEUED (hardware busy)")
                + (f"; ended job \"{revoked['name']}\" pid {revoked['pid']}" if revoked else "")
                + "; queue HELD until `adrvtrx-server resume`"
            )
            if not res["done"]:
                print(f"hardware busy with {res['busy_with']}; the watchdog handles a stuck call")
                return EXIT_CRASH
        elif args.cmd == "kick":
            res = server_kick(cfg)
            k = res["kicked"]
            print(f"kicked \"{k['name']}\" pid {k['pid']} on {k['host']}; TX safe: {res['done']}")
        elif args.cmd == "stop":
            res = server_stop(cfg)
            print(f"server pid {res['pid']} stopping (TX safe, disconnect)")
        elif args.cmd == "resume":
            res = server_resume(cfg)
            if res["resumed"]:
                print(
                    f"queue resumed ({res['waiting']} waiting; held since {res['held']['since']})"
                )
            else:
                print("queue was not held")
    except (ServerUnavailable, ServerConnectionLost, TimeoutError) as exc:
        print(f"adrvtrx-server: {exc}", file=sys.stderr)
        if args.cmd == "safe":
            print("server unreachable: use `adrvtrx-server safe --direct`", file=sys.stderr)
        return EXIT_CRASH
    except RuntimeError as exc:
        print(f"adrvtrx-server: {exc}", file=sys.stderr)
        return EXIT_CRASH
    return EXIT_OK


def _safe_direct(cfg: Config, args: argparse.Namespace) -> int:
    pong = None if args.force else _answers_ping(cfg.server)
    if pong is not None:
        print(
            f"refusing --direct: adrvtrx-server pid {pong.get('pid')} answers on port "
            f"{cfg.server.port}; use `adrvtrx-server safe` (or --force)",
            file=sys.stderr,
        )
        return EXIT_CRASH
    try:
        factory, target = resolve_backend(args.backend)
    except BackendError as exc:
        print(f"adrvtrx-server: {exc}", file=sys.stderr)
        return EXIT_BAD_CONFIG
    print(f"connecting to {cfg.board.ip}:{cfg.board.port} directly ({target}) ...", flush=True)
    res = force_safe_fresh(
        cfg,
        factory,
        retries=cfg.server.connect_retries,
        delay_s=cfg.server.connect_retry_delay_s,
    )
    if res["ok"]:
        print(f"TX forced safe (TX mask read back: {res.get('tx_mask')})")
        return EXIT_OK
    print(f"force_safe FAILED: {res.get('error')}. {UNKNOWN_TX}", file=sys.stderr)
    return EXIT_CRASH


def _force_safe_main(cfg: Config, args: argparse.Namespace) -> int:
    try:
        factory, _target = resolve_backend(args.backend)
    except BackendError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return EXIT_BAD_CONFIG
    retries = args.retries if args.retries is not None else cfg.server.connect_retries
    res = force_safe_fresh(cfg, factory, retries=retries, delay_s=cfg.server.connect_retry_delay_s)
    print(json.dumps(res, default=str), flush=True)
    return EXIT_OK if res["ok"] else EXIT_CRASH


def _child_main(cfg: Config, args: argparse.Namespace) -> int:
    try:
        factory, target = resolve_backend(args.backend)
    except BackendError as exc:
        print(f"adrvtrx-server: {exc}", file=sys.stderr)
        return EXIT_BAD_CONFIG
    server = HardwareServer(
        cfg,
        factory,
        backend_name=target,
        program=not args.no_program,
        supervised=args.supervised,
        restarts=args.restarts,
        last_error=args.last_error,
        token=args.token,
    )
    try:
        server.start()
    except PortInUse as exc:
        print(f"adrvtrx-server: refusing to start: {exc}", file=sys.stderr, flush=True)
        return EXIT_PORT_IN_USE
    except BackendError as exc:
        print(f"adrvtrx-server: {exc}", file=sys.stderr, flush=True)
        return EXIT_BAD_CONFIG

    def on_signal(signum, _frame):
        server.signal_stop = f"signal {signum}"

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, on_signal)
            except (OSError, ValueError):
                pass
    if args.supervised:

        def watch_supervisor() -> None:
            # os.read, not sys.stdin: a daemon thread blocked in the buffered
            # reader holds its lock and crashes interpreter shutdown.
            try:
                while os.read(stdin_fd, 4096):
                    pass
            except OSError:
                pass
            server.request_stop("supervisor gone (stdin closed)")

        stdin_fd = sys.stdin.fileno()

        threading.Thread(target=watch_supervisor, name="adrvtrx-parent", daemon=True).start()
    try:
        return server.serve_forever()
    except Exception as exc:  # noqa: BLE001 - serve_forever already stopped safely
        print(f"adrvtrx-server: crashed: {exc!r}", file=sys.stderr)
        return EXIT_CRASH


if __name__ == "__main__":  # pragma: no cover - entry point for the supervisor's children
    sys.exit(main())
