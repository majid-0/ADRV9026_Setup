"""A simulated board behind the real :class:`~adrvtrx.radio.Radio` (no DLL, no pythonnet).

:class:`FakeRadio` is ``Radio`` over :class:`FakeBridge`, a stand-in for the
TES DLL surface that ``Radio`` calls (connect, program, enables, attenuation,
gain, LO, PerformTx, PerformRx). Every ``Radio`` code path runs unchanged; only
the bottom layer is simulated. It is the hardware server's ``--backend fake``
for dry runs, and the backend of the server tests.

The model:

* Register state: connection, programmed flag, Rx/TX enable masks, TX
  attenuation, Rx/ORx gain index and LO frequencies. ``TxAttenSet`` is
  rejected until the board is programmed, like the real device.
* ``PerformTx`` must get exactly eight arrays built by the bridge (the .NET
  ``Int32[]`` stand-in); a numpy array that skipped the conversion is an error.
* ``PerformRx`` returns the full ``rxInitChannelMask`` set ``[ch0_I, ch0_Q, ...]``.
  An enabled ORx input whose TX (``[tx_to_orx]``) is enabled carries that TX's
  waveform through a Rapp PA (``p = 2``) driven by ``10**(-atten/20)``, a small
  delay, the ORx gain ``(index - 210) * 0.5`` dB, noise and the ADC rail. Every
  capture starts at a random point of the loop (IMMEDIATE trigger).

Set ``ADRVTRX_FAKE_STATE`` to a JSON file to keep the register state across
processes (the real board keeps its registers when a client dies) and to log
every register write to ``<file>.events.jsonl``. Creating ``<file>.refuse``
makes ``Connect`` fail. ``ADRVTRX_FAKE_DELAYS`` is a JSON object of seconds
per DLL call (for example ``{"PerformRx": 30}``) to simulate slow or stuck calls.
"""

from __future__ import annotations

import json
import os
import time
import zlib
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._enums import RX_SINGLE, TX_SINGLE, RxChannel, TxChannel
from .config import Config
from .radio import Radio

__all__ = ["FAKE_STATE_ENV", "FAKE_DELAYS_ENV", "FakeBoard", "FakeBridge", "FakeRadio"]

FAKE_STATE_ENV = "ADRVTRX_FAKE_STATE"
FAKE_DELAYS_ENV = "ADRVTRX_FAKE_DELAYS"

#: PA drive at 0 dB attenuation, relative to the Rapp saturation level.
PA_DRIVE = 6.0
#: ORx output at PA saturation and gain index 210, as a fraction of full scale.
ORX_LEVEL = 0.73
DELAY_SAMPLES = 5
NOISE_CODES = 0.5
DEFAULT_RATE_HZ = 10e6
DEFAULT_BITS = 12

_TX_NAMES = [c.name for c in TX_SINGLE]
_RX_NAMES = [c.name for c in RX_SINGLE]


class _Struct:
    """A .NET struct stand-in: nested fields spring into existence on first use."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _Struct()
        setattr(self, name, value)
        return value


class _Types:
    """``Types.<struct_name>()`` -> a fresh :class:`_Struct`."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Struct


class EnumValue(str):
    """A DLL enum member: its ``Type.MEMBER`` name, and a stable int for ``int()``."""

    def __int__(self) -> int:
        return zlib.crc32(self.encode())


class NetArray(list):
    """Stand-in for a .NET ``Int32[]`` built by the bridge."""


class NetArrayList(list):
    """Stand-in for a .NET ``ArrayList`` built by the bridge."""


def _member(enum_value: Any) -> str:
    """``"adi_adrv9010_TxChannels_e.ADI_ADRV9010_TX1"`` -> ``"TX1"``."""
    return str(enum_value).rsplit(".", 1)[-1].replace("ADI_ADRV9010_", "")


def _bits(mask: int, names: list[str]) -> list[str]:
    return [name for k, name in enumerate(names) if int(mask) & (1 << k)]


def _rapp(v: np.ndarray, p: float = 2.0) -> np.ndarray:
    return v / (1.0 + np.abs(v) ** (2 * p)) ** (1.0 / (2 * p))


def _full_scale(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def _env_delays() -> dict[str, float]:
    raw = os.environ.get(FAKE_DELAYS_ENV)
    return {str(k): float(v) for k, v in json.loads(raw).items()} if raw else {}


def _profile_numbers(config: Config) -> tuple[float, int, int]:
    """ORx rate and TX / Rx bits from the config's profile, or the fake defaults."""
    try:
        from .profile import read_profile

        info = read_profile(config.profile_path)
        return float(info.orx_rate_hz), int(info.tx_bits), int(info.rx_bits)
    except Exception:  # noqa: BLE001 - no profile on this machine: use defaults
        return DEFAULT_RATE_HZ, DEFAULT_BITS, DEFAULT_BITS


class FakeBoard:
    """Register state and DLL calls of the simulated ADS9 + ADRV9026."""

    def __init__(
        self,
        config: Config,
        *,
        state_path: str | os.PathLike[str] | None = None,
        delays: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        self.config = config
        self.state_path = Path(state_path) if state_path else None
        self.delays = dict(delays or {})
        self.rng = np.random.default_rng(seed)
        self.rate_hz, self.tx_bits, self.rx_bits = _profile_numbers(config)
        self.connected = False  # this process's socket, never persisted
        self.waves: dict[str, np.ndarray] = {}  # TX name -> complex codes (playback RAM)
        self.post_init: Any = None
        self.calls: deque[str] = deque(maxlen=10_000)  # recent DLL ops, newest last
        self.state: dict[str, Any] = {
            "programmed": False,
            "rx_mask": 0,
            "tx_mask": 0,
            "tx_atten_mdb": dict.fromkeys(_TX_NAMES, 0),
            "rx_gain": dict.fromkeys(_RX_NAMES, 195),
            "lo_hz": {"LO1": 0, "LO2": 0, "AUX": 0},
            "tx_len": 0,
            "pid": None,
        }
        self._load()

    # -- persistence -----------------------------------------------------------

    @property
    def events_path(self) -> Path | None:
        return None if self.state_path is None else self.state_path.with_suffix(".events.jsonl")

    def _load(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        for _ in range(50):
            try:
                self.state.update(json.loads(self.state_path.read_text()))
                return
            except (OSError, ValueError):  # mid-replace by another process
                time.sleep(0.02)

    def _save(self) -> None:
        if self.state_path is None:
            return
        self.state["pid"] = os.getpid()
        tmp = self.state_path.with_name(f"{self.state_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.state))
        for _ in range(50):
            try:
                os.replace(tmp, self.state_path)
                return
            except PermissionError:  # Windows: a reader has it open
                time.sleep(0.02)
        raise OSError(f"could not update fake board state {self.state_path}")

    def _event(self, op: str, **fields: Any) -> None:
        self.calls.append(op)
        path = self.events_path
        if path is None:
            return
        line = json.dumps({"t": time.time(), "pid": os.getpid(), "op": op, **fields})
        with open(path, "a") as fh:
            fh.write(line + "\n")

    def _delay(self, op: str) -> None:
        seconds = self.delays.get(op, 0.0)
        if seconds > 0:
            time.sleep(seconds)

    # -- link / client -----------------------------------------------------------

    def is_connected(self) -> bool:
        return self.connected

    def connect(self, ip: str, port: int) -> None:
        self._delay("Connect")
        if self.state_path is not None and self.state_path.with_suffix(".refuse").exists():
            self._event("ConnectRefused", ip=ip, port=int(port))
            raise ConnectionError(f"fake ADS9 at {ip}:{port} refused the connection")
        self._load()
        self.connected = True
        self._event("Connect", ip=ip, port=int(port))
        self._save()

    def disconnect(self) -> None:
        self.connected = False
        self._event("Disconnect")

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("fake board: not connected")

    # -- programming -------------------------------------------------------------

    def config_file_load(self, path: str | None = None) -> None:
        self._require_connected()
        self._event("ConfigFileLoad", path=path)

    def init_struct_get(self) -> _Struct:
        return _Struct()

    def utility_init_struct_set(self, post: Any) -> None:
        self.post_init = post

    def clock_config(self, *args: Any) -> None:
        self._event("ClockConfig", args=[int(a) for a in args])

    def program(self) -> None:
        self._require_connected()
        self._delay("Program")
        rc = getattr(self.post_init, "radioCtrlInit", None)
        for name, attr in (("LO1", "lo1PllFreq_Hz"), ("LO2", "lo2PllFreq_Hz")):
            value = getattr(rc, attr, None) if rc is not None else None
            if isinstance(value, (int, float)):
                self.state["lo_hz"][name] = int(value)
        self.state.update(programmed=True, rx_mask=0, tx_mask=0)
        self.waves.clear()
        self._event("Program")
        self._save()

    # -- enables / attenuation / gain ------------------------------------------------

    def rx_tx_enable_set(self, rx_mask: int, tx_mask: int) -> None:
        self._require_connected()
        self.state["rx_mask"] = int(rx_mask)
        self.state["tx_mask"] = int(tx_mask)
        self._event("RxTxEnableSet", rx=int(rx_mask), tx=int(tx_mask))
        self._save()

    def rx_tx_enable_get(self, _rx: int, _tx: int) -> tuple[int, int, int]:
        self._require_connected()
        return 0, int(self.state["rx_mask"]), int(self.state["tx_mask"])

    def tx_atten_set(self, items: list, count: int) -> None:
        self._require_connected()
        if not self.state["programmed"]:
            raise RuntimeError("Invalid Tx attenuation control mode (fake board not programmed)")
        for item in list(items)[: int(count)]:
            for name in _bits(item.txChannelMask, _TX_NAMES):
                self.state["tx_atten_mdb"][name] = int(item.txAttenuation_mdB)
            self._event("TxAttenSet", mask=int(item.txChannelMask), mdb=int(item.txAttenuation_mdB))
        self._save()

    def tx_atten_get(self, channel: Any, placeholder: Any) -> tuple[int, Any]:
        self._require_connected()
        placeholder.txAttenuation_mdB = int(self.state["tx_atten_mdb"][_member(channel)])
        return 0, placeholder

    def rx_gain_set(self, items: list, count: int) -> None:
        self._require_connected()
        for item in list(items)[: int(count)]:
            for name in _bits(item.rxChannelMask, _RX_NAMES):
                self.state["rx_gain"][name] = int(item.gainIndex)
            self._event("RxGainSet", mask=int(item.rxChannelMask), index=int(item.gainIndex))
        self._save()

    def rx_gain_get(self, channel: Any, placeholder: Any) -> tuple[int, Any]:
        self._require_connected()
        placeholder.gainIndex = int(self.state["rx_gain"][_member(channel)])
        return 0, placeholder

    def rx_dec_power_get(self, _channel: Any, _placeholder: int) -> tuple[int, int]:
        self._require_connected()
        return 0, 20000  # -20 dBFS

    # -- PLL ------------------------------------------------------------------------

    def pll_frequency_set(self, pll: Any, freq_hz: int) -> None:
        self._require_connected()
        name = _member(pll).replace("_PLL", "")
        self.state["lo_hz"][name] = int(freq_hz)
        self._event("PllFrequencySet", pll=name, hz=int(freq_hz))
        self._save()

    def pll_frequency_get(self, pll: Any, _placeholder: int) -> tuple[int, int]:
        self._require_connected()
        return 0, int(self.state["lo_hz"][_member(pll).replace("_PLL", "")])

    def pll_status_get(self, _placeholder: int) -> tuple[int, int]:
        self._require_connected()
        return 0, 0xF if self.state["programmed"] else 0

    # -- PerformTx / PerformRx ------------------------------------------------------

    def perform_tx(self, _trig: Any, tx_data: Any, channel_mask: int, continuous: int) -> None:
        self._require_connected()
        self._delay("PerformTx")
        if not isinstance(tx_data, NetArrayList):
            raise TypeError(f"PerformTx expects the bridge's ArrayList, got {type(tx_data)}")
        if len(tx_data) != 8 or not all(isinstance(a, NetArray) for a in tx_data):
            raise TypeError("PerformTx expects eight Int32[] arrays (I and Q for TX1..TX4)")
        for k, name in enumerate(_TX_NAMES):
            i = np.asarray(tx_data[2 * k], dtype=np.float64)
            q = np.asarray(tx_data[2 * k + 1], dtype=np.float64)
            self.waves[name] = i + 1j * q
        self.state["tx_len"] = len(tx_data[0])
        self._event(
            "PerformTx", mask=int(channel_mask), n=len(tx_data[0]), continuous=int(continuous)
        )
        self._save()

    def perform_rx(self, _trig: Any, _mask: int, capture_time_ms: float, _timeout_ms: int):
        self._require_connected()
        self._delay("PerformRx")
        from .capture import orx_slot_for, returned_channel_order, tx_for_orx

        n = int(round(float(capture_time_ms) * 1e-3 * self.rate_hz))
        order = returned_channel_order(self.config.channels.rx_init_mask)
        data = [self._noise(n) for _ in order]
        for orx in (RxChannel.ORX1, RxChannel.ORX2, RxChannel.ORX3, RxChannel.ORX4):
            if not self.state["rx_mask"] & int(orx):
                continue
            tx = tx_for_orx(orx, self.config.tx_to_orx)
            slot = orx_slot_for(orx, order)
            if tx is None or slot is None or not self.state["tx_mask"] & int(tx):
                continue
            data[slot] = data[slot] + self._loopback(tx, orx, n)
        rail = _full_scale(self.rx_bits)
        out = NetArrayList()
        for z in data:
            i = np.clip(np.round(z.real), -(rail + 1), rail).astype(np.int32)
            q = np.clip(np.round(z.imag), -(rail + 1), rail).astype(np.int32)
            out.extend([i, q])
        self._event("PerformRx", n=n)
        return out

    def _noise(self, n: int) -> np.ndarray:
        return NOISE_CODES * (self.rng.normal(size=n) + 1j * self.rng.normal(size=n))

    def _loopback(self, tx: TxChannel, orx: RxChannel, n: int) -> np.ndarray:
        wave = self.waves.get(tx.name)
        if wave is None or len(wave) == 0:
            return np.zeros(n, dtype=np.complex128)
        atten_db = self.state["tx_atten_mdb"][tx.name] / 1000.0
        drive = PA_DRIVE * 10 ** (-atten_db / 20.0)
        pa = _rapp(wave / _full_scale(self.tx_bits) * drive)
        gain_db = (self.state["rx_gain"][orx.name] - 210) * 0.5
        y = np.roll(pa, DELAY_SAMPLES) * ORX_LEVEL * _full_scale(self.rx_bits)
        y = y * 10 ** (gain_db / 20.0)
        start = int(self.rng.integers(len(y)))
        return y[(start + np.arange(n)) % len(y)]


class FakeBridge:
    """The :class:`~adrvtrx._clr.ClrBridge` surface, wired to a :class:`FakeBoard`."""

    def __init__(self, board: FakeBoard):
        self.model = board
        b = board
        self.Types = _Types()
        self.ns = SimpleNamespace(FpgaTypes=None)
        self.Array = None
        device = SimpleNamespace(
            ConfigFileLoad=b.config_file_load,
            InitStructGet=b.init_struct_get,
            UtilityInitStructSet=b.utility_init_struct_set,
            RadioCtrl=SimpleNamespace(
                RxTxEnableSet=b.rx_tx_enable_set, RxTxEnableGet=b.rx_tx_enable_get
            ),
            Tx=SimpleNamespace(TxAttenSet=b.tx_atten_set, TxAttenGet=b.tx_atten_get),
            Rx=SimpleNamespace(
                RxGainSet=b.rx_gain_set,
                RxGainGet=b.rx_gain_get,
                RxDecPowerGet=b.rx_dec_power_get,
            ),
        )
        board_api = SimpleNamespace(
            Client=SimpleNamespace(Connect=b.connect, Disconnect=b.disconnect),
            Adrv9010Device=device,
            ClockConfig=b.clock_config,
            Program=b.program,
            PerformTx=b.perform_tx,
            PerformRx=b.perform_rx,
        )
        adrv = SimpleNamespace(
            RadioCtrl=SimpleNamespace(
                PllFrequencySet=b.pll_frequency_set,
                PllFrequencyGet=b.pll_frequency_get,
                PllStatusGet=b.pll_status_get,
            )
        )
        self.link = SimpleNamespace(
            platform=SimpleNamespace(board=board_api),
            IsConnected=b.is_connected,
            Adrv9010Get=lambda _index: adrv,
        )

    def enum(self, enum_type_name: str, member: str) -> EnumValue:
        return EnumValue(f"{enum_type_name}.{member}")

    def new_array(self, _type_name: str, length: int) -> list:
        return [None] * length

    def int_array(self, values) -> NetArray:
        return NetArray(int(v) for v in values)

    def array_list(self, items=()) -> NetArrayList:
        return NetArrayList(items)


class FakeRadio(Radio):
    """The real :class:`~adrvtrx.radio.Radio` over a simulated board.

    ``FakeRadio(config)`` is a hardware-server backend (``--backend fake``).
    ``state_path`` defaults to ``$ADRVTRX_FAKE_STATE`` and ``delays`` to
    ``$ADRVTRX_FAKE_DELAYS``.
    """

    def __init__(
        self,
        config: Config,
        bridge: FakeBridge | None = None,
        *,
        state_path: str | os.PathLike[str] | None = None,
        delays: dict[str, float] | None = None,
        seed: int | None = None,
    ):
        if bridge is None:
            board = FakeBoard(
                config,
                state_path=state_path or os.environ.get(FAKE_STATE_ENV) or None,
                delays=_env_delays() if delays is None else delays,
                seed=seed,
            )
            bridge = FakeBridge(board)
        super().__init__(config, bridge)

    @property
    def board_model(self) -> FakeBoard:
        """The simulated board (register state, waveforms, call log)."""
        return self._bridge.model
