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
  waveform through the PA model (:class:`PaModel`), a small delay, the ORx gain
  ``(index - 210) * orx_db_per_index`` dB (0.5 unless a band sets its own), noise and the ADC
  rail. Every capture starts at a
  random point of the loop (IMMEDIATE trigger).

The PA model (:class:`PaModel`, on one period of the looping waveform, circular):

* ``v = x / full_scale * drive * 10**((G(f) - atten) / 20)``, with ``x`` the TX
  codes and ``G(f)`` the small-signal gain in dB at the TX's LO frequency ``f``.
* ``w = pre_taps * v`` (FIR: linear memory before the nonlinearity).
* ``g = w / (1 + (|w| / sat(f))**(2p))**(1/(2p))`` (Rapp, smoothness ``p``),
  then AM/PM: ``g *= exp(1j * am_pm_deg * pi/180 * |g / sat|**2)``.
* Nonlinear memory: ``g += g * sum_m nl_memory[m-1] * |g[n-m] / sat|**2``
  (bounded: the envelope is taken after compression).
* ``y = post_taps * g`` (FIR), delayed by ``delay_samples``.
* Frequency, with ``df = (f - f0_hz)`` in GHz: ``G = gain_db +
  gain_slope_db_per_ghz * df + gain_curve_db_per_ghz2 * df**2``,
  ``sat = 10**((sat_db + sat_slope_db_per_ghz * df) / 20)``, and every memory
  tap after the first is scaled by ``1 + memory_slope_per_ghz * df``. The memory
  makes a wide signal (100 MHz) harder to linearize than a narrow one (40 MHz).
* Drift ``d`` dB, added to ``G`` (``sat`` moves by ``drift_sat_fraction * d``):
  ``drift_db_per_hour`` times the hours since the board object was created,
  plus ``drift_step_db`` once ``drift_step_after_s`` seconds have passed, plus
  the number in ``drift_file`` (read at every capture: drift on demand).
* ``noise_codes`` (ORx noise per I and Q) and ``seed`` make it deterministic.

Several TX into one PA (dual band): ``shared = [["TX1", "TX4"]]``. Each TX of a group
gets its own ``w`` (its own attenuation, LO, drive and filters), and the other
members' average power uses up its headroom: band b compresses against
``sat_b / sqrt(1 + coupling * sum mean(|w_other|**2) / sat_b**2)``, so lowering one
band's attenuation compresses the other band at its peaks while its small-signal
gain stays (the gain-compression lock sees it). ``coupling`` sets the strength (with
a 4 dB lock, about 0.1 dB of compression per dB of the other band at 1, 0.3 at 4,
0.45 at 8). Only TX that are enabled feed the PA. Each ORx sees its own band; with
``other_in_window`` it also sees the other members' outputs, shifted by their LO
offset, when that offset is inside the ORx window (closely spaced bands). ``tx_skew_samples``
delays a TX before the PA (``{"TX4": 2}``), and ``tx_image_db`` adds the TX IQ
image ``10**(tx_image_db/20) * conj(x)`` before the drive. ``bands`` holds
per-TX overrides of any other parameter (``{"TX4": {"smoothness": 1.2}}``), as a
fitted preset needs.

The default ``PaModel()`` is the simple memoryless Rapp (``drive = 6``, ``p = 2``,
no memory, no frequency dependence, no drift). ``PaModel.rich()`` is a PA with
memory whose gain-compression lock moves with frequency. Select it with
``ADRVTRX_FAKE_PA``: ``rich``, or a ``.json`` / ``.toml`` file of parameters
(``preset = "rich"`` starts from the rich values), or ``FakeRadio(..., pa=...)``.

Set ``ADRVTRX_FAKE_STATE`` to a JSON file to keep the register state across
processes (the real board keeps its registers when a client dies) and to log
every register write to ``<file>.events.jsonl``. Creating ``<file>.refuse``
makes ``Connect`` fail. ``ADRVTRX_FAKE_DELAYS`` is a JSON object of seconds
per DLL call (for example ``{"PerformRx": 30}``) to simulate slow or stuck calls;
its key ``PerformTx_s_per_msample`` adds a load time proportional to the buffer
length (seconds per million samples per channel). Program clears the TX RAM.
"""

from __future__ import annotations

import json
import os
import sys
import time
import zlib
from collections import deque
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from ._enums import RX_SINGLE, TX_SINGLE, RxChannel, TxChannel
from .config import Config, lo_for_tx
from .radio import Radio

__all__ = [
    "FAKE_STATE_ENV",
    "FAKE_DELAYS_ENV",
    "FAKE_PA_ENV",
    "PaModel",
    "FakeBoard",
    "FakeBridge",
    "FakeRadio",
]

FAKE_STATE_ENV = "ADRVTRX_FAKE_STATE"
FAKE_DELAYS_ENV = "ADRVTRX_FAKE_DELAYS"
FAKE_PA_ENV = "ADRVTRX_FAKE_PA"

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


def _circular_fir(x: np.ndarray, taps) -> np.ndarray:
    """``sum_k taps[k] * x[n - k]`` on one period of a looping signal."""
    taps = [float(t) for t in taps]
    if taps == [1.0]:
        return x
    y = np.zeros_like(x)
    for k, tap in enumerate(taps):
        if tap:
            y += tap * np.roll(x, k)
    return y


@dataclass
class PaModel:
    """TX -> PA -> ORx model of the fake board; the module docstring has the equations.

    ``PaModel()`` is the simple memoryless Rapp; :meth:`rich` adds memory,
    frequency dependence and AM/PM.
    """

    drive: float = 6.0  # PA input at 0 dB atten and 0 dB gain for a full-scale peak
    smoothness: float = 2.0  # Rapp p
    f0_hz: float = 2.2e9
    gain_db: float = 0.0
    gain_slope_db_per_ghz: float = 0.0
    gain_curve_db_per_ghz2: float = 0.0
    sat_db: float = 0.0
    sat_slope_db_per_ghz: float = 0.0
    am_pm_deg: float = 0.0
    pre_taps: list[float] = field(default_factory=lambda: [1.0])
    post_taps: list[float] = field(default_factory=lambda: [1.0])
    nl_memory: list[float] = field(default_factory=list)
    memory_slope_per_ghz: float = 0.0
    delay_samples: int = 5
    orx_level: float = 0.73  # ORx output at saturation, gain index 210, x full scale
    orx_db_per_index: float = 0.5  # the ORx gain table's step (per TX in ``bands``: its ORx path)
    noise_codes: float = 0.5
    seed: int | None = None
    drift_db_per_hour: float = 0.0
    drift_step_db: float = 0.0
    drift_step_after_s: float = 0.0
    drift_file: str = ""
    drift_sat_fraction: float = 0.5
    shared: list[list[str]] = field(default_factory=list)  # TX groups that drive one PA
    coupling: float = 1.0  # how much the other members' mean power cuts a band's headroom
    other_in_window: bool = False  # ORx also sees the other members when inside its window
    tx_skew_samples: dict[str, int] = field(default_factory=dict)  # TX name -> delay before the PA
    tx_image_db: float = -300.0  # TX IQ image (conj(x)) relative to x; -300 = none
    bands: dict[str, dict[str, Any]] = field(default_factory=dict)  # TX name -> overrides

    def __post_init__(self) -> None:
        known = {f.name for f in fields(self)} - {"bands", "shared"}
        for tx, overrides in self.bands.items():
            unknown = sorted(set(overrides) - known)
            if unknown:
                raise ValueError(f"unknown fake PA parameter(s) {unknown} for {tx}")
        for group in self.shared:
            if len(group) != len(set(group)) or any(tx not in _TX_NAMES for tx in group):
                raise ValueError(f"shared PA group {group}: distinct names among {_TX_NAMES}")

    def for_tx(self, tx: str) -> PaModel:
        """This model with ``bands[tx]``'s overrides (itself when there are none)."""
        overrides = self.bands.get(tx)
        return replace(self, bands={}, **overrides) if overrides else self

    def group_of(self, tx: str) -> list[str]:
        """The TX names sharing ``tx``'s PA (just ``[tx]`` when it has its own)."""
        return next((list(g) for g in self.shared if tx in g), [tx])

    @classmethod
    def rich(cls, **overrides: Any) -> PaModel:
        """A PA with memory, AM/PM, and gain / saturation / memory that vary with frequency."""
        params: dict[str, Any] = dict(RICH_PA)
        params.update(overrides)
        return cls(**params)

    @classmethod
    def from_spec(cls, spec) -> PaModel:
        """``None`` / ``""`` / ``"simple"``, ``"rich"``, a dict, or a ``.json`` / ``.toml`` file.

        Keys starting with ``_`` are comments (JSON has none), e.g. ``"_about"``.
        """
        if isinstance(spec, PaModel):
            return spec
        if spec is None or str(spec).strip() in ("", "simple"):
            return cls()
        if isinstance(spec, dict):
            params = dict(spec)
        elif str(spec).strip() == "rich":
            return cls.rich()
        else:
            path = Path(spec)
            if path.suffix.lower() == ".toml":
                if sys.version_info >= (3, 11):
                    import tomllib
                else:  # pragma: no cover - 3.9 / 3.10
                    import tomli as tomllib
                params = tomllib.loads(path.read_text())
            else:
                params = json.loads(path.read_text())
        params = {k: v for k, v in params.items() if not str(k).startswith("_")}
        preset = params.pop("preset", "simple")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(params) - known)
        if unknown:
            raise ValueError(f"unknown fake PA parameter(s) {unknown}; known: {sorted(known)}")
        return cls.rich(**params) if preset == "rich" else cls(**params)

    def drift_db(self, elapsed_s: float) -> float:
        """Gain drift in dB after ``elapsed_s`` seconds (rate + step + drift file)."""
        drift = self.drift_db_per_hour * elapsed_s / 3600.0
        if self.drift_step_db and elapsed_s >= self.drift_step_after_s:
            drift += self.drift_step_db
        if self.drift_file:
            try:
                drift += float(Path(self.drift_file).read_text().strip() or 0.0)
            except (OSError, ValueError):
                pass
        return drift

    def _front(
        self, x_norm: np.ndarray, atten_db: float, lo_hz: float, drift_db: float, tx: str = ""
    ) -> tuple[np.ndarray, float, float]:
        """PA input ``w`` (drive, IQ image, skew, pre-filter), saturation and memory scale."""
        df = (float(lo_hz) - self.f0_hz) / 1e9 if lo_hz else 0.0
        gain_db = (
            self.gain_db
            + self.gain_slope_db_per_ghz * df
            + self.gain_curve_db_per_ghz2 * df * df
            + drift_db
        )
        sat_db = self.sat_db + self.sat_slope_db_per_ghz * df + self.drift_sat_fraction * drift_db
        mem = 1.0 + self.memory_slope_per_ghz * df
        x = np.asarray(x_norm, dtype=np.complex128)
        if self.tx_image_db > -300.0:
            x = x + 10 ** (self.tx_image_db / 20.0) * np.conj(x)
        skew = int(self.tx_skew_samples.get(tx, 0))
        if skew:
            x = np.roll(x, skew)
        gain = self.drive * 10 ** ((gain_db - atten_db) / 20.0)
        w = _circular_fir(x * gain, self._scaled(self.pre_taps, mem))
        return w, 10 ** (sat_db / 20.0), mem

    @staticmethod
    def _scaled(taps, mem: float) -> list[float]:
        return [taps[0]] + [t * mem for t in taps[1:]] if taps else [1.0]

    def _back(self, w: np.ndarray, drive_power: np.ndarray, sat: float, mem: float) -> np.ndarray:
        """Compression on ``drive_power`` (Rapp), AM/PM, memory, post-filter and delay."""
        p = self.smoothness
        g = w / (1.0 + (drive_power / sat**2) ** p) ** (1.0 / (2 * p))
        if self.am_pm_deg:
            g = g * np.exp(1j * np.deg2rad(self.am_pm_deg) * np.abs(g / sat) ** 2)
        if self.nl_memory:
            env = np.abs(g / sat) ** 2
            g = g + g * sum(b * mem * np.roll(env, m) for m, b in enumerate(self.nl_memory, 1))
        y = _circular_fir(g, self._scaled(self.post_taps, mem))
        return np.roll(y, int(self.delay_samples))

    def output(
        self, x_norm: np.ndarray, atten_db: float, lo_hz: float, drift_db: float = 0.0, tx: str = ""
    ) -> np.ndarray:
        """PA output for one period of ``x_norm`` (TX codes / full scale), delayed."""
        model = self.for_tx(tx) if tx else self
        w, sat, mem = model._front(x_norm, atten_db, lo_hz, drift_db, tx)
        return model._back(w, np.abs(w) ** 2, sat, mem)

    def shared_outputs(
        self, inputs: dict[str, tuple[np.ndarray, float, float]], drift_db: float = 0.0
    ) -> dict[str, np.ndarray]:
        """Outputs of TX that drive one PA: ``{tx: (x_norm, atten_db, lo_hz)}`` -> ``{tx: y}``.

        The waveforms must have the same length (one ``PerformTx`` loads them all).
        """
        fronts = {}
        for tx, (x_norm, atten_db, lo_hz) in inputs.items():
            model = self.for_tx(tx)
            fronts[tx] = (model, *model._front(x_norm, atten_db, lo_hz, drift_db, tx))
        mean_power = {tx: float(np.mean(np.abs(w) ** 2)) for tx, (_m, w, _s, _k) in fronts.items()}
        out = {}
        for tx, (model, w, sat, mem) in fronts.items():
            others = sum(pw for o, pw in mean_power.items() if o != tx)
            headroom = sat / np.sqrt(1.0 + model.coupling * others / sat**2)
            out[tx] = model._back(w, np.abs(w) ** 2, headroom, mem)
        return out


#: Parameters of :meth:`PaModel.rich`.
RICH_PA: dict[str, Any] = {
    "drive": 5.35,
    "smoothness": 1.6,
    "gain_slope_db_per_ghz": -1.5,
    "gain_curve_db_per_ghz2": -0.5,
    "sat_slope_db_per_ghz": 0.6,
    "am_pm_deg": 12.0,
    "pre_taps": [1.0, 0.22, -0.08],
    "post_taps": [1.0, -0.12, 0.04],
    "nl_memory": [0.15, 0.08, 0.04],
    "memory_slope_per_ghz": 0.35,
}


def _full_scale(bits: int) -> int:
    return (1 << (bits - 1)) - 1


def _env_delays() -> dict[str, float]:
    raw = os.environ.get(FAKE_DELAYS_ENV)
    return {str(k): float(v) for k, v in json.loads(raw).items()} if raw else {}


def _profile_numbers(config: Config, path: str | None = None) -> tuple[float, int, int]:
    """ORx rate and TX / Rx bits from ``path`` (default: the config's profile), or defaults."""
    try:
        from .profile import read_profile

        info = read_profile(config.profile_path if path is None else path)
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
        pa: PaModel | dict[str, Any] | str | None = None,
    ):
        self.config = config
        self.state_path = Path(state_path) if state_path else None
        self.delays = dict(delays or {})
        self.pa = PaModel.from_spec(pa)
        self.rng = np.random.default_rng(self.pa.seed if seed is None else seed)
        self.clock = time.time  # tests may replace it (drift)
        self.created = self.clock()
        self.rate_hz, self.tx_bits, self.rx_bits = _profile_numbers(config)
        self.connected = False  # this process's socket, never persisted
        self.waves: dict[str, np.ndarray] = {}  # TX name -> complex codes (playback RAM)
        self.post_init: Any = None
        self.profile_loaded: str | None = None  # the last ConfigFileLoad(path)
        self.calls: deque[str] = deque(maxlen=10_000)  # recent DLL ops, newest last
        # One PerformRx is one snapshot: every channel starts at the same point of the loop,
        # and each PA (a shared group) is computed once for all the ORx that see it.
        self._snapshot_start: int | None = None
        self._snapshot_pa: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
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
        if path is not None:
            self.profile_loaded = path  # what Program will run (its rates and bits)
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
        # program(profile=...) may load another profile: rates and bits follow the loaded one
        self.rate_hz, self.tx_bits, self.rx_bits = _profile_numbers(
            self.config, self.profile_loaded
        )
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
        per_msample = self.delays.get("PerformTx_s_per_msample", 0.0)
        if per_msample > 0 and len(tx_data):
            time.sleep(per_msample * len(tx_data[0]) / 1e6)  # load time grows with length
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
        self._snapshot_start, self._snapshot_pa = None, {}  # a new snapshot
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
        return self.pa.noise_codes * (self.rng.normal(size=n) + 1j * self.rng.normal(size=n))

    def tx_lo_hz(self, tx: TxChannel) -> int:
        """The LO frequency ``tx`` uses (``[clocks]`` tx12_lo / tx34_lo)."""
        return int(self.state["lo_hz"].get(lo_for_tx(self.config.clocks, tx), 0))

    def _pa_input(self, name: str):
        """``(x_norm, atten_db, lo_hz)`` of an enabled TX with a waveform, else None."""
        wave = self.waves.get(name)
        tx = TxChannel[name]
        if wave is None or len(wave) == 0 or not self.state["tx_mask"] & int(tx):
            return None
        atten_db = self.state["tx_atten_mdb"][name] / 1000.0
        return wave / _full_scale(self.tx_bits), atten_db, self.tx_lo_hz(tx)

    def pa_outputs(self, tx: TxChannel) -> dict[str, np.ndarray]:
        """Noise-free PA output (saturation = 1) of ``tx`` and of the enabled TX sharing its PA.

        ``tx`` itself is included whether or not it is enabled (its waveform must be loaded).
        """
        drift = self.pa.drift_db(self.clock() - self.created)
        group = self.pa.group_of(tx.name)
        inputs = {n: inp for n in group if n != tx.name and (inp := self._pa_input(n)) is not None}
        wave = self.waves.get(tx.name)
        if wave is None or len(wave) == 0:
            return {}
        own = wave / _full_scale(self.tx_bits)
        atten_db = self.state["tx_atten_mdb"][tx.name] / 1000.0
        if not inputs:
            return {tx.name: self.pa.output(own, atten_db, self.tx_lo_hz(tx), drift, tx.name)}
        inputs[tx.name] = (own, atten_db, self.tx_lo_hz(tx))
        return self.pa.shared_outputs(inputs, drift)

    def pa_output(self, tx: TxChannel) -> np.ndarray | None:
        """Noise-free PA output for one period of ``tx``'s waveform (saturation = 1)."""
        return self.pa_outputs(tx).get(tx.name)

    def _loopback(self, tx: TxChannel, orx: RxChannel, n: int) -> np.ndarray:
        group = tuple(self.pa.group_of(tx.name))
        outs = self._snapshot_pa.get(group)
        if outs is None or tx.name not in outs:  # once per snapshot for the PA of the group
            outs = self._snapshot_pa[group] = self.pa_outputs(tx)
        pa = outs.get(tx.name)
        if pa is None:
            return np.zeros(n, dtype=np.complex128)
        scale = self.pa.orx_level * _full_scale(self.rx_bits)
        step = self.pa.for_tx(tx.name).orx_db_per_index
        scale *= 10 ** ((self.state["rx_gain"][orx.name] - 210) * step / 20.0)
        if self._snapshot_start is None:  # PerformTx loads equal-length buffers
            self._snapshot_start = int(self.rng.integers(len(pa)))
        start = self._snapshot_start
        idx = (start + np.arange(n)) % len(pa)
        y = pa[idx]
        if self.pa.other_in_window:
            t = (start + np.arange(n)) / self.rate_hz
            for name, other in outs.items():
                offset = self.tx_lo_hz(TxChannel[name]) - self.tx_lo_hz(tx)
                if name != tx.name and abs(offset) < self.rate_hz / 2:
                    y = y + other[idx] * np.exp(2j * np.pi * offset * t)
        return y * scale


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
    ``state_path`` defaults to ``$ADRVTRX_FAKE_STATE``, ``delays`` to
    ``$ADRVTRX_FAKE_DELAYS`` and ``pa`` to ``$ADRVTRX_FAKE_PA`` (:class:`PaModel`).
    """

    def __init__(
        self,
        config: Config,
        bridge: FakeBridge | None = None,
        *,
        state_path: str | os.PathLike[str] | None = None,
        delays: dict[str, float] | None = None,
        seed: int | None = None,
        pa: PaModel | dict[str, Any] | str | None = None,
    ):
        if bridge is None:
            board = FakeBoard(
                config,
                state_path=state_path or os.environ.get(FAKE_STATE_ENV) or None,
                delays=_env_delays() if delays is None else delays,
                seed=seed,
                pa=os.environ.get(FAKE_PA_ENV) if pa is None else pa,
            )
            bridge = FakeBridge(board)
        super().__init__(config, bridge)

    @property
    def board_model(self) -> FakeBoard:
        """The simulated board (register state, waveforms, call log)."""
        return self._bridge.model
