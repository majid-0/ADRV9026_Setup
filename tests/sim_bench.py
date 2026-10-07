"""A simulated TX1 -> PA -> ORx1 bench behind the ``Radio`` surface the workflow uses.

The real :func:`adrvtrx.transmit.transmit_bands`, :func:`adrvtrx.capture.capture`
and the ORx AGC run unchanged against it. The model:

* ``perform_tx`` stores the TX1 codes and plays them in a loop.
* PA: memoryless Rapp (``p = 2``, saturation 1.0) driven by
  ``codes / full_scale * DRIVE * 10**(-atten/20)``.
* A fixed fractional path delay, applied circularly on the period.
* ORx gain ``(index - 210) * 0.5`` dB on top of ``ORX_SCALE`` codes per unit,
  additive noise, rounding, and clipping at the 12-bit rail.
* Every capture starts at a random offset inside the loop (IMMEDIATE trigger).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adrvtrx._enums import RxChannel, TxChannel
from adrvtrx.capture import orx_slot_for, returned_channel_order
from adrvtrx.config import Config, DllConfig
from adrvtrx.radio import MAX_TX_ATTEN_DB
from adrvtrx.waveform import full_scale, prepare_tx

FS = 10e6
BW = 2e6
N = 4096
BITS = 12
DRIVE = 6.0
ORX_SCALE = 1500.0
DELAY = 0.37
NOISE = 0.5


class _Bridge:
    def int_array(self, values):
        return list(values)

    def array_list(self, items=()):
        return list(items)


def rapp(v: np.ndarray, p: float = 2.0, sat: float = 1.0) -> np.ndarray:
    return v / (1.0 + (np.abs(v) / sat) ** (2 * p)) ** (1.0 / (2 * p))


def inverse_rapp(y: np.ndarray, p: float = 2.0, sat: float = 1.0) -> np.ndarray:
    return y / (1.0 - (np.abs(y) / sat) ** (2 * p)) ** (1.0 / (2 * p))


def drive_for(atten_db: float) -> float:
    return DRIVE * 10 ** (-atten_db / 20.0)


def make_signal(n: int = N, seed: int = 0) -> np.ndarray:
    """Band-limited complex Gaussian with unit peak, occupying ``0.9 * BW``."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n, 1.0 / FS)
    spec = (rng.normal(size=n) + 1j * rng.normal(size=n)) * (np.abs(f) < 0.45 * BW)
    x = np.fft.ifft(spec)
    return x / np.abs(x).max()


def reference_codes(x_norm: np.ndarray) -> np.ndarray:
    i, q = prepare_tx(x_norm, BITS)
    return i.astype(np.float64) + 1j * q.astype(np.float64)


def _fractional_delay(y: np.ndarray, d: float) -> np.ndarray:
    freqs = np.fft.fftfreq(len(y))
    return np.fft.ifft(np.fft.fft(y) * np.exp(-2j * np.pi * freqs * d))


class SimRadio:
    def __init__(self, seed: int = 1):
        self.config = Config(dll=DllConfig(install_dir=Path("C:/nonexistent")))
        self.bridge = _Bridge()
        self.rng = np.random.default_rng(seed)
        self._en_rx = 0
        self._en_tx = 0
        self.atten = MAX_TX_ATTEN_DB
        self.gain = 195
        self.loop: np.ndarray | None = None
        self.tx_on = False
        self.lo_hz: dict[str, int] = {}
        self.retunes: list[tuple[str, int]] = []
        self.gain_sets: list[int] = []
        self.transmitted: list[np.ndarray] = []
        self.safe_calls = 0
        self.disconnected = False

    # -- Radio surface --------------------------------------------------------

    def rx_tx_enable(self, rx_mask: int, tx_mask: int) -> None:
        self._en_rx = int(rx_mask)
        self._en_tx = int(tx_mask)
        self.tx_on = bool(tx_mask) and self.loop is not None

    def disable_tx(self) -> None:
        self.rx_tx_enable(self._en_rx, 0)

    def set_rx_enable(self, mask: int) -> None:
        self.rx_tx_enable(mask, self._en_tx)

    def set_tx_atten(self, channel, atten_db: float) -> None:
        if channel in (TxChannel.TX1, TxChannel.ALL):
            self.atten = float(atten_db)

    def set_rx_gain(self, channel, gain_index: int) -> None:
        if channel == RxChannel.ORX1:
            self.gain = int(gain_index)
            self.gain_sets.append(self.gain)

    def retune_lo(self, pll: str, freq_hz: int) -> int:
        self.lo_hz[pll] = int(freq_hz)
        self.retunes.append((pll, int(freq_hz)))
        return 0xF

    def safe_state(self) -> None:
        self.safe_calls += 1
        self.atten = MAX_TX_ATTEN_DB
        self.disable_tx()

    def disconnect(self) -> None:
        self.disconnected = True

    def perform_tx(self, tx_data, channel_mask: int, *, trig=None, continuous=True) -> None:
        self.disable_tx()
        i = np.asarray(tx_data[0], dtype=np.float64)
        q = np.asarray(tx_data[1], dtype=np.float64)
        self.loop = i + 1j * q
        self.transmitted.append(self.loop.copy())
        self.rx_tx_enable(self._en_rx, self._en_tx | int(channel_mask))

    def perform_rx(self, channel_mask: int, capture_time_ms: float, *, trig=None, timeout_ms=0):
        n = int(round(capture_time_ms * 1e-3 * FS))
        order = returned_channel_order(channel_mask)
        raw = [np.zeros(n, dtype=np.int32) for _ in range(2 * len(order))]
        slot = orx_slot_for(RxChannel.ORX1, order)
        codes = self._orx_codes(n)
        raw[2 * slot] = codes.real.astype(np.int32)
        raw[2 * slot + 1] = codes.imag.astype(np.int32)
        return raw

    # -- model ----------------------------------------------------------------

    def pa_period(self, codes: np.ndarray | None = None) -> np.ndarray:
        """Noise-free PA output for one period (unit = saturation), undelayed."""
        loop = self.loop if codes is None else codes
        return rapp(loop / full_scale(BITS) * drive_for(self.atten))

    def _orx_codes(self, n: int) -> np.ndarray:
        rail = full_scale(BITS)
        if self.tx_on and self.loop is not None:
            y = _fractional_delay(self.pa_period(), DELAY)
            y = y * ORX_SCALE * 10 ** ((self.gain - 210) * 0.5 / 20.0)
            start = int(self.rng.integers(len(y)))
            y = y[(start + np.arange(n)) % len(y)]
        else:
            y = np.zeros(n, dtype=np.complex128)
        y = y + NOISE * (self.rng.normal(size=n) + 1j * self.rng.normal(size=n))
        i = np.clip(np.round(y.real), -(rail + 1), rail)
        q = np.clip(np.round(y.imag), -(rail + 1), rail)
        return i + 1j * q
