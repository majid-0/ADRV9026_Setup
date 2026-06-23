"""Gain / level analysis and the software ORx leveling loop.

``clip_report`` and ``peak_window`` are pure numpy (unit-testable). ``level_orx``
drives a :class:`~adrvtrx.radio.Radio`-like object using the flag-based path
confirmed in Task 0 (``RxDecPowerGet`` in mdBFS + manual ``RxGainSet``), with the
IQ-derived clip metric as a cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ._enums import RxChannel
from .waveform import full_scale

__all__ = ["ClipReport", "clip_report", "peak_window", "level_orx", "LevelResult"]


@dataclass
class ClipReport:
    peak_dbfs: float
    railed_samples: int
    peak_index: int
    n_samples: int

    @property
    def is_clipping(self) -> bool:
        return self.railed_samples > 0


def clip_report(i_int: np.ndarray, q_int: np.ndarray, n_bits: int) -> ClipReport:
    """Per-rail clip metrics from raw integer IQ.

    Peak is taken on max(|I|, |Q|) (the quantity that actually rails an ADC code),
    expressed in dBFS relative to full scale.
    """
    i = np.abs(np.asarray(i_int, dtype=np.int64))
    q = np.abs(np.asarray(q_int, dtype=np.int64))
    n = int(max(i.size, q.size))
    if n == 0:
        return ClipReport(peak_dbfs=float("-inf"), railed_samples=0, peak_index=-1, n_samples=0)
    per_sample_peak = np.maximum(i, q)
    fs = full_scale(n_bits)
    peak = int(per_sample_peak.max())
    peak_index = int(per_sample_peak.argmax())
    railed = int(np.count_nonzero(per_sample_peak >= fs))
    peak_dbfs = 20.0 * np.log10(peak / fs) if peak > 0 else float("-inf")
    return ClipReport(
        peak_dbfs=float(peak_dbfs),
        railed_samples=railed,
        peak_index=peak_index,
        n_samples=n,
    )


def peak_window(
    i_int: np.ndarray,
    q_int: np.ndarray,
    window_samples: int,
    *,
    peak_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``window_samples``-long slice of IQ centered on the signal peak."""
    i = np.asarray(i_int)
    q = np.asarray(q_int)
    n = i.size
    if window_samples >= n:
        return i, q
    if peak_index is None:
        peak_index = int(np.maximum(np.abs(i), np.abs(q)).argmax())
    half = window_samples // 2
    start = max(0, min(peak_index - half, n - window_samples))
    sl = slice(start, start + window_samples)
    return i[sl], q[sl]


class _Leveler(Protocol):
    """Minimal interface ``level_orx`` needs from a radio driver."""

    def rx_dec_power_dbfs(self, channel: RxChannel) -> float: ...
    def set_rx_gain(self, channel: RxChannel, gain_index: int) -> None: ...
    def get_rx_gain(self, channel: RxChannel) -> int: ...


@dataclass
class LevelResult:
    converged: bool
    final_gain_index: int
    final_dbfs: float
    iterations: int


def level_orx(
    radio: _Leveler,
    channel: RxChannel,
    *,
    target_dbfs: float = -12.0,
    tolerance_db: float = 2.0,
    max_iterations: int = 12,
    gain_min: int = 0,
    gain_max: int = 255,
) -> LevelResult:
    """Step the ORx manual gain index until measured DEC power lands in the window.

    Higher gain index = more gain on the ADRV902x Rx gain table, so when the
    measured level is below target we *raise* the index and vice versa. One LSB of
    gain index ~ a fraction of a dB; we step proportionally to the error.
    """
    gain = radio.get_rx_gain(channel)
    measured = radio.rx_dec_power_dbfs(channel)
    for it in range(1, max_iterations + 1):
        error = target_dbfs - measured
        if abs(error) <= tolerance_db:
            return LevelResult(True, gain, measured, it - 1)
        # ~0.5 dB per gain index step; round away from zero so we always move.
        step = int(np.sign(error) * max(1, round(abs(error) / 0.5)))
        new_gain = int(np.clip(gain + step, gain_min, gain_max))
        if new_gain == gain:
            break  # hit a rail, cannot improve further
        gain = new_gain
        radio.set_rx_gain(channel, gain)
        measured = radio.rx_dec_power_dbfs(channel)
    return LevelResult(abs(target_dbfs - measured) <= tolerance_db, gain, measured, max_iterations)
