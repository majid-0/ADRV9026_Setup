"""Waveform IO: tab-delimited IQ load, normalize/quantize for TX, float save.

Conventions (decided with the user):
  * TX input files  : 2 columns ``I<TAB>Q``, one complex sample per row, no header.
  * Capture output  : normalized float IQ in [-1, 1] (divided by full scale),
                      same 2-column ``I<TAB>Q`` layout.

Quantization targets the JESD transport word width ``Np`` read from the loaded
profile (``jesd204Np``), i.e. full-scale magnitude = ``2**(Np-1) - 1``. For
StdUseCase102, Np = 16 -> +/-32767. All functions here are pure numpy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "load_tab_iq",
    "save_tab_iq_float",
    "normalize",
    "quantize",
    "prepare_tx",
    "full_scale",
    "samples_for_duration",
]


def full_scale(n_bits: int) -> int:
    """Max positive code for a signed ``n_bits`` sample (``2**(n_bits-1) - 1``)."""
    if n_bits < 2:
        raise ValueError(f"n_bits must be >= 2, got {n_bits}")
    return (1 << (n_bits - 1)) - 1


def samples_for_duration(capture_time_ms: float, sample_rate_khz: float) -> int:
    """Number of IQ samples for a duration at a given sample rate."""
    return int(round(capture_time_ms * 1e-3 * sample_rate_khz * 1e3))


def load_tab_iq(path: str | Path) -> np.ndarray:
    """Load a 2-column ``I<TAB>Q`` file into a complex128 array."""
    data = np.loadtxt(path, delimiter="\t", dtype=np.float64, ndmin=2)
    if data.shape[1] != 2:
        raise ValueError(f"{path}: expected 2 tab-separated columns (I, Q), got {data.shape[1]}")
    return data[:, 0] + 1j * data[:, 1]


def normalize(iq: np.ndarray) -> np.ndarray:
    """Scale so the peak magnitude is 1.0. A zero signal is returned unchanged."""
    peak = np.max(np.abs(iq)) if iq.size else 0.0
    if peak == 0.0:
        return iq.astype(np.complex128, copy=True)
    return iq / peak


def quantize(iq_norm: np.ndarray, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a unit-scaled complex array to signed ``n_bits`` integer I and Q.

    Returns ``(i_int, q_int)`` as int32 arrays clipped to the signed range.
    """
    fs = full_scale(n_bits)
    lo, hi = -(fs + 1), fs
    i_int = np.clip(np.round(iq_norm.real * fs), lo, hi).astype(np.int32)
    q_int = np.clip(np.round(iq_norm.imag * fs), lo, hi).astype(np.int32)
    return i_int, q_int


def prepare_tx(
    iq: np.ndarray, n_bits: int, *, do_normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Full TX pipeline: (optionally normalize) then quantize to ``n_bits``.

    Returns ``(i_int, q_int)`` ready to be packed for ``PerformTx``.
    """
    work = normalize(iq) if do_normalize else iq
    return quantize(work, n_bits)


def save_tab_iq_float(
    i_int: np.ndarray,
    q_int: np.ndarray,
    path: str | Path,
    n_bits: int,
) -> None:
    """Save captured integer IQ as normalized float (/ full scale) in 2-col tab form."""
    fs = float(full_scale(n_bits))
    out = np.empty((len(i_int), 2), dtype=np.float64)
    out[:, 0] = np.asarray(i_int, dtype=np.float64) / fs
    out[:, 1] = np.asarray(q_int, dtype=np.float64) / fs
    np.savetxt(path, out, delimiter="\t", fmt="%.9g")
