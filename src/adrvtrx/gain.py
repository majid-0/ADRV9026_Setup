"""Gain / level analysis and the software ORx auto-gain-control (AGC).

``clip_report`` and ``peak_window`` are pure numpy (unit-testable). The AGC is
split into pure, hardware-free, callback-driven stages so it stays unit-testable:

* :func:`autolevel_orx` -- Stages A (coarse) + B (fine trim) on short captures.
* :func:`verify_no_clip` -- Stage C, a final clip check on the FULL signal.

Both drive off a caller-supplied ``measure()`` that returns ``(peak_dbfs, railed)``
computed from a capture via :func:`clip_report`. They never touch hardware and
never raise: a FATAL condition is returned as a :class:`LevelResult` with
``fatal=True`` and a message in ``reason`` -- the orchestrator
(``capture.autolevel_capture``) performs disable+disconnect and raises
:class:`AgcError`.

Bench facts that shape this design (see docs/development_debugging.md):
* ORx has no hardware AGC and ``RxGainGet`` returns 0, so we TRACK the gain index
  in software and never read it back.
* ``RxDecPowerGet`` is range-compressed and must NOT be used for leveling.
* The gain table is clean & MONOTONIC from index 185 up to 250 (~0.5 dB/index)
  with ``railed == 0`` the whole way; index 255 rails hard. So the captured-IQ
  ``railed`` count is the true clip detector (peak dBFS compresses near full scale).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .waveform import full_scale

__all__ = [
    "ClipReport",
    "clip_report",
    "peak_window",
    "autolevel_orx",
    "verify_no_clip",
    "LevelResult",
    "AgcError",
    "ORX_GAIN_MIN",
    "ORX_GAIN_MAX",
    "ORX_DB_PER_INDEX",
]

# ORx gain table, characterized on the bench (TX2->ORx2, UC98 link-sharing):
# clean and MONOTONIC from index 185 up to 250 at ~0.5 dB/index (railed==0 the
# whole way); index 255 rails hard. So 185 is the usable floor and 255 the rail.
ORX_GAIN_MIN = 185
ORX_GAIN_MAX = 255
ORX_DB_PER_INDEX = 0.50


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
    # dBFS reference is 2**(N-1) (=32768 for N=16), per ADI's `/32768` convention;
    # the rail is the max representable magnitude, 2**(N-1)-1.
    fs_ref = 1 << (n_bits - 1)
    rail = full_scale(n_bits)
    peak = int(per_sample_peak.max())
    peak_index = int(per_sample_peak.argmax())
    railed = int(np.count_nonzero(per_sample_peak >= rail))
    peak_dbfs = 20.0 * np.log10(peak / fs_ref) if peak > 0 else float("-inf")
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


class AgcError(Exception):
    """Raised by the orchestrator when the AGC hits a FATAL condition.

    The pure stages (:func:`autolevel_orx`, :func:`verify_no_clip`) never raise;
    they flag ``fatal=True`` and the orchestrator translates that into a
    disable-TX + disconnect + ``AgcError`` so the bench is left safe.
    """


@dataclass
class LevelResult:
    converged: bool
    final_gain_index: int
    final_dbfs: float
    iterations: int
    reason: str = ""
    railed: int = 0
    at_max_gain: bool = False
    fatal: bool = False


def autolevel_orx(
    set_gain,
    measure,
    *,
    target_dbfs: float = -1.0,
    tol_up_db: float = 0.3,
    tol_down_db: float = 0.6,
    gain_min: int = ORX_GAIN_MIN,
    gain_max: int = ORX_GAIN_MAX,
    db_per_index: float = ORX_DB_PER_INDEX,
    max_iterations: int = 16,
) -> LevelResult:
    """Software ORx AGC -- Stage A (coarse) + Stage B (fine trim).

    Drives off a TRUSTED, caller-supplied metric: ``measure()`` returns
    ``(peak_dbfs, railed)`` from a short capture (``clip_report``). ``set_gain(idx)``
    applies an ORx gain-table index. The loop stays pure/hardware-free.

    INVARIANT: ``railed == 0`` at every accepted gain. The accepted band is the
    ASYMMETRIC window ``[target - tol_down, target + tol_up]`` (tighter toward the
    rail). On a FATAL floor condition (clipping or already-in-band at ``gain_min``)
    it returns ``fatal=True`` with a message in ``reason`` instead of raising -- the
    orchestrator disables TX, disconnects, and raises :class:`AgcError`.

    If the signal is below band even at ``gain_max`` it ACCEPTS ``gain_max`` as the
    best achievable (``at_max_gain=True``, not fatal).
    """
    band_lo = target_dbfs - tol_down_db
    band_hi = target_dbfs + tol_up_db

    def _clamp(g: int) -> int:
        return int(min(max(g, gain_min), gain_max))

    # -- Stage A: start at the floor, one computed jump --------------------------
    gain = gain_min
    set_gain(gain)
    peak, railed = measure()
    if railed > 0:
        return LevelResult(
            False,
            gain,
            peak,
            0,
            f"TX too strong: clips at minimum gain ({gain_min}) -- reduce TX power / add a pad.",
            railed=railed,
            fatal=True,
        )
    if peak >= band_lo:
        return LevelResult(
            False,
            gain,
            peak,
            0,
            f"TX too strong for target: minimum gain ({gain_min}) already at/above band "
            f"-- reduce TX power.",
            railed=railed,
            fatal=True,
        )
    steps = round((target_dbfs - peak) / db_per_index)
    gain = _clamp(gain_min + steps)
    set_gain(gain)

    # -- Stage B: fine trim ------------------------------------------------------
    fine_iters = 0
    for _ in range(max_iterations):
        fine_iters += 1
        peak, railed = measure()
        if railed > 0:  # clip veto -- always back off
            gain = _clamp(gain - 1)
            set_gain(gain)
            continue
        if peak > band_hi:  # too hot
            step = max(1, round((peak - target_dbfs) / db_per_index))
            gain = _clamp(gain - step)
            set_gain(gain)
            continue
        if peak < band_lo:  # too cold
            step = max(1, round((target_dbfs - peak) / db_per_index))
            new = gain + step
            if new >= gain_max:
                if gain >= gain_max:  # already maxed and still below band -> accept
                    return LevelResult(
                        False,
                        gain_max,
                        peak,
                        fine_iters,
                        "max gain reached (signal below target, best achievable)",
                        railed=railed,
                        at_max_gain=True,
                    )
                new = gain_max
            gain = _clamp(new)
            set_gain(gain)
            continue
        # railed == 0 and in band -> converged
        return LevelResult(True, gain, peak, fine_iters, "converged in band", railed=railed)

    return LevelResult(False, gain, peak, fine_iters, "max iterations reached", railed=railed)


def verify_no_clip(
    set_gain,
    measure_full,
    start_gain: int,
    *,
    target_dbfs: float = -1.0,
    tol_up_db: float = 0.3,
    gain_min: int = ORX_GAIN_MIN,
    db_per_index: float = ORX_DB_PER_INDEX,
) -> LevelResult:
    """Stage C -- verify the settled gain on the FULL signal, backing off on clip.

    A short coarse capture can miss the waveform's true peak, so re-check at the
    full waveform duration: while it rails OR exceeds ``target + tol_up``, drop the
    gain one index and recapture. Colder than band is fine (no action). If it backs
    all the way to ``gain_min`` and STILL rails, that is FATAL (``fatal=True``).

    ``measure_full()`` returns ``(peak_dbfs, railed)`` from a full-duration capture.
    ``db_per_index`` is accepted for API symmetry (Stage C steps one index at a time).
    """
    band_hi = target_dbfs + tol_up_db
    gain = int(start_gain)
    set_gain(gain)
    peak, railed = measure_full()
    steps = 0
    while railed > 0 or peak > band_hi:
        if gain <= gain_min:
            if railed > 0:
                return LevelResult(
                    False,
                    gain,
                    peak,
                    steps,
                    f"TX too strong: still clips at minimum gain ({gain_min}) on the full "
                    f"signal -- reduce TX power.",
                    railed=railed,
                    fatal=True,
                )
            break  # hot but not railing and cannot reduce further -- best achievable
        gain -= 1
        steps += 1
        set_gain(gain)
        peak, railed = measure_full()
    converged = railed == 0 and peak <= band_hi
    reason = "verified: no clip on full signal" if converged else "full signal hot but not railing"
    return LevelResult(converged, gain, peak, steps, reason, railed=railed)
