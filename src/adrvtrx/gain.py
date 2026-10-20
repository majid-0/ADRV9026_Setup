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

import math
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
    "ORX_STEP_BOUNDS",
    "OrxStep",
    "relevel_index",
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


def relevel_index(
    peak_dbfs: float, railed: int, *, target_dbfs: float, band_hi: float, db_per_index: float
) -> int:
    """ORx gain index change for a capture that is not leveled (railed, hot or cold).

    One index down when it railed with its peak still in band (the peak dBFS compresses near
    full scale, the railed count does not); otherwise the peak's distance from ``target_dbfs`` in
    indices, at least one. The re-level rule of the single-band and the multi-band searches
    (:class:`OrxStep` learns its ``db_per_index``).
    """
    if railed > 0 or peak_dbfs > band_hi:
        if railed > 0 and not peak_dbfs > band_hi:
            return -1
        return -max(1, int(round((peak_dbfs - target_dbfs) / db_per_index)))
    return max(1, int(round((target_dbfs - peak_dbfs) / db_per_index)))


#: The range a learnt ORx gain step is held in, dB per index.
ORX_STEP_BOUNDS = (0.25, 1.5)


class OrxStep:
    """One ORx's gain re-level that learns its step (dB per index), for every re-level of that ORx.

    The gain table is not 0.5 dB per index everywhere (ORX3 moves about 0.8 dB per index near
    202..204 on the bench), so a re-level sized with 0.5 overshoots and bounces across the window
    for ever. Here:

    * dB per index is the secant of the last two captures at different gains and the same TX
      drive (``drive``: anything that names it, e.g. the attenuations), neither railed, held in
      :data:`ORX_STEP_BOUNDS`; until there are two, ``db_per_index`` (0.5) is assumed. It is kept
      across drives (it belongs to the gain table).
    * Once a re-level overshoots (the peak crosses the window), the steps are 1 index, until the
      drive changes.
    * If a 1-index step still jumps the window, :attr:`jumped` is set and :meth:`closest` names
      the capture to take: the one closest to the target with no railed samples.

    :meth:`relevel` records a capture that is not leveled and returns the index change;
    :meth:`indices` converts a level change in dB with the learnt step.
    """

    def __init__(self, db_per_index: float = ORX_DB_PER_INDEX, bounds=ORX_STEP_BOUNDS):
        self.db_per_index = float(db_per_index)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.single = False  # 1-index steps (a re-level overshot at this drive)
        self.jumped = False  # a 1-index step jumped the window at this drive
        self._caps: list[tuple] = []  # (drive, gain, peak, railed, side) at the current drive

    def _side(self, peak: float, railed: int, band_lo: float, band_hi: float) -> int:
        if railed > 0 or peak > band_hi:
            return 1
        return -1 if peak < band_lo else 0

    def observe(self, gain: int, peak: float, railed: int, *, band_lo: float, band_hi: float,
                drive=None) -> None:  # fmt: skip
        """Record a capture (leveled or not): it learns the step and detects an overshoot."""
        side = self._side(peak, railed, band_lo, band_hi)
        if self._caps and self._caps[-1][0] != drive:  # a new drive: a new re-level episode
            self._caps, self.single, self.jumped = [], False, False
        if self._caps:
            _d, g0, p0, r0, s0 = self._caps[-1]
            if g0 != gain and r0 == 0 and railed == 0 and math.isfinite(p0 + peak):
                slope = (peak - p0) / (gain - g0)
                if slope > 0:  # noise or a wrong reading cannot teach a falling table
                    self.db_per_index = float(
                        np.clip(round(slope, 2), *self.bounds)
                    )  # 0.01 dB: no rounding flips
            if s0 * side < 0:  # the peak crossed the window
                self.jumped = self.jumped or abs(gain - g0) <= 1
                self.single = True
        self._caps.append((drive, int(gain), float(peak), int(railed), side))

    def relevel(self, gain: int, peak: float, railed: int, *, target_dbfs: float, band_lo: float,
                band_hi: float, drive=None) -> int:  # fmt: skip
        """Index change for a capture that is not leveled (recorded, see :meth:`observe`)."""
        self.observe(gain, peak, railed, band_lo=band_lo, band_hi=band_hi, drive=drive)
        return self.step(peak, railed, target_dbfs=target_dbfs, band_hi=band_hi)

    def step(self, peak: float, railed: int, *, target_dbfs: float, band_hi: float) -> int:
        """Index change for a capture already recorded (:meth:`observe`)."""
        k = relevel_index(
            peak, railed, target_dbfs=target_dbfs, band_hi=band_hi, db_per_index=self.db_per_index
        )
        return (1 if k > 0 else -1) if self.single else k

    def closest(self, target_dbfs: float) -> tuple[int, float] | None:
        """``(gain, peak)`` of the capture at this drive closest to the target, not railed."""
        ok = [(g, p) for _d, g, p, r, _s in self._caps if r == 0 and math.isfinite(p)]
        return min(ok, key=lambda gp: abs(gp[1] - target_dbfs)) if ok else None

    def indices(self, db: float) -> int:
        """Gain indices for a level change of ``db`` dB, with the learnt step."""
        return int(round(float(db) / self.db_per_index))


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
    start_gain: int | None = None,
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

    ``start_gain`` (a gain already known to be close, e.g. predicted from an earlier
    level) skips Stage A: Stage B trims from there, and a clip at ``gain_min`` is FATAL
    at once (Stage A's floor check). Stage A's other FATAL check does NOT run: a signal
    already in (or above) the band at ``gain_min`` ("TX too strong for target") is not an
    error from a start gain; a reading in band at ``gain_min`` is accepted, with no
    headroom left below it. ``None`` (the default) runs Stage A as always.
    """
    band_lo = target_dbfs - tol_down_db
    band_hi = target_dbfs + tol_up_db

    def _clamp(g: int) -> int:
        return int(min(max(g, gain_min), gain_max))

    if start_gain is not None:
        gain = _clamp(int(start_gain))
        set_gain(gain)
        return _fine_trim(
            set_gain,
            measure,
            gain,
            (float("nan"), 0),
            band_lo,
            band_hi,
            target_dbfs,
            gain_min,
            gain_max,
            db_per_index,
            max_iterations,
            floor_is_fatal=True,
        )

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
    return _fine_trim(
        set_gain,
        measure,
        gain,
        (peak, railed),
        band_lo,
        band_hi,
        target_dbfs,
        gain_min,
        gain_max,
        db_per_index,
        max_iterations,
    )


def _fine_trim(
    set_gain,
    measure,
    gain: int,
    last: tuple[float, int],
    band_lo: float,
    band_hi: float,
    target_dbfs: float,
    gain_min: int,
    gain_max: int,
    db_per_index: float,
    max_iterations: int,
    *,
    floor_is_fatal: bool = False,
) -> LevelResult:
    """Stage B of :func:`autolevel_orx` from ``gain`` (already set); ``last`` = the reading
    before it ``(peak_dbfs, railed)``, reported if no iteration runs."""

    def _clamp(g: int) -> int:
        return int(min(max(g, gain_min), gain_max))

    peak, railed = last
    fine_iters = 0
    level = OrxStep(db_per_index)  # learns the table's step; 1 index after an overshoot
    for _ in range(max_iterations):
        fine_iters += 1
        peak, railed = measure()
        level.observe(gain, peak, railed, band_lo=band_lo, band_hi=band_hi)
        if railed > 0 and floor_is_fatal and gain <= gain_min:  # Stage A's floor check
            return LevelResult(
                False,
                gain,
                peak,
                fine_iters,
                f"TX too strong: clips at minimum gain ({gain_min}) -- "
                "reduce TX power / add a pad.",
                railed=railed,
                fatal=True,
            )
        if railed > 0:  # clip veto -- always back off
            gain = _clamp(gain - 1)
            set_gain(gain)
            continue
        outside = not band_lo <= peak <= band_hi
        closest = level.closest(target_dbfs) if level.jumped and outside else None
        if closest is not None:  # 1-index steps jump the window: the capture closest to target
            if closest[0] == gain:
                return LevelResult(
                    False,
                    gain,
                    peak,
                    fine_iters,
                    "1-index steps jump the window: the capture closest to the target taken",
                    railed=railed,
                )
            gain = _clamp(closest[0])
            set_gain(gain)
            continue
        if peak > band_hi:  # too hot
            step = -level.step(peak, railed, target_dbfs=target_dbfs, band_hi=band_hi)
            gain = _clamp(gain - step)
            set_gain(gain)
            continue
        if peak < band_lo:  # too cold
            step = level.step(peak, railed, target_dbfs=target_dbfs, band_hi=band_hi)
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
