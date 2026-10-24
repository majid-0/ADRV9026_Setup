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
  with ``railed == 0`` the whole way; index 255 rails hard.
* The ORx saturates BELOW digital full scale with no railed code (UC102, 2026-10-21: an
  overdriven ORx pins its peak near -0.35 dBFS), so a capture is clipped when it railed OR
  its peak is at or above the profile's clip threshold, its measured ceiling minus
  :data:`ORX_CLIP_MARGIN_DB` (:func:`orx_clipped`, :func:`orx_clip_dbfs`); the AGC window
  must stay below that (:func:`check_orx_window`). Overdriven, it can also return a capture
  of all zeros (peak -inf): the stages here treat it as too hot, lower the gain
  :data:`EMPTY_STEP_IDX` indices and measure again, up to :data:`EMPTY_RETAKES` times, then
  FATAL; a peak that is not finite never reaches the gain arithmetic.
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
    "ORX_CEILING_DEFAULT_DBFS",
    "ORX_CLIP_MARGIN_DB",
    "ORX_CLIP_DBFS",
    "ORX_START_MARGIN_IDX",
    "EMPTY_RETAKES",
    "EMPTY_STEP_IDX",
    "orx_clip_dbfs",
    "orx_clipped",
    "check_orx_window",
]

# ORx gain table, characterized on the bench (TX2->ORx2, UC98 link-sharing):
# clean and MONOTONIC from index 185 up to 250 at ~0.5 dB/index (railed==0 the
# whole way); index 255 rails hard. So 185 is the usable floor and 255 the rail.
ORX_GAIN_MIN = 185
ORX_GAIN_MAX = 255
ORX_DB_PER_INDEX = 0.50

#: The ORx saturates below digital full scale, and without railed codes. Bench, 2026-10-21 (UC102
#: link sharing, 40 MHz, 245.76 MSPS, 16-bit): ORX2 and ORX3 overdriven both read peaks of
#: -0.35 / -0.41 / -0.40 dBFS and never a code at the rail. Each profile's ceiling (dBFS) is in
#: the config's ``[orx_ceiling]`` table (``Config.orx_ceiling_dbfs``); a profile not measured
#: yet gets this one.
ORX_CEILING_DEFAULT_DBFS = -0.45
#: A peak at or above the ceiling minus this is a clip (:func:`orx_clip_dbfs`).
ORX_CLIP_MARGIN_DB = 0.2
#: The clip threshold of a profile with the default ceiling: a peak at or above it is a clip
#: (:func:`orx_clipped`), exactly like a railed capture; the AGC window must stay below it
#: (:func:`check_orx_window`). The stages take the profile's own as ``clip_dbfs``.
ORX_CLIP_DBFS = round(ORX_CEILING_DEFAULT_DBFS - ORX_CLIP_MARGIN_DB, 4)
#: A fine trim from a predicted gain (``start_gain``) starts this many indices below it and
#: trims up to the window: it approaches from below, never from the clip.
ORX_START_MARGIN_IDX = 4
#: A capture with no data (all zeros / no samples: the ORx was overdriven) is taken again this
#: many times, the gain :data:`EMPTY_STEP_IDX` indices lower each time (the same gain at the
#: floor, or where the caller fixes the gain); still empty it is FATAL / ``AgcError``.
EMPTY_RETAKES = 2
EMPTY_STEP_IDX = 2


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


def orx_clip_dbfs(
    ceiling_dbfs: float = ORX_CEILING_DEFAULT_DBFS, margin_db: float = ORX_CLIP_MARGIN_DB
) -> float:
    """The clip threshold of a profile whose ORx saturates at ``ceiling_dbfs``: the ceiling minus
    ``margin_db`` (a peak at or above it is a clip, :func:`orx_clipped`)."""
    return round(float(ceiling_dbfs) - float(margin_db), 4)


def orx_clipped(peak_dbfs: float, railed: int, clip_dbfs: float = ORX_CLIP_DBFS) -> bool:
    """An ORx capture clipped: a sample at the rail, or a peak at or above ``clip_dbfs`` (the
    profile's clip threshold, :func:`orx_clip_dbfs`: the ORx saturates below full scale with no
    railed code). Never in band: the AGC lowers the gain, and at the gain floor it is FATAL
    ("clips at minimum gain")."""
    return railed > 0 or peak_dbfs >= clip_dbfs


def check_orx_window(
    target_dbfs: float, tol_up_db: float, clip_dbfs: float = ORX_CLIP_DBFS
) -> None:
    """``ValueError`` unless the AGC window's top, ``target_dbfs + tol_up_db``, is below
    ``clip_dbfs`` (a window reaching it would accept a clipped capture)."""
    top = float(target_dbfs) + float(tol_up_db)
    if not top < clip_dbfs:
        raise ValueError(
            f"ORx AGC window reaches the ORx clip: target_dbfs {target_dbfs} + tol_up_db "
            f"{tol_up_db} = {top:.2f} dBFS, must be below {clip_dbfs} dBFS (the profile's ORx "
            f"ceiling minus {ORX_CLIP_MARGIN_DB} dB: it saturates there with no railed code)"
        )


def no_data_reason(gain: int, retakes: int = EMPTY_RETAKES) -> str:
    """The FATAL / ``AgcError`` reason of an ORx that returned no data (all zeros) ``retakes`` + 1
    times in a row, the last at ``gain``."""
    return (
        f"returned no data (all zeros) after {retakes} retakes at gain {gain} (each retake "
        f"{EMPTY_STEP_IDX} indices lower, down to the floor): the ORx was likely overdriven -- "
        "reduce TX power / add a pad."
    )


def relevel_index(
    peak_dbfs: float, railed: int, *, target_dbfs: float, band_hi: float, db_per_index: float
) -> int:
    """ORx gain index change for a capture that is not leveled (railed, hot or cold).

    One index down when it railed with its peak still in band (the peak dBFS compresses near
    full scale, the railed count does not); otherwise the peak's distance from ``target_dbfs`` in
    indices, at least one (a peak clipped below full scale, :func:`orx_clipped`, is above the
    window, :func:`check_orx_window`: hot, down). The re-level rule of the single-band and the
    multi-band searches (:class:`OrxStep` learns its ``db_per_index``). The searches never pass a
    peak that is not finite (an empty capture is retaken first); if one comes, it is
    :data:`EMPTY_STEP_IDX` down, with no arithmetic on it.
    """
    if not math.isfinite(peak_dbfs):  # no data (an overdriven ORx reads zeros): down
        return -EMPTY_STEP_IDX
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
      across drives (it belongs to the gain table). A clipped capture (:func:`orx_clipped` with
      ``clip_dbfs``, the profile's threshold) teaches no step and is never the closest.
    * Once a re-level overshoots (the peak crosses the window), the steps are 1 index, until the
      drive changes.
    * If a 1-index step still jumps the window, :attr:`jumped` is set and :meth:`closest` names
      the capture to take: the one closest to the target with no railed samples.

    :meth:`relevel` records a capture that is not leveled and returns the index change;
    :meth:`indices` converts a level change in dB with the learnt step.
    """

    def __init__(
        self,
        db_per_index: float = ORX_DB_PER_INDEX,
        bounds=ORX_STEP_BOUNDS,
        clip_dbfs: float = ORX_CLIP_DBFS,
    ):
        self.db_per_index = float(db_per_index)
        self.clip_dbfs = float(clip_dbfs)
        self.bounds = (float(bounds[0]), float(bounds[1]))
        self.single = False  # 1-index steps (a re-level overshot at this drive)
        self.jumped = False  # a 1-index step jumped the window at this drive
        self._caps: list[tuple] = []  # (drive, gain, peak, railed, side) at the current drive

    def _side(self, peak: float, railed: int, band_lo: float, band_hi: float) -> int:
        if orx_clipped(peak, railed, self.clip_dbfs) or peak > band_hi:
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
            clip = self.clip_dbfs
            clipped = orx_clipped(p0, r0, clip) or orx_clipped(peak, railed, clip)  # pinned
            if g0 != gain and not clipped and math.isfinite(p0 + peak):
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
        """``(gain, peak)`` of the capture at this drive closest to the target, not clipped."""
        clip = self.clip_dbfs
        ok = [
            (g, p)
            for _d, g, p, r, _s in self._caps
            if math.isfinite(p) and not orx_clipped(p, r, clip)
        ]
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
    clip_dbfs: float = ORX_CLIP_DBFS,
    start_margin_idx: int = ORX_START_MARGIN_IDX,
) -> LevelResult:
    """Software ORx AGC -- Stage A (coarse) + Stage B (fine trim).

    Drives off a TRUSTED, caller-supplied metric: ``measure()`` returns
    ``(peak_dbfs, railed)`` from a short capture (``clip_report``). ``set_gain(idx)``
    applies an ORx gain-table index. The loop stays pure/hardware-free.

    INVARIANT: no clip (:func:`orx_clipped`: no railed sample, peak below
    ``clip_dbfs``, the profile's clip threshold) at every accepted gain. The accepted band
    is the ASYMMETRIC window ``[target - tol_down, target + tol_up]`` (tighter toward the
    rail), its top below ``clip_dbfs`` (else ``ValueError``). A reading with no data (peak
    not finite: an all-zero or empty capture, the ORx was overdriven) is too hot: the gain
    goes :data:`EMPTY_STEP_IDX` indices down (Stage A: it is at the floor already) and it
    measures again, up to :data:`EMPTY_RETAKES` times; still no data is FATAL. On a FATAL
    floor condition (clipping or already-in-band at ``gain_min``)
    it returns ``fatal=True`` with a message in ``reason`` instead of raising -- the
    orchestrator disables TX, disconnects, and raises :class:`AgcError`.

    If the signal is below band even at ``gain_max`` it ACCEPTS ``gain_max`` as the
    best achievable (``at_max_gain=True``, not fatal).

    ``start_gain`` (a gain already known to be close, e.g. predicted from an earlier
    level) skips Stage A: Stage B trims from ``start_margin_idx`` indices below it
    (:data:`ORX_START_MARGIN_IDX`), so it approaches the window from below and never starts
    in the clip (a predicted gain can be a few dB hot), and a clip at ``gain_min`` is FATAL
    at once (Stage A's floor check). Stage A's other FATAL check does NOT run: a signal
    already in (or above) the band at ``gain_min`` ("TX too strong for target") is not an
    error from a start gain; a reading in band at ``gain_min`` is accepted, with no
    headroom left below it. ``None`` (the default) runs Stage A as always.
    """
    check_orx_window(target_dbfs, tol_up_db, clip_dbfs)
    band_lo = target_dbfs - tol_down_db
    band_hi = target_dbfs + tol_up_db

    def _clamp(g: int) -> int:
        return int(min(max(g, gain_min), gain_max))

    if start_gain is not None:
        gain = _clamp(int(start_gain) - max(0, int(start_margin_idx)))
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
            clip_dbfs=clip_dbfs,
        )

    # -- Stage A: start at the floor, one computed jump --------------------------
    gain = gain_min
    set_gain(gain)
    peak, railed = measure()
    empties = 0
    while not math.isfinite(peak) and empties < EMPTY_RETAKES:  # no data: at the floor, retake
        empties += 1
        peak, railed = measure()
    if not math.isfinite(peak):
        return _no_data(gain, empties, railed)
    if orx_clipped(peak, railed, clip_dbfs):
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
        clip_dbfs=clip_dbfs,
    )


def _no_data(gain: int, iterations: int, railed: int) -> LevelResult:
    """FATAL: no data (peak not finite) after :data:`EMPTY_RETAKES` retakes, the last at
    ``gain``: nothing to step on."""
    return LevelResult(
        False,
        gain,
        float("-inf"),
        iterations,
        f"ORx {no_data_reason(gain)}",
        railed=railed,
        fatal=True,
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
    clip_dbfs: float = ORX_CLIP_DBFS,
) -> LevelResult:
    """Stage B of :func:`autolevel_orx` from ``gain`` (already set); ``last`` = the reading
    before it ``(peak_dbfs, railed)``, reported if no iteration runs. A reading with no data
    (peak not finite) lowers the gain :data:`EMPTY_STEP_IDX` indices and measures again, up to
    :data:`EMPTY_RETAKES` times in a row; then FATAL."""

    def _clamp(g: int) -> int:
        return int(min(max(g, gain_min), gain_max))

    peak, railed = last
    fine_iters = 0
    level = OrxStep(db_per_index, clip_dbfs=clip_dbfs)  # learns the step; 1 index on overshoot
    empties = 0  # readings with no data in a row
    for _ in range(max_iterations):
        fine_iters += 1
        peak, railed = measure()
        if not math.isfinite(peak):  # no data: overdriven -- lower the gain and measure again
            if empties >= EMPTY_RETAKES:
                return _no_data(gain, fine_iters, railed)
            empties += 1
            gain = _clamp(gain - EMPTY_STEP_IDX)
            set_gain(gain)
            continue
        empties = 0
        level.observe(gain, peak, railed, band_lo=band_lo, band_hi=band_hi)
        clipped = orx_clipped(peak, railed, clip_dbfs)
        if clipped and floor_is_fatal and gain <= gain_min:  # Stage A's check
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
        # A peak at or above clip_dbfs (pinned, no railed code) is above band_hi: "too hot" backs
        # off by its distance from the target, a lower bound of how far over the clip it is.
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
        # not clipped and in band -> converged
        return LevelResult(True, gain, peak, fine_iters, "converged in band", railed=railed)

    if empties:  # the last reading had no data: no level to report
        return _no_data(gain, fine_iters, railed)
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
    clip_dbfs: float = ORX_CLIP_DBFS,
) -> LevelResult:
    """Stage C -- verify the settled gain on the FULL signal, backing off on clip.

    A short coarse capture can miss the waveform's true peak, so re-check at the
    full waveform duration: while it clips (:func:`orx_clipped` with ``clip_dbfs``) OR
    exceeds ``target + tol_up``, drop the gain one index and recapture. Colder than band is
    fine (no action). If it backs all the way to ``gain_min`` and STILL clips, that is FATAL
    (``fatal=True``). A reading with no data (peak not finite) drops the gain
    :data:`EMPTY_STEP_IDX` indices and recaptures, up to :data:`EMPTY_RETAKES` times in a row;
    then FATAL.

    ``measure_full()`` returns ``(peak_dbfs, railed)`` from a full-duration capture.
    ``db_per_index`` is accepted for API symmetry (Stage C steps one index at a time).
    """
    check_orx_window(target_dbfs, tol_up_db, clip_dbfs)
    band_hi = target_dbfs + tol_up_db
    gain = int(start_gain)
    set_gain(gain)
    peak, railed = measure_full()
    steps = 0
    empties = 0  # readings with no data in a row
    while not math.isfinite(peak) or orx_clipped(peak, railed, clip_dbfs) or peak > band_hi:
        if not math.isfinite(peak):  # no data: overdriven -- lower the gain and recapture
            if empties >= EMPTY_RETAKES:
                return _no_data(gain, steps, railed)
            empties += 1
            gain = max(int(gain_min), gain - EMPTY_STEP_IDX)
            steps += 1
            set_gain(gain)
            peak, railed = measure_full()
            continue
        empties = 0
        if gain <= gain_min:
            if orx_clipped(peak, railed, clip_dbfs):
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
    converged = not orx_clipped(peak, railed, clip_dbfs) and peak <= band_hi
    reason = "verified: no clip on full signal" if converged else "full signal hot but not railing"
    return LevelResult(converged, gain, peak, steps, reason, railed=railed)
