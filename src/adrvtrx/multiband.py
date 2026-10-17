"""Several TX bands at once: compression search, operating point and saving.

On the dual-band bench, TX1 and TX4 drive one PA, so each band's gain compression depends on
every band's attenuation. This module locks every band to its own compression target together,
backs them off together and saves one case folder.

:func:`search_multiband_compression` is hardware-free (``set_atten``, ``set_gain`` and
``measure`` per band, like :func:`adrvtrx.compression.search_tx_compression`). One step sets
every band's attenuation and ORx gain and takes one reading per band; a band whose ORx is not
leveled moves only its ORx gain and is measured again before its reading counts (up to
``max_relevel`` times; still not leveled, it keeps its attenuation that step and its reading
drives nothing). A compression reading that is not finite (a failed alignment) does not count
either: the band is measured again at the same gain. Every capture is one ``history`` row.
Moves (``method``):

* ``"walk"``: every band by the single-band rule (one coarse step toward the target, then fine
  steps sized to its error, 1 dB of compression assumed per dB).
* ``"slopes"``: one coarse step per band, then a slope matrix ``J`` (how much each band's
  compression moves per dB of each band's attenuation) learnt from the steps taken: it starts
  from each band's own slope (its last two readings, no cross terms) and every step corrects it
  with what happened (Broyden, :func:`update_slopes`; a cross term never exceeds its band's own
  slope), so a cross-band effect is learnt when it exists. The move solves
  ``J @ da = target - c``. When a step made a band's error grow, that step walks and ``J`` starts
  again from the own slopes of the next step.

Both: a band's coarse step is taken at its first reading that counts; moves capped at
``coarse_step_db`` and rounded to the 0.05 dB attenuator step; the ORx gain follows the
predicted output peak (the drive change minus the compression change); a band never goes below
its ``atten_min_db`` (a band that needs more drive than that stops there, the others go on);
clip guard per band (``min_top_slope``); at most ``max_steps`` steps and no move after the last
one, then the step with the smallest worst-band error among the steps where every reading
counted is kept, or else the last step: the radio is left where it was measured. An ORx that
still clips at its minimum gain stops the search (``fatal``), that band one fine step up.

:func:`find_multiband_operating_point` runs it on hardware (a :class:`~adrvtrx.radio.Radio` or
the hardware server's ``RemoteRadio``): retune, play every TX in one ``PerformTx``, ORx AGC per
band, the search, then per backoff every band ``+k`` dB, ORx AGC per band, one capture per band
and the caller's ``check`` on them (nothing after a fatal search). The steps are separate
functions too (:func:`play_bands`, :func:`lock_bands`, :func:`capture_case`) for a campaign that
runs them as separate jobs. :class:`CaseWriter` saves a case folder (write-once, checked empty
before the run; ``case.json`` carries the lock, :func:`lock_summary`).
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ._enums import RxChannel, TxChannel, channel_name
from .capture import aligned_capture, autolevel_for
from .compression import fine_step, quantize_atten, relevel_index
from .conditions import save_aligned_iq
from .config import ClocksConfig, lo_for_tx
from .gain import ORX_DB_PER_INDEX, ORX_GAIN_MAX, ORX_GAIN_MIN
from .metrics import PA_CLIP_SLOPE, gain_compression_db, period_metrics, window_compression_db
from .radio import MAX_TX_ATTEN_DB
from .transmit import transmit_bands
from .waveform import prepare_tx

__all__ = [
    "METHODS",
    "BandSpec",
    "BandLock",
    "MultibandSearch",
    "BandCapture",
    "CaseResult",
    "search_multiband_compression",
    "own_slopes",
    "update_slopes",
    "play_bands",
    "measure_band",
    "lock_bands",
    "capture_case",
    "find_multiband_operating_point",
    "case_name",
    "CaseWriter",
    "lock_summary",
]

METHODS = ("slopes", "walk")


@dataclass
class BandSpec:
    """One band: its path, carrier, signal and compression target.

    ``signal`` is the complex waveform at any scale (it is normalized and quantized to TX codes).
    ``atten_min_db`` is the band's hard limit: the search never goes below it.
    """

    tx: TxChannel
    orx: RxChannel
    lo: str
    freq_hz: int
    bw_mhz: float
    signal: Any
    target_db: float = 4.0
    atten_start_db: float = 20.0
    atten_min_db: float = 10.0
    signal_name: str = ""

    @property
    def name(self) -> str:
        return channel_name(self.tx)

    def describe(self) -> dict[str, Any]:
        """The band as plain values (``case.json``, and the band columns of ``conditions.csv``)."""
        return {
            "name": self.name,
            "tx": self.name,
            "orx": channel_name(self.orx),
            "lo": self.lo,
            "freq_hz": int(self.freq_hz),
            "bw_mhz": self.bw_mhz,
            "signal": self.signal_name,
            "target_db": self.target_db,
            "atten_start_db": self.atten_start_db,
            "atten_min_db": self.atten_min_db,
        }


@dataclass
class BandLock:
    """One band at the end of the search.

    ``status``: ``converged``, ``at limit`` (needs more drive than its limit allows),
    ``clip-limited`` (the PA clips before the target) or ``not converged``.
    """

    name: str
    atten_db: float
    orx_gain: int
    compression_db: float
    papr_compression_db: float
    top_slope: float
    peak_dbfs: float
    railed: int
    floor_db: float
    status: str


@dataclass
class MultibandSearch:
    """The search.

    ``steps`` counts steps (one reading per band each); ``captures`` counts every capture, ORx
    re-leveling included. ``history`` has one row per band per reading.
    """

    converged: bool
    reason: str
    method: str
    bands: list[BandLock]
    steps: int
    captures: int
    fatal: bool = False
    history: list[dict] = field(default_factory=list)


def own_slopes(
    da: np.ndarray, dc: np.ndarray, *, slope_bounds: tuple[float, float] = (0.2, 1.5)
) -> np.ndarray:
    """``J`` from one step: each band's own slope ``-dc/da`` (0.5 if it did not move), no cross
    terms, held in ``slope_bounds`` (dB of compression per dB of attenuation, as a magnitude).
    """
    da, dc = np.asarray(da, dtype=float), np.asarray(dc, dtype=float)
    own = np.array([(-dc[i] / da[i]) if abs(da[i]) > 1e-9 else 0.5 for i in range(len(da))])
    own = np.where(np.isfinite(own), own, 0.5)
    return np.diag(-np.clip(own, *slope_bounds))


def update_slopes(
    jac: np.ndarray,
    da: np.ndarray,
    dc: np.ndarray,
    *,
    slope_bounds: tuple[float, float] = (0.2, 1.5),
) -> np.ndarray:
    """One Broyden update of ``J``: attenuations moved by ``da``, compressions by ``dc``.

    ``J + (dc - J da) da^T / (da^T da)`` makes ``J @ da == dc`` for this step and leaves ``J``
    unchanged across directions it did not explore. Each own slope (diagonal) is then held in
    ``slope_bounds``, and each cross term to at most its row's own slope,
    ``|J[i, j]| <= |J[i, i]|``: band ``j`` reaches band ``i``'s compression only through the
    headroom its average power takes in the shared PA, while band ``i``'s own peaks drive it into
    compression directly, so a dB of the other band's attenuation moves it less than a dB of its
    own. Without the bound one noisy step can teach a cross term larger than the own slope, and
    the solve then moves the wrong band. ``J`` itself is returned when the step is too small or
    not finite.
    """
    jac = np.asarray(jac, dtype=float)
    da, dc = np.asarray(da, dtype=float), np.asarray(dc, dtype=float)
    norm = float(da @ da)
    if not norm > 1e-12:
        return jac
    new = jac + np.outer(dc - jac @ da, da) / norm
    smin, smax = slope_bounds
    for i in range(len(da)):
        new[i, i] = -float(np.clip(-new[i, i], smin, smax))
        own = abs(new[i, i])
        for j in range(len(da)):
            if j != i:
                new[i, j] = float(np.clip(new[i, j], -own, own))
    return new if np.all(np.isfinite(new)) else jac  # never learn from a non-finite step


@dataclass(frozen=True)
class _Rules:
    """The settings of one search (its arguments, checked)."""

    targets: np.ndarray
    names: list[str]
    method: str
    coarse_step_db: float
    fine_step_db: float
    comp_tol_db: float
    target_dbfs: float
    band_lo: float  # the ORx AGC band, dBFS
    band_hi: float
    gain_min: int
    gain_max: int
    db_per_index: float
    atten_max_db: float
    max_relevel: int
    min_top_slope: float | None
    slope_bounds: tuple[float, float]

    def clamp_gain(self, g: float) -> int:
        return int(min(max(int(round(g)), self.gain_min), self.gain_max))


@dataclass
class _State:
    """What the search carries from one step to the next."""

    a: np.ndarray  # attenuation per band (dB), applied at the next step
    g: list[int]  # ORx gain index per band, applied at the next step
    floor: np.ndarray  # lowest attenuation per band: its limit, raised by the clip guard
    clip_floor: list[bool]  # the floor was raised by the clip guard
    coarse_pending: list[bool]  # the band's coarse step is still to come
    jac: np.ndarray | None = None  # the slope matrix J (slopes method)
    prev: tuple[np.ndarray, np.ndarray] | None = None  # (a, c) of the last step that counted
    prev_err: np.ndarray | None = None  # errors of the last step (NaN where they did not count)
    best: dict | None = None  # the counted, unclipped step with the smallest worst-band error
    last: dict | None = None  # the last step, as measured
    history: list[dict] = field(default_factory=list)
    captures: int = 0


@dataclass
class _Reading:
    """A band's reading in a step: what counts, and its history row."""

    comp: float
    papr: float
    slope: float
    peak: float
    railed: int
    counts: bool  # leveled (or below the AGC band at maximum gain) with a finite compression
    row: dict | None = None  # None: a band never read (after a fatal stop)
    fatal: bool = False  # the ORx clips at its minimum gain


@dataclass
class _Classes:
    """One step's readings, classified."""

    c: np.ndarray  # compression per band
    err: np.ndarray  # target - compression
    ok: list[bool]  # the reading counts
    trusted: bool  # every reading counts
    clipped: list[bool]  # the clip guard: top slope below min_top_slope
    within: list[bool]  # within tolerance
    stuck: list[bool]  # below target at its limit
    worst: float  # largest |err|


def _read_band(i: int, step: int, rules: _Rules, state: _State, measure, set_gain) -> _Reading:
    """Band ``i``'s reading: captured again (ORx gain moved, attenuation not) until it counts.

    Every capture that does not count is a history row with its re-level action; after
    ``max_relevel`` of them the last reading is returned not counting (the band holds). A capture
    railed or hot at the minimum gain returns ``fatal`` (its row is not recorded yet).
    """
    nan = float("nan")
    for attempt in range(rules.max_relevel + 1):
        peak, railed, comp, *rest = measure(i)
        state.captures += 1
        extra = dict(rest[0]) if rest and rest[0] is not None else {}
        peak, railed, comp = float(peak), int(railed), float(comp)
        in_band = railed == 0 and rules.band_lo <= peak <= rules.band_hi
        at_max = railed == 0 and peak < rules.band_lo and state.g[i] >= rules.gain_max
        leveled = in_band and math.isfinite(comp)  # NaN: a failed alignment
        cold = at_max and math.isfinite(comp)
        row = {
            "step": step,
            "band": rules.names[i],
            "atten_db": float(state.a[i]),
            "orx_gain": int(state.g[i]),
            "peak_dbfs": peak,
            "railed": railed,
            "gain_compression_db": comp,
            "papr_compression_db": float(extra.get("papr_compression_db", nan)),
            "top_slope": float(extra.get("top_slope", nan)),
            "leveled": leveled,
            "error_db": float(rules.targets[i] - comp),
            "move_db": 0.0,
            "action": "",
            "slopes": "",
        }
        reading = _Reading(
            comp, row["papr_compression_db"], row["top_slope"], peak, railed, leveled or cold, row
        )
        if leveled or cold:
            return reading
        if (railed > 0 or peak > rules.band_hi) and state.g[i] <= rules.gain_min:
            reading.fatal = True
            return reading
        if in_band or at_max:  # level fine, compression not finite: measure again
            row["action"] = "measure again (compression not finite)"
        else:
            k = relevel_index(
                peak,
                railed,
                target_dbfs=rules.target_dbfs,
                band_hi=rules.band_hi,
                db_per_index=rules.db_per_index,
            )  # the single-band search's re-level rule
            state.g[i] = rules.clamp_gain(state.g[i] + k)
            row["action"] = f"ORx gain {k:+d}"
        if attempt == rules.max_relevel:  # this reading drives nothing: the band holds
            row["action"] += f"; not leveled after {attempt + 1} captures, atten held"
        state.history.append(row)
        set_gain(i, int(state.g[i]))
    return reading


def _stop_fatal(i: int, reading: _Reading, readings: list[_Reading], rules: _Rules, state: _State):
    """Band ``i``'s ORx clips at its minimum gain: record the rows of the bands read before it
    ("stopped") and its own, raise its attenuation one fine step; the bands after it are unread.
    Returns every band's reading and error."""
    for r in readings:
        if r.counts:
            r.row["action"] = "stopped"
            state.history.append(r.row)
    reading.row["action"] = "ORx clips at minimum gain: TX atten up one fine step, stop"
    state.history.append(reading.row)
    state.a[i] = quantize_atten(min(rules.atten_max_db, state.a[i] + rules.fine_step_db))
    reading.counts = False
    nan = float("nan")
    unread = _Reading(nan, nan, nan, nan, 0, False)
    readings = readings + [reading] + [unread] * (len(rules.names) - len(readings) - 1)
    return readings, rules.targets - np.array([r.comp for r in readings])


def _classify(readings: list[_Reading], rules: _Rules, state: _State) -> _Classes:
    c = np.array([r.comp for r in readings])
    err = rules.targets - c
    ok = [r.counts for r in readings]
    clipped = [
        rules.min_top_slope is not None
        and r.counts
        and math.isfinite(r.slope)
        and r.slope < rules.min_top_slope
        for r in readings
    ]
    within = [ok[i] and abs(e) <= rules.comp_tol_db for i, e in enumerate(err)]
    stuck = [
        ok[i] and e > rules.comp_tol_db and state.a[i] <= state.floor[i] + 1e-9
        for i, e in enumerate(err)
    ]
    worst = float(np.max(np.abs(err)))
    return _Classes(c, err, ok, all(ok), clipped, within, stuck, worst)


def _remember(readings: list[_Reading], cls: _Classes, state: _State) -> None:
    """Keep the step as the last one, and as the best one when it is."""
    if (
        cls.trusted
        and not any(cls.clipped)
        and (state.best is None or cls.worst < state.best["worst"])
    ):
        state.best = {
            "worst": cls.worst,
            "a": state.a.copy(),
            "g": list(state.g),
            "readings": readings,
            "err": cls.err.copy(),
        }
    state.last = {
        "readings": readings,
        "err": cls.err,
        "a": state.a.copy(),
        "g": [int(r.row["orx_gain"]) for r in readings],  # as captured
    }


def _fixed_moves(readings: list[_Reading], cls: _Classes, rules: _Rules, state: _State):
    """The moves set by a rule rather than solved for: clip guard, at limit, held, coarse.

    Returns ``(moves, fixed, actions)``. The clip guard raises the band's floor; a band's coarse
    step is spent at its first reading that counts.
    """
    n = len(readings)
    mv, fixed, actions = np.zeros(n), [False] * n, [""] * n
    tol, fine, coarse = rules.comp_tol_db, rules.fine_step_db, rules.coarse_step_db
    for i in range(n):
        if cls.clipped[i]:
            over = -cls.err[i]
            up = fine_step(over, fine, coarse) if over > 0 else fine
            new = quantize_atten(min(rules.atten_max_db, state.a[i] + up))
            state.floor[i] = max(state.floor[i], new)
            state.clip_floor[i] = True
            mv[i], fixed[i] = new - state.a[i], True
            actions[i] = f"clip guard (top slope {readings[i].slope:.3f})"
        elif cls.stuck[i]:
            fixed[i], actions[i] = True, "at limit"
        elif not cls.ok[i]:
            fixed[i] = True  # not leveled: its row is in the history already, atten held
        elif state.coarse_pending[i]:  # the band's first reading that counts: single-band rule
            if cls.err[i] > tol:
                mv[i], actions[i] = -coarse, "coarse"
            elif cls.err[i] < -tol:
                mv[i], actions[i] = fine_step(cls.err[i], fine, coarse), "fine"
            else:
                actions[i] = "within tolerance"
            fixed[i] = True
        if cls.ok[i]:
            state.coarse_pending[i] = False
    return mv, fixed, actions


def _error_grew(err: np.ndarray, prev_err: np.ndarray | None, free: list[int], tol: float) -> bool:
    """A free band's error grew since the last step (by more than 0.05 dB, outside tolerance)."""
    return prev_err is not None and any(
        abs(err[i]) > abs(prev_err[i]) + 0.05 and abs(err[i]) > tol for i in free
    )


def _learn_slopes(cls: _Classes, rules: _Rules, state: _State, grew: bool) -> None:
    """Update ``J`` from the step since the last counted one (slopes method)."""
    if grew:
        state.jac = None  # this step walks; J starts again from the own slopes of the next one
    if rules.method == "slopes" and cls.trusted and state.prev is not None and not grew:
        da, dc = state.a - state.prev[0], cls.c - state.prev[1]
        if state.jac is None:
            state.jac = own_slopes(da, dc, slope_bounds=rules.slope_bounds)
        else:
            state.jac = update_slopes(state.jac, da, dc, slope_bounds=rules.slope_bounds)


def _walk_move(err: float, rules: _Rules) -> float:
    step_db = fine_step(err, rules.fine_step_db, rules.coarse_step_db)  # the single-band rule
    return -step_db if err > 0 else step_db


def _free_moves(mv, actions, fixed, free, err, rules: _Rules, jac, use_walk: bool) -> None:
    """The free bands' moves (in place): the walk, or ``J @ da = err`` given the fixed moves."""
    tol = rules.comp_tol_db
    for i in free:
        if abs(err[i]) <= tol and use_walk:
            actions[i] = "within tolerance"
    if not free:
        return
    if use_walk or jac is None:
        for i in free:
            if abs(err[i]) > tol:
                mv[i] = _walk_move(err[i], rules)
                actions[i] = "walk (fallback)" if rules.method == "slopes" else "walk"
        return
    fx = [i for i in range(len(mv)) if fixed[i]]
    rhs = err[free] - (jac[np.ix_(free, fx)] @ mv[fx] if fx else 0.0)
    try:
        sol = np.linalg.solve(jac[np.ix_(free, free)], rhs)
        if not np.all(np.isfinite(sol)):
            raise np.linalg.LinAlgError("non-finite move")
        mv[free] = sol
        for i in free:
            actions[i] = "slopes"
    except np.linalg.LinAlgError:
        for i in free:
            if abs(err[i]) > tol:
                mv[i] = _walk_move(err[i], rules)
                actions[i] = "walk (singular slopes)"


def _new_attens(mv, actions, free, err, rules: _Rules, state: _State) -> np.ndarray:
    """Moves capped at the coarse step, at least one attenuator step for a band still out of
    tolerance (slopes), then held between the floor and the maximum and quantized."""
    mv = np.clip(mv, -rules.coarse_step_db, rules.coarse_step_db)
    for i in free:
        if abs(err[i]) > rules.comp_tol_db and abs(mv[i]) < 0.05 and actions[i] == "slopes":
            mv[i] = -0.05 if err[i] > 0 else 0.05
    return np.array(
        [
            quantize_atten(min(max(state.a[i] + mv[i], state.floor[i]), rules.atten_max_db))
            for i in range(len(mv))
        ]
    )


def _predict_gains(g: list[int], moved: np.ndarray, jac, rules: _Rules) -> list[int]:
    """The ORx gain follows the output peak, which moves by the drive change minus the
    compression change: predicted from the slopes when known, else from a typical own slope of
    0.5 dB per dB."""
    dc = jac @ moved if jac is not None else -0.5 * moved
    return [
        rules.clamp_gain(g[i] + round((moved[i] + dc[i]) / rules.db_per_index))
        for i in range(len(g))
    ]


def _band_locks(readings: list[_Reading], err, rules: _Rules, state: _State) -> list[BandLock]:
    out = []
    tol = rules.comp_tol_db
    for i, r in enumerate(readings):
        below = r.counts and err[i] > tol and state.a[i] <= state.floor[i] + 1e-9
        if r.counts and abs(err[i]) <= tol:
            status = "converged"
        elif state.clip_floor[i] and below:
            status = "clip-limited"
        elif below:
            status = "at limit"
        else:
            status = "not converged"
        out.append(
            BandLock(
                rules.names[i],
                float(state.a[i]),
                int(state.g[i]),
                r.comp,
                r.papr,
                r.slope,
                r.peak,
                r.railed,
                float(state.floor[i]),
                status,
            )
        )
    return out


def search_multiband_compression(
    set_atten: Callable[[int, float], None],
    set_gain: Callable[[int, int], None],
    measure: Callable[[int], tuple],
    *,
    targets_db: Sequence[float],
    start_attens_db: Sequence[float],
    atten_mins_db: Sequence[float],
    orx_gains: Sequence[int],
    names: Sequence[str] | None = None,
    method: str = "slopes",
    coarse_step_db: float = 2.5,
    fine_step_db: float = 0.2,
    comp_tol_db: float = 0.1,
    target_dbfs: float = -1.0,
    tol_up_db: float = 0.3,
    tol_down_db: float = 0.6,
    gain_min: int = ORX_GAIN_MIN,
    gain_max: int = ORX_GAIN_MAX,
    db_per_index: float = ORX_DB_PER_INDEX,
    atten_max_db: float = MAX_TX_ATTEN_DB,
    max_steps: int = 20,
    max_relevel: int = 6,
    min_top_slope: float | None = PA_CLIP_SLOPE,
    slope_bounds: tuple[float, float] = (0.2, 1.5),
    on_step: Callable[[list[dict]], None] | None = None,
) -> MultibandSearch:
    """Lock every band within ``±comp_tol_db`` of its target together (see the module docstring).

    ``measure(i)`` -> ``(peak_dbfs, railed, compression_db, extra)`` for band ``i``, ``extra``
    holding ``papr_compression_db`` and ``top_slope``. ``on_step(rows)`` gets each step's rows
    (one per band).
    The radio is left at the returned attenuations and gains.

    The defaults differ from the single-band :func:`~adrvtrx.compression.search_tx_compression`
    (coarse 10 dB, fine 0.25 dB, tolerance 0.3 dB, clip guard off), and stay so: changing either
    set would change locks already taken with it.

    * ``coarse_step_db=2.5``: a band's move also moves every other band's compression (shared
      PA), so the first jump is kept small enough for the bands to stay where their slopes can be
      learnt; one band alone can take 10 dB from a safe start.
    * ``comp_tol_db=0.1``, ``fine_step_db=0.2``: the search steers on gain compression (all the
      samples of the top 2.5 % of the peaks; the single-band default steers on the PAPR of one
      window around the peak, hence its 0.3 dB). The tolerance window is then 0.2 dB wide, and
      near the lock a dB of attenuation moves about 0.5 to 1 dB of compression, so a 0.25 dB fine
      step could step over it; 0.2 dB (four attenuator steps) does not.
    * ``min_top_slope=PA_CLIP_SLOPE``: one band's drive can push another band's peaks into
      clipping, so the clip guard is on; single band keeps it opt-in, as it was introduced.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    n = len(targets_db)
    if not (len(start_attens_db) == len(atten_mins_db) == len(orx_gains) == n):
        raise ValueError(
            "targets, start attenuations, limits and ORx gains need one value per band"
        )
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if not comp_tol_db > 0:
        raise ValueError(f"comp_tol_db must be positive, got {comp_tol_db}")
    rules = _Rules(
        targets=np.asarray(targets_db, dtype=float),
        names=list(names) if names is not None else [f"band{i + 1}" for i in range(n)],
        method=method,
        coarse_step_db=coarse_step_db,
        fine_step_db=fine_step_db,
        comp_tol_db=comp_tol_db,
        target_dbfs=target_dbfs,
        band_lo=target_dbfs - tol_down_db,
        band_hi=target_dbfs + tol_up_db,
        gain_min=gain_min,
        gain_max=gain_max,
        db_per_index=db_per_index,
        atten_max_db=atten_max_db,
        max_relevel=max_relevel,
        min_top_slope=min_top_slope,
        slope_bounds=slope_bounds,
    )
    floor = np.array([quantize_atten(m) for m in atten_mins_db])
    for i, s in enumerate(start_attens_db):
        if floor[i] > s:
            raise ValueError(f"{rules.names[i]}: limit {floor[i]} dB is above the start {s} dB")
    state = _State(
        a=np.array(
            [quantize_atten(min(max(s, f), atten_max_db)) for s, f in zip(start_attens_db, floor)]
        ),
        g=[rules.clamp_gain(x) for x in orx_gains],
        floor=floor,
        clip_floor=[False] * n,
        coarse_pending=[True] * n,  # until the band's first reading that counts
    )

    def apply() -> None:
        for i in range(n):
            set_atten(i, float(state.a[i]))
            set_gain(i, int(state.g[i]))

    def finish(converged: bool, reason: str, readings, err, *, fatal=False) -> MultibandSearch:
        apply()
        return MultibandSearch(
            converged,
            reason,
            method,
            _band_locks(readings, err, rules, state),
            len({h["step"] for h in state.history}),
            state.captures,
            fatal,
            state.history,
        )

    def report(rows: list[dict]) -> None:
        if on_step:
            on_step(rows)

    for step in range(max_steps):
        apply()
        readings: list[_Reading] = []
        for i in range(n):
            reading = _read_band(i, step, rules, state, measure, set_gain)
            if reading.fatal:
                readings, err = _stop_fatal(i, reading, readings, rules, state)
                return finish(
                    False,
                    f"{rules.names[i]}: ORx still clips at minimum gain",
                    readings,
                    err,
                    fatal=True,
                )
            readings.append(reading)
        cls = _classify(readings, rules, state)
        _remember(readings, cls, state)

        if not any(cls.clipped) and all(w or s for w, s in zip(cls.within, cls.stuck)):
            rows = [r.row for r in readings]
            for row in rows:
                row["action"] = "converged" if all(cls.within) else row["action"] or "done"
            state.history.extend(rows)
            report(rows)
            if all(cls.within):
                return finish(True, "every band within tolerance", readings, cls.err)
            held = ", ".join(rules.names[i] for i, s in enumerate(cls.stuck) if s)
            return finish(
                False,
                f"{held} at the attenuation limit below target; the others within tolerance",
                readings,
                cls.err,
            )
        if step == max_steps - 1:  # no move after the last step: nothing would measure it
            rows = [r.row for r in readings]
            for i, row in enumerate(rows):
                if cls.ok[i]:
                    row["action"] = "max steps"
                    state.history.append(row)
            report(rows)
            break

        mv, fixed, actions = _fixed_moves(readings, cls, rules, state)
        free = [i for i in range(n) if not fixed[i]]
        grew = _error_grew(cls.err, state.prev_err, free, comp_tol_db)
        _learn_slopes(cls, rules, state, grew)
        use_walk = method == "walk" or grew or not cls.trusted
        _free_moves(mv, actions, fixed, free, cls.err, rules, state.jac, use_walk)
        new_a = _new_attens(mv, actions, free, cls.err, rules, state)
        moved = new_a - state.a
        rows = []
        slopes = "" if state.jac is None else json.dumps(np.round(state.jac, 4).tolist())
        for i, r in enumerate(readings):
            if cls.ok[i]:  # a band not leveled has its row already (its own action, no move)
                r.row.update(move_db=float(moved[i]), action=actions[i], slopes=slopes)
                state.history.append(r.row)
            rows.append(r.row)
        report(rows)
        if np.all(np.abs(moved) < 1e-9) and cls.trusted:
            return finish(False, "no band can move further (limits)", readings, cls.err)
        state.prev = (state.a.copy(), cls.c.copy()) if cls.trusted else None
        state.prev_err = np.where(cls.ok, cls.err, np.nan)  # what did not count cannot grow
        state.g = _predict_gains(state.g, moved, state.jac, rules)
        state.a = new_a

    if state.best is not None:
        state.a, state.g = state.best["a"], state.best["g"]
        return finish(
            False,
            "max steps; left at the step closest to the targets",
            state.best["readings"],
            state.best["err"],
        )
    state.a, state.g = state.last["a"], list(state.last["g"])
    return finish(
        False,
        "max steps; left at the last step measured",
        state.last["readings"],
        state.last["err"],
    )


# -- hardware ---------------------------------------------------------------------------------


def play_bands(radio, bands: Sequence[BandSpec], *, tx_bits: int) -> list[np.ndarray]:
    """Play every band's signal on its TX in one ``PerformTx`` (they start together).

    Returns each band's reference in TX codes (what ``prepare_tx`` plays). The signals must have
    equal length.
    """
    refs = []
    for b in bands:
        i_tx, q_tx = prepare_tx(np.asarray(b.signal, dtype=np.complex128), tx_bits)
        refs.append(i_tx.astype(np.float64) + 1j * q_tx.astype(np.float64))
    if len({len(r) for r in refs}) != 1:
        raise ValueError(
            f"every band's signal must have the same length, got {[len(r) for r in refs]}"
        )
    transmit_bands(radio, {b.tx: r for b, r in zip(bands, refs)}, tx_bits, do_normalize=False)
    return refs


def measure_band(
    radio,
    orx: RxChannel,
    ref,
    *,
    rx_bits: int,
    fs: float,
    oversample: int = 2,
    window_s: float = 1e-4,
    fraction: float | None = None,
):
    """One capture of ``orx`` (``oversample`` periods).

    Returns ``(peak_dbfs, railed, gain_compression_db, extra)``.

    ``extra`` holds ``papr_compression_db``, ``top_slope``, the raw capture (``raw``, complex
    codes) and the
    aligned period (``y_aligned``) with its ``delay_samples``.

    ``fraction`` (the path's ``delay % 1`` from an earlier full alignment) aligns with
    :func:`~adrvtrx.align.align_with_fraction` instead of the full
    :func:`~adrvtrx.align.estimate_and_align` (no upsampled refinement): the search's readings
    use it; a case capture aligns in full.
    """
    cap = aligned_capture(
        radio, orx, ref, bits=rx_bits, fs=fs, oversample=oversample, fraction=fraction
    )
    papr, _pin, _pout = window_compression_db(cap.x_aligned, cap.y_aligned, fs, window_s)
    gain_comp, slope = gain_compression_db(cap.x_aligned, cap.y_aligned)
    extra = {
        "papr_compression_db": papr,
        "top_slope": slope,
        "raw": cap.iq,
        "y_aligned": cap.y_aligned,
        "delay_samples": cap.delay_samples,
    }
    return cap.clip.peak_dbfs, cap.clip.railed_samples, gain_comp, extra


def _check_los(bands: Sequence[BandSpec], clocks) -> None:
    """Each band's ``lo`` must be the LO ``[clocks]`` routes its TX to, and one LO one frequency."""
    freqs: dict[str, int] = {}
    for b in bands:
        if isinstance(clocks, ClocksConfig) and b.lo != lo_for_tx(clocks, b.tx):
            raise ValueError(
                f"{b.name}: lo is {b.lo!r}, but [clocks] routes {b.name} to "
                f"{lo_for_tx(clocks, b.tx)!r}"
            )
        hz = freqs.setdefault(b.lo, int(b.freq_hz))
        if hz != int(b.freq_hz):
            raise ValueError(f"{b.lo} is shared by bands at {hz} Hz and {int(b.freq_hz)} Hz")


def lock_bands(
    radio,
    bands: Sequence[BandSpec],
    *,
    tx_bits: int,
    rx_bits: int,
    fs: float,
    retune: bool = True,
    method: str = "slopes",
    oversample: int = 2,
    target_dbfs: float = -1.0,
    tol_up_db: float = 0.3,
    tol_down_db: float = 0.6,
    on_step=None,
    **search_kw: Any,
) -> tuple[MultibandSearch, list[np.ndarray]]:
    """Retune, play every band, ORx AGC per band at the start attenuations, then the search.

    Returns the search and each band's reference (TX codes). TX keeps playing (the caller
    captures next). A band's first reading leveled in the AGC band is aligned in full; its
    fractional delay is kept and the band's later readings only search the integer lag
    (:func:`measure_band` ``fraction``). Before the radio is touched, each band's ``lo`` is
    checked against the LO the radio's ``[clocks]`` routes its TX to, and bands that share an
    LO must share its frequency (``ValueError``).
    """
    _check_los(bands, getattr(getattr(radio, "config", None), "clocks", None))
    radio.disable_tx()
    for b in bands:
        radio.set_tx_atten(b.tx, quantize_atten(b.atten_start_db))
    if retune:
        for lo, hz in {b.lo: int(b.freq_hz) for b in bands}.items():
            radio.retune_lo(lo, hz)
    refs = play_bands(radio, bands, tx_bits=tx_bits)
    gains = [
        int(
            autolevel_for(
                radio,
                b.orx,
                r,
                bits=rx_bits,
                fs=fs,
                oversample=oversample,
                target_dbfs=target_dbfs,
                tol_up_db=tol_up_db,
                tol_down_db=tol_down_db,
            ).final_gain_index
        )
        for b, r in zip(bands, refs)
    ]
    fractions: dict[int, float] = {}  # per band: delay % 1 of its first leveled reading

    def measure(i: int):
        reading = measure_band(
            radio,
            bands[i].orx,
            refs[i],
            rx_bits=rx_bits,
            fs=fs,
            oversample=oversample,
            fraction=fractions.get(i),
        )
        peak, railed = reading[0], reading[1]
        if i not in fractions and railed == 0 and -tol_down_db <= peak - target_dbfs <= tol_up_db:
            fractions[i] = float(reading[3]["delay_samples"]) % 1.0
        return reading

    search = search_multiband_compression(
        lambda i, db: radio.set_tx_atten(bands[i].tx, db),
        lambda i, gi: radio.set_rx_gain(bands[i].orx, gi),
        measure,
        targets_db=[b.target_db for b in bands],
        start_attens_db=[b.atten_start_db for b in bands],
        atten_mins_db=[b.atten_min_db for b in bands],
        orx_gains=gains,
        names=[b.name for b in bands],
        method=method,
        target_dbfs=target_dbfs,
        tol_up_db=tol_up_db,
        tol_down_db=tol_down_db,
        on_step=on_step,
        **search_kw,
    )
    return search, refs


@dataclass
class BandCapture:
    """One band at one state: the raw capture (complex codes), the aligned period, metrics.

    ``ref`` is the band's transmitted reference (TX codes), so a check needs nothing else.
    """

    name: str
    atten_db: float
    orx_gain: int
    raw: np.ndarray
    y_aligned: np.ndarray
    peak_dbfs: float
    railed: int
    metrics: dict[str, float]
    tries: int = 1
    check: dict = field(default_factory=dict)
    ref: np.ndarray | None = None


def capture_case(
    radio,
    bands: Sequence[BandSpec],
    refs: Sequence[np.ndarray],
    attens_db: Sequence[float],
    *,
    rx_bits: int,
    fs: float,
    oversample: int = 2,
    agc: bool = True,
    target_dbfs: float = -1.0,
    tol_up_db: float = 0.3,
    tol_down_db: float = 0.6,
    check: Callable[[list[BandCapture]], dict[str, dict]] | None = None,
    retakes: int = 3,
    min_top_slope: float = PA_CLIP_SLOPE,
    gains: Sequence[int] | None = None,
) -> list[BandCapture]:
    """Every band at ``attens_db`` (TX playing): ORx AGC, one capture per band, then ``check``.

    ``check(captures)`` -> ``{band name: {"ok": bool, ...}}`` sees every band of the state at once
    (pairing needs both). A failing band is captured again (re-leveled first when it railed), up to
    ``retakes`` times;
    its last check result stays in :attr:`BandCapture.check`.

    ``agc=False`` captures at ``gains`` (one ORx gain index per band, set here) without leveling.
    The ORx gain cannot be read back (``RxGainGet`` returns 0 on the chip; the gain is tracked in
    software), so ``agc=False`` needs ``gains``: each :class:`BandCapture` records the gain set.
    With ``agc=True``, ``gains`` (optional) is where each band's AGC starts its fine trim instead
    of the gain floor (a gain predicted from the lock); a railed retake levels from the floor.
    """
    if gains is not None and len(gains) != len(bands):
        raise ValueError(f"gains needs one ORx gain index per band, got {list(gains)}")
    if not agc and gains is None:
        raise ValueError(
            "agc=False needs gains= (the ORx gain of each band): the ORx gain cannot be read back"
        )
    for b, db in zip(bands, attens_db):
        radio.set_tx_atten(b.tx, quantize_atten(db))
    used: dict[str, int] = {}  # the ORx gain in effect per band (set here, never read back)
    out: list[BandCapture | None] = [None] * len(bands)

    def take(i: int, tries: int, relevel: bool) -> BandCapture:
        b, ref = bands[i], refs[i]
        if relevel or b.name not in used:
            if agc:
                level = autolevel_for(
                    radio,
                    b.orx,
                    ref,
                    bits=rx_bits,
                    fs=fs,
                    oversample=oversample,
                    target_dbfs=target_dbfs,
                    tol_up_db=tol_up_db,
                    tol_down_db=tol_down_db,
                    start_gain=None if gains is None or tries > 1 else int(gains[i]),
                )
                used[b.name] = int(level.final_gain_index)
            else:
                used[b.name] = int(gains[i])  # type: ignore[index]
                radio.set_rx_gain(b.orx, used[b.name])
        peak, railed, _comp, extra = measure_band(
            radio, b.orx, ref, rx_bits=rx_bits, fs=fs, oversample=oversample
        )
        metrics = period_metrics(
            ref,
            extra["y_aligned"],
            fs=fs,
            bw_hz=float(b.bw_mhz) * 1e6,
            rx_bits=rx_bits,
            delay_samples=extra["delay_samples"],
            min_top_slope=min_top_slope,
        )
        return BandCapture(
            b.name,
            float(quantize_atten(attens_db[i])),
            int(used[b.name]),
            extra["raw"],
            extra["y_aligned"],
            float(peak),
            int(railed),
            metrics,
            tries,
            ref=ref,
        )

    for i in range(len(bands)):
        out[i] = take(i, 1, relevel=True)
    if check is None:
        return out  # type: ignore[return-value]
    for attempt in range(retakes + 1):
        results = check(out) or {}
        failing = []
        for i, c in enumerate(out):
            res = dict(results.get(c.name, {"ok": True}))
            c.check = res
            if not res.get("ok", True):
                failing.append(i)
        if not failing or attempt == retakes:
            break
        for i in failing:
            out[i] = take(i, out[i].tries + 1, relevel=out[i].railed > 0)
    return out  # type: ignore[return-value]


@dataclass
class CaseResult:
    """A case: the lock, and per backoff the captures of every band."""

    search: MultibandSearch
    refs: list[np.ndarray]
    captures: dict[int, list[BandCapture]]
    case_dir: Path | None = None


def find_multiband_operating_point(
    radio,
    bands: Sequence[BandSpec],
    *,
    tx_bits: int,
    rx_bits: int,
    fs: float,
    backoffs: Sequence[int] = (0, 1, 2, 3, 4),
    method: str = "slopes",
    check: Callable[[list[BandCapture]], dict[str, dict]] | None = None,
    retakes: int = 3,
    writer: CaseWriter | None = None,
    retune: bool = True,
    oversample: int = 2,
    on_step=None,
    **search_kw: Any,
) -> CaseResult:
    """Lock every band together, then capture every backoff (every band ``+k`` dB together).

    With ``writer`` (a :class:`CaseWriter`), the inputs, every aligned output, ``conditions.csv``
    and ``lock.csv`` are written. The case folder must be empty: that is checked before the radio
    is touched. If writing still fails, the error raised carries the result as ``case_result``
    (the captures are not lost). TX is disabled when the call ends or fails. A backoff is added
    to the lock, so it must be ``>= 0`` (a band never goes below its limit).

    A fatal search (the ORx still clips at its minimum gain) captures and writes nothing: the
    result has the search and no captures. A lock that did not converge (a band at its limit,
    clip-limited) is the operating point at that limit: its backoffs are captured.
    """
    if any(k < 0 for k in backoffs):
        raise ValueError(f"backoffs must be >= 0, got {list(backoffs)}")
    _check_los(bands, getattr(getattr(radio, "config", None), "clocks", None))
    if writer is not None:
        writer.require_empty()
    captures: dict[int, list[BandCapture]] = {}
    try:
        search, refs = lock_bands(
            radio,
            bands,
            tx_bits=tx_bits,
            rx_bits=rx_bits,
            fs=fs,
            retune=retune,
            method=method,
            oversample=oversample,
            on_step=on_step,
            **search_kw,
        )
        locked = [bl.atten_db for bl in search.bands]
        for k in () if search.fatal else backoffs:
            attens = [min(a + int(k), MAX_TX_ATTEN_DB) for a in locked]
            captures[int(k)] = capture_case(
                radio,
                bands,
                refs,
                attens,
                rx_bits=rx_bits,
                fs=fs,
                oversample=oversample,
                check=check,
                retakes=retakes,
                gains=_backoff_gains(search, int(k), search_kw),
            )
    finally:
        try:
            radio.disable_tx()
        except Exception:  # noqa: BLE001 - leave the bench safe on any path
            pass
    result = CaseResult(search, refs, captures)
    if writer is not None and not search.fatal:
        try:
            writer.write_inputs(refs)
            for k, caps in captures.items():
                for c in caps:
                    writer.write_output(k, c.name, c.y_aligned)
            writer.write_conditions(search, captures)
            writer.write_lock(search)
        except Exception as exc:
            exc.case_result = result  # the bench run is done: keep its captures
            raise
        result.case_dir = writer.dir
    return result


def _backoff_gains(search: MultibandSearch, k: int, search_kw: dict) -> list[int]:
    """Each band's ORx gain predicted at ``k`` dB of backoff, where its AGC starts.

    The search's rule: the output peak moves by the drive change minus the compression change,
    with a typical own slope of 0.5 dB per dB, so the gain goes up ``k / 2`` dB. (Up the full
    ``k`` dB overshoots: from a 4 dB lock the peak falls only ~1.5 dB in 4 dB of backoff, and a
    railed start costs the AGC one index per capture.)
    """
    lo = int(search_kw.get("gain_min", ORX_GAIN_MIN))
    hi = int(search_kw.get("gain_max", ORX_GAIN_MAX))
    per_db = 1.0 / float(search_kw.get("db_per_index", ORX_DB_PER_INDEX))
    return [int(min(max(bl.orx_gain + round(0.5 * k * per_db), lo), hi)) for bl in search.bands]


# -- saving -----------------------------------------------------------------------------------


def _token(value: float) -> str:
    """``1440.0`` -> ``1440``; ``758.5`` -> ``758p5`` (file-name safe)."""
    v = float(value)
    return f"{v:g}".replace(".", "p") if not v.is_integer() else str(int(v))


def case_name(bands: Sequence[BandSpec]) -> str:
    """``TX1-TX4_1440-1860MHz_80-80BW_4-4GC``: paths, carriers, bandwidths, compression targets."""
    return (
        "-".join(b.name for b in bands)
        + "_"
        + "-".join(_token(b.freq_hz / 1e6) for b in bands)
        + "MHz"
        + "_"
        + "-".join(_token(b.bw_mhz) for b in bands)
        + "BW"
        + "_"
        + "-".join(_token(b.target_db) for b in bands)
        + "GC"
    )


class CaseWriter:
    """One case folder ``<run_dir>/<case_name>/``, write-once (an existing file is an error).

    ``meta`` (plan row, filters, pads, profile, programming id, ...) goes into ``case.json`` and,
    flattened,
    into every ``conditions.csv`` row it names in ``row_meta``.

    Making a writer creates nothing: the case folder appears with the first file written (or
    :meth:`path`, which names a file about to be written), after :meth:`require_empty` has run.
    """

    def __init__(
        self,
        run_dir: str | Path,
        bands: Sequence[BandSpec],
        *,
        tx_bits: int,
        rx_bits: int,
        fs: float,
        meta: dict | None = None,
        row_meta: Sequence[str] = ("case", "version", "profile", "program_id"),
    ):
        self.bands = list(bands)
        self.name = case_name(bands)
        self.dir = Path(run_dir) / self.name  # created by the first write, not here
        self.tx_bits, self.rx_bits, self.fs = tx_bits, rx_bits, fs
        self.meta = dict(meta or {})
        self.row_meta = list(row_meta)
        self.search: MultibandSearch | None = None  # the lock, once written (for case.json)

    def path(self, suffix: str) -> Path:
        """``<dir>/<case>_<suffix>``, a case file about to be written: the folder is created."""
        self.dir.mkdir(parents=True, exist_ok=True)
        return self.dir / f"{self.name}_{suffix}"

    def require_empty(self) -> None:
        """Raise ``FileExistsError`` unless the case folder is empty (check before a run)."""
        found = sorted(p.name for p in self.dir.iterdir()) if self.dir.is_dir() else []
        if found:
            shown = ", ".join(found[:3]) + (", ..." if len(found) > 3 else "")
            raise FileExistsError(
                f"{self.dir} is not empty ({shown}): a case folder is written once (move it "
                "aside to redo; write case.json after the run)"
            )

    def _new(self, path: Path) -> Path:
        if path.exists():
            raise FileExistsError(
                f"{path} exists: files are written once (move the case folder aside to redo)"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_inputs(self, refs: Sequence[np.ndarray]) -> list[Path]:
        out = []
        for b, ref in zip(self.bands, refs):
            p = self._new(self.path(f"{b.name}_in.txt"))
            save_aligned_iq(ref, p, self.tx_bits)
            out.append(p)
        return out

    def write_output(self, backoff_db: int, band: str, y_aligned: np.ndarray) -> Path:
        p = self._new(self.path(f"{int(backoff_db)}dB_{band}_out.txt"))
        save_aligned_iq(y_aligned, p, self.rx_bits)
        return p

    def condition_rows(
        self, search: MultibandSearch, captures: dict[int, list[BandCapture]]
    ) -> list[dict]:
        locks = {bl.name: bl for bl in search.bands}
        freqs = "/".join(_token(b.freq_hz / 1e6) for b in self.bands)
        rows = []
        for k, caps in captures.items():
            for b, c in zip(self.bands, caps):
                bl = locks[b.name]
                band = b.describe()
                rows.append(
                    {
                        **{key: self.meta.get(key, "") for key in self.row_meta},
                        "band": band["name"],
                        **{key: band[key] for key in ("tx", "orx", "lo", "freq_hz", "bw_mhz")},
                        "signal": band["signal"],
                        "freqs_mhz": freqs,
                        "target_db": band["target_db"],
                        "backoff_db": int(k),
                        "locked_atten_db": bl.atten_db,
                        "tx_atten_db": c.atten_db,
                        "orx_gain": c.orx_gain,
                        "lock_status": bl.status,
                        "lock_converged": search.converged,
                        "lock_reason": search.reason,
                        "search_method": search.method,
                        "search_steps": search.steps,
                        "search_captures": search.captures,
                        "peak_dbfs": c.peak_dbfs,
                        "railed": c.railed,
                        **c.metrics,
                        "fs_hz": self.fs,
                        "tries": c.tries,
                        "check_ok": bool(c.check.get("ok", True)) if c.check else "",
                        "check": json.dumps(c.check, default=str) if c.check else "",
                        "out_file": self.path(f"{int(k)}dB_{b.name}_out.txt").name,
                        "in_file": self.path(f"{b.name}_in.txt").name,
                    }
                )
        return rows

    def _write_csv(self, name: str, rows: list[dict]) -> Path:
        p = self._new(self.dir / name)
        cols: list[str] = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        with open(p, "x", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        return p

    def write_conditions(
        self, search: MultibandSearch, captures: dict[int, list[BandCapture]]
    ) -> Path:
        self.search = search
        return self._write_csv("conditions.csv", self.condition_rows(search, captures))

    def write_lock(self, search: MultibandSearch) -> Path:
        self.search = search
        return self._write_csv("lock.csv", [dict(h) for h in search.history])

    def write_case_json(
        self, extra: dict | None = None, *, search: MultibandSearch | None = None
    ) -> Path:
        """``case.json``: the bands, ``fs_hz``, the lock (:func:`lock_summary` of ``search``, or of
        the search this writer wrote), then ``meta`` and ``extra`` (which win on a shared key).
        """
        p = self._new(self.dir / "case.json")
        search = search if search is not None else self.search
        doc = {
            "case_name": self.name,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "bands": [b.describe() for b in self.bands],
            "fs_hz": self.fs,
            **({"lock": lock_summary(search)} if search is not None else {}),
            **self.meta,
            **(extra or {}),
        }
        p.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
        return p

    def write_text(self, name: str, text: str) -> Path:
        """Any other file of the case (e.g. ``report_0dB.md``), write-once."""
        p = self._new(self.dir / name)
        p.write_text(text, encoding="utf-8")
        return p


def lock_summary(search: MultibandSearch) -> list[dict]:
    """The lock per band as plain dicts (for ``case.json``)."""
    return [
        {
            "band": bl.name,
            "atten_db": bl.atten_db,
            "orx_gain": bl.orx_gain,
            "gain_compression_db": bl.compression_db,
            "top_slope": bl.top_slope,
            "status": bl.status,
        }
        for bl in search.bands
    ]
