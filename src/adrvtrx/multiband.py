"""Several TX bands at once: compression search, operating point and saving.

On the dual-band bench, TX1 and TX4 drive one PA, so each band's gain compression depends on
every band's attenuation. This module locks every band to its own compression target together,
backs them off together and saves one case folder.

:func:`search_multiband_compression` is hardware-free (``set_atten``, ``set_gain`` per band and
``measure(bands)``, one snapshot reading those bands). One step sets every band's attenuation and
ORx gain and takes ONE snapshot: a capture that reads every band's ORx (``PerformRx`` returns them
all), so a step costs one capture, not one per band (``snapshot=False``: one capture per band,
for a pair the ``two_orx_snapshot`` bench check failed). A reading counts when it has no railed
sample, a peak between ``target_dbfs - 3`` and 0 dBFS and a finite compression, even outside the
ORx AGC window (its gain is still corrected for the next step), or below the window at the
maximum gain. A reading that does not count moves only its ORx gain and the band is captured
again (up to ``max_relevel`` times; still not counting, it keeps its attenuation that step). The
re-level learns each ORx's dB per index (:class:`~adrvtrx.gain.OrxStep`). A railed capture, or
one hotter than the window, at the minimum ORx gain stops the search (``fatal``), that band one
clip step (0.2 dB) up. When no band moved for two steps in a row and every band not within
tolerance was at its limit or held at an unchanged ORx gain, the search stops. Every band's
reading is one ``history`` row.

The compression per band depends on every attenuation, ``c ~ c0 + J da``: ``J`` holds each
band's own slope (dB of compression per dB of attenuation, negative: less attenuation, more
compression) on the diagonal and the cross-band coupling off it. ``J`` starts from
``slope_prior`` (own 0.31, cross 0.06 dB/dB: the first bench lock, S20-100 TX2 + TX3) and every
step that follows a counted step updates it with what was measured (Broyden,
:func:`update_slopes`), held to own slopes in 0.1 .. 1.5 and each cross term to at most half its
row's own slope (diagonally dominant, so invertible). Moves (``method``):

* ``"joint"`` (default): every band moves every step, ``J @ da = target - c`` solved for all of
  them together. A band within tolerance takes no move unless the other bands' moves would push
  it out (the cross terms predict it). The trust region: every move is capped at ``max_move_db``
  (4 dB); a step that makes the worst band's error grow is undone halfway and the cap halves;
  two steps in a row that improve it restore the cap.
* ``"pingpong"``: one band at a time. The active band runs a 1-D secant search (the other bands
  held) until it is within tolerance or has taken ``turn_steps`` (3) steps, then the next band;
  rounds repeat until every band is within tolerance in one snapshot. Each snapshot reads every
  band anyway, so the cross slopes are learnt too: the active band aims where the other bands'
  coming moves will leave it (the joint solve's move for that band).

Both: moves are continuous, quantized to the 0.05 dB attenuator step (at least one step for a
band out of tolerance); a band never goes below its ``atten_min_db`` (a band that needs more
drive than that is held there, the others are solved with it fixed); clip guard per band
(``min_top_slope``); at most ``max_steps`` steps and no move after the last one, then the step
with the smallest worst-band error among the steps where every reading counted is kept, or else
the last step: the radio is left where it was measured. The walk and the old ``slopes`` method
were removed (``joint`` replaces both).

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
from .capture import aligned_capture, aligned_captures, autolevel_for, shares_orx_adc
from .compression import fine_step, quantize_atten
from .conditions import save_aligned_iq
from .config import ClocksConfig, lo_for_tx
from .gain import ORX_DB_PER_INDEX, ORX_GAIN_MAX, ORX_GAIN_MIN, OrxStep
from .metrics import PA_CLIP_SLOPE, gain_compression_db, period_metrics, window_compression_db
from .radio import MAX_TX_ATTEN_DB
from .transmit import transmit_bands
from .waveform import prepare_tx

__all__ = [
    "METHODS",
    "SLOPE_PRIOR",
    "BandSpec",
    "BandLock",
    "MultibandSearch",
    "BandCapture",
    "CaseResult",
    "search_multiband_compression",
    "prior_slopes",
    "update_slopes",
    "play_bands",
    "measure_band",
    "measure_bands",
    "lock_bands",
    "capture_case",
    "find_multiband_operating_point",
    "case_name",
    "CaseWriter",
    "lock_summary",
    "lock_details",
    "timing_text",
    "confirm_lock",
    "make_probes",
    "probe_cut",
    "probe_length",
    "PROBE_MS",
]

METHODS = ("joint", "pingpong")
#: Methods that are gone, and why (a clear error for a config that still names one).
REMOVED_METHODS = {
    "walk": "the walk was removed: it moved 1 dB per dB of compression error (the bench PA gives "
    "about 0.3), so every move was a third of what was needed",
    "slopes": "'slopes' was replaced by 'joint' (one snapshot per step, a slope prior, bounds on "
    "the slopes and a trust region)",
}
#: Removed search arguments and what replaces them.
REMOVED_ARGS = {
    "coarse_step_db": "max_move_db caps every move (there is no coarse first step any more)",
    "fine_step_db": "moves are continuous, quantized to the 0.05 dB attenuator step",
    "slope_bounds": "the slope bounds are fixed: own 0.1 .. 1.5, cross at most half the own",
}
#: ``(own, cross)`` slope prior, dB of compression per dB of attenuation, as magnitudes: the
#: first bench lock (S20-100, TX2 0.56 -> 2.70 dB over 6.8 dB; TX3 +0.43 dB for TX2's +7 dB).
SLOPE_PRIOR = (0.31, 0.06)
#: Own slopes are held in this range; each cross term to at most half its row's own slope.
OWN_SLOPE_BOUNDS = (0.1, 1.5)
#: The clip guard's step up, and the fatal stop's (the single-band fine step).
CLIP_STEP_DB = 0.2
#: A reading counts with its peak this far below ``target_dbfs`` (and up to 0 dBFS, not railed).
USABLE_BELOW_DB = 3.0
#: The worst error "grew" (trust region) when it rose by more than this (reading noise).
GROW_MARGIN_DB = 0.05
#: Pingpong: a band's turn ends within tolerance or after this many steps.
TURN_STEPS = 3


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

    ``steps`` counts steps; ``snapshots`` the ``measure`` calls (one per step, plus ORx
    re-levels); ``captures`` every capture (a snapshot is one, or one per band read without
    ``snapshot``). ``history`` has one row per band per reading. ``slopes`` is the final ``J``.
    ``timing`` holds ``captures_s`` (each snapshot's time, capture and reading) and, from
    :func:`lock_bands`, ``agc_s``, ``swaps_s`` and ``total_s``. ``probe`` (a lock on the probe
    signal, :func:`lock_bands` ``probe_ms``) and ``confirm`` (the full-signal confirmation,
    :func:`confirm_lock`) describe them.
    """

    converged: bool
    reason: str
    method: str
    bands: list[BandLock]
    steps: int
    captures: int
    fatal: bool = False
    history: list[dict] = field(default_factory=list)
    snapshots: int = 0
    snapshot: bool = True
    slopes: list = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    probe: dict = field(default_factory=dict)
    confirm: dict = field(default_factory=dict)


def prior_slopes(n: int, prior: Sequence[float] = SLOPE_PRIOR) -> np.ndarray:
    """``J`` before any step: ``-own`` on the diagonal, ``-cross`` off it (``prior`` as magnitudes:
    less attenuation, more compression)."""
    own, cross = float(prior[0]), float(prior[1])
    return -(np.full((n, n), cross) + np.eye(n) * (own - cross))


def _bound_slopes(jac: np.ndarray, own_bounds: tuple[float, float]) -> np.ndarray:
    out = jac.copy()
    for i in range(len(out)):
        out[i, i] = -float(np.clip(-out[i, i], *own_bounds))
        half = abs(out[i, i]) / 2.0
        for j in range(len(out)):
            if j != i:
                out[i, j] = float(np.clip(out[i, j], -half, half))
    return out


def update_slopes(
    jac: np.ndarray,
    da: np.ndarray,
    dc: np.ndarray,
    *,
    own_bounds: tuple[float, float] = OWN_SLOPE_BOUNDS,
    rows: Sequence[int] | None = None,
) -> np.ndarray:
    """One Broyden update of ``J``: attenuations moved by ``da``, compressions by ``dc``.

    ``J + (dc - J da) da^T / (da^T da)`` makes ``J @ da == dc`` for this step and leaves ``J``
    unchanged across directions it did not explore (a step of one band alone learns that band's
    column exactly: its own slope and its effect on the others). Each own slope (diagonal) is
    then held in ``own_bounds`` and each cross term to at most half its row's own slope,
    ``|J[i, j]| <= |J[i, i]| / 2``: band ``j`` reaches band ``i``'s compression only through the
    headroom its average power takes in the shared PA, so a dB of the other band's attenuation
    moves it less than a dB of its own; the bound keeps ``J`` diagonally dominant, so the solve
    never moves the wrong band after a noisy step. ``J`` itself is returned when the step is too
    small or not finite. ``rows``: only those bands' rows learn (the bands whose compression was
    read at both ends of the step); the others keep theirs.
    """
    jac = np.asarray(jac, dtype=float)
    da, dc = np.asarray(da, dtype=float), np.asarray(dc, dtype=float)
    norm = float(da @ da)
    rows = list(range(len(da))) if rows is None else list(rows)
    if not norm > 1e-12 or not rows:
        return jac
    new = jac.copy()
    new[rows] += np.outer(dc[rows] - jac[rows] @ da, da) / norm
    new = _bound_slopes(new, own_bounds)
    return new if np.all(np.isfinite(new)) else jac  # never learn from a non-finite step


@dataclass(frozen=True)
class _Rules:
    """The settings of one search (its arguments, checked)."""

    targets: np.ndarray
    names: list[str]
    method: str
    max_move_db: float
    comp_tol_db: float
    target_dbfs: float
    band_lo: float  # the ORx AGC window, dBFS
    band_hi: float
    gain_min: int
    gain_max: int
    atten_max_db: float
    max_relevel: int
    min_top_slope: float | None
    snapshot: bool
    turn_steps: int

    def clamp_gain(self, g: float) -> int:
        return int(min(max(int(round(g)), self.gain_min), self.gain_max))


@dataclass
class _State:
    """What the search carries from one step to the next."""

    a: np.ndarray  # attenuation per band (dB), applied at the next step
    g: list[int]  # ORx gain index per band, applied at the next step
    floor: np.ndarray  # lowest attenuation per band: its limit, raised by the clip guard
    clip_floor: list[bool]  # the floor was raised by the clip guard
    jac: np.ndarray  # the slope matrix J
    orx: list[OrxStep]  # each band's ORx re-level (learns its dB per index)
    cap: float  # the trust region: the largest move per band this step
    improving: int = 0  # steps in a row whose worst error fell
    undone: bool = False  # the last move undid half of the one before
    last_move: np.ndarray | None = None  # the attenuation move of the last step
    prev: tuple[np.ndarray, np.ndarray] | None = (
        None  # (a, c) of the last step (c NaN: not counted)
    )
    prev_worst: float | None = None  # worst error of the last step (bands that could move)
    active: int | None = None  # pingpong: the band whose turn it is
    turn: int = 0  # pingpong: steps taken in that turn
    best: dict | None = None  # the counted, unclipped step with the smallest worst-band error
    last: dict | None = None  # the last step, as measured
    history: list[dict] = field(default_factory=list)
    captures: int = 0
    snapshots: int = 0
    capture_s: list[float] = field(default_factory=list)
    idle_steps: int = 0  # steps in a row where nothing moved and no band could (see _idle)


@dataclass
class _Reading:
    """A band's reading in a step: what counts, and its history row."""

    comp: float
    papr: float
    slope: float
    peak: float
    railed: int
    counts: bool  # no railed sample, peak usable, finite compression (see the module docstring)
    row: dict | None = None  # None: a band never read
    fatal: bool = False  # the ORx clips (or is hot) at its minimum gain
    in_window: bool = False  # inside the ORx AGC window


def _evaluate(i: int, m, step: int, dt: float, rules: _Rules, state: _State) -> _Reading:
    """Band ``i``'s reading ``m`` (``(peak, railed, comp, extra)``) of a snapshot, classified."""
    nan = float("nan")
    peak, railed, comp, *rest = m
    extra = dict(rest[0]) if rest and rest[0] is not None else {}
    peak, railed, comp = float(peak), int(railed), float(comp)
    g = int(state.g[i])
    in_window = railed == 0 and rules.band_lo <= peak <= rules.band_hi
    usable = railed == 0 and rules.target_dbfs - USABLE_BELOW_DB <= peak <= 0.0
    cold_at_max = railed == 0 and peak < rules.band_lo and g >= rules.gain_max
    fatal = (railed > 0 or peak > rules.band_hi) and g <= rules.gain_min
    orx = state.orx[i]
    orx.observe(g, peak, railed, band_lo=rules.band_lo, band_hi=rules.band_hi, drive=tuple(state.a))
    closest = orx.closest(rules.target_dbfs) if orx.jumped else None
    taken = closest is not None and closest[0] == g and railed == 0  # 1-index steps jump it
    counts = not fatal and math.isfinite(comp) and (usable or cold_at_max or taken)
    row = {
        "step": step,
        "band": rules.names[i],
        "atten_db": float(state.a[i]),
        "orx_gain": g,
        "peak_dbfs": peak,
        "railed": railed,
        "gain_compression_db": comp,
        "papr_compression_db": float(extra.get("papr_compression_db", nan)),
        "top_slope": float(extra.get("top_slope", nan)),
        "leveled": in_window and math.isfinite(comp),
        "counts": counts,
        "error_db": float(rules.targets[i] - comp),
        "move_db": 0.0,
        "action": "",
        "slopes": "",
        "capture_s": round(float(dt), 3),
    }
    return _Reading(
        comp,
        row["papr_compression_db"],
        row["top_slope"],
        peak,
        railed,
        counts,
        row,
        fatal,
        in_window,
    )


def _read_step(step: int, rules: _Rules, state: _State, measure, set_gain):
    """Every band's reading of this step: one snapshot, then again for the bands whose reading
    does not count (ORx gain moved, attenuation not), until each counts.

    Every capture that does not count is a history row with its re-level action; after
    ``max_relevel`` of them the last reading is returned not counting (the band holds). Returns
    ``(readings, fatal band or None)``; a fatal band's row is not recorded yet.
    """
    n = len(rules.names)
    readings: list[_Reading | None] = [None] * n
    pending = list(range(n))
    for attempt in range(rules.max_relevel + 1):
        t0 = time.perf_counter()
        got = list(measure(list(pending)))
        dt = time.perf_counter() - t0
        state.snapshots += 1
        state.captures += 1 if rules.snapshot else len(pending)
        state.capture_s.append(round(dt, 3))
        for i, m in zip(pending, got):
            readings[i] = _evaluate(i, m, step, dt, rules, state)
        fatal = [i for i in pending if readings[i].fatal]
        if fatal:
            return readings, fatal[0]
        again = []
        for i in pending:
            r = readings[i]
            if r.counts:
                continue
            orx = state.orx[i]
            closest = orx.closest(rules.target_dbfs) if orx.jumped else None
            usable_level = r.railed == 0 and (
                rules.target_dbfs - USABLE_BELOW_DB <= r.peak <= 0.0
                or (r.peak < rules.band_lo and state.g[i] >= rules.gain_max)
            )
            if usable_level:  # level fine, compression not finite: measure again
                r.row["action"] = "measure again (compression not finite)"
            elif closest is not None:  # back to the closest capture: it is taken next time
                k = closest[0] - state.g[i]
                state.g[i] = rules.clamp_gain(closest[0])
                r.row["action"] = f"ORx gain {k:+d} (the closest capture)"
            else:
                k = orx.step(r.peak, r.railed, target_dbfs=rules.target_dbfs, band_hi=rules.band_hi)
                state.g[i] = rules.clamp_gain(state.g[i] + k)
                r.row["action"] = f"ORx gain {k:+d}"
            if attempt == rules.max_relevel:  # this reading drives nothing: the band holds
                r.row["action"] += f"; not leveled after {attempt + 1} captures, atten held"
            else:
                again.append(i)
            state.history.append(r.row)
            set_gain(i, int(state.g[i]))
        pending = again
        if not pending:
            break
    return readings, None


def _stop_fatal(i: int, readings: list[_Reading], rules: _Rules, state: _State):
    """Band ``i``'s ORx clips (or is hot) at its minimum gain: record the other bands' rows of
    this step ("stopped") and its own, raise its attenuation one clip step. Returns every band's
    reading and error."""
    recorded = {id(h) for h in state.history}
    for k, r in enumerate(readings):
        if k != i and r is not None and r.row is not None and id(r.row) not in recorded:
            r.row["action"] = "stopped"
            state.history.append(r.row)
    reading = readings[i]
    reading.row["action"] = "ORx clips at minimum gain: TX atten up one clip step, stop"
    state.history.append(reading.row)
    state.a[i] = quantize_atten(min(rules.atten_max_db, state.a[i] + CLIP_STEP_DB))
    reading.counts = False
    return readings, rules.targets - np.array([r.comp for r in readings])


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
    """The moves set by a rule rather than solved for: clip guard, at limit, not counted.

    Returns ``(moves, fixed, actions)``. The clip guard raises the band's floor.
    """
    n = len(readings)
    mv, fixed, actions = np.zeros(n), [False] * n, [""] * n
    for i in range(n):
        if cls.clipped[i]:
            over = -cls.err[i]
            up = fine_step(over, CLIP_STEP_DB, rules.max_move_db) if over > 0 else CLIP_STEP_DB
            new = quantize_atten(min(rules.atten_max_db, state.a[i] + up))
            state.floor[i] = max(state.floor[i], new)
            state.clip_floor[i] = True
            mv[i], fixed[i] = new - state.a[i], True
            actions[i] = f"clip guard (top slope {readings[i].slope:.3f})"
        elif cls.stuck[i]:
            fixed[i], actions[i] = True, "at limit"
        elif not cls.ok[i]:
            fixed[i] = True  # not counted: its row is in the history already, atten held
    return mv, fixed, actions


def _solve(mv, free: list[int], fixed: list[int], err, jac) -> bool:
    """``J @ da = err`` for the ``free`` bands given the ``fixed`` bands' moves (in place)."""
    if not free:
        return True
    rhs = err[free] - (jac[np.ix_(free, fixed)] @ mv[fixed] if fixed else 0.0)
    try:
        sol = np.linalg.solve(jac[np.ix_(free, free)], rhs)
    except np.linalg.LinAlgError:
        return False
    if not np.all(np.isfinite(sol)):
        return False
    mv[free] = sol
    return True


def _worst_free(cls: _Classes, fixed: list[bool]) -> float | None:
    """The worst |error| among the counted bands a move can help (not fixed by a rule)."""
    errs = [abs(cls.err[i]) for i in range(len(fixed)) if cls.ok[i] and not fixed[i]]
    return max(errs) if errs else None


def _trust(cls: _Classes, fixed: list[bool], rules: _Rules, state: _State, only=None) -> bool:
    """The trust region's bookkeeping for this step; True when the last step must be undone
    halfway (it made the worst error grow; never twice in a row). ``only``: the error of that
    band alone (pingpong's active band)."""
    if only is None:
        worst = _worst_free(cls, fixed)
    else:
        worst = abs(cls.err[only]) if cls.ok[only] and not fixed[only] else None
    prev, state.prev_worst = state.prev_worst, worst
    if worst is None or prev is None:
        state.undone = False
        return False
    if worst > prev + GROW_MARGIN_DB and not state.undone and state.last_move is not None:
        state.cap = max(state.cap / 2.0, 0.05)
        state.improving, state.undone = 0, True
        return True
    state.undone = False
    state.improving = state.improving + 1 if worst < prev else 0
    if state.improving >= 2:
        state.cap = rules.max_move_db
    return False


def _joint_moves(mv, fixed, actions, cls: _Classes, rules: _Rules, state: _State) -> list[int]:
    """Every band at once (in place); returns the bands whose move was solved for."""
    n, tol = len(mv), rules.comp_tol_db
    hold = [i for i in range(n) if not fixed[i] and cls.within[i]]
    free = [i for i in range(n) if not fixed[i] and not cls.within[i]]
    fx = [i for i in range(n) if fixed[i]]
    ok = _solve(mv, free, fx + hold, cls.err, state.jac)
    if ok and hold:  # a band within tolerance moves when the others' moves would push it out
        pushed = [i for i in hold if abs(cls.err[i] - state.jac[i] @ mv) > tol]
        if pushed:
            free = sorted(free + pushed)
            hold = [i for i in hold if i not in pushed]
            ok = _solve(mv, free, fx + hold, cls.err, state.jac)
    for i in hold:
        actions[i] = "within tolerance"
    for i in free:
        actions[i] = "joint" if ok else "held (singular slopes)"
        if not ok:
            mv[i] = 0.0
    return free if ok else []


def _pingpong_moves(mv, fixed, actions, cls: _Classes, rules: _Rules, state: _State) -> list[int]:
    """One band at a time (in place): the active band's move of the joint solve (it aims where
    the other bands' coming moves will leave it), the others held. Returns the moved band."""
    n, tol = len(mv), rules.comp_tol_db
    movable = [i for i in range(n) if not fixed[i]]
    plan = np.zeros(n)
    plan[[i for i in range(n) if fixed[i]]] = mv[[i for i in range(n) if fixed[i]]]
    if not _solve(plan, movable, [i for i in range(n) if fixed[i]], cls.err, state.jac):
        for i in movable:
            actions[i] = "held (singular slopes)"
        return []
    aim = {i: abs(state.jac[i, i] * plan[i]) for i in movable}  # error left after the others

    def needs(i: int) -> bool:
        return aim[i] > tol or not cls.within[i]

    a = state.active
    if a is None or a not in movable or aim[a] <= tol or state.turn >= rules.turn_steps:
        others = [i for i in movable if i != a and needs(i)]
        if not others and a in movable and needs(a) and state.turn < rules.turn_steps:
            others = [a]
        elif not others:
            others = [i for i in movable if needs(i)]
        a = max(others, key=lambda i: (aim[i], abs(cls.err[i]))) if others else None
        if a != state.active:
            state.turn = 0
            state.prev_worst = None if a is None else abs(cls.err[a])  # the new turn's baseline
        state.active = a
    for i in movable:
        actions[i] = "within tolerance" if cls.within[i] else "held (other band's turn)"
    if a is None:
        return []
    mv[a] = plan[a] if aim[a] > tol else cls.err[a] / state.jac[a, a]
    actions[a] = f"pingpong (turn step {state.turn + 1})"
    state.turn += 1
    return [a]


def _new_attens(mv, solved, err, rules: _Rules, state: _State) -> np.ndarray:
    """Solved moves capped at the trust region, at least one attenuator step for a band still
    out of tolerance, then every band held between its floor and the maximum, quantized."""
    for i in solved:
        mv[i] = float(np.clip(mv[i], -state.cap, state.cap))
        if abs(err[i]) > rules.comp_tol_db and abs(mv[i]) < 0.05:
            mv[i] = -0.05 if err[i] > 0 else 0.05
    return np.array(
        [
            quantize_atten(min(max(state.a[i] + mv[i], state.floor[i]), rules.atten_max_db))
            for i in range(len(mv))
        ]
    )


def _idle(moved: np.ndarray, cls: _Classes, gains_before: list[int], gains: list[int]) -> bool:
    """No band moved, and every band not within tolerance is stuck at its limit or held with
    its ORx gain where the step found it (re-leveling got nowhere, e.g. compression readings that
    are not finite): the next step can only capture the same thing again. A held band whose gain
    moved is still leveling."""
    return bool(np.all(np.abs(moved) < 1e-9)) and all(
        w or s or (not ok and g0 == g1)
        for w, s, ok, g0, g1 in zip(cls.within, cls.stuck, cls.ok, gains_before, gains)
    )


#: Steps in a row with nothing to do before the search stops (held bands re-level meanwhile).
IDLE_STEPS = 2


def _idle_reason(cls: _Classes, rules: _Rules) -> str:
    parts = []
    for label, flags in (
        ("not leveled at an unchanged ORx gain", [not ok for ok in cls.ok]),
        ("at the attenuation limit", cls.stuck),
        ("within tolerance", cls.within),
    ):
        names = [rules.names[i] for i, f in enumerate(flags) if f]
        if names:
            parts.append(f"{', '.join(names)} {label}")
    return f"no band could move for {IDLE_STEPS} steps ({'; '.join(parts)})"


def _next_gains(readings: list[_Reading], moved: np.ndarray, rules: _Rules, state: _State):
    """The ORx gain follows the output peak, which moves by the drive change minus the
    compression change (predicted from ``J``), with each ORx's learnt dB per index. A reading
    that counted outside the AGC window is also brought back to ``target_dbfs``."""
    dc = state.jac @ moved
    out = []
    for i, r in enumerate(readings):
        level = rules.target_dbfs - r.peak if r.counts and not r.in_window else 0.0
        out.append(rules.clamp_gain(state.g[i] + state.orx[i].indices(moved[i] + dc[i] + level)))
    return out


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


def _check_method(method: str) -> None:
    if method in REMOVED_METHODS:
        raise ValueError(f"method {method!r}: {REMOVED_METHODS[method]}; use one of {METHODS}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def _check_prior(prior) -> tuple[float, float]:
    try:
        own, cross = (float(v) for v in prior)
    except (TypeError, ValueError):
        raise ValueError(f"slope_prior is (own, cross) in dB/dB, got {prior!r}") from None
    lo, hi = OWN_SLOPE_BOUNDS
    if not (lo <= own <= hi and 0.0 <= cross <= own / 2.0):
        raise ValueError(
            f"slope_prior {prior!r}: own in {lo} .. {hi} dB/dB, cross in 0 .. own / 2 dB/dB"
        )
    return own, cross


def search_multiband_compression(
    set_atten: Callable[[int, float], None],
    set_gain: Callable[[int, int], None],
    measure: Callable[[list[int]], Sequence[tuple]],
    *,
    targets_db: Sequence[float],
    start_attens_db: Sequence[float],
    atten_mins_db: Sequence[float],
    orx_gains: Sequence[int],
    names: Sequence[str] | None = None,
    method: str = "joint",
    max_move_db: float = 4.0,
    slope_prior: Sequence[float] = SLOPE_PRIOR,
    slopes: Sequence[Sequence[float]] | None = None,
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
    turn_steps: int = TURN_STEPS,
    snapshot: bool = True,
    on_step: Callable[[list[dict]], None] | None = None,
    **removed: Any,
) -> MultibandSearch:
    """Lock every band within ``±comp_tol_db`` of its target together (see the module docstring).

    ``measure(bands)`` -> one ``(peak_dbfs, railed, compression_db, extra)`` per band in
    ``bands`` (indices), from one snapshot, ``extra`` holding ``papr_compression_db`` and
    ``top_slope``. ``snapshot=False`` counts one capture per band read (the hardware side then
    captures each band alone). ``slopes`` starts ``J`` from a matrix (a search continued) instead
    of ``slope_prior``. ``on_step(rows)`` gets each step's rows (one per band). The radio is left
    at the returned attenuations and gains.

    The defaults differ from the single-band :func:`~adrvtrx.compression.search_tx_compression`
    (tolerance 0.3 dB, clip guard off), and stay so:

    * ``comp_tol_db=0.1``: the search steers on gain compression (all the samples of the top
      2.5 % of the peaks; the single-band default steers on the PAPR of one window around the
      peak, hence its 0.3 dB).
    * ``min_top_slope=PA_CLIP_SLOPE``: one band's drive can push another band's peaks into
      clipping, so the clip guard is on; single band keeps it opt-in, as it was introduced.
    """
    if removed:
        k = sorted(removed)[0]
        if k in REMOVED_ARGS:
            raise ValueError(f"{k} was removed: {REMOVED_ARGS[k]}")
        raise TypeError(f"search_multiband_compression() got an unexpected keyword argument {k!r}")
    _check_method(method)
    prior = _check_prior(slope_prior)
    n = len(targets_db)
    if not (len(start_attens_db) == len(atten_mins_db) == len(orx_gains) == n):
        raise ValueError(
            "targets, start attenuations, limits and ORx gains need one value per band"
        )
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if not comp_tol_db > 0:
        raise ValueError(f"comp_tol_db must be positive, got {comp_tol_db}")
    if not max_move_db > 0:
        raise ValueError(f"max_move_db must be positive, got {max_move_db}")
    rules = _Rules(
        targets=np.asarray(targets_db, dtype=float),
        names=list(names) if names is not None else [f"band{i + 1}" for i in range(n)],
        method=method,
        max_move_db=float(max_move_db),
        comp_tol_db=comp_tol_db,
        target_dbfs=target_dbfs,
        band_lo=target_dbfs - tol_down_db,
        band_hi=target_dbfs + tol_up_db,
        gain_min=gain_min,
        gain_max=gain_max,
        atten_max_db=atten_max_db,
        max_relevel=max_relevel,
        min_top_slope=min_top_slope,
        snapshot=bool(snapshot),
        turn_steps=int(turn_steps),
    )
    floor = np.array([quantize_atten(m) for m in atten_mins_db])
    for i, s in enumerate(start_attens_db):
        if floor[i] > s:
            raise ValueError(f"{rules.names[i]}: limit {floor[i]} dB is above the start {s} dB")
    jac = prior_slopes(n, prior)
    if slopes is not None:
        jac = _bound_slopes(np.asarray(slopes, dtype=float).reshape(n, n), OWN_SLOPE_BOUNDS)
    state = _State(
        a=np.array(
            [quantize_atten(min(max(s, f), atten_max_db)) for s, f in zip(start_attens_db, floor)]
        ),
        g=[rules.clamp_gain(x) for x in orx_gains],
        floor=floor,
        clip_floor=[False] * n,
        jac=jac,
        orx=[OrxStep(db_per_index) for _ in range(n)],
        cap=float(max_move_db),
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
            snapshots=state.snapshots,
            snapshot=rules.snapshot,
            slopes=np.round(state.jac, 4).tolist(),
            timing={"captures_s": list(state.capture_s)},
        )

    def report(rows: list[dict]) -> None:
        if on_step:
            on_step(rows)

    stop = "max steps"
    for step in range(max_steps):
        apply()
        gains_before = list(state.g)
        readings, fatal_band = _read_step(step, rules, state, measure, set_gain)
        if fatal_band is not None:
            readings, err = _stop_fatal(fatal_band, readings, rules, state)
            return finish(
                False,
                f"{rules.names[fatal_band]}: ORx still clips at minimum gain",
                readings,
                err,
                fatal=True,
            )
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

        if state.prev is not None:  # learn J from the step since the last one
            da, dc = state.a - state.prev[0], cls.c - state.prev[1]
            rows = [i for i in range(n) if cls.ok[i] and np.isfinite(state.prev[1][i])]
            state.jac = update_slopes(state.jac, da, dc, rows=rows)
        mv, fixed, actions = _fixed_moves(readings, cls, rules, state)
        only = None if method == "joint" else state.active  # pingpong: the active band's error
        if _trust(cls, fixed, rules, state, only):  # the last step made it worse: undo it halfway
            solved = []
            for i in range(n):
                if not fixed[i]:
                    mv[i] = -0.5 * state.last_move[i]
                    actions[i] = f"undo half (error grew; cap {state.cap:g} dB)"
        elif method == "joint":
            solved = _joint_moves(mv, fixed, actions, cls, rules, state)
        else:
            solved = _pingpong_moves(mv, fixed, actions, cls, rules, state)
        new_a = _new_attens(mv, solved, cls.err, rules, state)
        moved = new_a - state.a
        rows = []
        slopes_text = json.dumps(np.round(state.jac, 4).tolist())
        for i, r in enumerate(readings):
            if cls.ok[i]:  # a band not counted has its row already (its own action, no move)
                r.row.update(move_db=float(moved[i]), action=actions[i], slopes=slopes_text)
                state.history.append(r.row)
            rows.append(r.row)
        report(rows)
        if np.all(np.abs(moved) < 1e-9) and cls.trusted:
            return finish(False, "no band can move further (limits)", readings, cls.err)
        idle = _idle(moved, cls, gains_before, state.g)
        state.idle_steps = state.idle_steps + 1 if idle else 0
        if state.idle_steps >= IDLE_STEPS:  # a band that never levels: stop capturing it
            stop = _idle_reason(cls, rules)
            break
        state.prev = (state.a.copy(), np.where(cls.ok, cls.c, np.nan))
        state.last_move = moved
        state.g = _next_gains(readings, moved, rules, state)
        state.a = new_a

    if state.best is not None:
        state.a, state.g = state.best["a"], state.best["g"]
        return finish(
            False,
            f"{stop}; left at the step closest to the targets",
            state.best["readings"],
            state.best["err"],
        )
    state.a, state.g = state.last["a"], list(state.last["g"])
    return finish(
        False,
        f"{stop}; left at the last step measured",
        state.last["readings"],
        state.last["err"],
    )


# -- hardware ---------------------------------------------------------------------------------

#: The probe's default length, ms (0.05 ms: 24,576 samples at 491.52 MSPS).
PROBE_MS = 0.05
#: The probe lengths accepted, ms.
PROBE_MS_RANGE = (0.005, 0.5)


def tx_codes(bands: Sequence[BandSpec], *, tx_bits: int) -> list[np.ndarray]:
    """Each band's signal in TX codes (what ``prepare_tx`` plays: unit peak, full scale)."""
    refs = []
    for b in bands:
        i_tx, q_tx = prepare_tx(np.asarray(b.signal, dtype=np.complex128), tx_bits)
        refs.append(i_tx.astype(np.float64) + 1j * q_tx.astype(np.float64))
    return refs


def play_codes(radio, bands: Sequence[BandSpec], codes: Sequence[np.ndarray], *, tx_bits: int):
    """Play each band's TX codes as they are (no normalisation) on its TX in one ``PerformTx``
    (they start together; equal lengths)."""
    if len({len(r) for r in codes}) != 1:
        raise ValueError(
            f"every band's signal must have the same length, got {[len(r) for r in codes]}"
        )
    transmit_bands(radio, {b.tx: r for b, r in zip(bands, codes)}, tx_bits, do_normalize=False)


def play_bands(radio, bands: Sequence[BandSpec], *, tx_bits: int) -> list[np.ndarray]:
    """Play every band's signal on its TX in one ``PerformTx`` (they start together).

    Returns each band's reference in TX codes (what ``prepare_tx`` plays). The signals must have
    equal length.
    """
    refs = tx_codes(bands, tx_bits=tx_bits)
    play_codes(radio, bands, refs, tx_bits=tx_bits)
    return refs


def probe_length(probe_ms: float, fs: float, block: int = 1) -> int:
    """``round(probe_ms * fs)`` samples, rounded up to a multiple of the TX memory's ``block``."""
    if not PROBE_MS_RANGE[0] <= float(probe_ms) <= PROBE_MS_RANGE[1]:
        raise ValueError(f"probe_ms {probe_ms} outside {PROBE_MS_RANGE[0]} .. {PROBE_MS_RANGE[1]}")
    block = max(1, int(block))
    n = int(round(float(probe_ms) * 1e-3 * float(fs)))
    return max(block, -(-n // block) * block)


def _papr(x: np.ndarray) -> float:
    p = np.abs(x) ** 2
    return float(10 * np.log10(p.max() / p.mean()))


def probe_cut(ref: np.ndarray, n: int) -> tuple[np.ndarray, dict]:
    """The probe of one band: ``n`` samples of ``ref`` (TX codes, one period of a loop) around
    its peak, the same codes (no new normalisation, so the peak drive is the full signal's).

    Centred on ``argmax|ref|``, the start moves by up to ``n // 20`` samples to make the wrap jump
    ``|w[0] - w[-1]|`` (the probe loops) smallest. Returns ``(w, info)``: ``n``, ``start`` (in
    ``ref``, circular), ``shift`` (from the centred cut), the jump and the centred cut's jump in
    RMS units of ``ref``, the PAPR of the probe and of ``ref`` and the probe's mean power against
    ``ref``'s (dB).
    """
    ref = np.asarray(ref)
    m = len(ref)
    if not 2 <= n <= m:
        raise ValueError(f"a probe of {n} samples from a signal of {m}")
    rms = float(np.sqrt(np.mean(np.abs(ref) ** 2)))
    centred = int(np.argmax(np.abs(ref))) - n // 2
    shifts = np.arange(-(n // 20), n // 20 + 1)
    starts = centred + shifts
    jumps = np.abs(ref[starts % m] - ref[(starts + n - 1) % m])
    k = int(np.argmin(jumps))  # ties: the first, i.e. the most negative shift; centred wins below
    if jumps[k] >= jumps[n // 20]:
        k = n // 20
    start = int(starts[k] % m)
    w = ref[(start + np.arange(n)) % m]
    info = {
        "n": int(n),
        "start": start,
        "shift": int(shifts[k]),
        "jump_rms": float(jumps[k] / rms),
        "jump_centred_rms": float(jumps[n // 20] / rms),
        "papr_db": round(_papr(w), 3),
        "papr_full_db": round(_papr(ref), 3),
        "mean_power_db": round(float(10 * np.log10(np.mean(np.abs(w) ** 2) / rms**2)), 3),
    }
    return w, info


def make_probes(refs: Sequence[np.ndarray], fs: float, probe_ms: float, *, block: int = 1):
    """Every band's probe (:func:`probe_cut`), all of one length (the TX loads them in one
    ``PerformTx``): ``(probes, info)``, ``info`` per band plus ``probe_ms``, ``n``, ``block``."""
    n = probe_length(probe_ms, fs, block)
    probes, per = [], []
    for r in refs:
        w, info = probe_cut(r, n)
        probes.append(w)
        per.append(info)
    return probes, {"probe_ms": float(probe_ms), "n": n, "block": int(block), "bands": per}


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
    return _reading(cap, fs, window_s)


def _reading(cap, fs: float, window_s: float):
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


def measure_bands(
    radio,
    orxs: Sequence[RxChannel],
    refs: Sequence,
    *,
    rx_bits: int,
    fs: float,
    oversample: int = 2,
    window_s: float = 1e-4,
    fractions: Sequence[float | None] | None = None,
) -> list[tuple]:
    """:func:`measure_band` of several ORx from ONE snapshot (one ``PerformRx``): one reading per
    ORx, in order (:func:`~adrvtrx.capture.aligned_captures`)."""
    caps = aligned_captures(
        radio, list(orxs), list(refs), bits=rx_bits, fs=fs, oversample=oversample,
        fractions=fractions,
    )  # fmt: skip
    return [_reading(cap, fs, window_s) for cap in caps]


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


def _band_reader(radio, bands, refs, *, rx_bits, fs, oversample, snapshot, fractions, learn):
    """``measure(idx)`` for the search: one snapshot of those bands (or one capture each), with
    each band's ``fractions`` (``learn``: the fraction of a band's first reading in ``learn``'s
    window, ``(target_dbfs, tol_up_db, tol_down_db)``, is kept)."""

    def measure(idx: list[int]):
        if snapshot and len(idx) > 1:
            got = measure_bands(
                radio, [bands[i].orx for i in idx], [refs[i] for i in idx], rx_bits=rx_bits,
                fs=fs, oversample=oversample, fractions=[fractions.get(i) for i in idx],
            )  # fmt: skip
        else:
            got = [
                measure_band(
                    radio, bands[i].orx, refs[i], rx_bits=rx_bits, fs=fs, oversample=oversample,
                    fraction=fractions.get(i),
                )  # fmt: skip
                for i in idx
            ]
        if learn is not None:
            target, up, down = learn
            for i, (peak, railed, _c, extra) in zip(idx, got):
                if i not in fractions and railed == 0 and -down <= peak - target <= up:
                    fractions[i] = float(extra["delay_samples"]) % 1.0
        return got

    return measure


def lock_bands(
    radio,
    bands: Sequence[BandSpec],
    *,
    tx_bits: int,
    rx_bits: int,
    fs: float,
    retune: bool = True,
    method: str = "joint",
    oversample: int = 2,
    target_dbfs: float = -1.0,
    tol_up_db: float = 0.3,
    tol_down_db: float = 0.6,
    probe_ms: float | None = PROBE_MS,
    probe_block: int = 1,
    snapshot: bool | None = None,
    on_step=None,
    **search_kw: Any,
) -> tuple[MultibandSearch, list[np.ndarray]]:
    """Retune, play every band, ORx AGC per band at the start attenuations, then the search.

    With ``probe_ms`` (default 0.05 ms) every band plays its probe (:func:`make_probes`: the
    full signal's codes around its peak, ``probe_block`` the TX memory's block), the ORx AGC and
    every search reading capture ``oversample`` probe periods, each aligned in full, and after
    the lock every band swaps back to its full signal at the same attenuations and gains (the
    caller confirms on it, :func:`confirm_lock`). ``probe_ms=None`` plays the full signals
    throughout: a band's first reading inside the AGC window is aligned in full, its fractional
    delay kept, and its later readings only search the integer lag (:func:`measure_band`
    ``fraction``).

    One snapshot per step reads every band (``snapshot=None``: unless two bands' ORx share an
    ORx ADC; ``False``: one capture per band, for a pair whose ``two_orx_snapshot`` bench check
    failed; the search records which). Returns the search and each band's FULL reference (TX
    codes); TX keeps playing the full signals (the caller captures next). ``search.timing``
    gets ``agc_s`` (per band), ``swaps_s`` (each waveform load) and ``total_s``;
    ``search.probe`` the probe's description. Before the radio is touched, each band's ``lo``
    is checked against the LO the radio's ``[clocks]`` routes its TX to, and bands that share an
    LO must share its frequency (``ValueError``).
    """
    t_start = time.perf_counter()
    _check_method(method)
    _check_los(bands, getattr(getattr(radio, "config", None), "clocks", None))
    refs = tx_codes(bands, tx_bits=tx_bits)
    probes, probe_info = (
        (None, {}) if probe_ms is None else make_probes(refs, fs, probe_ms, block=probe_block)
    )
    played = refs if probes is None else probes
    if snapshot is None:
        snapshot = not shares_orx_adc([b.orx for b in bands])
    radio.disable_tx()
    for b in bands:
        radio.set_tx_atten(b.tx, quantize_atten(b.atten_start_db))
    if retune:
        for lo, hz in {b.lo: int(b.freq_hz) for b in bands}.items():
            radio.retune_lo(lo, hz)
    swaps = []
    t0 = time.perf_counter()
    play_codes(radio, bands, played, tx_bits=tx_bits)
    swaps.append(round(time.perf_counter() - t0, 3))
    gains, agc_s = [], []
    for b, r in zip(bands, played):
        t0 = time.perf_counter()
        level = autolevel_for(
            radio,
            b.orx,
            r,
            bits=rx_bits,
            fs=fs,
            oversample=oversample,
            target_dbfs=target_dbfs,
            tol_up_db=tol_up_db,
            tol_down_db=tol_down_db,
            gain_min=int(search_kw.get("gain_min", ORX_GAIN_MIN)),
            gain_max=int(search_kw.get("gain_max", ORX_GAIN_MAX)),
        )
        gains.append(int(level.final_gain_index))
        agc_s.append(round(time.perf_counter() - t0, 3))
    measure = _band_reader(
        radio, bands, played, rx_bits=rx_bits, fs=fs, oversample=oversample, snapshot=snapshot,
        fractions={}, learn=None if probes is not None else (target_dbfs, tol_up_db, tol_down_db),
    )  # fmt: skip
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
        snapshot=snapshot,
        on_step=on_step,
        **search_kw,
    )
    signal = "full" if probes is None else "probe"
    for h in search.history:
        h["signal"] = signal
    if probes is not None:  # back to the full signals, same attenuations and gains
        t0 = time.perf_counter()
        play_codes(radio, bands, refs, tx_bits=tx_bits)
        swaps.append(round(time.perf_counter() - t0, 3))
        search.probe = probe_info
    search.timing.update(
        agc_s=agc_s, swaps_s=swaps, total_s=round(time.perf_counter() - t_start, 3)
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
    gain_min: int = ORX_GAIN_MIN,
    gain_max: int = ORX_GAIN_MAX,
) -> list[BandCapture]:
    """Every band at ``attens_db`` (TX playing): ORx AGC, one capture per band, then ``check``.

    The AGC levels each band's ORx peak to ``target_dbfs`` (window ``+tol_up_db`` /
    ``-tol_down_db``) within the gain indices ``gain_min`` .. ``gain_max``.

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
                    gain_min=gain_min,
                    gain_max=gain_max,
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


def confirm_lock(
    radio,
    bands: Sequence[BandSpec],
    refs: Sequence[np.ndarray],
    search: MultibandSearch,
    *,
    rx_bits: int,
    fs: float,
    oversample: int = 2,
    confirm_steps: int = 2,
    check: Callable[[list[BandCapture]], dict[str, dict]] | None = None,
    retakes: int = 3,
    on_step=None,
    **search_kw: Any,
) -> list[BandCapture]:
    """The lock on the probe, confirmed on the full signals (TX playing them, :func:`lock_bands`).

    The 0 dB backoff capture is the confirmation: :func:`capture_case` at the locked
    attenuations (ORx AGC with its Stage C level check on the full signal, ``check``), then each
    band's gain compression against its target. A band outside ``±comp_tol_db`` continues the
    search from there on full captures (the reading just taken is its first step; ``J`` as the
    probe search left it), for at most ``confirm_steps`` moves, and the 0 dB capture is taken
    again where it ends. Returns the 0 dB captures; ``search`` is updated in place: the bands,
    ``converged`` and ``reason`` of the full signal, the fallback's rows (``signal`` "full") in
    ``history``, and ``confirm``: ``offset_db`` per band (the full GC minus the probe GC at the
    same attenuation), ``gc_full_db``, ``fallback_steps``, and ``converged_on`` ("probe",
    "fallback" or "").
    """
    tol = float(search_kw.get("comp_tol_db", 0.1))
    targets = np.array([b.target_db for b in bands])
    orx = _orx_settings(search_kw)

    def zero_db(attens, gains):
        return capture_case(
            radio, bands, refs, attens, rx_bits=rx_bits, fs=fs, oversample=oversample,
            check=check, retakes=retakes, gains=gains, **orx,
        )  # fmt: skip

    t0 = time.perf_counter()
    caps = zero_db([bl.atten_db for bl in search.bands], [bl.orx_gain for bl in search.bands])
    confirm_s = search.timing.setdefault("confirm_s", [])
    confirm_s.append(round(time.perf_counter() - t0, 3))
    gc = np.array([c.metrics["gain_compression_db"] for c in caps])
    confirm = {
        "gc_full_db": {c.name: round(float(v), 4) for c, v in zip(caps, gc)},
        "offset_db": {
            bl.name: round(float(v - bl.compression_db), 4) for bl, v in zip(search.bands, gc)
        },
        "fallback_steps": 0,
        "converged_on": "",
    }
    search.confirm = confirm
    off = ~(np.abs(targets - gc) <= tol)
    if not off.any():
        confirm["converged_on"] = "probe"
        if not search.converged:
            search.reason += "; every band within tolerance on the full signal"
        search.converged = True
        for bl in search.bands:
            bl.status = "converged"
        return caps
    if confirm_steps <= 0:
        search.converged = False
        search.reason += "; the full signal is off its target (no fallback steps)"
        return caps
    first = [
        (c.peak_dbfs, c.railed, c.metrics["gain_compression_db"],
         {"papr_compression_db": c.metrics["papr_compression_db"],
          "top_slope": c.metrics["amam_top_slope"]})
        for c in caps
    ]  # fmt: skip
    fractions = {i: float(c.metrics["delay_samples"]) % 1.0 for i, c in enumerate(caps)}
    reader = _band_reader(
        radio, bands, refs, rx_bits=rx_bits, fs=fs, oversample=oversample,
        snapshot=search.snapshot, fractions=fractions, learn=None,
    )  # fmt: skip
    pending = {"first": True}

    def measure(idx: list[int]):
        if pending.pop("first", False) and len(idx) == len(bands):
            return [first[i] for i in idx]  # the confirmation capture is the first reading
        return reader(idx)

    kw = {k: v for k, v in search_kw.items() if k not in ("max_steps", "slopes", "method")}
    fb = search_multiband_compression(
        lambda i, db: radio.set_tx_atten(bands[i].tx, db),
        lambda i, gi: radio.set_rx_gain(bands[i].orx, gi),
        measure,
        targets_db=list(targets),
        start_attens_db=[c.atten_db for c in caps],
        atten_mins_db=[min(b.atten_min_db, c.atten_db) for b, c in zip(bands, caps)],
        orx_gains=[c.orx_gain for c in caps],
        names=[b.name for b in bands],
        method="joint",
        slopes=search.slopes,
        max_steps=int(confirm_steps) + 1,
        snapshot=search.snapshot,
        on_step=on_step,
        **kw,
    )
    fb.captures -= 1 if search.snapshot else len(bands)  # its first reading was not captured
    fb.snapshots -= 1
    for h in fb.history:
        h["step"] += search.steps
        h["signal"] = "full"
    search.history.extend(fb.history)
    search.steps += fb.steps
    search.captures += fb.captures
    search.snapshots += fb.snapshots
    search.timing.setdefault("captures_s", []).extend(fb.timing.get("captures_s", [])[1:])
    search.bands, search.slopes = fb.bands, fb.slopes
    search.converged = fb.converged
    search.reason = f"{fb.reason} (full-signal fallback after the probe lock: {search.reason})"
    confirm["fallback_steps"] = fb.steps - 1
    confirm["converged_on"] = "fallback" if fb.converged else ""
    attens = [bl.atten_db for bl in fb.bands]
    if any(abs(a - c.atten_db) > 1e-9 for a, c in zip(attens, caps)):
        t0 = time.perf_counter()
        caps = zero_db(attens, [bl.orx_gain for bl in fb.bands])
        confirm_s.append(round(time.perf_counter() - t0, 3))
    return caps


def find_multiband_operating_point(
    radio,
    bands: Sequence[BandSpec],
    *,
    tx_bits: int,
    rx_bits: int,
    fs: float,
    backoffs: Sequence[int] = (0, 1, 2, 3, 4),
    method: str = "joint",
    check: Callable[[list[BandCapture]], dict[str, dict]] | None = None,
    retakes: int = 3,
    writer: CaseWriter | None = None,
    retune: bool = True,
    oversample: int = 2,
    probe_ms: float | None = PROBE_MS,
    probe_block: int = 1,
    confirm_steps: int = 2,
    snapshot: bool | None = None,
    on_step=None,
    **search_kw: Any,
) -> CaseResult:
    """Lock every band together, then capture every backoff (every band ``+k`` dB together).

    The lock runs on the probe (``probe_ms``, :func:`lock_bands`) and is confirmed on the full
    signals by the 0 dB capture (:func:`confirm_lock`: at most ``confirm_steps`` full-signal
    steps when it is off); ``probe_ms=None`` locks on the full signals as before, and the 0 dB
    capture is a backoff like the others. ``search.timing["total_s"]`` is the whole call.

    With ``writer`` (a :class:`CaseWriter`), the inputs, every aligned output, ``conditions.csv``
    and ``lock.csv`` are written. The case folder must be empty: that is checked before the radio
    is touched. If writing still fails, the error raised carries the result as ``case_result``
    (the captures are not lost). TX is disabled when the call ends or fails. A backoff is added
    to the lock, so it must be ``>= 0`` (a band never goes below its limit).

    A fatal search (the ORx still clips at its minimum gain) captures and writes nothing: the
    result has the search and no captures. A lock that did not converge (a band at its limit,
    clip-limited) is the operating point at that limit: its backoffs are captured.
    """
    t_start = time.perf_counter()
    if any(k < 0 for k in backoffs):
        raise ValueError(f"backoffs must be >= 0, got {list(backoffs)}")
    _check_method(method)
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
            probe_ms=probe_ms,
            probe_block=probe_block,
            snapshot=snapshot,
            on_step=on_step,
            **search_kw,
        )
        if search.probe and not search.fatal:
            captures[0] = confirm_lock(
                radio, bands, refs, search, rx_bits=rx_bits, fs=fs, oversample=oversample,
                confirm_steps=confirm_steps, check=check, retakes=retakes, on_step=on_step,
                **search_kw,
            )  # fmt: skip
        locked = [bl.atten_db for bl in search.bands]
        for k in () if search.fatal else backoffs:
            if int(k) in captures:
                continue
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
                **_orx_settings(search_kw),
            )
        captures = {int(k): captures[int(k)] for k in backoffs if int(k) in captures}
        search.timing["total_s"] = round(time.perf_counter() - t_start, 3)
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


def _orx_settings(search_kw: dict) -> dict:
    """The ORx AGC settings of a call's ``search_kw`` (level, window, gain range), so the backoff
    captures level the ORx exactly as the lock did."""
    keys = ("target_dbfs", "tol_up_db", "tol_down_db", "gain_min", "gain_max")
    return {k: search_kw[k] for k in keys if k in search_kw}


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
                        "search_snapshot": search.snapshot,
                        "lock_signal": "probe" if search.probe else "full",
                        "lock_converged_on": search.confirm.get("converged_on", ""),
                        "probe_offset_db": search.confirm.get("offset_db", {}).get(b.name, ""),
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
            **(
                {"lock": lock_summary(search), **lock_details(search)} if search is not None else {}
            ),
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
    """The lock per band as plain dicts (for ``case.json``): with a probe lock, also the band's
    probe-to-full GC offset (``probe_offset_db``, full GC minus probe GC at the same attenuation).
    :func:`lock_details` has the rest (timing, probe, confirmation)."""
    offsets = search.confirm.get("offset_db", {})
    return [
        {
            "band": bl.name,
            "atten_db": bl.atten_db,
            "orx_gain": bl.orx_gain,
            "gain_compression_db": bl.compression_db,
            "top_slope": bl.top_slope,
            "status": bl.status,
            **({"probe_offset_db": offsets[bl.name]} if bl.name in offsets else {}),
        }
        for bl in search.bands
    ]


def lock_details(search: MultibandSearch) -> dict:
    """What ``case.json`` keeps of the lock besides :func:`lock_summary`: ``lock_method``,
    ``lock_steps``, ``lock_snapshots``, ``lock_captures``, ``lock_snapshot`` (one snapshot per
    step, or one capture per band), ``lock_timing`` (each capture, each waveform swap, the AGC,
    the confirmation and the total, s), ``lock_probe`` and ``lock_confirm``."""
    return {
        "lock_method": search.method,
        "lock_steps": search.steps,
        "lock_snapshots": search.snapshots,
        "lock_captures": search.captures,
        "lock_snapshot": search.snapshot,
        "lock_timing": search.timing,
        "lock_probe": search.probe,
        "lock_confirm": search.confirm,
    }


def timing_text(search: MultibandSearch) -> str:
    """The lock's timing in a few lines (notebook 01 prints it under the lock)."""
    t = search.timing
    caps = t.get("captures_s", [])
    lines = [
        f"lock timing: {t.get('total_s', float('nan')):.1f} s in all; {search.steps} steps, "
        f"{search.snapshots} snapshots, {search.captures} captures "
        f"({'one snapshot per step' if search.snapshot else 'one capture per band'})"
    ]
    if caps:
        lines.append(
            f"  captures: {sum(caps):.1f} s, {np.mean(caps):.2f} s each (min {min(caps):.2f}, "
            f"max {max(caps):.2f})"
        )
    for key, label in (("agc_s", "ORx AGC per band"), ("swaps_s", "waveform swaps"),
                       ("confirm_s", "0 dB confirmation")):  # fmt: skip
        if t.get(key):
            lines.append(f"  {label}: " + ", ".join(f"{v:.2f}" for v in t[key]) + " s")
    c = search.confirm
    if c:
        offs = ", ".join(f"{k} {v:+.3f}" for k, v in c.get("offset_db", {}).items())
        lines.append(
            f"  probe -> full GC offset: {offs} dB; converged on: {c.get('converged_on') or 'no'}"
            f" ({c.get('fallback_steps', 0)} full-signal steps)"
        )
    return "\n".join(lines)
