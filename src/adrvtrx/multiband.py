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
``coarse_step_db`` and rounded to the 0.05 dB attenuator step; the ORx
gain follows the predicted output peak (the drive change minus the compression change); a band
never goes below its ``atten_min_db`` (a band that needs more drive than that stops there, the
others go on); clip guard per band (``min_top_slope``); at most ``max_steps`` steps (no move
after the last one), then the step with the smallest worst-band error among the steps where
every reading counted is kept, or else the last step's attenuations and gains: the radio is
always left where it was measured.

:func:`find_multiband_operating_point` runs it on hardware (a :class:`~adrvtrx.radio.Radio` or
the hardware server's ``RemoteRadio``): retune, play every TX in one ``PerformTx``, ORx AGC per
band, the search, then per backoff every band ``+k`` dB, ORx AGC per band, one capture per band
and the caller's ``check`` on them. The steps are separate functions too (:func:`play_bands`,
:func:`lock_bands`, :func:`capture_case`) for a campaign that runs them as separate jobs.
:class:`CaseWriter` saves a case folder (write-once).
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

from ._enums import RxChannel, TxChannel
from .align import estimate_and_align
from .capture import autolevel_capture, capture
from .compression import _quantize_atten
from .conditions import save_aligned_iq
from .gain import ORX_DB_PER_INDEX, ORX_GAIN_MAX, ORX_GAIN_MIN, clip_report
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
        return self.tx.name if hasattr(self.tx, "name") else str(self.tx)


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


def _fine_step(error_db: float, fine: float, coarse: float) -> float:
    steps = max(1, math.floor((abs(error_db) + 1e-6) / fine))
    return min(coarse, steps * fine)


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
    names = list(names) if names is not None else [f"band{i + 1}" for i in range(n)]
    targets = np.asarray(targets_db, dtype=float)
    band_lo, band_hi = target_dbfs - tol_down_db, target_dbfs + tol_up_db
    nan = float("nan")

    def clamp_gain(g: float) -> int:
        return int(min(max(int(round(g)), gain_min), gain_max))

    floor = np.array([_quantize_atten(m) for m in atten_mins_db])
    for i, s in enumerate(start_attens_db):
        if floor[i] > s:
            raise ValueError(f"{names[i]}: limit {floor[i]} dB is above the start {s} dB")
    a = np.array(
        [_quantize_atten(min(max(s, f), atten_max_db)) for s, f in zip(start_attens_db, floor)]
    )
    g = [clamp_gain(x) for x in orx_gains]
    clip_floor = [False] * n
    history: list[dict] = []
    captures = 0
    jac: np.ndarray | None = None
    prev: tuple[np.ndarray, np.ndarray] | None = None
    prev_err: np.ndarray | None = None
    coarse_pending = [True] * n  # until the band's first reading that counts
    best: dict | None = None
    last: dict | None = None

    def apply() -> None:
        for i in range(n):
            set_atten(i, float(a[i]))
            set_gain(i, int(g[i]))

    def band_locks(readings, err) -> list[BandLock]:
        out = []
        for i, r in enumerate(readings):
            below = r["leveled"] and err[i] > comp_tol_db and a[i] <= floor[i] + 1e-9
            if r["leveled"] and abs(err[i]) <= comp_tol_db:
                status = "converged"
            elif clip_floor[i] and below:
                status = "clip-limited"
            elif below:
                status = "at limit"
            else:
                status = "not converged"
            out.append(
                BandLock(
                    names[i],
                    float(a[i]),
                    int(g[i]),
                    r["comp"],
                    r["papr"],
                    r["slope"],
                    r["peak"],
                    r["railed"],
                    float(floor[i]),
                    status,
                )
            )
        return out

    def finish(
        converged: bool, reason: str, readings, err, *, fatal: bool = False
    ) -> MultibandSearch:
        apply()
        return MultibandSearch(
            converged,
            reason,
            method,
            band_locks(readings, err),
            len({h["step"] for h in history}),
            captures,
            fatal,
            history,
        )

    for step in range(max_steps):
        apply()
        readings = []
        for i in range(n):
            for attempt in range(max_relevel + 1):
                peak, railed, comp, *rest = measure(i)
                captures += 1
                extra = dict(rest[0]) if rest and rest[0] is not None else {}
                peak, railed, comp = float(peak), int(railed), float(comp)
                in_band = railed == 0 and band_lo <= peak <= band_hi
                at_max = railed == 0 and peak < band_lo and g[i] >= gain_max
                leveled = in_band and math.isfinite(comp)  # NaN: a failed alignment
                cold = at_max and math.isfinite(comp)
                row = {
                    "step": step,
                    "band": names[i],
                    "atten_db": float(a[i]),
                    "orx_gain": int(g[i]),
                    "peak_dbfs": peak,
                    "railed": railed,
                    "gain_compression_db": comp,
                    "papr_compression_db": float(extra.get("papr_compression_db", nan)),
                    "top_slope": float(extra.get("top_slope", nan)),
                    "leveled": leveled,
                    "error_db": float(targets[i] - comp),
                    "move_db": 0.0,
                    "action": "",
                    "slopes": "",
                }
                if leveled or cold:
                    break
                if (railed > 0 or peak > band_hi) and g[i] <= gain_min:
                    for r in readings:  # the bands read before this one this step
                        if r["leveled"]:
                            r["row"]["action"] = "stopped"
                            history.append(r["row"])
                    row["action"] = "ORx clips at minimum gain: TX atten up one fine step, stop"
                    history.append(row)
                    a[i] = _quantize_atten(min(atten_max_db, a[i] + fine_step_db))
                    readings.append(
                        {
                            "comp": comp,
                            "papr": row["papr_compression_db"],
                            "slope": row["top_slope"],
                            "peak": peak,
                            "railed": railed,
                            "leveled": False,
                        }
                    )
                    unread = {
                        "comp": nan,
                        "papr": nan,
                        "slope": nan,
                        "peak": nan,
                        "railed": 0,
                        "leveled": False,
                    }
                    readings += [unread] * (n - len(readings))
                    err = targets - np.array([r["comp"] for r in readings])
                    return finish(
                        False,
                        f"{names[i]}: ORx still clips at minimum gain",
                        readings,
                        err,
                        fatal=True,
                    )
                if in_band or at_max:  # level fine, compression not finite: measure again
                    row["action"] = "measure again (compression not finite)"
                elif railed > 0 or peak > band_hi:
                    k = (
                        1
                        if (railed > 0 and not peak > band_hi)
                        else max(1, round((peak - target_dbfs) / db_per_index))
                    )
                    g[i] = clamp_gain(g[i] - k)
                    row["action"] = f"ORx gain -{k}"
                else:
                    k = max(1, round((target_dbfs - peak) / db_per_index))
                    g[i] = clamp_gain(g[i] + k)
                    row["action"] = f"ORx gain +{k}"
                if attempt == max_relevel:  # this reading drives nothing: the band holds
                    row["action"] += f"; not leveled after {attempt + 1} captures, atten held"
                history.append(row)
                set_gain(i, int(g[i]))
            readings.append(
                {
                    "comp": comp,
                    "papr": row["papr_compression_db"],
                    "slope": row["top_slope"],
                    "peak": peak,
                    "railed": railed,
                    "leveled": leveled or cold,
                    "row": row,
                }
            )
        c = np.array([r["comp"] for r in readings])
        err = targets - c
        ok = [r["leveled"] for r in readings]  # a reading that counts (leveled or cold)
        trusted = all(ok)
        clipped = [
            min_top_slope is not None
            and r["leveled"]
            and math.isfinite(r["slope"])
            and r["slope"] < min_top_slope
            for r in readings
        ]
        worst = float(np.max(np.abs(err)))
        if trusted and not any(clipped) and (best is None or worst < best["worst"]):
            best = {
                "worst": worst,
                "a": a.copy(),
                "g": list(g),
                "readings": readings,
                "err": err.copy(),
            }
        last = {
            "readings": readings,
            "err": err,
            "a": a.copy(),
            "g": [int(r["row"]["orx_gain"]) for r in readings],  # as captured
        }

        within = [ok[i] and abs(e) <= comp_tol_db for i, e in enumerate(err)]
        stuck = [ok[i] and e > comp_tol_db and a[i] <= floor[i] + 1e-9 for i, e in enumerate(err)]
        if not any(clipped) and all(w or s for w, s in zip(within, stuck)):
            rows = [r["row"] for r in readings]
            for row in rows:
                row["action"] = "converged" if all(within) else row["action"] or "done"
            history.extend(rows)
            if on_step:
                on_step(rows)
            if all(within):
                return finish(True, "every band within tolerance", readings, err)
            held = ", ".join(names[i] for i, s in enumerate(stuck) if s)
            return finish(
                False,
                f"{held} at the attenuation limit below target; the others within tolerance",
                readings,
                err,
            )
        if step == max_steps - 1:  # no move after the last step: nothing would measure it
            rows = [r["row"] for r in readings]
            for i, row in enumerate(rows):
                if ok[i]:
                    row["action"] = "max steps"
                    history.append(row)
            if on_step:
                on_step(rows)
            break

        # -- moves ----------------------------------------------------------------------
        mv = np.zeros(n)
        fixed = [False] * n
        actions = [""] * n
        for i in range(n):
            if clipped[i]:
                over = -err[i]
                up = _fine_step(over, fine_step_db, coarse_step_db) if over > 0 else fine_step_db
                new = _quantize_atten(min(atten_max_db, a[i] + up))
                floor[i] = max(floor[i], new)
                clip_floor[i] = True
                mv[i], fixed[i] = new - a[i], True
                actions[i] = f"clip guard (top slope {readings[i]['slope']:.3f})"
            elif stuck[i]:
                fixed[i], actions[i] = True, "at limit"
            elif not ok[i]:
                fixed[i] = True  # not leveled: its row is in the history already, atten held
            elif coarse_pending[i]:  # the band's first reading that counts: the single-band rule
                if err[i] > comp_tol_db:
                    mv[i], actions[i] = -coarse_step_db, "coarse"
                elif err[i] < -comp_tol_db:
                    mv[i], actions[i] = _fine_step(err[i], fine_step_db, coarse_step_db), "fine"
                else:
                    actions[i] = "within tolerance"
                fixed[i] = True
            if ok[i]:
                coarse_pending[i] = False
        free = [i for i in range(n) if not fixed[i]]
        grew = prev_err is not None and any(
            abs(err[i]) > abs(prev_err[i]) + 0.05 and abs(err[i]) > comp_tol_db for i in free
        )
        use_walk = method == "walk" or grew or not trusted
        if grew:
            jac = None  # this step walks; J starts again from the own slopes of the next one
        if method == "slopes" and trusted and prev is not None and not grew:
            da, dc = a - prev[0], c - prev[1]
            if jac is None:
                jac = own_slopes(da, dc, slope_bounds=slope_bounds)
            else:
                jac = update_slopes(jac, da, dc, slope_bounds=slope_bounds)
        for i in free:
            if abs(err[i]) <= comp_tol_db and use_walk:
                actions[i] = "within tolerance"
        if not free:
            pass
        elif use_walk or jac is None:
            for i in free:
                if abs(err[i]) > comp_tol_db:
                    step_db = _fine_step(err[i], fine_step_db, coarse_step_db)
                    mv[i] = -step_db if err[i] > 0 else step_db
                    actions[i] = "walk (fallback)" if method == "slopes" else "walk"
        else:
            fx = [i for i in range(n) if fixed[i]]
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
                    if abs(err[i]) > comp_tol_db:
                        step_db = _fine_step(err[i], fine_step_db, coarse_step_db)
                        mv[i] = -step_db if err[i] > 0 else step_db
                        actions[i] = "walk (singular slopes)"
        mv = np.clip(mv, -coarse_step_db, coarse_step_db)
        # a band still outside tolerance moves at least one attenuator step toward its target
        for i in free:
            if abs(err[i]) > comp_tol_db and abs(mv[i]) < 0.05 and actions[i] == "slopes":
                mv[i] = -0.05 if err[i] > 0 else 0.05
        new_a = np.array(
            [_quantize_atten(min(max(a[i] + mv[i], floor[i]), atten_max_db)) for i in range(n)]
        )
        moved = new_a - a
        rows = []
        for i, r in enumerate(readings):
            row = r["row"]
            if ok[i]:  # a band not leveled has its row already (its own action, no move)
                row["move_db"] = float(moved[i])
                row["action"] = actions[i]
                row["slopes"] = "" if jac is None else json.dumps(np.round(jac, 4).tolist())
                history.append(row)
            rows.append(row)
        if on_step:
            on_step(rows)
        if np.all(np.abs(moved) < 1e-9) and trusted:
            return finish(False, "no band can move further (limits)", readings, err)
        prev = (a.copy(), c.copy()) if trusted else None
        prev_err = np.where(ok, err, np.nan)  # an error that did not count cannot have grown
        # The ORx gain follows the output peak, which moves by the drive change minus the
        # compression change:
        # predicted from the slopes when known, else from a typical own slope of 0.5 dB per dB.
        dc = jac @ moved if jac is not None else -0.5 * moved
        for i in range(n):
            g[i] = clamp_gain(g[i] + round((moved[i] + dc[i]) / db_per_index))
        a = new_a

    if best is not None:
        a, g = best["a"], best["g"]
        return finish(
            False,
            "max steps; left at the step closest to the targets",
            best["readings"],
            best["err"],
        )
    a, g = last["a"], list(last["g"])
    return finish(False, "max steps; left at the last step measured", last["readings"], last["err"])


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
):
    """One capture of ``orx`` (``oversample`` periods).

    Returns ``(peak_dbfs, railed, gain_compression_db, extra)``.

    ``extra`` holds ``papr_compression_db``, ``top_slope``, the raw capture (``raw``, complex
    codes) and the
    aligned period (``y_aligned``) with its ``delay_samples``.
    """
    ref = np.asarray(ref)
    ms = oversample * len(ref) / float(fs) * 1e3
    out = capture(radio, int(orx), ms, bits=rx_bits).channels[orx]
    rep = clip_report(out.i, out.q, rx_bits)
    x_al, y_al, delay = estimate_and_align(ref, out.iq, fs)
    papr, _pin, _pout = window_compression_db(x_al, y_al, fs, window_s)
    gain_comp, slope = gain_compression_db(x_al, y_al)
    extra = {
        "papr_compression_db": papr,
        "top_slope": slope,
        "raw": out.iq,
        "y_aligned": y_al,
        "delay_samples": float(delay),
    }
    return rep.peak_dbfs, rep.railed_samples, gain_comp, extra


def _agc(radio, orx, ref, *, rx_bits, fs, oversample, target_dbfs, tol_up_db, tol_down_db) -> int:
    agc = autolevel_capture(
        radio,
        orx,
        bits=rx_bits,
        orx_rate_hz=fs,
        target_dbfs=target_dbfs,
        tol_up_db=tol_up_db,
        tol_down_db=tol_down_db,
        verify_capture_ms=oversample * len(ref) / float(fs) * 1e3,
    )
    return int(agc.final_gain_index)


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
    captures next).
    """
    radio.disable_tx()
    for b in bands:
        radio.set_tx_atten(b.tx, _quantize_atten(b.atten_start_db))
    if retune:
        for lo in dict.fromkeys(b.lo for b in bands):
            radio.retune_lo(lo, int(next(b.freq_hz for b in bands if b.lo == lo)))
    refs = play_bands(radio, bands, tx_bits=tx_bits)
    gains = [
        _agc(
            radio,
            b.orx,
            r,
            rx_bits=rx_bits,
            fs=fs,
            oversample=oversample,
            target_dbfs=target_dbfs,
            tol_up_db=tol_up_db,
            tol_down_db=tol_down_db,
        )
        for b, r in zip(bands, refs)
    ]
    search = search_multiband_compression(
        lambda i, db: radio.set_tx_atten(bands[i].tx, db),
        lambda i, gi: radio.set_rx_gain(bands[i].orx, gi),
        lambda i: measure_band(
            radio, bands[i].orx, refs[i], rx_bits=rx_bits, fs=fs, oversample=oversample
        ),
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
    """
    if gains is not None and len(gains) != len(bands):
        raise ValueError(f"gains needs one ORx gain index per band, got {list(gains)}")
    if not agc and gains is None:
        raise ValueError(
            "agc=False needs gains= (the ORx gain of each band): the ORx gain cannot be read back"
        )
    for b, db in zip(bands, attens_db):
        radio.set_tx_atten(b.tx, _quantize_atten(db))
    used: dict[str, int] = {}  # the ORx gain in effect per band (set here, never read back)
    out: list[BandCapture | None] = [None] * len(bands)

    def take(i: int, tries: int, relevel: bool) -> BandCapture:
        b, ref = bands[i], refs[i]
        if relevel or b.name not in used:
            if agc:
                used[b.name] = _agc(
                    radio,
                    b.orx,
                    ref,
                    rx_bits=rx_bits,
                    fs=fs,
                    oversample=oversample,
                    target_dbfs=target_dbfs,
                    tol_up_db=tol_up_db,
                    tol_down_db=tol_down_db,
                )
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
            float(_quantize_atten(attens_db[i])),
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
        self.dir = Path(run_dir) / self.name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tx_bits, self.rx_bits, self.fs = tx_bits, rx_bits, fs
        self.meta = dict(meta or {})
        self.row_meta = list(row_meta)

    def path(self, suffix: str) -> Path:
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
                rows.append(
                    {
                        **{key: self.meta.get(key, "") for key in self.row_meta},
                        "band": b.name,
                        "tx": b.name,
                        "orx": b.orx.name if hasattr(b.orx, "name") else str(b.orx),
                        "lo": b.lo,
                        "freq_hz": int(b.freq_hz),
                        "bw_mhz": b.bw_mhz,
                        "signal": b.signal_name,
                        "freqs_mhz": freqs,
                        "target_db": b.target_db,
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
        return self._write_csv("conditions.csv", self.condition_rows(search, captures))

    def write_lock(self, search: MultibandSearch) -> Path:
        return self._write_csv("lock.csv", [dict(h) for h in search.history])

    def write_case_json(self, extra: dict | None = None) -> Path:
        p = self._new(self.dir / "case.json")
        doc = {
            "case_name": self.name,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "bands": [
                {
                    "name": b.name,
                    "tx": b.name,
                    "orx": getattr(b.orx, "name", str(b.orx)),
                    "lo": b.lo,
                    "freq_hz": int(b.freq_hz),
                    "bw_mhz": b.bw_mhz,
                    "signal": b.signal_name,
                    "target_db": b.target_db,
                    "atten_start_db": b.atten_start_db,
                    "atten_min_db": b.atten_min_db,
                }
                for b in self.bands
            ],
            "fs_hz": self.fs,
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
