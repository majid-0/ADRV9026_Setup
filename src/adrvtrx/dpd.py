"""Indirect-learning DPD (ILA) for :func:`adrvtrx.linearize.linearize`, with a hard peak limit.

Units: a waveform is normalized to DAC full scale, so 1.0 is the original input
peak and :data:`FULL_SCALE_DBM` (10 dBm). :func:`peak_dbm` reads a waveform's
peak in those units.

One DPD pass (:class:`IlaStep`), from the last capture to the next waveform:

1. :func:`normalize_pair`: ``x`` is divided by its peak, the capture ``z`` by the
   peak of the **first** capture of the run (iteration 0, no DPD; the *anchor*),
   and ``z`` is rotated onto ``x``. The ORx gain is fixed during the loop, so
   every capture is measured on the same scale.
2. The post-inverse is fitted from the normalized ``z`` to the transmitted ``u``
   **as is** (DAC units), by block least squares on the samples around the
   peak of ``z`` (:func:`adrvtrx.gmp.peak_block`).
3. :func:`limit_peak`: ``x`` is predistorted through the post-inverse at the
   output target, ``target_backoff_db`` below the anchor. That backoff is a rule
   decided once, not something that grows each pass. The peak guard raises it
   only as far as needed to keep the DPD peak at or below the limit (9.9 dBm by
   default). If no backoff up to the maximum meets it, the pass returns ``None``
   and the loop stops; nothing over the limit is transmitted.

The model is any object with ``fit(u, target, block)`` and ``predict(u)``, for
example :class:`adrvtrx.gmp.GMP`. Each call builds a fresh one.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .gmp import peak_block
from .metrics import papr_db

__all__ = [
    "FULL_SCALE_DBM",
    "peak_dbm",
    "normalize_pair",
    "PeakLimit",
    "limit_peak",
    "ANCHORS",
    "TARGET_BACKOFF_DB",
    "IlaStep",
    "iteration_table",
]

FULL_SCALE_DBM = 10.0
"""Power of a waveform peak at DAC full scale (1.0, the original input peak)."""


def peak_dbm(s) -> float:
    """``10 + 20*log10(max|s|)`` for a waveform normalized to DAC full scale.

    ``-inf`` for an empty or all-zero signal, NaN if it holds a NaN.
    """
    a = np.abs(np.asarray(s))
    if a.size == 0:
        return float("-inf")
    peak = float(a.max())
    if not math.isfinite(peak):
        return float("nan") if math.isnan(peak) else float("inf")
    if peak <= 0.0:
        return float("-inf")
    return FULL_SCALE_DBM + 20.0 * math.log10(peak)


def normalize_pair(
    x, y, rotate: bool = True, y_scale: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``x / max|x|`` and ``y / y_scale`` (default ``max|y|``), with ``y`` rotated onto ``x``.

    The rotation is the phase of ``vdot(x, y)`` (the least-squares gain of ``y``
    on ``x``), so the loop phase is removed. ``rotate=False`` keeps it.
    ``y_scale`` holds the scale of an earlier capture (the DPD anchor), so a
    change in the output level stays visible. The two signals must be aligned
    and the same length.
    """
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    if len(x) != len(y):
        raise ValueError(f"normalize_pair: lengths differ ({len(x)} vs {len(y)})")
    px = float(np.abs(x).max()) if x.size else 0.0
    py = float(np.abs(y).max()) if y.size else 0.0
    if px <= 0.0 or py <= 0.0:
        raise ValueError("normalize_pair: a signal is empty or all zero")
    if y_scale is not None:
        if not (math.isfinite(y_scale) and y_scale > 0.0):
            raise ValueError(f"normalize_pair: y_scale must be positive, got {y_scale}")
        py = float(y_scale)
    xn = x / px
    yn = y / py
    if rotate:
        yn = yn * np.exp(-1j * np.angle(np.vdot(xn, yn)))
    return xn, yn


@dataclass
class PeakLimit:
    """Result of :func:`limit_peak`.

    ``waveform`` is the DPD output at ``margin_db``; ``peak_dbm`` is its peak.
    ``ok`` is False when even the maximum margin does not meet the limit (then
    ``waveform`` is the output at the maximum margin and must not be sent).
    ``tries`` lists every ``(margin_db, peak_dbm)`` evaluated, in order.
    """

    waveform: np.ndarray
    margin_db: float
    peak_dbm: float
    ok: bool
    tries: list[tuple[float, float]] = field(default_factory=list)


def limit_peak(
    dpd: Callable[[np.ndarray], np.ndarray],
    x,
    *,
    peak_limit_dbm: float = 9.9,
    start_margin_db: float = 0.2,
    max_margin_db: float = 4.0,
    tol_db: float = 0.01,
    coarse_step_db: float = 0.25,
) -> PeakLimit:
    """Smallest input backoff ``m >= start_margin_db`` with ``peak_dbm(dpd(x * 10**(-m/20)))``
    at or below ``peak_limit_dbm``.

    ``x`` is the input at its full peak (normalized). The output peak need not
    fall monotonically with ``m``: the margins from ``start_margin_db`` to
    ``max_margin_db`` are tried in ``coarse_step_db`` steps, and the first step
    that meets the limit is refined by bisection against the step before it down
    to ``tol_db``. The returned margin is always one that was evaluated and met
    the limit; ``ok`` is False if none up to ``max_margin_db`` did.
    """
    if not start_margin_db <= max_margin_db:
        raise ValueError(
            f"start_margin_db ({start_margin_db}) must be <= max_margin_db ({max_margin_db})"
        )
    if tol_db <= 0 or coarse_step_db <= 0:
        raise ValueError("tol_db and coarse_step_db must be positive")
    x = np.asarray(x, dtype=np.complex128)
    tries: list[tuple[float, float]] = []

    def run(m: float) -> tuple[np.ndarray, float]:
        w = np.asarray(dpd(x * 10.0 ** (-m / 20.0)), dtype=np.complex128)
        p = peak_dbm(w)
        tries.append((m, p))
        return w, p

    def good(p: float) -> bool:
        return math.isfinite(p) and p <= peak_limit_dbm

    n_steps = int(math.ceil((max_margin_db - start_margin_db) / coarse_step_db - 1e-9))
    grid = [min(start_margin_db + i * coarse_step_db, max_margin_db) for i in range(n_steps + 1)]
    bad_m = None
    w = np.empty(0, dtype=np.complex128)
    p = float("nan")
    for m in grid:
        w, p = run(m)
        if not good(p):
            bad_m = m
            continue
        if bad_m is None:
            return PeakLimit(w, m, p, True, tries)
        lo, hi, best = bad_m, m, (m, w, p)
        while hi - lo > tol_db:
            mid = 0.5 * (lo + hi)
            wm, pm = run(mid)
            if good(pm):
                hi, best = mid, (mid, wm, pm)
            else:
                lo = mid
        m_best, w_best, p_best = best
        return PeakLimit(w_best, m_best, p_best, good(p_best), tries)
    return PeakLimit(w, grid[-1], p, False, tries)


def _plain_nmse_db(target: np.ndarray, pred: np.ndarray) -> float:
    """``10*log10(sum|target - pred|^2 / sum|target|^2)`` with no gain removed."""
    den = float(np.vdot(target, target).real)
    if den <= 0.0:
        return float("nan")
    err = target - pred
    num = float(np.vdot(err, err).real)
    if not math.isfinite(num):
        return float("nan")
    return float(10.0 * np.log10(num / den)) if num > 0 else float("-inf")


ANCHORS = ("first", "each")

TARGET_BACKOFF_DB = 0.15
"""Default output target of :class:`IlaStep`, in dB below the iteration-0 output peak."""


class IlaStep:
    """Indirect-learning ``step(x, u, z, it)`` for :func:`adrvtrx.linearize.linearize`.

    ``model_factory()`` returns a fresh model with ``fit(u, target, block)`` and
    ``predict(u)``, e.g. ``lambda: GMP(5, 5, 2)``. Each call:

    1. normalizes ``z`` and rotates it onto ``x`` (:func:`normalize_pair`). With
       ``anchor="first"`` (the default) every capture is divided by the peak of
       the first one (iteration 0, the PA without DPD), so the output target stays
       put. ``anchor="each"`` divides each capture by its own peak (the target
       then slides down by the backoff on every pass);
    2. fits the post-inverse ``z -> u`` on the ``n_train`` samples around the peak
       of the normalized ``z``. The target is the transmitted ``u`` as is (DAC
       units), so the DPD output is in DAC units too;
    3. predistorts ``x / max|x|`` at the output target ``target_backoff_db`` below
       the anchor. :func:`limit_peak` starts there and adds backoff (the *guard*)
       only while the DPD peak would be above ``peak_limit_dbm``, up to
       ``max_margin_db`` in total. Returns the waveform.

    A call with ``it == 0`` starts a run: it takes the anchor from that ``z`` and
    clears ``history`` and ``reason``.

    ``history`` gets one dict per call: ``iteration`` (the iteration that will
    transmit the new waveform, ``it + 1``), ``z_peak_db`` (the peak of the capture
    the pass learned from, relative to the anchor), ``post_inverse_nmse_db`` (the
    fit over the whole period, no gain removed), ``target_backoff_db``,
    ``guard_db`` (backoff the peak guard added), ``margin_db`` (their sum),
    ``dpd_peak_dbm``, ``papr_expansion_db`` (``papr(next u) - papr(x)``) and
    ``ok``. If the limit cannot be met, the call returns ``None`` (the loop stops)
    and ``reason`` says why. ``model`` is the last fitted post-inverse and
    ``anchor_peak`` the peak of the first capture.
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        *,
        n_train: int = 8192,
        target_backoff_db: float = TARGET_BACKOFF_DB,
        peak_limit_dbm: float = 9.9,
        max_margin_db: float = 4.0,
        anchor: str = "first",
        tol_db: float = 0.01,
        coarse_step_db: float = 0.25,
        verbose: bool = False,
    ):
        if anchor not in ANCHORS:
            raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor!r}")
        if not 0.0 <= target_backoff_db <= max_margin_db:
            raise ValueError(
                f"target_backoff_db ({target_backoff_db}) must be in [0, max_margin_db "
                f"({max_margin_db})]"
            )
        self.model_factory = model_factory
        self.n_train = int(n_train)
        self.target_backoff_db = float(target_backoff_db)
        self.peak_limit_dbm = float(peak_limit_dbm)
        self.max_margin_db = float(max_margin_db)
        self.anchor = anchor
        self.tol_db = float(tol_db)
        self.coarse_step_db = float(coarse_step_db)
        self.verbose = verbose
        self.history: list[dict[str, Any]] = []
        self.reason = ""
        self.model = None
        self.anchor_peak: float | None = None

    def __call__(self, x, u, z, it: int) -> np.ndarray | None:
        z = np.asarray(z, dtype=np.complex128)
        z_peak = float(np.abs(z).max()) if z.size else 0.0
        if it == 0 or self.anchor_peak is None:
            if not z_peak > 0.0:
                raise ValueError("IlaStep: the first capture is empty or all zero")
            self.anchor_peak = z_peak
            self.history = []
            self.reason = ""
        scale = self.anchor_peak if self.anchor == "first" else None
        x_n, z_n = normalize_pair(x, z, y_scale=scale)
        u = np.asarray(u, dtype=np.complex128)
        block = peak_block(z_n, self.n_train)
        model = self.model_factory()
        model.fit(z_n, u, block)
        self.model = model
        fit_nmse = _plain_nmse_db(u, np.asarray(model.predict(z_n), dtype=np.complex128))

        lim = limit_peak(
            model.predict,
            x_n,
            peak_limit_dbm=self.peak_limit_dbm,
            start_margin_db=self.target_backoff_db,
            max_margin_db=self.max_margin_db,
            tol_db=self.tol_db,
            coarse_step_db=self.coarse_step_db,
        )
        row = {
            "iteration": it + 1,
            "z_peak_db": 20.0 * math.log10(z_peak / self.anchor_peak),
            "post_inverse_nmse_db": fit_nmse,
            "target_backoff_db": self.target_backoff_db,
            "guard_db": lim.margin_db - self.target_backoff_db,
            "margin_db": lim.margin_db,
            "dpd_peak_dbm": lim.peak_dbm,
            "papr_expansion_db": papr_db(lim.waveform) - papr_db(x_n),
            "ok": lim.ok,
        }
        self.history.append(row)
        if self.verbose:
            print(
                f"    step {it} -> {it + 1}: capture peak {row['z_peak_db']:+.2f} dB vs it 0, "
                f"post-inverse NMSE {fit_nmse:.2f} dB, backoff {self.target_backoff_db:.2f} "
                f"+ guard {row['guard_db']:.2f} dB, DPD peak {lim.peak_dbm:.2f} dBm, "
                f"PAPR expansion {row['papr_expansion_db']:.2f} dB"
            )
        if not lim.ok:
            self.reason = (
                f"iteration {it + 1}: DPD peak {lim.peak_dbm:.2f} dBm is above the "
                f"{self.peak_limit_dbm:.2f} dBm limit even with a {lim.margin_db:.2f} dB backoff"
            )
            return None
        return lim.waveform


_RECORD_KEYS = (
    "aclr_lower_dbc",
    "aclr_upper_dbc",
    "nmse_db",
    "gain_compression_db",
    "papr_compression_db",
    "papr_expansion_db",
    "tx_clipped",
    "railed",
)
_PASS_KEYS = (
    "target_backoff_db",
    "guard_db",
    "margin_db",
    "dpd_peak_dbm",
    "post_inverse_nmse_db",
)


def iteration_table(records: Sequence[Any], history: Sequence[dict]) -> list[dict[str, Any]]:
    """One row per loop iteration: the capture's figures and the DPD pass that made its ``u``.

    ``records`` are :class:`~adrvtrx.conditions.DutRecord` (``LinearizeResult.records``)
    and ``history`` is :attr:`IlaStep.history`. Columns:

    * from the capture: ``iteration``, ``aclr_lower_dbc``, ``aclr_upper_dbc``,
      ``aclr_worst_dbc`` (the larger), ``nmse_db``, ``output_peak_db`` (complex
      peak of the aligned capture relative to iteration 0, from
      ``rms_dbfs + papr_out_db``; the ORx gain is fixed, so this is the output
      level kept), ``gain_compression_db`` and ``papr_compression_db`` (``x -> z``:
      the PA's at iteration 0, what is left after it), ``pa_papr_compression_db``
      (``papr(u) - papr(z)``, the PA on its actual input), ``papr_expansion_db``
      (``papr(u) - papr(x)``, of the transmitted codes), ``tx_clipped``, ``railed``;
    * from the pass that made that iteration's ``u`` (NaN at iteration 0, which
      sends ``x``): ``target_backoff_db``, ``guard_db``, ``margin_db``,
      ``dpd_peak_dbm``, ``post_inverse_nmse_db``.
    """
    by_it = {int(h["iteration"]): h for h in history}
    nan = float("nan")
    rows = []
    out0 = None
    for k, rec in enumerate(records):
        row: dict[str, Any] = {"iteration": k}
        for key in _RECORD_KEYS:
            row[key] = getattr(rec, key)
        row["aclr_worst_dbc"] = max(rec.aclr_lower_dbc, rec.aclr_upper_dbc)
        out_peak = rec.rms_dbfs + rec.papr_out_db
        out0 = out_peak if out0 is None else out0
        row["output_peak_db"] = out_peak - out0
        row["pa_papr_compression_db"] = rec.papr_dpd_db - rec.papr_out_db
        h = by_it.get(k, {})
        for key in _PASS_KEYS:
            row[key] = h.get(key, nan)
        rows.append(row)
    return rows
