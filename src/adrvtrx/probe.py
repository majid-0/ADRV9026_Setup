"""The probe: a short signal that a lock can play instead of the full signals, true to their peaks.

A lock steers on gain compression (:func:`~adrvtrx.metrics.gain_compression_db`). That averages
the gain over every sample at or above :data:`~adrvtrx.metrics.PEAK_REGION` of the signal's
maximum, which is dozens of separate peak events in a 100 MHz signal (S20-100: 41 in TX2, 27 in
TX3). The probe keeps a window of real signal around every one of them, and keeps the same
sample indices in every band, so the bands play together exactly as in the full transmission.
This module is pure numpy:

* :func:`peak_events`: the local maxima of ``|x|`` at or above ``threshold * max|x|``.
* :func:`select_windows`: the peaks of every band ranked together (each against its own
  maximum), a ``window`` centred on each peak not yet covered, windows merged; the shared index
  ranges in time order. ``budget`` (samples) stops it early, highest peaks kept.
* :func:`build_probe`: every band cut at those ranges and joined in time order (gaps dropped,
  codes as they are). Each join is placed in the ``crossfade`` samples beyond its windows' edges
  (outside every window, so at least ``window / 2`` samples of real signal stay around each
  peak), where both bands are low and the jump smallest, and crossfaded over ``crossfade``
  samples of the real continuations. The length is rounded up to a multiple of ``multiple`` and
  of ``block``. The build check (``ok`` / ``reason``): per band, PAPR within 0.1 dB of the full
  signal and mean power within 0.2 dB, every peak included (unless ``budget``), and
  ``n <= len(signal) / 4``.
* :func:`probe_fidelity`: on a saved full-signal capture, the compression on the probe's sample
  positions against the full period, per band (the offline check of a probe design);
  :func:`fidelity_report` for a whole case (``probe_fidelity.json``).

:class:`ProbeSettings` holds the settings (:func:`~adrvtrx.multiband.lock_bands` ``probe``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .metrics import PEAK_REGION, gain_compression_db

__all__ = [
    "ProbeSettings",
    "Probe",
    "peak_events",
    "select_windows",
    "build_probe",
    "probe_fidelity",
    "fidelity_report",
    "PAPR_TOL_DB",
    "POWER_TOL_DB",
]

#: The build check: per band, the probe's PAPR and mean power against the full signal's.
PAPR_TOL_DB = 0.1
POWER_TOL_DB = 0.2
#: The longest probe, as a fraction of the signal (a longer one would save little time).
MAX_FRACTION = 0.25


@dataclass(frozen=True)
class ProbeSettings:
    """The probe's settings (task6_fast_lock.md 6b, C): ``window`` 50 .. 20,000 samples,
    ``crossfade`` 0 .. window / 4, ``budget`` None or at least one window (samples), ``block`` the
    TX memory's block 1 .. 4096, ``multiple`` (16: two periods are whole 4-byte words at every
    rate of a profile)."""

    window: int = 500
    crossfade: int = 16
    budget: int | None = None
    block: int = 1
    multiple: int = 16
    threshold: float = PEAK_REGION

    def __post_init__(self) -> None:
        if not 50 <= int(self.window) <= 20_000:
            raise ValueError(f"probe window {self.window} outside 50 .. 20000 samples")
        if not 0 <= int(self.crossfade) <= int(self.window) // 4:
            raise ValueError(f"probe crossfade {self.crossfade} outside 0 .. window / 4")
        if self.budget is not None and int(self.budget) < int(self.window):
            raise ValueError(f"probe budget {self.budget} is less than one window")
        if not 1 <= int(self.block) <= 4096:
            raise ValueError(f"probe block {self.block} outside 1 .. 4096")
        if int(self.multiple) < 1 or not 0 < float(self.threshold) <= 1:
            raise ValueError("probe multiple >= 1 and 0 < threshold <= 1")

    def build(self, signals: Sequence[np.ndarray]) -> Probe:
        return build_probe(
            signals,
            window=self.window,
            crossfade=self.crossfade,
            multiple=self.multiple,
            block=self.block,
            budget=self.budget,
            threshold=self.threshold,
        )


@dataclass
class Probe:
    """A probe of every band (:func:`build_probe`).

    ``codes`` per band (complex TX codes, one period of the loop); ``ranges`` the kept index
    ranges of the full signal, ``(start, stop)`` in time order (a range across the period's end
    is two ranges); ``positions`` the full-signal index of each probe sample (-1 inside a
    crossfade); ``joins`` one dict per join (``at`` its probe index, ``cut`` and ``resume`` the
    full-signal indices it leaves and resumes at, ``jump_before`` / ``jump_after`` the largest
    sample step there without / with the crossfade, in RMS units of the band); ``stats`` per
    band; ``ok`` and ``reason`` from the build check.
    """

    codes: list[np.ndarray]
    ranges: list[tuple[int, int]]
    positions: np.ndarray
    n: int
    joins: list[dict]
    stats: list[dict]
    ok: bool
    reason: str
    settings: dict = field(default_factory=dict)

    def describe(self) -> dict:
        """The build result as plain values (``case.json`` ``lock_probe``), without the codes."""
        return {
            "ok": self.ok,
            "reason": self.reason,
            "n": self.n,
            "ranges": [list(r) for r in self.ranges],
            "joins": self.joins,
            "stats": self.stats,
            "settings": self.settings,
        }


def peak_events(x, threshold: float = PEAK_REGION) -> np.ndarray:
    """Indices of the local maxima of ``|x|`` (one period of a loop: circular) at or above
    ``threshold * max|x|``, highest first."""
    a = np.abs(np.asarray(x))
    if a.size == 0 or not a.max() > 0:
        return np.zeros(0, dtype=int)
    peak = (a >= np.roll(a, 1)) & (a > np.roll(a, -1)) & (a >= threshold * a.max())
    idx = np.flatnonzero(peak)
    return idx[np.argsort(-a[idx], kind="stable")]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, stop)`` of every run of True, in index order (no wrap handling)."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _fill_gaps(mask: np.ndarray, gap: int) -> np.ndarray:
    """Gaps of False shorter than ``gap`` between True runs filled (circular)."""
    if gap <= 0 or mask.all() or not mask.any():
        return mask
    n = len(mask)
    shift = int(np.flatnonzero(mask)[0])  # start the scan on a covered sample
    m = np.roll(mask, -shift)
    for start, stop in _runs(~m):
        if stop - start < gap:
            m[start:stop] = True
    return np.roll(m, shift)[:n]


def _select(signals, window: int, threshold: float, budget: int | None, gap: int = 0):
    """:func:`select_windows`, also returning the coverage mask and each band's peaks."""
    signals = [np.asarray(s) for s in signals]
    n = len(signals[0])
    if any(len(s) != n for s in signals):
        raise ValueError(f"the bands' signals differ in length: {[len(s) for s in signals]}")
    if window > n:
        raise ValueError(f"a window of {window} samples in a signal of {n}")
    peaks = [peak_events(s, threshold) for s in signals]
    ranked = sorted(
        (
            (float(np.abs(s[i]) / np.abs(s).max()), b, int(i))
            for b, (s, idx) in enumerate(zip(signals, peaks))
            for i in idx
        ),
        key=lambda t: -t[0],
    )
    mask = np.zeros(n, dtype=bool)
    offsets = np.arange(window) - window // 2
    for _level, _b, i in ranked:
        cells = (i + offsets) % n
        if mask[cells].all():
            continue  # its whole window is covered by higher peaks' windows
        if budget is not None and np.count_nonzero(mask) + np.count_nonzero(~mask[cells]) > budget:
            break  # the budget is reached: the highest peaks are kept
        mask[cells] = True
    mask = _fill_gaps(mask, gap)
    return _runs(mask), mask, peaks


def select_windows(
    signals: Sequence[np.ndarray],
    window: int = 500,
    threshold: float = PEAK_REGION,
    budget: int | None = None,
) -> list[tuple[int, int]]:
    """The shared index ranges ``(start, stop)`` in time order, one set for every band.

    The peaks of every band (:func:`peak_events`) are ranked together, highest first, each
    against its own signal's maximum, so the absolute maxima come first. A window of ``window``
    samples is centred on each peak, skipped when the higher peaks' windows already cover all of
    it (so every peak keeps ``window / 2`` samples of real signal on each side); overlapping
    windows merge. It stops when
    every peak is covered, or before a window would take the coverage past ``budget`` samples.
    A window across the period's end is two ranges (the loop makes them contiguous).
    """
    return _select(signals, int(window), float(threshold), budget)[0]


def _papr_db(x: np.ndarray) -> float:
    p = np.abs(x) ** 2
    return float(10 * np.log10(p.max() / p.mean()))


def _cut(signals, rms, end: int, start: int, d_max: int, e_max: int, total: int | None = None):
    """The join from ``end`` (exclusive) to ``start``: the cut moved ``d`` samples later
    (``0..d_max``) and the resume ``e`` samples earlier (``0..e_max``), where every band is low and
    the jump smallest (worst band). ``total``: ``d + e`` fixed. Returns ``(d, e)``."""
    best, best_score = (0, 0), math.inf
    for d in range(d_max + 1):
        for e in range(e_max + 1):
            if total is not None and d + e != total:
                continue
            score = 0.0
            for s, r in zip(signals, rms):
                a, b = s[(end + d - 1) % len(s)], s[(start - e) % len(s)]
                score = max(score, (abs(a - b) + 0.5 * (abs(a) + abs(b))) / r)
            if score < best_score:
                best, best_score = (d, e), score
    return best


def build_probe(
    signals: Sequence[np.ndarray],
    window: int = 500,
    crossfade: int = 16,
    multiple: int = 16,
    block: int = 1,
    budget: int | None = None,
    threshold: float = PEAK_REGION,
) -> Probe:
    """Every band's probe at the shared ranges of :func:`select_windows` (see the module
    docstring). ``signals`` are the bands' TX codes, one period each, of one length; the probe's
    codes are theirs, unscaled, and the crossfades are rounded to whole codes."""
    signals = [np.asarray(s, dtype=np.complex128) for s in signals]
    if not signals:
        raise ValueError("build_probe needs at least one band")
    window, crossfade = int(window), int(crossfade)
    n_full = len(signals[0])
    unit = math.lcm(max(1, int(multiple)), max(1, int(block)))
    room = 3 * crossfade + unit  # a gap shorter than this is kept: no join there
    ranges, mask, peaks = _select(signals, window, float(threshold), budget, room)
    rms = [float(np.sqrt(np.mean(np.abs(s) ** 2))) or 1.0 for s in signals]
    settings = {"window": window, "crossfade": crossfade, "multiple": int(multiple),
                "block": int(block), "budget": budget, "threshold": float(threshold)}  # fmt: skip
    if mask.all():
        ranges = [(0, n_full)]
    # each range runs [start, stop); range k joins range k + 1 (the last joins the first)
    starts = [r[0] for r in ranges]
    stops = [r[1] for r in ranges]
    k_ranges = len(ranges)
    joined = [(stops[k] % n_full) != (starts[(k + 1) % k_ranges] % n_full) for k in range(k_ranges)]
    if mask.all():
        joined = [False]
    moves = []
    for k in range(k_ranges):
        if joined[k]:
            moves.append(
                _cut(signals, rms, stops[k], starts[(k + 1) % k_ranges], crossfade, crossfade)
            )
        else:
            moves.append((0, 0))
    total = sum(b - a for a, b in ranges) + sum(
        d + e + crossfade for (d, e), j in zip(moves, joined) if j
    )
    pad = (-total) % unit
    if pad:
        last = max((k for k in range(k_ranges) if joined[k]), default=None)
        if last is None:  # one run around the whole loop: nothing to cut
            pass
        else:
            d0, e0 = moves[last]
            moves[last] = _cut(
                signals,
                rms,
                stops[last],
                starts[(last + 1) % k_ranges],
                crossfade + unit,
                crossfade,
                total=d0 + e0 + pad,
            )
    # the segments of real signal, in time order: [resume of the join before, cut of this one)
    segs = []
    for k in range(k_ranges):
        a = starts[k] - (moves[k - 1][1] if joined[k - 1] else 0)
        b = stops[k] + (moves[k][0] if joined[k] else 0)
        segs.append((a, b))
    t = (np.arange(crossfade) + 0.5) / max(crossfade, 1)
    codes: list[list[np.ndarray]] = [[] for _ in signals]
    positions: list[np.ndarray] = []
    joins = []
    at = 0
    for k, (a, b) in enumerate(segs):
        idx = np.arange(a, b) % n_full
        positions.append(idx)
        for c, s in zip(codes, signals):
            c.append(s[idx])
        at += b - a
        if not joined[k]:
            continue
        resume = segs[(k + 1) % k_ranges][0]
        before, after = 0.0, 0.0
        for c, s, r in zip(codes, signals, rms):
            tail = s[(b + np.arange(crossfade)) % n_full]  # the cut's real continuation
            head = s[(resume - crossfade + np.arange(crossfade)) % n_full]  # leads into the resume
            fade = (1 - t) * tail + t * head
            fade = np.round(fade.real) + 1j * np.round(fade.imag)
            c.append(fade)
            path = np.concatenate([[s[(b - 1) % n_full]], fade, [s[resume % n_full]]])
            before = max(before, abs(s[(b - 1) % n_full] - s[resume % n_full]) / r)
            after = max(after, float(np.max(np.abs(np.diff(path)))) / r)
        positions.append(np.full(crossfade, -1))
        joins.append(
            {
                "at": int(at),
                "cut": int(b % n_full),
                "resume": int(resume % n_full),
                "jump_before": round(float(before), 4),
                "jump_after": round(after, 4),
            }
        )
        at += crossfade
    probe_codes = [np.concatenate(c) for c in codes]
    pos = np.concatenate(positions) if positions else np.zeros(0, dtype=int)
    n = len(pos)
    kept = np.zeros(n_full, dtype=bool)
    kept[pos[pos >= 0]] = True
    stats, problems = [], []
    for b, (s, p, pk) in enumerate(zip(signals, probe_codes, peaks)):
        papr, papr_full = _papr_db(p), _papr_db(s)
        power = float(10 * np.log10(np.mean(np.abs(p) ** 2) / np.mean(np.abs(s) ** 2)))
        covered = int(np.count_nonzero(kept[pk]))
        stats.append(
            {
                "band": b,
                "papr_db": round(papr, 3),
                "papr_full_db": round(papr_full, 3),
                "mean_power_db": round(power, 3),
                "peaks_covered": covered,
                "peaks_total": int(len(pk)),
            }
        )
        if abs(papr - papr_full) > PAPR_TOL_DB:
            problems.append(f"band {b}: PAPR {papr:.2f} dB, the full signal {papr_full:.2f} dB")
        if abs(power) > POWER_TOL_DB:
            problems.append(f"band {b}: mean power {power:+.2f} dB from the full signal")
        if covered < len(pk) and budget is None:
            problems.append(f"band {b}: {covered} of {len(pk)} peaks covered")
    if n * 4 > n_full:
        problems.append(f"{n} samples is more than a quarter of the signal ({n_full})")
    return Probe(
        probe_codes,
        [(int(a % n_full), int(a % n_full + b - a)) for a, b in segs],
        pos,
        n,
        joins,
        stats,
        not problems,
        "; ".join(problems),
        settings,
    )


def probe_fidelity(x, y, probe: Probe) -> dict:
    """The compression on ``probe``'s sample positions against the full period, per band.

    ``x`` the reference, ``y`` the aligned output: one band's arrays, or one per band (a saved
    full-signal capture, e.g. a case's ``*_in.txt`` / ``*_<k>dB_*_out.txt``). Returns
    ``{"bands": [{"full_db", "probe_db", "offset_db"}, ...]}`` (offset = probe minus full).
    """
    xs = [x] if np.ndim(x[0]) == 0 else list(x)
    ys = [y] if np.ndim(y[0]) == 0 else list(y)
    pos = probe.positions[probe.positions >= 0]
    out = []
    for xb, yb in zip(xs, ys):
        xb, yb = np.asarray(xb), np.asarray(yb)
        full = gain_compression_db(xb, yb)[0]
        kept = gain_compression_db(xb[pos], yb[pos])[0]
        out.append(
            {
                "full_db": round(full, 4),
                "probe_db": round(kept, 4),
                "offset_db": round(kept - full, 4),
            }
        )
    return {"bands": out, "n": int(probe.n)}


def fidelity_report(
    refs: Sequence[np.ndarray],
    outputs: dict,
    names: Sequence[str],
    settings: ProbeSettings | None = None,
) -> dict:
    """``probe_fidelity.json`` of a case: the probe ``settings`` (default :class:`ProbeSettings`)
    built from the case's references (TX codes) and checked on its captures.

    ``outputs``: ``{backoff_db: [aligned output per band]}``. Returns the probe's build result
    (:meth:`Probe.describe`) and, per backoff, per band, the full and probe compression and their
    offset (:func:`probe_fidelity`). It runs whether the lock used the probe or not.
    """
    try:
        probe = (settings or ProbeSettings()).build(refs)
    except ValueError as exc:  # e.g. a signal shorter than one window: nothing to check
        return {"probe": None, "error": str(exc), "backoffs": {}, "worst_offset_db": None}
    backoffs = {}
    for k, ys in sorted(outputs.items()):
        got = probe_fidelity(list(refs), list(ys), probe)["bands"]
        backoffs[str(int(k))] = dict(zip(names, got))
    worst = max(
        (abs(b["offset_db"]) for per in backoffs.values() for b in per.values()), default=None
    )
    return {"probe": probe.describe(), "backoffs": backoffs, "worst_offset_db": worst}
