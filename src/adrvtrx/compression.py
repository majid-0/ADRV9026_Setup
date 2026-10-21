"""Find the TX attenuation where the PA reaches a target compression.

:func:`search_tx_compression` is hardware-free, the same shape as
:func:`adrvtrx.gain.autolevel_orx`: the caller supplies ``set_tx_atten``,
``set_orx_gain`` and ``measure``. ``measure`` returns
``(peak_dbfs, railed, compression_db)`` from a capture already aligned to the
reference, optionally followed by a dict of extra readings that is recorded in
the history (``papr_compression_db``, ``gain_compression_db``, ``top_slope``).

A compression reading only counts once the ORx peak is inside the AGC band with
no clip (:func:`adrvtrx.gain.orx_clipped`: no railed sample, peak at most
``ORX_CLIP_DBFS``), or the signal is below the band at maximum gain. Lowering TX
attenuation lowers the ORx gain by the same number of dB in that step so the ADC
does not clip on the next capture. Attenuation never goes below ``atten_min_db``.

Clip guard: with ``min_top_slope`` set, a trusted reading whose ``top_slope`` is
below it counts as PA clipping. The search raises the attenuation and never goes
below that point again; if the target needs more drive than that, it stops at the
last unclipped attenuation with ``clip_limited=True``.

:func:`find_compression_point` runs the ORx AGC at the start attenuation and then
the search against live hardware. ``lock_on`` picks what the search steers on:
``"papr"`` (:func:`adrvtrx.metrics.window_compression_db`, the default) or
``"gain"`` (:func:`adrvtrx.metrics.gain_compression_db`). Both, and the AM/AM top
slope, are measured and recorded at every step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ._enums import RxChannel, TxChannel
from .capture import aligned_capture, autolevel_for
from .gain import (
    ORX_DB_PER_INDEX,
    ORX_GAIN_MAX,
    ORX_GAIN_MIN,
    OrxStep,
    check_orx_window,
    orx_clipped,
    relevel_index,
)
from .metrics import gain_compression_db, window_compression_db
from .radio import MAX_TX_ATTEN_DB

LOCK_METRICS = ("papr", "gain")

__all__ = [
    "LOCK_METRICS",
    "CompressionResult",
    "search_tx_compression",
    "find_compression_point",
    "fine_step",
    "relevel_index",
    "quantize_atten",
]


@dataclass
class CompressionResult:
    """``compression_db`` is the metric the search steered on (``lock_on``) at the result.

    ``papr_compression_db``, ``gain_compression_db`` and ``top_slope`` are the
    readings at the result when ``measure`` reports them, else NaN.
    """

    converged: bool
    final_atten_db: float
    final_orx_gain: int
    compression_db: float
    peak_dbfs: float
    railed: int
    iterations: int
    reason: str
    fatal: bool = False
    at_atten_floor: bool = False
    clip_limited: bool = False
    lock_on: str = "papr"
    papr_compression_db: float = float("nan")
    gain_compression_db: float = float("nan")
    top_slope: float = float("nan")
    history: list[dict] = field(default_factory=list)


def quantize_atten(db: float) -> float:
    """``db`` on the TX attenuator's 0.05 dB step."""
    return round(float(db) * 20.0) / 20.0


_quantize_atten = quantize_atten  # the earlier private name


def fine_step(error_db: float, fine_db: float, coarse_db: float) -> float:
    """TX attenuation step for a compression error of ``error_db``: the largest multiple of
    ``fine_db`` not larger than the error (at least one), capped at ``coarse_db``.

    The fine-step rule of the single-band search and of the multi-band walk.
    """
    steps = max(1, math.floor((abs(error_db) + 1e-6) / fine_db))
    return min(coarse_db, steps * fine_db)


def search_tx_compression(
    set_tx_atten,
    set_orx_gain,
    measure,
    *,
    target_compression_db: float,
    start_atten_db: float,
    atten_min_db: float,
    orx_gain: int,
    coarse_step_db: float = 10.0,
    fine_step_db: float = 0.25,
    comp_tol_db: float = 0.3,
    target_dbfs: float = -2.0,
    tol_up_db: float = 0.3,
    tol_down_db: float = 0.6,
    gain_min: int = ORX_GAIN_MIN,
    gain_max: int = ORX_GAIN_MAX,
    db_per_index: float = ORX_DB_PER_INDEX,
    atten_max_db: float = MAX_TX_ATTEN_DB,
    max_iterations: int = 24,
    min_top_slope: float | None = None,
    on_step=None,
) -> CompressionResult:
    """Walk TX attenuation until compression is within ``±comp_tol_db`` of the target.

    ``measure()`` -> ``(peak_dbfs, railed, compression_db)``, or the same followed by
    a dict of extra readings that is copied into every history record.

    The first move toward more compression is ``coarse_step_db``. Later moves are
    the largest multiple of ``fine_step_db`` not larger than the remaining error,
    capped at ``coarse_step_db``. ORx gain moves by the same dB
    (``round(delta / db_per_index)`` indices). The radio is left at the returned
    attenuation and gain.

    The ORx re-level learns the gain table's step (:class:`~adrvtrx.gain.OrxStep`, shared with
    the multi-band search): ``db_per_index`` until two captures at one attenuation tell it, then
    their secant; 1-index steps after an overshoot; the closest capture when even those jump the
    window.

    With ``min_top_slope`` set, a trusted reading whose ``extra["top_slope"]`` is
    below it is PA clipping: the attenuation goes up (by the remaining error, at
    least one fine step), and that attenuation becomes the new floor. A target the
    PA cannot reach without clipping ends at that floor with ``clip_limited=True``.

    The multi-band search (:func:`adrvtrx.multiband.search_multiband_compression`) uses the
    same rules with other defaults (coarse 2.5 dB, fine 0.2 dB, tolerance 0.1 dB, clip guard on);
    its docstring says why.

    A clipped ORx capture (:func:`~adrvtrx.gain.orx_clipped`) never counts and lowers the ORx
    gain; at the minimum gain it stops the search (``fatal``). A reading with no data (peak not
    finite) stops it too (``fatal``). The window's top must be below
    :data:`~adrvtrx.gain.ORX_CLIP_DBFS` (``ValueError``).
    """
    check_orx_window(target_dbfs, tol_up_db)
    if atten_min_db > start_atten_db:
        raise ValueError(
            f"atten_min_db ({atten_min_db}) must be <= start_atten_db ({start_atten_db})"
        )
    if comp_tol_db <= 0:
        raise ValueError("comp_tol_db must be positive")

    band_lo = target_dbfs - tol_down_db
    band_hi = target_dbfs + tol_up_db
    comp_lo = target_compression_db - comp_tol_db
    comp_hi = target_compression_db + comp_tol_db

    def clamp_gain(g: int) -> int:
        return int(min(max(int(g), gain_min), gain_max))

    def in_band(peak: float, railed: int) -> bool:
        return not orx_clipped(peak, railed) and band_lo <= peak <= band_hi

    def step_for(error_db: float) -> float:
        return fine_step(error_db, fine_step_db, coarse_step_db)

    atten = quantize_atten(max(atten_min_db, min(start_atten_db, atten_max_db)))
    gain = clamp_gain(orx_gain)
    orx = OrxStep(db_per_index)  # the re-level learns the gain table's step
    floor = quantize_atten(atten_min_db)  # raised by the clip guard
    guard_floor = False
    coarse_pending = True
    history: list[dict] = []
    best: dict | None = None
    nan = float("nan")

    def finish(
        converged: bool,
        reason: str,
        *,
        fatal: bool = False,
        at_floor: bool = False,
        clip_limited: bool = False,
        comp: float = nan,
        peak: float = nan,
        railed: int = 0,
        extra: dict | None = None,
    ) -> CompressionResult:
        set_tx_atten(atten)
        set_orx_gain(gain)
        extra = extra or {}
        return CompressionResult(
            converged=converged,
            final_atten_db=atten,
            final_orx_gain=gain,
            compression_db=comp,
            peak_dbfs=peak,
            railed=railed,
            iterations=len(history),
            reason=reason,
            fatal=fatal,
            at_atten_floor=at_floor,
            clip_limited=clip_limited,
            papr_compression_db=float(extra.get("papr_compression_db", nan)),
            gain_compression_db=float(extra.get("gain_compression_db", nan)),
            top_slope=float(extra.get("top_slope", nan)),
            history=history,
        )

    def record(rec: dict, action: str) -> None:
        rec["action"] = action
        history.append(rec)
        if on_step:
            on_step(rec)

    for _ in range(max_iterations):
        set_tx_atten(atten)
        set_orx_gain(gain)
        reading = measure()
        peak, railed, comp = float(reading[0]), int(reading[1]), float(reading[2])
        extra = dict(reading[3]) if len(reading) > 3 and reading[3] is not None else {}
        if not math.isfinite(peak):  # no data (an empty capture): nothing to level on
            rec = {"atten_db": atten, "orx_gain": gain, "peak_dbfs": peak, "railed": railed,
                   "compression_db": comp, **extra, "orx_ok": False}  # fmt: skip
            record(rec, "ORx returned no data, stop")
            return finish(
                False,
                "ORx returned no data (peak not finite): the ORx was likely driven into clipping",
                fatal=True,
                comp=comp,
                peak=peak,
                railed=railed,
                extra=extra,
            )
        leveled = in_band(peak, railed)
        cold_at_max = (not leveled) and railed == 0 and peak < band_lo and gain >= gain_max
        trusted = leveled or cold_at_max
        orx.observe(gain, peak, railed, band_lo=band_lo, band_hi=band_hi, drive=atten)
        closest = orx.closest(target_dbfs) if orx.jumped and not trusted else None
        if closest is not None and closest[0] == gain and not orx_clipped(peak, railed):
            trusted = True  # 1-index steps jump the window: the capture closest to the target
        rec = {
            "atten_db": atten,
            "orx_gain": gain,
            "peak_dbfs": peak,
            "railed": railed,
            "compression_db": comp,
            **extra,
            "orx_ok": leveled,
        }
        slope = float(extra.get("top_slope", nan))
        clipped = (
            min_top_slope is not None and trusted and math.isfinite(slope) and slope < min_top_slope
        )
        if min_top_slope is not None:
            rec["pa_clipped"] = clipped

        if clipped:
            over = comp - target_compression_db
            up = step_for(over) if over > 0 else fine_step_db
            new = quantize_atten(min(atten_max_db, atten + up))
            if new <= atten + 1e-9:
                record(rec, "PA clip guard at attenuation ceiling")
                return finish(
                    False,
                    "PA clips even at the attenuation ceiling",
                    clip_limited=True,
                    comp=comp,
                    peak=peak,
                    railed=railed,
                    extra=extra,
                )
            delta = new - atten
            atten = new
            floor = max(floor, atten)
            guard_floor = True
            gain = clamp_gain(gain + orx.indices(delta))
            record(rec, f"PA clip guard (top slope {slope:.3f}); TX atten +{delta:.2f} dB")
            continue

        if trusted:
            err = abs(comp - target_compression_db)
            if best is None or err < best["err"]:
                best = {
                    "err": err,
                    "atten": atten,
                    "gain": gain,
                    "comp": comp,
                    "peak": peak,
                    "railed": railed,
                    "extra": extra,
                }
            if leveled and comp_lo <= comp <= comp_hi:
                record(rec, "converged")
                return finish(
                    True,
                    "compression inside tolerance",
                    comp=comp,
                    peak=peak,
                    railed=railed,
                    extra=extra,
                )

            error = target_compression_db - comp  # > 0: need more compression
            if error > 0:
                step = coarse_step_db if coarse_pending else step_for(error)
                coarse_pending = False
                new = quantize_atten(atten - step)
                if new < floor:
                    if atten <= floor + 1e-9:
                        if guard_floor:
                            record(rec, "clip guard floor")
                            return finish(
                                False,
                                "the PA clips before the target; left at the last unclipped "
                                "attenuation",
                                clip_limited=True,
                                comp=comp,
                                peak=peak,
                                railed=railed,
                                extra=extra,
                            )
                        record(rec, "atten floor")
                        return finish(
                            False,
                            "attenuation floor reached before the target",
                            at_floor=True,
                            comp=comp,
                            peak=peak,
                            railed=railed,
                            extra=extra,
                        )
                    new = floor
                delta = atten - new
                atten = new
                gain = clamp_gain(gain - orx.indices(delta))
                action = f"TX atten -{delta:.2f} dB"
            else:
                new = quantize_atten(min(atten_max_db, atten + step_for(error)))
                if new <= atten + 1e-9:
                    record(rec, "atten ceiling")
                    return finish(
                        False,
                        "attenuation ceiling reached before the target",
                        comp=comp,
                        peak=peak,
                        railed=railed,
                        extra=extra,
                    )
                delta = new - atten
                atten = new
                gain = clamp_gain(gain + orx.indices(delta))
                action = f"TX atten +{delta:.2f} dB"
        elif (orx_clipped(peak, railed) or peak > band_hi) and gain <= gain_min:
            new = quantize_atten(min(atten_max_db, atten + fine_step_db))
            record(rec, f"ORx railed at gain floor; TX atten +{new - atten:.2f} dB, stop")
            atten = new
            return finish(
                False,
                "ORx still clips at minimum gain; TX attenuation raised one step and stopped",
                fatal=True,
                comp=comp,
                peak=peak,
                railed=railed,
                extra=extra,
            )
        elif closest is not None:  # back to the closest capture: it is taken next time
            k = closest[0] - gain
            gain = clamp_gain(closest[0])
            action = f"ORx gain {k:+d} (1-index steps jump the window: the closest capture)"
        else:
            k = orx.step(peak, railed, target_dbfs=target_dbfs, band_hi=band_hi)
            gain = clamp_gain(gain + k)
            action = f"ORx gain {k:+d}"

        record(rec, action)

    if best is not None:
        atten = best["atten"]
        gain = best["gain"]
        return finish(
            False,
            "max iterations; left at the closest compression",
            clip_limited=guard_floor,
            comp=best["comp"],
            peak=best["peak"],
            railed=best["railed"],
            extra=best["extra"],
        )
    return finish(False, "max iterations")


def find_compression_point(
    radio,
    tx: TxChannel,
    orx: RxChannel,
    ref,
    *,
    rx_bits: int,
    fs: float,
    target_compression_db: float,
    start_atten_db: float,
    atten_min_db: float,
    lock_on: str = "papr",
    min_top_slope: float | None = None,
    window_s: float = 1e-4,
    oversample: int = 2,
    coarse_step_db: float = 10.0,
    fine_step_db: float = 0.25,
    comp_tol_db: float = 0.3,
    target_dbfs: float = -2.0,
    tol_up_db: float = 0.3,
    tol_down_db: float = 0.6,
    max_iterations: int = 24,
    on_step=None,
) -> CompressionResult:
    """ORx AGC at ``start_atten_db``, then :func:`search_tx_compression` on hardware.

    ``ref`` is the transmitted reference in TX codes (what ``prepare_tx`` returns
    for the playing waveform). TX must already be running. Each measurement
    captures ``oversample`` periods, aligns one to ``ref`` and scores both
    :func:`~adrvtrx.metrics.window_compression_db` on ``window_s`` around the peak
    and :func:`~adrvtrx.metrics.gain_compression_db` (with the AM/AM top slope) on
    the whole aligned period. ``lock_on`` picks the one the search steers on:
    ``"papr"`` or ``"gain"``. ``min_top_slope`` turns on the clip guard (see
    :func:`search_tx_compression`). ``fs`` is the ORx rate. A fatal AGC result
    raises :class:`~adrvtrx.gain.AgcError` after leaving the bench safe.
    """
    if lock_on not in LOCK_METRICS:
        raise ValueError(f"lock_on must be one of {LOCK_METRICS}, got {lock_on!r}")
    ref = np.asarray(ref)

    radio.set_tx_atten(tx, quantize_atten(start_atten_db))
    agc = autolevel_for(
        radio,
        orx,
        ref,
        bits=rx_bits,
        fs=fs,
        oversample=oversample,
        target_dbfs=target_dbfs,
        tol_up_db=tol_up_db,
        tol_down_db=tol_down_db,
    )

    def measure() -> tuple[float, int, float, dict]:
        cap = aligned_capture(radio, orx, ref, bits=rx_bits, fs=fs, oversample=oversample)
        papr_comp, _pin, _pout = window_compression_db(cap.x_aligned, cap.y_aligned, fs, window_s)
        gain_comp, slope = gain_compression_db(cap.x_aligned, cap.y_aligned)
        comp = gain_comp if lock_on == "gain" else papr_comp
        extra = {
            "papr_compression_db": papr_comp,
            "gain_compression_db": gain_comp,
            "top_slope": slope,
        }
        return cap.clip.peak_dbfs, cap.clip.railed_samples, comp, extra

    result = search_tx_compression(
        lambda db: radio.set_tx_atten(tx, db),
        lambda g: radio.set_rx_gain(orx, g),
        measure,
        target_compression_db=target_compression_db,
        start_atten_db=start_atten_db,
        atten_min_db=atten_min_db,
        orx_gain=agc.final_gain_index,
        coarse_step_db=coarse_step_db,
        fine_step_db=fine_step_db,
        comp_tol_db=comp_tol_db,
        target_dbfs=target_dbfs,
        tol_up_db=tol_up_db,
        tol_down_db=tol_down_db,
        max_iterations=max_iterations,
        min_top_slope=min_top_slope,
        on_step=on_step,
    )
    result.lock_on = lock_on
    return result
