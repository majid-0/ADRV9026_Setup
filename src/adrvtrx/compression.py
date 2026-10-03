"""Find the TX attenuation where the PA reaches a target compression.

:func:`search_tx_compression` is hardware-free, the same shape as
:func:`adrvtrx.gain.autolevel_orx`: the caller supplies ``set_tx_atten``,
``set_orx_gain`` and ``measure``. ``measure`` returns
``(peak_dbfs, railed, compression_db)`` from a capture already aligned to the
reference, optionally followed by a dict of extra readings that is recorded in
the history (``papr_compression_db``, ``gain_compression_db``, ``top_slope``).

A compression reading only counts once the ORx peak is inside the AGC band with
no railed sample (or the signal is below the band at maximum gain). Lowering TX
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
from .align import estimate_and_align
from .capture import autolevel_capture, capture
from .gain import ORX_DB_PER_INDEX, ORX_GAIN_MAX, ORX_GAIN_MIN, clip_report
from .metrics import gain_compression_db, window_compression_db
from .radio import MAX_TX_ATTEN_DB

LOCK_METRICS = ("papr", "gain")

__all__ = [
    "LOCK_METRICS",
    "CompressionResult",
    "search_tx_compression",
    "find_compression_point",
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


def _quantize_atten(db: float) -> float:
    """TX attenuator step is 0.05 dB."""
    return round(float(db) * 20.0) / 20.0


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
    target_dbfs: float = -1.0,
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

    With ``min_top_slope`` set, a trusted reading whose ``extra["top_slope"]`` is
    below it is PA clipping: the attenuation goes up (by the remaining error, at
    least one fine step), and that attenuation becomes the new floor. A target the
    PA cannot reach without clipping ends at that floor with ``clip_limited=True``.
    """
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
        return railed == 0 and band_lo <= peak <= band_hi

    def fine_step(error_db: float) -> float:
        steps = max(1, math.floor((abs(error_db) + 1e-6) / fine_step_db))
        return min(coarse_step_db, steps * fine_step_db)

    atten = _quantize_atten(max(atten_min_db, min(start_atten_db, atten_max_db)))
    gain = clamp_gain(orx_gain)
    floor = _quantize_atten(atten_min_db)  # raised by the clip guard
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
        leveled = in_band(peak, railed)
        cold_at_max = (not leveled) and railed == 0 and peak < band_lo and gain >= gain_max
        trusted = leveled or cold_at_max
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
            up = fine_step(over) if over > 0 else fine_step_db
            new = _quantize_atten(min(atten_max_db, atten + up))
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
            gain = clamp_gain(gain + int(round(delta / db_per_index)))
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
                step = coarse_step_db if coarse_pending else fine_step(error)
                coarse_pending = False
                new = _quantize_atten(atten - step)
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
                gain = clamp_gain(gain - int(round(delta / db_per_index)))
                action = f"TX atten -{delta:.2f} dB"
            else:
                new = _quantize_atten(min(atten_max_db, atten + fine_step(error)))
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
                gain = clamp_gain(gain + int(round(delta / db_per_index)))
                action = f"TX atten +{delta:.2f} dB"
        elif (railed > 0 or peak > band_hi) and gain <= gain_min:
            new = _quantize_atten(min(atten_max_db, atten + fine_step_db))
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
        elif railed > 0 or peak > band_hi:
            if railed > 0 and not peak > band_hi:
                step_idx = 1
            else:
                step_idx = max(1, int(round((peak - target_dbfs) / db_per_index)))
            gain = clamp_gain(gain - step_idx)
            action = f"ORx gain -{step_idx}"
        else:
            step_idx = max(1, int(round((target_dbfs - peak) / db_per_index)))
            gain = clamp_gain(gain + step_idx)
            action = f"ORx gain +{step_idx}"

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
    target_dbfs: float = -1.0,
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
    capture_ms = oversample * len(ref) / float(fs) * 1e3

    radio.set_tx_atten(tx, _quantize_atten(start_atten_db))
    agc = autolevel_capture(
        radio,
        orx,
        bits=rx_bits,
        orx_rate_hz=fs,
        target_dbfs=target_dbfs,
        tol_up_db=tol_up_db,
        tol_down_db=tol_down_db,
        verify_capture_ms=capture_ms,
    )

    def measure() -> tuple[float, int, float, dict]:
        out = capture(radio, int(orx), capture_ms, bits=rx_bits).channels[orx]
        rep = clip_report(out.i, out.q, rx_bits)
        x_al, y_al, _delay = estimate_and_align(ref, out.iq, fs)
        papr_comp, _pin, _pout = window_compression_db(x_al, y_al, fs, window_s)
        gain_comp, slope = gain_compression_db(x_al, y_al)
        comp = gain_comp if lock_on == "gain" else papr_comp
        extra = {
            "papr_compression_db": papr_comp,
            "gain_compression_db": gain_comp,
            "top_slope": slope,
        }
        return rep.peak_dbfs, rep.railed_samples, comp, extra

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
