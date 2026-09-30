"""Find the TX attenuation where the PA reaches a target PAPR compression.

:func:`search_tx_compression` is hardware-free, the same shape as
:func:`adrvtrx.gain.autolevel_orx`: the caller supplies ``set_tx_atten``,
``set_orx_gain`` and ``measure``. ``measure`` returns
``(peak_dbfs, railed, compression_db)`` from a capture already aligned to the
reference. Compression is :func:`adrvtrx.metrics.window_compression_db`.

A compression reading only counts once the ORx peak is inside the AGC band with
no railed sample (or the signal is below the band at maximum gain). Lowering TX
attenuation lowers the ORx gain by the same number of dB in that step so the ADC
does not clip on the next capture. Attenuation never goes below ``atten_min_db``.

:func:`find_compression_point` runs the ORx AGC at the start attenuation and then
the search against live hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ._enums import RxChannel, TxChannel
from .align import estimate_and_align
from .capture import autolevel_capture, capture
from .gain import ORX_DB_PER_INDEX, ORX_GAIN_MAX, ORX_GAIN_MIN, clip_report
from .metrics import window_compression_db
from .radio import MAX_TX_ATTEN_DB

__all__ = [
    "CompressionResult",
    "search_tx_compression",
    "find_compression_point",
]


@dataclass
class CompressionResult:
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
    on_step=None,
) -> CompressionResult:
    """Walk TX attenuation until compression is within ``±comp_tol_db`` of the target.

    ``measure()`` -> ``(peak_dbfs, railed, compression_db)``.

    The first move toward more compression is ``coarse_step_db``. Later moves are
    the largest multiple of ``fine_step_db`` not larger than the remaining error,
    capped at ``coarse_step_db``. ORx gain moves by the same dB
    (``round(delta / db_per_index)`` indices). The radio is left at the returned
    attenuation and gain.
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
    coarse_pending = True
    history: list[dict] = []
    best: dict | None = None

    def finish(
        converged: bool,
        reason: str,
        *,
        fatal: bool = False,
        at_floor: bool = False,
        comp: float = float("nan"),
        peak: float = float("nan"),
        railed: int = 0,
    ) -> CompressionResult:
        set_tx_atten(atten)
        set_orx_gain(gain)
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
        peak, railed, comp = measure()
        peak, railed, comp = float(peak), int(railed), float(comp)
        leveled = in_band(peak, railed)
        cold_at_max = (not leveled) and railed == 0 and peak < band_lo and gain >= gain_max
        rec = {
            "atten_db": atten,
            "orx_gain": gain,
            "peak_dbfs": peak,
            "railed": railed,
            "compression_db": comp,
            "orx_ok": leveled,
        }

        if leveled or cold_at_max:
            err = abs(comp - target_compression_db)
            if best is None or err < best["err"]:
                best = {
                    "err": err,
                    "atten": atten,
                    "gain": gain,
                    "comp": comp,
                    "peak": peak,
                    "railed": railed,
                }
            if leveled and comp_lo <= comp <= comp_hi:
                record(rec, "converged")
                return finish(
                    True, "compression inside tolerance", comp=comp, peak=peak, railed=railed
                )

            error = target_compression_db - comp  # > 0: need more compression
            if error > 0:
                step = coarse_step_db if coarse_pending else fine_step(error)
                coarse_pending = False
                new = _quantize_atten(atten - step)
                if new < atten_min_db:
                    if atten <= atten_min_db + 1e-9:
                        record(rec, "atten floor")
                        return finish(
                            False,
                            "attenuation floor reached before the target",
                            at_floor=True,
                            comp=comp,
                            peak=peak,
                            railed=railed,
                        )
                    new = _quantize_atten(atten_min_db)
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
            comp=best["comp"],
            peak=best["peak"],
            railed=best["railed"],
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
    captures ``oversample`` periods, aligns one to ``ref`` and scores
    :func:`~adrvtrx.metrics.window_compression_db` on ``window_s`` around the peak.
    ``fs`` is the ORx rate. A fatal AGC result raises
    :class:`~adrvtrx.gain.AgcError` after leaving the bench safe.
    """
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

    def measure() -> tuple[float, int, float]:
        out = capture(radio, int(orx), capture_ms, bits=rx_bits).channels[orx]
        rep = clip_report(out.i, out.q, rx_bits)
        x_al, y_al, _delay = estimate_and_align(ref, out.iq, fs)
        comp, _pin, _pout = window_compression_db(x_al, y_al, fs, window_s)
        return rep.peak_dbfs, rep.railed_samples, comp

    return search_tx_compression(
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
        on_step=on_step,
    )
