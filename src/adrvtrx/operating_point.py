"""Find and capture one PA operating point: transmit, lock, optional backoff, capture, one CSV row.

:func:`find_operating_point` is the procedure of ``notebooks/pa_operating_point.ipynb``
(and one point of ``pa_operating_sweep.ipynb``) as one call:

1. TX off, the start attenuation set, the LO tuned; then the signal is transmitted
   (normalized and quantized by ``prepare_tx``).
2. :func:`adrvtrx.compression.find_compression_point` runs the ORx AGC and the
   attenuation search (PAPR or gain lock, optional PA clip guard). Its result is
   the lock, backoff 0 dB.
3. With ``backoff_db > 0``: the attenuation goes to ``lock + backoff`` and the ORx
   AGC runs again there (as in the sweep notebook).
4. :func:`adrvtrx.conditions.capture_point` takes one aligned capture.
5. The :class:`~adrvtrx.conditions.OperatingCondition` row is built (name from
   :func:`~adrvtrx.conditions.condition_name`). With ``save_dir``, the reference,
   the aligned capture and the CSV row are written (the row is appended to an
   existing CSV with the same columns).

TX is disabled when the call ends or fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._enums import RxChannel, TxChannel
from .capture import AgcResult, autolevel_capture
from .compression import CompressionResult, _quantize_atten, find_compression_point
from .conditions import (
    ConditionLog,
    OperatingCondition,
    PointCapture,
    capture_point,
    condition_name,
    save_aligned_iq,
)
from .metrics import PA_CLIP_SLOPE
from .radio import MAX_TX_ATTEN_DB
from .transmit import transmit_bands
from .waveform import full_scale, prepare_tx

__all__ = ["OperatingPoint", "find_operating_point"]


@dataclass
class OperatingPoint:
    """One operating point.

    ``condition`` is the CSV row, ``point`` the aligned capture at
    ``condition.tx_atten_db`` and ``search`` the attenuation search (the lock).
    ``ref`` is the transmitted reference in TX codes and ``x`` the same normalized
    (1.0 = full scale), the input a DPD loop uses. ``agc`` is the AGC run after the
    backoff (``None`` at backoff 0). ``csv_path`` is the CSV the row went to.
    """

    condition: OperatingCondition
    point: PointCapture
    search: CompressionResult
    ref: np.ndarray
    x: np.ndarray
    agc: AgcResult | None = None
    csv_path: Path | None = None


def find_operating_point(
    radio,
    tx: TxChannel,
    orx: RxChannel,
    signal,
    *,
    tx_bits: int,
    rx_bits: int,
    fs: float,
    freq_hz: int,
    bw_mhz: int,
    target_compression_db: float,
    start_atten_db: float,
    atten_min_db: float,
    signal_name: str = "signal.txt",
    lock_on: str = "papr",
    min_top_slope: float | None = None,
    window_s: float = 1e-4,
    coarse_step_db: float = 10.0,
    fine_step_db: float = 0.25,
    comp_tol_db: float = 0.3,
    target_dbfs: float = -1.0,
    tol_up_db: float = 0.3,
    tol_down_db: float = 0.6,
    max_iterations: int = 24,
    backoff_db: int = 0,
    oversample: int = 2,
    lo: str = "LO1",
    save_dir: str | Path | None = None,
    csv_name: str | None = None,
    on_step=None,
) -> OperatingPoint:
    """Lock the PA at the target compression, back off, capture, and build the CSV row.

    ``signal`` is the waveform at any scale; it is normalized and quantized to TX
    codes. ``signal_name`` goes in the ``signal`` column and names the reference
    file ``{stem}_in.txt``. The search arguments are those of
    :func:`~adrvtrx.compression.find_compression_point`; ``fs`` is the ORx rate.
    The capture uses ``min_top_slope`` (or ``PA_CLIP_SLOPE``) for ``pa_clipped``.

    With ``save_dir``, writes the reference and ``{name}_out.txt`` there and
    appends the row to ``csv_name`` (default ``{TX}_conditions.csv``). The radio is
    left at the condition's attenuation and ORx gain, with TX off.
    """
    signal = np.asarray(signal, dtype=np.complex128)
    i_tx, q_tx = prepare_tx(signal, tx_bits)
    ref = i_tx.astype(np.float64) + 1j * q_tx.astype(np.float64)
    x = ref / float(full_scale(tx_bits))
    backoff_db = int(backoff_db)
    if backoff_db < 0:
        raise ValueError(f"backoff_db must be >= 0, got {backoff_db}")
    name = condition_name(tx, backoff_db, freq_hz, bw_mhz)
    tx_name = tx.name if hasattr(tx, "name") else str(tx)
    in_name = f"{Path(signal_name).stem}_in.txt"
    out_name = f"{name}_out.txt"

    agc = None
    try:
        radio.disable_tx()
        radio.set_tx_atten(tx, _quantize_atten(start_atten_db))
        radio.retune_lo(lo, int(freq_hz))
        transmit_bands(radio, {tx: signal}, tx_bits)
        search = find_compression_point(
            radio,
            tx,
            orx,
            ref,
            rx_bits=rx_bits,
            fs=fs,
            target_compression_db=target_compression_db,
            start_atten_db=start_atten_db,
            atten_min_db=atten_min_db,
            lock_on=lock_on,
            min_top_slope=min_top_slope,
            window_s=window_s,
            oversample=oversample,
            coarse_step_db=coarse_step_db,
            fine_step_db=fine_step_db,
            comp_tol_db=comp_tol_db,
            target_dbfs=target_dbfs,
            tol_up_db=tol_up_db,
            tol_down_db=tol_down_db,
            max_iterations=max_iterations,
            on_step=on_step,
        )
        locked = search.final_atten_db
        atten = locked
        gain = search.final_orx_gain
        if backoff_db > 0:
            atten = _quantize_atten(min(locked + backoff_db, MAX_TX_ATTEN_DB))
            radio.set_tx_atten(tx, atten)
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
            gain = agc.final_gain_index
        point = capture_point(
            radio,
            orx,
            ref,
            rx_bits=rx_bits,
            fs=fs,
            bw_hz=bw_mhz * 1_000_000,
            oversample=oversample,
            min_top_slope=PA_CLIP_SLOPE if min_top_slope is None else min_top_slope,
        )
    finally:
        try:
            radio.disable_tx()
        except Exception:  # noqa: BLE001 - leave the bench safe on any path
            pass

    condition = OperatingCondition(
        name=name,
        signal=Path(signal_name).name,
        freq_hz=int(freq_hz),
        bw_mhz=int(bw_mhz),
        backoff_db=backoff_db,
        locked_atten_db=locked,
        tx_atten_db=atten,
        orx_gain=int(gain),
        compression_db=search.compression_db,
        converged=search.converged,
        peak_dbfs=point.clip.peak_dbfs,
        railed=point.clip.railed_samples,
        in_file=in_name,
        out_file=out_name,
        lock_on=search.lock_on,
        **point.metrics,
    )

    csv_path = None
    if save_dir is not None:
        save = Path(save_dir)
        save.mkdir(parents=True, exist_ok=True)
        save_aligned_iq(ref, save / in_name, tx_bits)
        save_aligned_iq(point.y_aligned, save / out_name, rx_bits)
        csv_path = save / (csv_name or f"{tx_name}_conditions.csv")
        with ConditionLog(csv_path, append=True) as log:
            log.append(condition)

    return OperatingPoint(
        condition=condition,
        point=point,
        search=search,
        ref=ref,
        x=x,
        agc=agc,
        csv_path=csv_path,
    )
