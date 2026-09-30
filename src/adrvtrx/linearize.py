"""Online linearization: transmit, capture, align, let the user's ``step`` pick the next waveform.

The loop holds one saved condition (LO, TX attenuation, ORx gain) for every
iteration and never runs the ORx AGC, so iterations are comparable. Each
capture is aligned to the waveform actually transmitted, ``u``, and scored
against the original input ``x`` exactly like :mod:`adrvtrx.replay`.

The DPD algorithm lives entirely in ``step(x, u, z, it)``: all three are
normalized floats of the same length, ``z`` aligned to ``u``. It returns the next
``u`` or ``None`` to stop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ._enums import RxChannel, TxChannel
from .conditions import (
    DUT_FIELDS,
    ConditionLog,
    DutRecord,
    OperatingCondition,
    capture_point,
    save_aligned_iq,
)
from .replay import dut_record, stored_tx, transmit_stored
from .waveform import full_scale

__all__ = ["StepFn", "LinearizeResult", "linearize"]

StepFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int], Optional[np.ndarray]]


@dataclass
class LinearizeResult:
    """``records[k]`` is the capture of iteration ``k``; iteration 0 transmits ``u0``."""

    records: list[DutRecord] = field(default_factory=list)
    u: np.ndarray | None = None
    z: np.ndarray | None = None
    reason: str = ""


def linearize(
    radio,
    condition: OperatingCondition,
    x,
    step: StepFn,
    *,
    tx: TxChannel,
    orx: RxChannel,
    tx_bits: int,
    rx_bits: int,
    fs: float,
    n_iter: int = 5,
    u0=None,
    stop_on_rail: bool = True,
    save_dir: str | Path | None = None,
    label: str = "ila",
    oversample: int = 2,
    lo: str = "LO1",
    on_iter: Callable[[int, DutRecord], None] | None = None,
) -> LinearizeResult:
    """Run up to ``n_iter`` transmit/capture iterations at ``condition``.

    ``x`` is the normalized original input. ``u0`` defaults to ``x``. The loop
    stops after ``n_iter`` captures, when ``step`` returns ``None``, or when the
    ORx rails and ``stop_on_rail`` is set. With ``save_dir``, every ``u`` and
    ``z`` is written with ``{label}.csv`` (DUT columns plus ``iteration``).
    TX is disabled when the loop ends or fails.
    """
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    x_norm = np.asarray(x, dtype=np.complex128)
    x_codes = x_norm * float(full_scale(tx_bits))
    rx_scale = float(full_scale(rx_bits))
    u_norm = x_norm if u0 is None else np.asarray(u0, dtype=np.complex128)

    save = Path(save_dir) if save_dir is not None else None
    log = None
    if save is not None:
        save.mkdir(parents=True, exist_ok=True)
        log = ConditionLog(save / f"{label}.csv", DUT_FIELDS + ("iteration",))

    result = LinearizeResult()
    try:
        radio.disable_tx()
        radio.retune_lo(lo, int(condition.freq_hz))
        radio.set_tx_atten(tx, condition.tx_atten_db)
        radio.set_rx_gain(orx, int(condition.orx_gain))

        for it in range(n_iter):
            if len(u_norm) != len(x_norm):
                raise ValueError(
                    f"iteration {it}: waveform has {len(u_norm)} samples, input {len(x_norm)}"
                )
            u = stored_tx(u_norm, tx_bits)
            transmit_stored(radio, tx, u)
            point = capture_point(
                radio,
                orx,
                u.codes,
                rx_bits=rx_bits,
                fs=fs,
                bw_hz=condition.bw_mhz * 1_000_000,
                oversample=oversample,
                metric_ref=x_codes,
            )
            in_file = out_file = ""
            if save is not None:
                in_file = f"{condition.name}_{label}_it{it}_u.txt"
                out_file = f"{condition.name}_{label}_it{it}_z.txt"
                save_aligned_iq(u.codes, save / in_file, tx_bits)
                save_aligned_iq(point.y_aligned, save / out_file, rx_bits)
            record = dut_record(condition, point, x_codes, u, in_file=in_file, out_file=out_file)
            result.records.append(record)
            result.u = u.normalized
            result.z = point.y_aligned / rx_scale
            if log is not None:
                log.append(record, iteration=it)
            if on_iter is not None:
                on_iter(it, record)

            if stop_on_rail and record.railed > 0:
                result.reason = "railed"
                break
            if it == n_iter - 1:
                result.reason = "n_iter"
                break
            nxt = step(x_norm, result.u, result.z, it)
            if nxt is None:
                result.reason = "step returned None"
                break
            u_norm = np.asarray(nxt, dtype=np.complex128)
    finally:
        try:
            radio.disable_tx()
        except Exception:  # noqa: BLE001 - leave the bench safe on any path
            pass
        if log is not None:
            log.close()
    return result
