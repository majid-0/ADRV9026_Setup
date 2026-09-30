"""Offline linearization: play a stored waveform at every condition of a capture CSV.

The waveform ``u`` (for example a DPD output) is sent as stored: it is multiplied
by TX full scale, rounded and clipped, and nothing rescales it. LO, TX
attenuation and ORx gain come from the CSV row; the ORx AGC does not run, so
the capture is directly comparable with the original one.

Each capture is aligned to ``u`` (what the DAC played). Quality figures are
against the original input ``x`` of that row. Results go to ``{label}.csv`` and
``{name}_{label}_dut.txt`` in ``out_dir``. Columns: ``docs/dpd_workflow_spec.md``
section 4.2.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np

from ._enums import RxChannel, TxChannel
from .conditions import (
    DUT_FIELDS,
    ConditionLog,
    DutRecord,
    OperatingCondition,
    PointCapture,
    capture_point,
    condition_keys,
    load_aligned_iq,
    load_conditions,
    save_aligned_iq,
)
from .metrics import papr_db
from .transmit import transmit_bands
from .waveform import full_scale

__all__ = [
    "StoredTx",
    "stored_tx",
    "transmit_stored",
    "dut_record",
    "replay_conditions",
]

WaveformSource = Union[str, Path, np.ndarray]


@dataclass
class StoredTx:
    """A normalized waveform as the DAC will play it. ``codes`` is complex TX codes."""

    codes: np.ndarray
    peak_dbfs: float
    clipped: int
    bits: int

    @property
    def normalized(self) -> np.ndarray:
        return self.codes / float(full_scale(self.bits))


def stored_tx(u, tx_bits: int) -> StoredTx:
    """Scale a normalized waveform to TX codes with no renormalization.

    ``u * full_scale`` is rounded and clipped to the signed range, the same as
    ``prepare_tx(..., do_normalize=False)``. ``peak_dbfs`` is
    ``max(|I|, |Q|)`` before the clip against ``2**(bits-1)``; ``clipped``
    counts samples with I or Q outside the range after rounding.
    """
    fs = full_scale(tx_bits)
    lo, hi = -(fs + 1), fs
    raw = np.asarray(u, dtype=np.complex128) * fs
    i = np.round(raw.real)
    q = np.round(raw.imag)
    over = (i < lo) | (i > hi) | (q < lo) | (q > hi)
    peak = float(np.max(np.maximum(np.abs(raw.real), np.abs(raw.imag)))) if raw.size else 0.0
    peak_dbfs = 20.0 * np.log10(peak / float(1 << (tx_bits - 1))) if peak > 0 else float("-inf")
    codes = np.clip(i, lo, hi) + 1j * np.clip(q, lo, hi)
    return StoredTx(codes, float(peak_dbfs), int(np.count_nonzero(over)), tx_bits)


def transmit_stored(radio, tx: TxChannel, u: StoredTx) -> None:
    """Start looping playback of ``u`` exactly as stored."""
    transmit_bands(radio, {tx: u.codes}, u.bits, do_normalize=False)


def dut_record(
    condition: OperatingCondition,
    point: PointCapture,
    x_codes,
    u: StoredTx,
    *,
    in_file: str,
    out_file: str,
) -> DutRecord:
    """Build one DUT row from a capture of ``u`` scored against ``x_codes``."""
    papr_u = papr_db(u.codes)
    return DutRecord(
        **condition_keys(condition),
        **point.metrics,
        peak_dbfs=point.clip.peak_dbfs,
        railed=point.clip.railed_samples,
        papr_dpd_db=papr_u,
        papr_expansion_db=papr_u - papr_db(x_codes),
        tx_peak_dbfs=u.peak_dbfs,
        tx_clipped=u.clipped,
        ref_file=condition.in_file,
        in_file=in_file,
        out_file=out_file,
    )


def _preflight(rows, csv_dir: Path, waveform_for) -> dict[str, WaveformSource]:
    missing: list[str] = []
    sources: dict[str, WaveformSource] = {}
    for row in rows:
        ref = csv_dir / row.in_file
        if not ref.is_file():
            missing.append(f"{row.name}: reference {ref}")
        try:
            src = waveform_for(row)
        except Exception as exc:  # noqa: BLE001 - report every row before failing
            missing.append(f"{row.name}: waveform_for raised {exc!r}")
            continue
        if isinstance(src, (str, Path)):
            if not Path(src).is_file():
                missing.append(f"{row.name}: waveform {src}")
            sources[row.name] = Path(src)
        else:
            sources[row.name] = np.asarray(src)
    if missing:
        raise FileNotFoundError("replay cannot start:\n" + "\n".join(missing))
    return sources


def replay_conditions(
    radio,
    csv_path: str | Path,
    waveform_for: Callable[[OperatingCondition], WaveformSource],
    *,
    tx: TxChannel,
    orx: RxChannel,
    tx_bits: int,
    rx_bits: int,
    fs: float,
    out_dir: str | Path,
    label: str = "dpd",
    oversample: int = 2,
    lo: str = "LO1",
    file_scale: float = 1.0,
    on_row: Callable[[DutRecord], None] | None = None,
) -> Path:
    """Replay a waveform at every row of ``csv_path``. Returns the DUT CSV path.

    ``waveform_for(row)`` returns the normalized waveform for that row, or a path
    to an ``I<TAB>Q`` file holding it. A file holds ``u * file_scale``: 1.0 for
    normalized files, 2048 for files written in TX codes scaled by ``2**11``.
    Paths are loaded one row at a time. Every reference and every waveform path is
    checked before the first transmission. Rows run sorted by
    ``(freq_hz, bw_mhz, backoff_db)``; the LO is retuned, with TX off, only when
    the frequency changes. TX is disabled when the replay ends or fails.
    """
    csv_path = Path(csv_path)
    csv_dir = csv_path.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(load_conditions(csv_path), key=lambda r: (r.freq_hz, r.bw_mhz, r.backoff_db))
    sources = _preflight(rows, csv_dir, waveform_for)

    tx_scale = float(full_scale(tx_bits))
    refs: dict[str, np.ndarray] = {}
    log_path = out_dir / f"{label}.csv"
    current_freq = None
    with ConditionLog(log_path, DUT_FIELDS) as log:
        try:
            for row in rows:
                if row.in_file not in refs:
                    refs[row.in_file] = load_aligned_iq(csv_dir / row.in_file) * tx_scale
                x_codes = refs[row.in_file]

                src = sources[row.name]
                if isinstance(src, Path):
                    u_norm = load_aligned_iq(src) / float(file_scale)
                    in_file = src.name
                else:
                    u_norm = src
                    in_file = f"{row.name}_{label}.txt"
                    save_aligned_iq(u_norm * tx_scale, out_dir / in_file, tx_bits)
                if len(u_norm) != len(x_codes):
                    raise ValueError(
                        f"{row.name}: waveform has {len(u_norm)} samples, "
                        f"reference {len(x_codes)}"
                    )
                u = stored_tx(u_norm, tx_bits)

                if row.freq_hz != current_freq:
                    radio.disable_tx()
                    radio.retune_lo(lo, int(row.freq_hz))
                    current_freq = row.freq_hz
                radio.set_tx_atten(tx, row.tx_atten_db)
                radio.set_rx_gain(orx, int(row.orx_gain))
                transmit_stored(radio, tx, u)

                point = capture_point(
                    radio,
                    orx,
                    u.codes,
                    rx_bits=rx_bits,
                    fs=fs,
                    bw_hz=row.bw_mhz * 1_000_000,
                    oversample=oversample,
                    metric_ref=x_codes,
                )
                out_file = f"{row.name}_{label}_dut.txt"
                save_aligned_iq(point.y_aligned, out_dir / out_file, rx_bits)
                record = dut_record(row, point, x_codes, u, in_file=in_file, out_file=out_file)
                log.append(record)
                if on_row is not None:
                    on_row(record)
        finally:
            try:
                radio.disable_tx()
            except Exception:  # noqa: BLE001 - leave the bench safe on any path
                pass
    return log_path
