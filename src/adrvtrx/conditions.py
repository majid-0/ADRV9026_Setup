"""Saved operating conditions: one reference per signal, one ORx file per capture, one CSV.

A condition is one signal, one LO frequency, one TX attenuation and one ORx gain
index. Replay loads the CSV, sets that LO, attenuation and gain, and does not
run the ORx AGC again.

The transmit reference is the same for every frequency and backoff of a signal,
so it is written once. Each capture writes only its time-aligned ORx period.
Columns are defined in ``docs/dpd_workflow_spec.md`` section 4.

IQ files are normalized float ``I<TAB>Q`` (1.0 = full scale), the same layout
as :func:`adrvtrx.waveform.save_tab_iq_float`.

The columns ``lock_on``, ``gain_compression_db``, ``amam_top_slope`` and
``pa_clipped`` come last and have defaults, so CSVs written before they existed
still load (``papr``, NaN, NaN, false).
"""

from __future__ import annotations

import csv
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from ._enums import RxChannel
from .align import estimate_and_align
from .capture import capture
from .gain import ClipReport, clip_report
from .metrics import PA_CLIP_SLOPE, period_metrics
from .waveform import full_scale, load_tab_iq

__all__ = [
    "CSV_FIELDS",
    "DUT_FIELDS",
    "OperatingCondition",
    "DutRecord",
    "ConditionLog",
    "PointCapture",
    "capture_point",
    "condition_keys",
    "condition_name",
    "load_conditions",
    "load_dut_records",
    "save_aligned_iq",
    "load_aligned_iq",
]

_INT_FIELDS = frozenset({"freq_hz", "bw_mhz", "backoff_db", "orx_gain", "railed", "tx_clipped"})
_BOOL_FIELDS = frozenset({"converged", "pa_clipped"})
_STR_FIELDS = frozenset({"name", "signal", "ref_file", "in_file", "out_file", "lock_on"})
_FLOAT_FMT = {"delay_samples": ".2f", "delay_ns": ".1f", "corr": ".4f", "amam_top_slope": ".3f"}


def _format(key: str, value: Any) -> str:
    if key in _STR_FIELDS:
        return str(value)
    if key in _BOOL_FIELDS:
        return "true" if value else "false"
    if key in _INT_FIELDS:
        return str(int(value))
    return format(float(value), _FLOAT_FMT.get(key, ".2f"))


def _parse(key: str, text: str) -> Any:
    if key in _STR_FIELDS:
        return text
    if key in _BOOL_FIELDS:
        return str(text).strip().lower() in ("true", "1", "yes")
    if key in _INT_FIELDS:
        return int(float(text))
    return float(text)


class _Row:
    """CSV conversion shared by the record dataclasses."""

    def to_row(self) -> dict[str, str]:
        return {k: _format(k, v) for k, v in asdict(self).items()}

    @classmethod
    def from_row(cls, row: dict[str, str]):
        """Columns the row does not have keep their defaults (older CSVs)."""
        return cls(**{f.name: _parse(f.name, row[f.name]) for f in fields(cls) if f.name in row})


@dataclass
class OperatingCondition(_Row):
    """One initial capture. ``compression_db`` is the search result at the lock.

    ``lock_on`` is the metric the search locked on (``papr`` or ``gain``), so
    ``compression_db`` is that metric. ``gain_compression_db``, ``amam_top_slope``
    and ``pa_clipped`` are measured on this capture (see ``metrics.period_metrics``).
    """

    name: str
    signal: str
    freq_hz: int
    bw_mhz: int
    backoff_db: int
    locked_atten_db: float
    tx_atten_db: float
    orx_gain: int
    compression_db: float
    converged: bool
    delay_samples: float
    delay_ns: float
    corr: float
    nmse_db: float
    papr_in_db: float
    papr_out_db: float
    papr_compression_db: float
    aclr_lower_dbc: float
    aclr_upper_dbc: float
    peak_dbfs: float
    rms_dbfs: float
    railed: int
    in_file: str
    out_file: str
    lock_on: str = "papr"
    gain_compression_db: float = float("nan")
    amam_top_slope: float = float("nan")
    pa_clipped: bool = False


@dataclass
class DutRecord(_Row):
    """One capture of a linearized waveform ``u``. Quality figures are against ``x``.

    Delay and ``corr`` come from aligning the DUT capture to ``u``.
    """

    name: str
    signal: str
    freq_hz: int
    bw_mhz: int
    backoff_db: int
    locked_atten_db: float
    tx_atten_db: float
    orx_gain: int
    compression_db: float
    converged: bool
    delay_samples: float
    delay_ns: float
    corr: float
    nmse_db: float
    papr_in_db: float
    papr_out_db: float
    papr_compression_db: float
    aclr_lower_dbc: float
    aclr_upper_dbc: float
    peak_dbfs: float
    rms_dbfs: float
    railed: int
    papr_dpd_db: float
    papr_expansion_db: float
    tx_peak_dbfs: float
    tx_clipped: int
    ref_file: str
    in_file: str
    out_file: str
    lock_on: str = "papr"
    gain_compression_db: float = float("nan")
    amam_top_slope: float = float("nan")
    pa_clipped: bool = False


CSV_FIELDS = tuple(f.name for f in fields(OperatingCondition))
DUT_FIELDS = tuple(f.name for f in fields(DutRecord))

_CONDITION_KEYS = (*CSV_FIELDS[: CSV_FIELDS.index("converged") + 1], "lock_on")


def condition_name(tx, backoff_db: int, freq_hz: int, bw_mhz: int) -> str:
    """``{TX}_{backoff}dB_{freq_MHz}MHz_{bw}BW``, e.g. ``TX1_0dB_2000MHz_100BW``."""
    tx_name = tx.name if hasattr(tx, "name") else str(tx)
    return f"{tx_name}_{int(backoff_db)}dB_{int(freq_hz) // 1_000_000}MHz_{int(bw_mhz)}BW"


def condition_keys(condition: OperatingCondition) -> dict[str, Any]:
    """The condition columns (``name`` .. ``converged``, and ``lock_on``) of a capture row."""
    return {k: getattr(condition, k) for k in _CONDITION_KEYS}


def save_aligned_iq(iq_codes, path: str | Path, bits: int) -> None:
    """Write complex integer-code samples as normalized ``I<TAB>Q`` (codes / full scale)."""
    scale = float(full_scale(bits))
    z = np.asarray(iq_codes)
    out = np.column_stack((np.real(z) / scale, np.imag(z) / scale))
    np.savetxt(path, out, delimiter="\t", fmt="%.9g")


def load_aligned_iq(path: str | Path) -> np.ndarray:
    """Load a normalized ``I<TAB>Q`` file (multiply by ``full_scale(bits)`` for codes)."""
    return load_tab_iq(path)


class ConditionLog:
    """CSV writer for records. The header is written on open; each row is flushed."""

    def __init__(self, path: str | Path, fields: tuple[str, ...] = CSV_FIELDS):
        self.path = Path(path)
        self.fields = tuple(fields)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=list(self.fields))
        self._writer.writeheader()
        self._file.flush()

    def append(self, record, **extra: Any) -> None:
        """Write one record. ``extra`` fills columns the record does not have."""
        row = record.to_row()
        row.update({k: str(v) for k, v in extra.items()})
        missing = [k for k in self.fields if k not in row]
        if missing:
            raise ValueError(f"{self.path.name}: record is missing columns {missing}")
        self._writer.writerow({k: row[k] for k in self.fields})
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> ConditionLog:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _load(csv_path: str | Path, cls):
    required = {
        f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
    }
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        header = set(reader.fieldnames or ())
        if not required.issubset(header):
            raise ValueError(f"{csv_path}: missing columns {sorted(required - header)}")
        return [cls.from_row(row) for row in reader]


def load_conditions(csv_path: str | Path) -> list[OperatingCondition]:
    """Read a capture CSV. Extra columns are ignored; a missing column without a
    default raises ``ValueError``."""
    return _load(csv_path, OperatingCondition)


def load_dut_records(csv_path: str | Path) -> list[DutRecord]:
    """Read a DUT CSV written by replay or the online loop."""
    return _load(csv_path, DutRecord)


@dataclass
class PointCapture:
    """One aligned capture. ``y_aligned`` is in ORx codes, the length of the reference."""

    y_aligned: np.ndarray
    delay_samples: float
    clip: ClipReport
    metrics: dict[str, float]


def capture_point(
    radio,
    orx: RxChannel,
    ref,
    *,
    rx_bits: int,
    fs: float,
    bw_hz: float,
    oversample: int = 2,
    metric_ref=None,
    min_top_slope: float = PA_CLIP_SLOPE,
) -> PointCapture:
    """Capture ``oversample`` periods, align one period to ``ref``, compute the metrics.

    TX must already be running. Gain and attenuation are not touched. The clip
    report is on the raw capture. The period is aligned to ``ref`` (the waveform
    being transmitted, in TX codes). Metrics are against ``metric_ref`` when given
    (the original input during replay), otherwise against ``ref``; ``corr`` is
    always against ``ref``. ``pa_clipped`` uses ``min_top_slope``.
    """
    ref = np.asarray(ref)
    capture_ms = oversample * len(ref) / float(fs) * 1e3
    out = capture(radio, int(orx), capture_ms, bits=rx_bits).channels[orx]
    rep = clip_report(out.i, out.q, rx_bits)
    _x_al, y_al, delay = estimate_and_align(ref, out.iq, fs)
    x = ref if metric_ref is None else np.asarray(metric_ref)
    if len(x) != len(y_al):
        raise ValueError(f"metric reference has {len(x)} samples, aligned period {len(y_al)}")
    metrics = period_metrics(
        x,
        y_al,
        fs=fs,
        bw_hz=bw_hz,
        rx_bits=rx_bits,
        delay_samples=delay,
        corr_ref=ref,
        min_top_slope=min_top_slope,
    )
    return PointCapture(y_aligned=y_al, delay_samples=float(delay), clip=rep, metrics=metrics)
