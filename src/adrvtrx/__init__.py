"""adrvtrx -- Python automation for ADRV9026 multi-band TX + synchronized ORx capture.

Hardware-free modules (config, waveform, profile, gain math, enums) import without
pythonnet; only the .NET boundary in ``_clr`` / ``radio`` needs it.
"""

from __future__ import annotations

from ._enums import RxChannel, RxTrigSource, TxChannel, TxTrigSource
from .align import apply_delay, estimate_and_align, estimate_delay, match_corr
from .bands import Band, make_bands, run_bands
from .capture import AgcResult, autolevel_capture, measure_delay
from .compression import CompressionResult, find_compression_point, search_tx_compression
from .conditions import (
    CSV_FIELDS,
    DUT_FIELDS,
    ConditionLog,
    DutRecord,
    OperatingCondition,
    PointCapture,
    capture_point,
    condition_name,
    load_aligned_iq,
    load_conditions,
    load_dut_records,
    save_aligned_iq,
)
from .config import Config, lo_for_tx, load_config
from .dpd import (
    FULL_SCALE_DBM,
    TARGET_BACKOFF_DB,
    IlaStep,
    PeakLimit,
    iteration_table,
    limit_peak,
    normalize_pair,
    peak_dbm,
)
from .gain import AgcError, ClipReport, autolevel_orx, clip_report, peak_window, verify_no_clip
from .gmp import GMP, peak_block
from .linearize import LinearizeResult, linearize
from .metrics import (
    PA_CLIP_SLOPE,
    aclr_db,
    gain_compression_db,
    inband_corr,
    nmse_db,
    papr_db,
    period_metrics,
    rms_dbfs,
    window_compression_db,
)
from .operating_point import OperatingPoint, find_operating_point
from .profile import ProfileInfo, read_profile
from .replay import replay_conditions, stored_tx
from .sweep import SweepAxis, run_sweep, sweep_points
from .sweep_plan import (
    SweepPlanSummary,
    apply_sweep_point,
    flatten_point,
    format_point_label,
    iter_sweep_points,
    max_power_db,
    run_planned_sweep,
    summarize_sweep_plan,
    sweep_defaults_from_config,
)
from .waveform import load_tab_iq, normalize, prepare_tx, quantize, save_tab_iq_float

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
    "lo_for_tx",
    "RxChannel",
    "TxChannel",
    "RxTrigSource",
    "TxTrigSource",
    "ProfileInfo",
    "read_profile",
    "load_tab_iq",
    "save_tab_iq_float",
    "normalize",
    "quantize",
    "prepare_tx",
    "clip_report",
    "ClipReport",
    "peak_window",
    "autolevel_orx",
    "verify_no_clip",
    "autolevel_capture",
    "AgcResult",
    "AgcError",
    "estimate_delay",
    "estimate_and_align",
    "apply_delay",
    "match_corr",
    "measure_delay",
    "Band",
    "make_bands",
    "run_bands",
    "SweepAxis",
    "run_sweep",
    "sweep_points",
    "iter_sweep_points",
    "summarize_sweep_plan",
    "SweepPlanSummary",
    "apply_sweep_point",
    "run_planned_sweep",
    "flatten_point",
    "format_point_label",
    "max_power_db",
    "sweep_defaults_from_config",
    "papr_db",
    "window_compression_db",
    "gain_compression_db",
    "PA_CLIP_SLOPE",
    "nmse_db",
    "aclr_db",
    "inband_corr",
    "rms_dbfs",
    "period_metrics",
    "CompressionResult",
    "search_tx_compression",
    "find_compression_point",
    "CSV_FIELDS",
    "DUT_FIELDS",
    "OperatingCondition",
    "DutRecord",
    "ConditionLog",
    "PointCapture",
    "capture_point",
    "condition_name",
    "load_conditions",
    "load_dut_records",
    "save_aligned_iq",
    "load_aligned_iq",
    "replay_conditions",
    "stored_tx",
    "linearize",
    "LinearizeResult",
    "OperatingPoint",
    "find_operating_point",
    "GMP",
    "peak_block",
    "FULL_SCALE_DBM",
    "TARGET_BACKOFF_DB",
    "peak_dbm",
    "normalize_pair",
    "PeakLimit",
    "limit_peak",
    "IlaStep",
    "iteration_table",
]


def __getattr__(name: str):
    """Lazily expose the hardware-facing Radio so importing the package stays light."""
    if name == "Radio":
        from .radio import Radio

        return Radio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
