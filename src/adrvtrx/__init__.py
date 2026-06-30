"""adrvtrx -- Python automation for ADRV9026 multi-band TX + synchronized ORx capture.

Hardware-free modules (config, waveform, profile, gain math, enums) import without
pythonnet; only the .NET boundary in ``_clr`` / ``radio`` needs it.
"""

from __future__ import annotations

from ._enums import RxChannel, RxTrigSource, TxChannel, TxTrigSource
from .align import apply_delay, estimate_and_align, estimate_delay, match_corr
from .bands import Band, make_bands, run_bands
from .capture import AgcResult, autolevel_capture, measure_delay
from .config import Config, lo_for_tx, load_config
from .gain import AgcError, ClipReport, autolevel_orx, clip_report, peak_window, verify_no_clip
from .profile import ProfileInfo, read_profile
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
]


def __getattr__(name: str):
    """Lazily expose the hardware-facing Radio so importing the package stays light."""
    if name == "Radio":
        from .radio import Radio

        return Radio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
