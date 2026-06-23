"""adrvtrx -- Python automation for ADRV9026 multi-band TX + synchronized ORx capture.

Hardware-free modules (config, waveform, profile, gain math, enums) import without
pythonnet; only the .NET boundary in ``_clr`` / ``radio`` needs it.
"""

from __future__ import annotations

from ._enums import RxChannel, RxTrigSource, TxChannel, TxTrigSource
from .bands import Band, make_bands, run_bands
from .config import Config, load_config
from .gain import ClipReport, clip_report, level_orx, peak_window
from .profile import ProfileInfo, read_profile
from .sweep import SweepAxis, run_sweep, sweep_points
from .waveform import load_tab_iq, normalize, prepare_tx, quantize, save_tab_iq_float

__version__ = "0.1.0"

__all__ = [
    "Config",
    "load_config",
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
    "level_orx",
    "Band",
    "make_bands",
    "run_bands",
    "SweepAxis",
    "run_sweep",
    "sweep_points",
]


def __getattr__(name: str):
    """Lazily expose the hardware-facing Radio so importing the package stays light."""
    if name == "Radio":
        from .radio import Radio

        return Radio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
