"""High-level session helpers tying config -> radio -> profile together."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from .config import Config, load_config
from .profile import ProfileInfo, read_profile
from .radio import Radio


@contextmanager
def session(
    config: Config | None = None, *, program: bool = True
) -> Iterator[tuple[Radio, ProfileInfo]]:
    """Open a connected, safe-stated radio and the profile datapath info.

    Yields ``(radio, profile_info)``. The radio is forced safe on entry and on
    every exit path (see :class:`~adrvtrx.radio.Radio`).
    """
    cfg = config or load_config()
    info = read_profile(cfg.profile_path)
    with Radio(cfg) as radio:
        if program:
            radio.program()
        yield radio, info


def verify_status(radio: Radio) -> dict:
    """Read back a quick health summary after programming."""
    return {
        "connected": radio._is_connected(),
        "lo1_hz": radio.get_lo("LO1"),
        "lo2_hz": radio.get_lo("LO2"),
        "pll_lock_status": radio.pll_lock_status(),
    }
