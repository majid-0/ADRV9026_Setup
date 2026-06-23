"""Read datapath parameters (bit depth + sample rates) from a TES ``.profile`` JSON.

Confirmed fields (see docs/api_notes.md):
  * ``framer[].jesd204Np``   -> Rx/ORx transport word width (bit depth)
  * ``deframer[].jesd204Np`` -> Tx transport word width
  * ``framer/.../rxOutputRate_kHz`` and ``deframer/.../txInputRate_kHz`` -> sample rates

Pure JSON parsing -- testable without hardware. The ``.profile`` files are valid
JSON despite the extension.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _iter_values(obj: Any, key: str):
    """Yield every value stored under ``key`` anywhere in a nested JSON structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                yield v
            yield from _iter_values(v, key)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_values(item, key)


def _first_positive(obj: Any, key: str) -> int | None:
    for v in _iter_values(obj, key):
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
    return None


@dataclass
class ProfileInfo:
    tx_bits: int
    rx_bits: int
    tx_rate_khz: int
    rx_rate_khz: int

    @property
    def tx_rate_hz(self) -> int:
        return self.tx_rate_khz * 1000

    @property
    def rx_rate_hz(self) -> int:
        return self.rx_rate_khz * 1000


def read_profile(path: str | Path) -> ProfileInfo:
    """Parse the datapath parameters needed for waveform/capture scaling."""
    data = json.loads(Path(path).read_text())

    framer = data.get("framer", [])
    deframer = data.get("deframer", [])

    rx_bits = _first_positive(framer, "jesd204Np") or _first_positive(data, "jesd204Np")
    tx_bits = _first_positive(deframer, "jesd204Np") or rx_bits
    tx_rate = _first_positive(data, "txInputRate_kHz")
    rx_rate = _first_positive(data, "rxOutputRate_kHz")

    missing = [
        name
        for name, val in (
            ("jesd204Np (rx)", rx_bits),
            ("jesd204Np (tx)", tx_bits),
            ("txInputRate_kHz", tx_rate),
            ("rxOutputRate_kHz", rx_rate),
        )
        if val is None
    ]
    if missing:
        raise ValueError(f"{path}: could not read profile field(s): {', '.join(missing)}")

    return ProfileInfo(tx_bits=tx_bits, rx_bits=rx_bits, tx_rate_khz=tx_rate, rx_rate_khz=rx_rate)
