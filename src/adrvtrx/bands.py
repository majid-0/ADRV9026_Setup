"""Band primitive and single/dual/"quad" orchestration.

A :class:`Band` ties one TX path + waveform to a capture channel (Rx or ORx),
a duration and an output name. Single/dual/quad = 1/2/4 bands sharing the two
real LOs across four distinct paths. All bands transmit together and are captured
in one sample-aligned snapshot (concern #2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._enums import RxChannel, RxTrigSource, TxChannel, is_orx
from .capture import CaptureResult, capture
from .gain import level_orx
from .transmit import transmit_bands


@dataclass
class Band:
    name: str
    tx_channel: TxChannel
    waveform: np.ndarray  # complex IQ
    capture_channel: RxChannel
    capture_time_ms: float = 1.0


@dataclass
class BandRun:
    captures: CaptureResult
    files: list[Path]


def _sof_trigger(bands: Sequence[Band]) -> RxTrigSource:
    """Align the snapshot to the start-of-frame of the lowest-numbered TX in use."""
    from ._enums import TX_SINGLE, TX_SOF_FOR

    for tx in TX_SINGLE:
        if any(b.tx_channel == tx for b in bands):
            return TX_SOF_FOR[tx]
    return RxTrigSource.IMMEDIATE


def run_bands(
    radio,
    bands: Sequence[Band],
    *,
    tx_bits: int,
    rx_bits: int,
    out_dir: str | Path,
    trig: RxTrigSource | None = None,
    level_orx_channels: bool = True,
    orx_target_dbfs: float = -12.0,
    orx_tolerance_db: float = 2.0,
) -> BandRun:
    """Transmit all bands, (optionally) level ORx, capture one snapshot, save per band."""
    # 1. Multi-band transmit (all TX channels start together).
    channel_to_iq = {b.tx_channel: b.waveform for b in bands}
    transmit_bands(radio, channel_to_iq, tx_bits, continuous=True)

    # 2. Software ORx leveling on any ORx capture channels (flag-based).
    if level_orx_channels:
        for b in bands:
            if is_orx(b.capture_channel):
                level_orx(
                    radio,
                    b.capture_channel,
                    target_dbfs=orx_target_dbfs,
                    tolerance_db=orx_tolerance_db,
                )

    # 3. One aligned snapshot over the union of capture channels.
    capture_mask = 0
    for b in bands:
        capture_mask |= int(b.capture_channel)
    trig = trig if trig is not None else _sof_trigger(bands)
    longest = max(b.capture_time_ms for b in bands)
    result = capture(radio, capture_mask, longest, trig=trig, bits=rx_bits)

    # 4. Save each band's capture channel under the band name.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for b in bands:
        cap = result.channels.get(b.capture_channel)
        if cap is None:
            continue
        path = out_dir / f"{b.name}.txt"
        cap.save(path)
        files.append(path)
    return BandRun(captures=result, files=files)


# -- convenience builders ------------------------------------------------------


def single_band(name, tx, waveform, capture_channel, capture_time_ms=1.0) -> list[Band]:
    return [Band(name, tx, waveform, capture_channel, capture_time_ms)]


def make_bands(specs: Sequence[tuple]) -> list[Band]:
    """Build bands from ``(name, tx_channel, waveform, capture_channel[, ms])`` tuples."""
    bands = []
    for spec in specs:
        bands.append(Band(*spec))
    return bands
