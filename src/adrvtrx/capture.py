"""Snapshot capture: trigger PerformRx, shape per-channel IQ, report clipping, save.

A capture is a single FPGA snapshot of all channels named in ``channels`` taken in
one ``PerformRx`` call, so they are mutually sample-aligned. Use a ``TXn_SOF``
trigger to align the snapshot to TX start-of-frame (concern #2).

The one seam confirmed on hardware is :func:`extract_channels` -- how the raw
int IQ buffers are read back out of the ``PerformRx`` result. Everything around it
(sample-count math, clip reporting, peak windowing, float save) is pure and tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._enums import RX_SINGLE, RxChannel, RxTrigSource
from .gain import ClipReport, clip_report, peak_window
from .waveform import samples_for_duration, save_tab_iq_float


@dataclass
class ChannelCapture:
    channel: RxChannel
    i: np.ndarray
    q: np.ndarray
    bits: int

    @property
    def iq(self) -> np.ndarray:
        return self.i.astype(np.float64) + 1j * self.q.astype(np.float64)

    def clip(self) -> ClipReport:
        return clip_report(self.i, self.q, self.bits)

    def peak_window(self, window_samples: int) -> ChannelCapture:
        i, q = peak_window(self.i, self.q, window_samples)
        return ChannelCapture(self.channel, i, q, self.bits)

    def save(self, path: str | Path) -> None:
        save_tab_iq_float(self.i, self.q, path, self.bits)


@dataclass
class CaptureResult:
    capture_time_ms: float
    trig: RxTrigSource
    channels: dict[RxChannel, ChannelCapture] = field(default_factory=dict)

    def save_all(self, directory: str | Path, prefix: str = "capture") -> list[Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for ch, cap in self.channels.items():
            path = directory / f"{prefix}_{ch.name}.txt"
            cap.save(path)
            written.append(path)
        return written


def channel_list(channel_mask: int) -> list[RxChannel]:
    """Expand a mask into its individual Rx/ORx channels (stable order)."""
    return [ch for ch in RX_SINGLE if channel_mask & int(ch)]


def extract_channels(perform_rx_result, channels: list[RxChannel], bits: int) -> dict:
    """Pull per-channel int IQ out of a ``PerformRx`` result via the DLL ScaleRx path.

    HARDWARE SEAM: the precise readback container is confirmed during bring-up
    (docs/api_notes.md flags this). The expected shape is a list of int[] buffers
    (I and Q per channel). Until confirmed on hardware, this raises clearly rather
    than guessing silently.
    """
    raise NotImplementedError(
        "extract_channels: confirm the PerformRx readback container on hardware "
        "(see docs/api_notes.md). Expected: per-channel int[] I/Q from ScaleRx/ScaleRxSingle."
    )


def capture(
    radio,
    channel_mask: int,
    capture_time_ms: float,
    *,
    trig: RxTrigSource = RxTrigSource.IMMEDIATE,
    timeout_ms: int = 1000,
    bits: int,
) -> CaptureResult:
    """Trigger a snapshot and return per-channel captures.

    ``bits`` is the Rx datapath width (``ProfileInfo.rx_bits``).
    """
    chans = channel_list(channel_mask)
    raw = radio.perform_rx(channel_mask, capture_time_ms, trig=trig, timeout_ms=timeout_ms)
    per_channel = extract_channels(raw, chans, bits)
    result = CaptureResult(capture_time_ms=capture_time_ms, trig=trig)
    for ch in chans:
        i, q = per_channel[ch]
        result.channels[ch] = ChannelCapture(ch, np.asarray(i), np.asarray(q), bits)
    return result


def expected_samples(capture_time_ms: float, rx_rate_khz: float) -> int:
    """Convenience: how many samples a capture of this duration should yield."""
    return samples_for_duration(capture_time_ms, rx_rate_khz)
