"""Snapshot capture: trigger PerformRx, shape per-channel IQ, report clipping, save.

A capture is one FPGA snapshot taken in a single ``PerformRx`` call, so all channels
are mutually sample-aligned. Use a ``TXn_SOF`` trigger to align to TX start-of-frame.

Confirmed on hardware (docs/api_notes.md): ``PerformRx`` ignores its mask argument
and returns the full ``rxInitChannelMask`` set as a flat indexable of already-scaled
int arrays, interleaved ``[ch0_I, ch0_Q, ...]`` in ascending channel-bit order. We
therefore index a wanted channel by its absolute position in that set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._enums import RX_SINGLE, RxChannel, RxTrigSource
from .gain import ClipReport, clip_report, peak_window
from .waveform import samples_for_duration, save_tab_iq_float

#: Single-bit channel for each rxInitChannelMask bit we can name (bits 8/9 are
#: internal/loopback Rx and map to None -- present in the readback but unnamed).
_RX_BIT_TO_CHANNEL = {int(ch): ch for ch in RX_SINGLE}


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


def returned_channel_order(rx_init_mask: int) -> list[RxChannel | None]:
    """Order of channels in a PerformRx readback: one entry per set bit of
    ``rxInitChannelMask`` (ascending). Named Rx/ORx channels map to their enum;
    internal/loopback bits (e.g. 0x100, 0x200) map to ``None``."""
    order: list[RxChannel | None] = []
    bit = 0
    while (1 << bit) <= rx_init_mask:
        m = 1 << bit
        if rx_init_mask & m:
            order.append(_RX_BIT_TO_CHANNEL.get(m))
        bit += 1
    return order


def extract_channels(perform_rx_result, order, wanted, bits: int) -> dict:
    """Extract ``wanted`` channels from a PerformRx result by absolute position.

    ``order`` is :func:`returned_channel_order` for the active ``rxInitChannelMask``;
    the result holds 2 arrays (I, Q) per entry in ``order``. Returns
    ``{channel: ChannelCapture}``.
    """
    out: dict[RxChannel, ChannelCapture] = {}
    for ch in wanted:
        if ch not in order:
            raise ValueError(
                f"{ch.name} is not in the active rxInitChannelMask "
                f"(returned channels: {[c.name for c in order if c]}). "
                f"Enable it in config [channels].rx_init_mask."
            )
        idx = order.index(ch)
        i = np.fromiter(perform_rx_result[2 * idx], dtype=np.int32)
        q = np.fromiter(perform_rx_result[2 * idx + 1], dtype=np.int32)
        out[ch] = ChannelCapture(ch, i, q, bits)
    return out


def capture(
    radio,
    channel_mask: int,
    capture_time_ms: float,
    *,
    trig: RxTrigSource = RxTrigSource.IMMEDIATE,
    timeout_ms: int = 1000,
    bits: int,
) -> CaptureResult:
    """Trigger a snapshot and return the requested channels.

    ``PerformRx`` returns the full ``rxInitChannelMask`` set regardless of mask, so
    we capture all of it and pick out ``channel_mask``'s channels by position.
    ``bits`` is the Rx datapath width (``ProfileInfo.rx_bits``).
    """
    rx_init = radio.config.channels.rx_init_mask
    order = returned_channel_order(rx_init)
    radio.enable_rx(rx_init & 0x0F)  # enable main-Rx framer; ORx rides it (link-sharing)
    raw = radio.perform_rx(rx_init, capture_time_ms, trig=trig, timeout_ms=timeout_ms)

    # Self-diagnose profile/mask mismatch: the readback must hold 2 arrays (I,Q)
    # per channel in rxInitChannelMask. If a profile returns a different set, the
    # positional mapping would be wrong -> fail clearly instead.
    n_arrays = _result_len(raw)
    if n_arrays is not None and n_arrays != 2 * len(order):
        raise RuntimeError(
            f"PerformRx returned {n_arrays} arrays ({n_arrays // 2} channels) but "
            f"rxInitChannelMask=0x{rx_init:X} implies {len(order)} channels. The "
            f"profile and rx_init_mask disagree -- set [channels].rx_init_mask to "
            f"match this profile's framer routing (run scripts/hw_smoke.py to see "
            f"the actual count)."
        )

    wanted = channel_list(channel_mask)
    result = CaptureResult(capture_time_ms=capture_time_ms, trig=trig)
    result.channels.update(extract_channels(raw, order, wanted, bits))
    return result


def _result_len(raw) -> int | None:
    """Length of a PerformRx result if knowable, else None (skip the check)."""
    try:
        return len(raw)
    except TypeError:
        return getattr(raw, "Count", None) or getattr(raw, "Length", None)


def expected_samples(capture_time_ms: float, rx_rate_khz: float) -> int:
    """Convenience: how many samples a capture of this duration should yield."""
    return samples_for_duration(capture_time_ms, rx_rate_khz)
