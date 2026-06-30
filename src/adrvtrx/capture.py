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

from ._enums import RX_SINGLE, TX_SOF_FOR, RxChannel, RxTrigSource, TxChannel, is_orx
from .gain import AgcError, ClipReport, autolevel_orx, clip_report, peak_window, verify_no_clip
from .waveform import samples_for_duration, save_tab_iq_float

#: Single-bit channel for each rxInitChannelMask bit we can name (bits 8/9 are
#: internal/loopback Rx and map to None -- present in the readback but unnamed).
_RX_BIT_TO_CHANNEL = {int(ch): ch for ch in RX_SINGLE}

#: Which physical ORx ADC each ORx front-end input muxes into. The ADRV9026 has
#: only 2 ORx ADCs: ORx1/ORx2 share ADC0, ORx3/ORx4 share ADC1. Confirmed on the
#: bench (TX2->ORx2 lands on ADC0; enabling ORx1/2 zeros the ADC1 slot) and by
#: ADI's rxDataCapture sample, whose readback slots are ...,ORx1,ORx3 (the 2 ADCs).
_ORX_ADC_INDEX = {
    RxChannel.ORX1: 0,
    RxChannel.ORX2: 0,
    RxChannel.ORX3: 1,
    RxChannel.ORX4: 1,
}


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


def orx_slot_positions(order) -> list[int]:
    """Readback positions of the physical ORx ADC slots, in ascending order.

    Only TWO ORx ADCs exist; they occupy the first two ORx positions of the
    readback (ADI's sample names them ORx1 and ORx3). Any further ORx bit
    positions are empty placeholders with no converter.
    """
    return [i for i, ch in enumerate(order) if ch is not None and is_orx(ch)]


def orx_slot_for(channel: RxChannel, order) -> int | None:
    """Readback position of the ORx ADC slot a given ORx input lands on.

    ORx1/ORx2 -> ADC0 (first ORx slot), ORx3/ORx4 -> ADC1 (second). Returns
    ``None`` if that ADC slot isn't present in this readback's order.
    """
    slots = orx_slot_positions(order)
    adc = _ORX_ADC_INDEX[channel]
    return slots[adc] if adc < len(slots) else None


def tx_for_orx(orx: RxChannel, tx_to_orx) -> TxChannel | None:
    """The TX channel an ORx input observes, per the tx_to_orx map (e.g. ``"TX2_ORX2"``)."""
    for entry in tx_to_orx:
        tx_part, _, orx_part = entry.partition("_")
        if orx_part == orx.name:
            return TxChannel[tx_part]
    return None


def auto_sof_trigger(channel_mask, tx_to_orx, default=RxTrigSource.IMMEDIATE) -> RxTrigSource:
    """SOF trigger of the lowest TX tied to any requested ORx; ``default`` if none.

    Lets a capture align to its source TX's start-of-frame automatically (capture
    ORx2 -> TX2_SOF). Falls back to ``default`` (IMMEDIATE) for non-ORx captures or
    when no ORx in the mask maps to a TX.
    """
    txs = [tx_for_orx(ch, tx_to_orx) for ch in channel_list(channel_mask) if is_orx(ch)]
    txs = sorted((t for t in txs if t is not None), key=int)
    return TX_SOF_FOR[txs[0]] if txs else default


def extract_channels(perform_rx_result, order, wanted, bits: int) -> dict:
    """Extract ``wanted`` channels from a PerformRx result by absolute position.

    ``order`` is :func:`returned_channel_order` for the active ``rxInitChannelMask``;
    the result holds 2 arrays (I, Q) per entry in ``order``. Main Rx channels sit
    at their own bit position. ORx is different: the 4 ORx inputs mux into just 2
    ADCs, so an ORx request resolves to one of the two physical ORx ADC slots
    (ORx1/ORx2 -> slot 0, ORx3/ORx4 -> slot 1) -- confirmed on the bench and
    against ADI's rxDataCapture sample. Returns ``{channel: ChannelCapture}``.
    """
    out: dict[RxChannel, ChannelCapture] = {}
    for ch in wanted:
        if is_orx(ch):
            idx = orx_slot_for(ch, order)
            if idx is None:
                raise ValueError(
                    f"{ch.name} maps to ORx ADC{_ORX_ADC_INDEX[ch]}, but the readback "
                    f"exposes {len(orx_slot_positions(order))} ORx slot(s) for this "
                    f"rxInitChannelMask. Enable ORx in config [channels].rx_init_mask."
                )
        elif ch in order:
            idx = order.index(ch)
        else:
            raise ValueError(
                f"{ch.name} is not in the active rxInitChannelMask "
                f"(returned channels: {[c.name for c in order if c]}). "
                f"Enable it in config [channels].rx_init_mask."
            )
        i = np.fromiter(perform_rx_result[2 * idx], dtype=np.int32)
        q = np.fromiter(perform_rx_result[2 * idx + 1], dtype=np.int32)
        out[ch] = ChannelCapture(ch, i, q, bits)
    return out


def capture(
    radio,
    channel_mask: int,
    capture_time_ms: float,
    *,
    trig: RxTrigSource | str = RxTrigSource.IMMEDIATE,
    timeout_ms: int = 1000,
    bits: int,
) -> CaptureResult:
    """Trigger a snapshot and return the requested channels.

    ``PerformRx`` returns the full ``rxInitChannelMask`` set regardless of mask, so
    we capture all of it and pick out ``channel_mask``'s channels by position.
    ``bits`` is the Rx datapath width (``ProfileInfo.rx_bits``).

    ``trig`` defaults to IMMEDIATE. ``trig="auto"`` would align to the start-of-frame
    of the TX tied to the requested ORx (capture ORx2 -> TX2_SOF) via the tx_to_orx
    map -- but on this bench TXn_SOF currently TIMES OUT (RxCaptureWait timeout): the
    continuous PerformTx path emits no TX start-of-frame for the FPGA to gate on.
    Keep IMMEDIATE (TX runs continuously, so the signal is always present) until SOF
    generation is sorted out.
    """
    rx_init = radio.config.channels.rx_init_mask
    order = returned_channel_order(rx_init)
    wanted = channel_list(channel_mask)
    if trig == "auto":
        trig = auto_sof_trigger(channel_mask, radio.config.tx_to_orx)
    # Enable the main-Rx framer (ORx rides it in link-sharing) plus any requested
    # ORx INPUT bits -- an ORx front-end reads zeros until its enable bit is set.
    # Absolute set (preserving TX) so a prior capture's ORx enable can't leak in.
    orx_bits = sum(int(ch) for ch in wanted if is_orx(ch))
    radio.rx_tx_enable((rx_init & 0x0F) | orx_bits, radio._en_tx)
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

    result = CaptureResult(capture_time_ms=capture_time_ms, trig=trig)
    result.channels.update(extract_channels(raw, order, wanted, bits))
    return result


def measure_delay(
    radio,
    channel: RxChannel,
    reference,
    *,
    bits: int,
    fs: float,
    oversample: int = 2,
    trig: RxTrigSource | str = RxTrigSource.IMMEDIATE,
    timeout_ms: int = 2000,
):
    """Measure the TX->ORx delay of ``channel`` against a transmitted ``reference``.

    Captures ``oversample`` x ``len(reference)`` samples (so the reference can be
    located inside the window regardless of where the IMMEDIATE capture started),
    then returns ``(delay_samples, delay_ns, corr)`` using :mod:`adrvtrx.align`.

    ``corr`` is the normalized complex correlation at the aligned position: ~1.0
    means the capture is a faithful copy of the reference; ``corr`` well below ~0.9
    means it is NOT (wrong channel, TX not transmitting, or bad level). This only
    captures -- the reference's TX must already be running. ``fs`` is the ORx rate
    (``ProfileInfo.orx_rate_hz``).
    """
    from .align import estimate_delay, match_corr

    ref = np.asarray(reference)
    capture_time_ms = oversample * len(ref) / float(fs) * 1e3
    res = capture(radio, int(channel), capture_time_ms, trig=trig, timeout_ms=timeout_ms, bits=bits)
    cap = res.channels[channel]
    y = cap.i.astype(np.float64) + 1j * cap.q.astype(np.float64)

    delay = estimate_delay(ref, y, fs)
    corr = match_corr(ref, y, fs)  # in-band: ~1.0 for a clean band even in dual-band
    return delay, delay / float(fs) * 1e9, corr


@dataclass
class AgcResult:
    """Outcome of :func:`autolevel_capture` (the full A/B/C software ORx AGC)."""

    final_gain_index: int
    converged: bool
    at_max_gain: bool
    reason: str
    coarse_peak_dbfs: float  # peak after Stages A+B (short captures)
    final_peak_dbfs: float  # peak after Stage C verify (full signal), or == coarse
    railed: int  # samples at the ADC rail at the final gain (0 when accepted)
    fine_iters: int  # Stage B trim iterations
    verify_steps: int  # Stage C back-off steps on the full signal


def autolevel_capture(
    radio,
    channel: RxChannel,
    *,
    bits: int,
    orx_rate_hz: float,
    target_dbfs: float = -1.0,
    tol_up_db: float = 0.3,
    tol_down_db: float = 0.6,
    coarse_ms: float = 0.1,
    verify_capture_ms: float | None = None,
    max_iterations: int = 16,
) -> AgcResult:
    """End-to-end software ORx AGC against live hardware (TX must already be running).

    Wraps the pure stages in :mod:`adrvtrx.gain`:

    * Builds ``set_gain`` and a coarse ``measure()`` (a ``coarse_ms`` capture ->
      ``clip_report`` -> ``(peak_dbfs, railed)``) and runs :func:`autolevel_orx`
      (Stages A+B) to settle the gain index.
    * If ``verify_capture_ms`` is given, builds a full-duration ``measure_full()``
      and runs :func:`verify_no_clip` (Stage C) to back off on full-signal clipping.

    On any FATAL stage result it leaves the bench safe -- ``radio.safe_state()`` +
    ``radio.disconnect()`` -- and raises :class:`~adrvtrx.gain.AgcError`. ORx
    ``RxGainGet`` is unusable, so the gain index is tracked entirely in software here.

    ``orx_rate_hz`` is the ORx datapath rate (``ProfileInfo.orx_rate_hz``); the caller
    typically derives ``verify_capture_ms`` from it and the waveform length.
    """

    def set_gain(g: int) -> None:
        radio.set_rx_gain(channel, g)

    def coarse_measure() -> tuple[float, int]:
        cap = capture(radio, int(channel), coarse_ms, bits=bits).channels[channel]
        rep = clip_report(cap.i, cap.q, bits)
        return rep.peak_dbfs, rep.railed_samples

    lr = autolevel_orx(
        set_gain,
        coarse_measure,
        target_dbfs=target_dbfs,
        tol_up_db=tol_up_db,
        tol_down_db=tol_down_db,
        max_iterations=max_iterations,
    )
    if lr.fatal:
        radio.safe_state()
        radio.disconnect()
        raise AgcError(lr.reason)

    result = AgcResult(
        final_gain_index=lr.final_gain_index,
        converged=lr.converged,
        at_max_gain=lr.at_max_gain,
        reason=lr.reason,
        coarse_peak_dbfs=lr.final_dbfs,
        final_peak_dbfs=lr.final_dbfs,
        railed=lr.railed,
        fine_iters=lr.iterations,
        verify_steps=0,
    )

    if verify_capture_ms is not None:

        def full_measure() -> tuple[float, int]:
            cap = capture(radio, int(channel), verify_capture_ms, bits=bits).channels[channel]
            rep = clip_report(cap.i, cap.q, bits)
            return rep.peak_dbfs, rep.railed_samples

        vr = verify_no_clip(
            set_gain,
            full_measure,
            lr.final_gain_index,
            target_dbfs=target_dbfs,
            tol_up_db=tol_up_db,
        )
        if vr.fatal:
            radio.safe_state()
            radio.disconnect()
            raise AgcError(vr.reason)
        result.final_gain_index = vr.final_gain_index
        result.final_peak_dbfs = vr.final_dbfs
        result.railed = vr.railed
        result.verify_steps = vr.iterations
        result.converged = lr.converged and vr.converged
        if vr.iterations > 0:  # Stage C backed the gain off the max
            result.at_max_gain = False
            result.reason = vr.reason

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
