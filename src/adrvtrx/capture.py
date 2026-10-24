"""Snapshot capture: trigger PerformRx, shape per-channel IQ, report clipping, save.

A capture is one FPGA snapshot taken in a single ``PerformRx`` call, so all channels
are mutually sample-aligned. The trigger is IMMEDIATE: ``TXn_SOF`` times out on this bench
(the TX datapath emits no start-of-frame), so a capture is aligned to its reference in
software (:mod:`adrvtrx.align`, :func:`aligned_capture`). Through the hardware server only
the arrays of the wanted channels travel (``perform_rx(..., keep=...)``).

Confirmed on hardware (docs/api_notes.md): ``PerformRx`` ignores its mask argument
and returns the full ``rxInitChannelMask`` set as a flat indexable of already-scaled
int arrays, interleaved ``[ch0_I, ch0_Q, ...]`` in ascending channel-bit order. We
therefore index a wanted channel by its absolute position in that set.
"""

from __future__ import annotations

import inspect
import math
import time
from dataclasses import dataclass, field
from functools import lru_cache, reduce
from pathlib import Path

import numpy as np

from ._enums import RX_SINGLE, TX_SOF_FOR, RxChannel, RxTrigSource, TxChannel, is_orx
from .align import align_with_fraction, estimate_and_align
from .gain import (
    EMPTY_RETAKES,
    ORX_CLIP_DBFS,
    ORX_GAIN_MAX,
    ORX_GAIN_MIN,
    ORX_START_MARGIN_IDX,
    AgcError,
    ClipReport,
    autolevel_orx,
    check_orx_window,
    clip_report,
    peak_window,
    verify_no_clip,
)
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


def orx_for_tx(tx: TxChannel, tx_to_orx) -> RxChannel | None:
    """The ORx input that observes ``tx``, per the tx_to_orx map (e.g. ``"TX1_ORX1"``)."""
    for entry in tx_to_orx:
        tx_part, _, orx_part = entry.partition("_")
        if tx_part == tx.name:
            return RxChannel[orx_part]
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


def channel_slots(order, wanted) -> dict:
    """Readback slot of each ``wanted`` channel: ``{channel: slot}`` (arrays ``2*slot``, ``+1``).

    Main Rx channels sit at their own bit position; an ORx input resolves to its physical ORx
    ADC slot (see :func:`extract_channels`). Raises ``ValueError`` for a channel the active
    ``rxInitChannelMask`` does not return.
    """
    slots: dict[RxChannel, int] = {}
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
        slots[ch] = idx
    return slots


def _int32(values) -> np.ndarray:
    """A numpy array (the hardware server's result) as is; a .NET array element by element."""
    if isinstance(values, np.ndarray):
        return np.asarray(values, dtype=np.int32)
    return np.fromiter(values, dtype=np.int32)


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
    for ch, idx in channel_slots(order, wanted).items():
        i = _int32(perform_rx_result[2 * idx])
        q = _int32(perform_rx_result[2 * idx + 1])
        out[ch] = ChannelCapture(ch, i, q, bits)
    return out


#: A capture is a whole number of this many samples at every rate of the profile (the FPGA
#: refuses a length whose bytes are not a multiple of 4: 0.01 ms at 491.52 MSPS is 4915.2 samples).
CAPTURE_WORD_SAMPLES = 4


def valid_capture_ms(capture_time_ms: float, rates_hz, word: int = CAPTURE_WORD_SAMPLES) -> float:
    """The shortest capture length >= ``capture_time_ms`` that is a whole multiple of ``word``
    samples at every rate in ``rates_hz`` (unchanged when it already is, or without rates)."""
    rates = [int(round(r)) for r in rates_hz or () if r and r > 0]
    if not rates:
        return float(capture_time_ms)
    g = reduce(math.gcd, rates)  # a length is valid at every rate iff it is k * word / g seconds
    quanta = float(capture_time_ms) * 1e-3 * g / word
    k = round(quanta) if abs(quanta - round(quanta)) < 1e-6 else math.ceil(quanta)
    return max(1, k) * word / g * 1e3


@lru_cache(maxsize=16)
def _rates_of(profile_path: str) -> tuple[int, ...]:
    from .profile import profile_rates_hz

    return tuple(profile_rates_hz(profile_path))


def profile_rates(radio) -> tuple[int, ...]:
    """The datapath rates of ``radio``'s profile (Hz); empty when its profile cannot be read."""
    try:
        return _rates_of(str(radio.config.profile_path))
    except Exception:  # noqa: BLE001 - no profile file on this machine: no rounding
        return ()


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
    ``bits`` is the Rx datapath width (``ProfileInfo.rx_bits``). The length is rounded up to one
    that is whole 4-sample words at every rate of the radio's profile (:func:`valid_capture_ms`).

    ``trig`` defaults to IMMEDIATE. ``trig="auto"`` would align to the start-of-frame
    of the TX tied to the requested ORx (capture ORx2 -> TX2_SOF) via the tx_to_orx
    map -- but on this bench TXn_SOF currently TIMES OUT (RxCaptureWait timeout): the
    continuous PerformTx path emits no TX start-of-frame for the FPGA to gate on.
    Keep IMMEDIATE (TX runs continuously, so the signal is always present) until SOF
    generation is sorted out.
    """
    capture_time_ms = valid_capture_ms(capture_time_ms, profile_rates(radio))
    rx_init = radio.config.channels.rx_init_mask
    order = returned_channel_order(rx_init)
    wanted = channel_list(channel_mask)
    if trig == "auto":
        trig = auto_sof_trigger(channel_mask, radio.config.tx_to_orx)
    # Enable the main-Rx framer (ORx rides it in link-sharing) plus any requested
    # ORx INPUT bits -- an ORx front-end reads zeros until its enable bit is set.
    # Absolute set (preserving TX) so a prior capture's ORx enable can't leak in.
    orx_bits = sum(int(ch) for ch in wanted if is_orx(ch))
    radio.set_rx_enable((rx_init & 0x0F) | orx_bits)
    # Through the hardware server only the wanted arrays travel (``keep``).
    keep = sorted(
        p for slot in channel_slots(order, wanted).values() for p in (2 * slot, 2 * slot + 1)
    )
    if _takes_keep(radio):
        raw = radio.perform_rx(
            rx_init, capture_time_ms, trig=trig, timeout_ms=timeout_ms, keep=keep
        )
    else:  # a radio stand-in without ``keep``
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


#: Between the retakes of an empty ORx capture at a fixed gain (:func:`orx_capture`).
EMPTY_RETAKE_S = 0.05


class EmptyCaptureError(AgcError):
    """An ORx capture with no data (all zeros or no samples) after its retakes at a fixed gain
    (:func:`orx_capture`); ``channels`` are the empty ones."""

    def __init__(self, message: str, channels=()):
        super().__init__(message)
        self.channels = tuple(channels)


def is_empty(cap: ChannelCapture) -> bool:
    """No data: no samples, or every I and Q sample zero."""
    return cap.i.size == 0 or not (np.any(cap.i) or np.any(cap.q))


def orx_capture(
    radio,
    channel_mask: int,
    capture_time_ms: float,
    *,
    bits: int,
    retakes: int = EMPTY_RETAKES,
    wait_s: float = EMPTY_RETAKE_S,
    trig: RxTrigSource | str = RxTrigSource.IMMEDIATE,
) -> CaptureResult:
    """:func:`capture` at a gain the caller fixed, whose ORx channels must carry data: an empty
    ORx (:func:`is_empty`) is captured again at the same gain up to ``retakes`` times, ``wait_s``
    apart; still empty raises :class:`EmptyCaptureError` (an :class:`~adrvtrx.gain.AgcError`).

    Bench, 2026-10-21 (UC102): an overdriven ORX2 returned a capture with every sample zero (peak
    -inf), and the next capture returned data again. The ORx AGC does not come here: it treats an
    empty capture as too hot and lowers the gain before it retakes (:func:`autolevel_orx`).
    """
    for attempt in range(int(retakes) + 1):
        res = capture(radio, channel_mask, capture_time_ms, trig=trig, bits=bits)
        empty = [ch for ch, cap in res.channels.items() if is_orx(ch) and is_empty(cap)]
        if not empty:
            return res
        if attempt < retakes:
            time.sleep(wait_s)
    names = "/".join(ch.name for ch in empty)
    raise EmptyCaptureError(
        f"{names} returned no data (all zeros) after {int(retakes)} retakes at the same (fixed) "
        "ORx gain: the ORx was likely overdriven -- reduce TX power / add a pad",
        empty,
    )


def orx_clip_for(radio) -> float:
    """The ORx clip threshold (dBFS) of the profile ``radio`` runs: its ``[orx_ceiling]`` minus
    the margin (``Config.orx_clip_dbfs``); :data:`~adrvtrx.gain.ORX_CLIP_DBFS` (the default
    ceiling's) for a radio with no config (a stand-in)."""
    clip_of = getattr(getattr(radio, "config", None), "orx_clip_dbfs", None)
    return float(clip_of()) if callable(clip_of) else ORX_CLIP_DBFS


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
    start_gain: int | None = None,
    gain_min: int = ORX_GAIN_MIN,
    gain_max: int = ORX_GAIN_MAX,
    clip_dbfs: float | None = None,
    start_margin_idx: int = ORX_START_MARGIN_IDX,
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

    ``start_gain`` (a predicted gain) skips Stage A and trims from ``start_margin_idx`` indices
    below it, up to the window (see :func:`autolevel_orx`). ``gain_min`` / ``gain_max`` bound
    the gain index (the floor is where Stage A starts and the lowest gain Stage C backs off to).

    ``clip_dbfs`` is the clip threshold (a peak at or above it is a clip, like a railed sample);
    ``None`` takes the profile's (:func:`orx_clip_for`). A window whose top reaches it is a
    ``ValueError`` before the radio is touched. An empty capture (all zeros: the ORx was
    overdriven) is too hot: the stages lower the gain and capture again; still empty after
    :data:`~adrvtrx.gain.EMPTY_RETAKES` retakes is FATAL (``AgcError``, the bench left safe).
    """
    clip = orx_clip_for(radio) if clip_dbfs is None else float(clip_dbfs)
    check_orx_window(target_dbfs, tol_up_db, clip)

    def set_gain(g: int) -> None:
        radio.set_rx_gain(channel, g)

    def measure(capture_ms: float) -> tuple[float, int]:
        # one capture: an empty one reads -inf, and the stages lower the gain before a retake
        cap = capture(radio, int(channel), capture_ms, bits=bits).channels[channel]
        rep = clip_report(cap.i, cap.q, bits)
        return rep.peak_dbfs, rep.railed_samples

    def coarse_measure() -> tuple[float, int]:
        return measure(coarse_ms)

    lr = autolevel_orx(
        set_gain,
        coarse_measure,
        target_dbfs=target_dbfs,
        tol_up_db=tol_up_db,
        tol_down_db=tol_down_db,
        gain_min=gain_min,
        gain_max=gain_max,
        max_iterations=max_iterations,
        start_gain=start_gain,
        clip_dbfs=clip,
        start_margin_idx=start_margin_idx,
    )
    if lr.fatal:
        radio.safe_state()
        radio.disconnect()
        raise AgcError(f"{channel.name}: {lr.reason}")

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
            return measure(verify_capture_ms)

        vr = verify_no_clip(
            set_gain,
            full_measure,
            lr.final_gain_index,
            target_dbfs=target_dbfs,
            tol_up_db=tol_up_db,
            gain_min=gain_min,
            clip_dbfs=clip,
        )
        if vr.fatal:
            radio.safe_state()
            radio.disconnect()
            raise AgcError(f"{channel.name}: {vr.reason}")
        result.final_gain_index = vr.final_gain_index
        result.final_peak_dbfs = vr.final_dbfs
        result.railed = vr.railed
        result.verify_steps = vr.iterations
        result.converged = lr.converged and vr.converged
        if vr.iterations > 0:  # Stage C backed the gain off the max
            result.at_max_gain = False
            result.reason = vr.reason

    return result


def reference_capture_ms(ref, fs: float, oversample: int = 2) -> float:
    """Capture length (ms) for ``oversample`` periods of the reference ``ref`` at ``fs``."""
    return oversample * len(ref) / float(fs) * 1e3


@dataclass
class AlignedCapture:
    """One capture of ``oversample`` reference periods, one period aligned to the reference.

    ``raw`` is the capture (``raw.iq`` complex codes is ``iq``), ``clip`` its clip report;
    ``x_aligned`` / ``y_aligned`` are the reference period and the captured period aligned to it,
    ``delay_samples`` the delay found.
    """

    raw: ChannelCapture
    iq: np.ndarray
    clip: ClipReport
    x_aligned: np.ndarray
    y_aligned: np.ndarray
    delay_samples: float


def aligned_capture(
    radio,
    channel: RxChannel,
    ref,
    *,
    bits: int,
    fs: float,
    oversample: int = 2,
    fraction: float | None = None,
    retakes: int | None = EMPTY_RETAKES,
) -> AlignedCapture:
    """Capture ``oversample`` periods of ``ref`` on ``channel``, report clipping, align a period.

    TX must already be playing ``ref`` (TX codes). The alignment is
    :func:`~adrvtrx.align.estimate_and_align`, or :func:`~adrvtrx.align.align_with_fraction`
    with ``fraction`` (the path's ``delay % 1``, known from an earlier full alignment). The
    capture-and-align step of the compression searches and of ``conditions.capture_point``.

    An empty ORx capture (all zeros) is retaken at the same gain up to ``retakes`` times
    (:func:`orx_capture`; still empty: :class:`EmptyCaptureError`). ``retakes=None`` (a search
    that sets the gain itself and lowers it first) returns it as it is: :func:`empty_aligned`.
    """
    ref = np.asarray(ref)
    ms = reference_capture_ms(ref, fs, oversample)
    if retakes is None:
        out = capture(radio, int(channel), ms, bits=bits)
    else:
        out = orx_capture(radio, int(channel), ms, bits=bits, retakes=retakes)
    return _aligned(out.channels[channel], ref, fraction, bits, fs)


def _aligned(raw: ChannelCapture, ref: np.ndarray, fraction, bits: int, fs: float):
    rep = clip_report(raw.i, raw.q, bits)
    iq = raw.iq
    if is_empty(raw):  # nothing to align: an empty capture (retakes=None)
        return empty_aligned(raw, rep, ref)
    if fraction is None:
        x_al, y_al, delay = estimate_and_align(ref, iq, fs)
    else:
        x_al, y_al, delay = align_with_fraction(ref, iq, fraction)
    return AlignedCapture(raw, iq, rep, x_al, y_al, float(delay))


def empty_aligned(raw: ChannelCapture, rep: ClipReport, ref: np.ndarray) -> AlignedCapture:
    """An :class:`AlignedCapture` of a capture with no data: peak -inf, the reference period
    against zeros, delay NaN (nothing was aligned)."""
    x = np.asarray(ref, dtype=complex)
    return AlignedCapture(raw, raw.iq, rep, x, np.zeros_like(x), float("nan"))


def aligned_captures(
    radio,
    channels,
    refs,
    *,
    bits: int,
    fs: float,
    oversample: int = 2,
    fractions=None,
    retakes: int | None = EMPTY_RETAKES,
) -> list[AlignedCapture]:
    """:func:`aligned_capture` of several channels from ONE snapshot (one ``PerformRx``).

    ``refs`` (one per channel, equal lengths: one ``PerformTx`` plays them) and ``fractions``
    (one per channel, ``None`` for a full alignment) as in :func:`aligned_capture`. The channels
    must sit on different readback slots (two ORx inputs on one ORx ADC cannot be read together).
    ``retakes`` as in :func:`aligned_capture` (``None``: an empty channel comes back as
    :func:`empty_aligned`, the others aligned).
    """
    refs = [np.asarray(r) for r in refs]
    if len({len(r) for r in refs}) != 1:
        raise ValueError(
            f"one snapshot needs equal reference lengths, got {[len(r) for r in refs]}"
        )
    fractions = list(fractions) if fractions is not None else [None] * len(channels)
    mask = 0
    for ch in channels:
        mask |= int(ch)
    ms = reference_capture_ms(refs[0], fs, oversample)
    if retakes is None:
        out = capture(radio, mask, ms, bits=bits)
    else:
        out = orx_capture(radio, mask, ms, bits=bits, retakes=retakes)
    return [
        _aligned(out.channels[ch], ref, frac, bits, fs)
        for ch, ref, frac in zip(channels, refs, fractions)
    ]


def shares_orx_adc(channels) -> bool:
    """Two of the ORx ``channels`` land on one ORx ADC (ORx1/ORx2 or ORx3/ORx4): one snapshot cannot
    read them both."""
    adcs = [_ORX_ADC_INDEX[ch] for ch in channels if ch in _ORX_ADC_INDEX]
    return len(adcs) != len(set(adcs))


def autolevel_for(
    radio,
    channel: RxChannel,
    ref,
    *,
    bits: int,
    fs: float,
    oversample: int = 2,
    target_dbfs: float = -1.0,
    tol_up_db: float = 0.3,
    tol_down_db: float = 0.6,
    start_gain: int | None = None,
    gain_min: int = ORX_GAIN_MIN,
    gain_max: int = ORX_GAIN_MAX,
    clip_dbfs: float | None = None,
    start_margin_idx: int = ORX_START_MARGIN_IDX,
) -> AgcResult:
    """:func:`autolevel_capture` for a capture of ``oversample`` periods of ``ref``: Stage C
    verifies the gain on exactly the capture the measurement takes. ``fs`` is the ORx rate;
    ``start_gain`` (a predicted gain: the trim starts ``start_margin_idx`` below it) and
    ``clip_dbfs`` (``None``: the profile's) as in :func:`autolevel_capture`."""
    return autolevel_capture(
        radio,
        channel,
        bits=bits,
        orx_rate_hz=fs,
        target_dbfs=target_dbfs,
        tol_up_db=tol_up_db,
        tol_down_db=tol_down_db,
        verify_capture_ms=reference_capture_ms(ref, fs, oversample),
        start_gain=start_gain,
        gain_min=gain_min,
        gain_max=gain_max,
        clip_dbfs=clip_dbfs,
        start_margin_idx=start_margin_idx,
    )


def capture_arrays(raw, keep=None) -> list[np.ndarray]:
    """A PerformRx result as plain ``int32`` numpy arrays ``[ch0_I, ch0_Q, ...]``.

    Same layout and indexing as the DLL result, so :func:`extract_channels`
    reads either. Used where the .NET object cannot travel (the hardware server).
    With ``keep`` (array positions), only those are converted; the others are
    empty arrays in their places.
    """
    n = _result_len(raw)
    items = raw if n is None else (raw[k] for k in range(n))
    if keep is None:
        return [np.fromiter(a, dtype=np.int32) for a in items]
    kept = {int(k) for k in keep}
    empty = np.zeros(0, dtype=np.int32)
    return [np.fromiter(a, dtype=np.int32) if k in kept else empty for k, a in enumerate(items)]


def _takes_keep(radio) -> bool:
    """Whether this radio's ``perform_rx`` accepts ``keep`` (Radio, RemoteRadio; not every
    stand-in). Asked of the method itself at every capture (a few microseconds), so a
    ``perform_rx`` replaced on one instance is judged by its own signature."""
    try:
        params = inspect.signature(radio.perform_rx).parameters
    except (TypeError, ValueError):
        return False
    return "keep" in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def _result_len(raw) -> int | None:
    """Length of a PerformRx result if knowable, else None (skip the check)."""
    try:
        return len(raw)
    except TypeError:
        return getattr(raw, "Count", None) or getattr(raw, "Length", None)


def expected_samples(capture_time_ms: float, rx_rate_khz: float) -> int:
    """Convenience: how many samples a capture of this duration should yield."""
    return samples_for_duration(capture_time_ms, rx_rate_khz)
