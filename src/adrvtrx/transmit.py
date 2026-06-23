"""Multi-band transmit: prepare per-channel waveforms and start PerformTx.

All TX channels named in one ``PerformTx`` call start together (deterministic
relative start across TX paths). For multi-band you pass one waveform per TX
channel; each rides its channel's LO (LO2 for tx12/tx34 by default).

``pack_channel`` is pure/tested. ``build_tx_data`` and the int packing format are
the hardware seam (docs/api_notes.md flags PerformTx int packing for bring-up).
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._enums import TX_SINGLE, TxChannel, TxTrigSource
from .waveform import prepare_tx

#: Default packing of a complex sample into one 32-bit int: I in the high 16 bits,
#: Q in the low 16 bits. Switch to "interleaved" if hardware bring-up shows the
#: FPGA expects [I0, Q0, I1, Q1, ...].
DEFAULT_PACK = "packed"


def pack_channel(i_int: np.ndarray, q_int: np.ndarray, mode: str = DEFAULT_PACK) -> np.ndarray:
    """Pack signed int I/Q into the int buffer layout PerformTx expects."""
    i = np.asarray(i_int, dtype=np.int64)
    q = np.asarray(q_int, dtype=np.int64)
    if mode == "packed":
        return ((i << 16) | (q & 0xFFFF)).astype(np.int32)
    if mode == "interleaved":
        out = np.empty(i.size + q.size, dtype=np.int32)
        out[0::2] = i.astype(np.int32)
        out[1::2] = q.astype(np.int32)
        return out
    raise ValueError(f"unknown pack mode {mode!r}")


def prepare_channel_buffer(
    iq: np.ndarray, bits: int, *, mode: str = DEFAULT_PACK, do_normalize: bool = True
) -> np.ndarray:
    """Full per-channel TX pipeline: normalize -> quantize(bits) -> pack."""
    i_int, q_int = prepare_tx(iq, bits, do_normalize=do_normalize)
    return pack_channel(i_int, q_int, mode)


def build_tx_data(bridge, channel_buffers: Mapping[TxChannel, np.ndarray]):
    """Build the ArrayList of per-channel int[] in TX channel order, return (data, mask)."""
    mask = 0
    items = []
    for ch in TX_SINGLE:
        if ch in channel_buffers:
            items.append(bridge.int_array(channel_buffers[ch].tolist()))
            mask |= int(ch)
    return bridge.array_list(items), mask


def transmit_bands(
    radio,
    channel_to_iq: Mapping[TxChannel, np.ndarray],
    bits: int,
    *,
    mode: str = DEFAULT_PACK,
    continuous: bool = True,
    trig: TxTrigSource = TxTrigSource.IMMEDIATE,
    do_normalize: bool = True,
) -> int:
    """Load one waveform per TX channel and start (looping) playback. Returns the mask."""
    buffers = {
        ch: prepare_channel_buffer(iq, bits, mode=mode, do_normalize=do_normalize)
        for ch, iq in channel_to_iq.items()
    }
    tx_data, mask = build_tx_data(radio.bridge, buffers)
    radio.perform_tx(tx_data, mask, trig=trig, continuous=continuous)
    return mask
