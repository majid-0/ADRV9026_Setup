"""Multi-band transmit: prepare per-channel waveforms and start PerformTx.

Confirmed against an ADI LoadTxData sample (docs/api_notes.md): ``PerformTx`` takes
an ``ArrayList`` of **separate I and Q int arrays per channel**, in ascending TX
channel order -- ``[Tx1_I, Tx1_Q, Tx2_I, Tx2_Q, ...]`` -- NOT a packed/interleaved
single array. All channels named in one call start together.

``prepare_channel_iq`` is pure/tested; ``build_tx_data`` marshals to .NET arrays.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._enums import TX_SINGLE, TxChannel, TxTrigSource
from .waveform import prepare_tx


def prepare_channel_iq(
    iq: np.ndarray, bits: int, *, do_normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel TX pipeline: (optionally normalize) then quantize to ``bits``.

    Returns ``(i_int, q_int)`` int arrays -- the format PerformTx wants per channel.
    Pass ``do_normalize=False`` to keep waveforms that are already integer-scaled
    (e.g. ADI test vectors with a fixed PeakScaling).
    """
    return prepare_tx(iq, bits, do_normalize=do_normalize)


def build_tx_data(bridge, channel_iq: Mapping[TxChannel, tuple[np.ndarray, np.ndarray]]):
    """Build the PerformTx ArrayList (I, Q arrays per channel) + channel mask.

    Channels are emitted in ascending order; each contributes its I array then its
    Q array, matching the proven ADI ``LoadTxData`` ordering.
    """
    mask = 0
    items = []
    for ch in TX_SINGLE:
        if ch in channel_iq:
            i_int, q_int = channel_iq[ch]
            items.append(bridge.int_array(np.asarray(i_int).tolist()))
            items.append(bridge.int_array(np.asarray(q_int).tolist()))
            mask |= int(ch)
    return bridge.array_list(items), mask


def transmit_bands(
    radio,
    channel_to_iq: Mapping[TxChannel, np.ndarray],
    bits: int,
    *,
    continuous: bool = True,
    trig: TxTrigSource = TxTrigSource.IMMEDIATE,
    do_normalize: bool = True,
) -> int:
    """Load one waveform per TX channel and start (looping) playback. Returns the mask."""
    channel_iq = {
        ch: prepare_channel_iq(iq, bits, do_normalize=do_normalize)
        for ch, iq in channel_to_iq.items()
    }
    tx_data, mask = build_tx_data(radio.bridge, channel_iq)
    radio.perform_tx(tx_data, mask, trig=trig, continuous=continuous)
    return mask
