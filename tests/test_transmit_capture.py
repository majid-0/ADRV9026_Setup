from __future__ import annotations

import numpy as np
import pytest

from adrvtrx._enums import RxChannel, TxChannel
from adrvtrx.capture import channel_list, extract_channels
from adrvtrx.transmit import build_tx_data, prepare_channel_iq


def test_channel_list_expands_mask():
    mask = int(RxChannel.RX1) | int(RxChannel.ORX1)
    assert channel_list(mask) == [RxChannel.RX1, RxChannel.ORX1]


def test_channel_list_all_rx_and_orx():
    chans = channel_list(0xFF)
    assert len(chans) == 8
    assert RxChannel.ORX4 in chans


def test_prepare_channel_iq_normalizes():
    iq = np.array([0.5 + 0j, -0.5 + 0j])
    i_int, q_int = prepare_channel_iq(iq, 16, do_normalize=True)
    assert i_int[0] == 32767  # peak 0.5 normalized to full scale


def test_prepare_channel_iq_keeps_integer_vectors():
    # Already-scaled vector, do_normalize=False -> values pass through (rounded).
    iq = np.array([1000 + 0j, -2000 + 0j])
    i_int, _ = prepare_channel_iq(iq, 16, do_normalize=False)
    assert i_int[0] == 1000
    assert i_int[1] == -2000


def test_build_tx_data_separate_iq_arrays_in_order(fake_bridge):
    bufs = {
        TxChannel.TX3: (np.array([3, 3]), np.array([30, 30])),
        TxChannel.TX1: (np.array([1, 1]), np.array([10, 10])),
    }
    data, mask = build_tx_data(fake_bridge, bufs)
    assert mask == (int(TxChannel.TX1) | int(TxChannel.TX3))
    # ascending channel order, each channel = I array then Q array
    assert list(data[0]) == [1, 1]  # TX1 I
    assert list(data[1]) == [10, 10]  # TX1 Q
    assert list(data[2]) == [3, 3]  # TX3 I
    assert list(data[3]) == [30, 30]  # TX3 Q


def test_extract_channels_interleaved_readback():
    # PerformRx returns [ch0_I, ch0_Q, ch1_I, ch1_Q, ...] in ascending order.
    raw = [
        np.array([1, 2, 3]),
        np.array([4, 5, 6]),
        np.array([7, 8, 9]),
        np.array([10, 11, 12]),
    ]
    chans = [RxChannel.RX1, RxChannel.ORX1]
    per = extract_channels(raw, chans, 16)
    assert list(per[RxChannel.RX1][0]) == [1, 2, 3]
    assert list(per[RxChannel.RX1][1]) == [4, 5, 6]
    assert list(per[RxChannel.ORX1][0]) == [7, 8, 9]
    assert list(per[RxChannel.ORX1][1]) == [10, 11, 12]


def test_extract_channels_too_few_buffers_raises():
    with pytest.raises(IndexError):
        extract_channels([np.array([1])], [RxChannel.RX1], 16)  # need I and Q
