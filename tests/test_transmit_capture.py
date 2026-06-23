from __future__ import annotations

import numpy as np
import pytest

from adrvtrx._enums import RxChannel, TxChannel
from adrvtrx.capture import channel_list, extract_channels, returned_channel_order
from adrvtrx.transmit import build_tx_data, prepare_channel_iq


def test_channel_list_expands_mask():
    mask = int(RxChannel.RX1) | int(RxChannel.ORX1)
    assert channel_list(mask) == [RxChannel.RX1, RxChannel.ORX1]


# --- TX: PerformTx needs exactly 8 arrays (4 channels x I/Q) ------------------


def test_prepare_channel_iq_normalizes():
    iq = np.array([0.5 + 0j, -0.5 + 0j])
    i_int, q_int = prepare_channel_iq(iq, 16, do_normalize=True)
    assert i_int[0] == 32767


def test_prepare_channel_iq_keeps_integer_vectors():
    iq = np.array([1000 + 0j, -2000 + 0j])
    i_int, _ = prepare_channel_iq(iq, 16, do_normalize=False)
    assert i_int[0] == 1000
    assert i_int[1] == -2000


def test_build_tx_data_emits_eight_arrays_zero_filled(fake_bridge):
    bufs = {TxChannel.TX2: (np.array([2, 2]), np.array([20, 20]))}
    data, mask = build_tx_data(fake_bridge, bufs)
    assert len(data) == 8  # one I + one Q for each of TX1..TX4
    assert mask == int(TxChannel.TX2)
    assert list(data[0]) == [0, 0]  # TX1 I zero-filled
    assert list(data[2]) == [2, 2]  # TX2 I
    assert list(data[3]) == [20, 20]  # TX2 Q
    assert list(data[4]) == [0, 0]  # TX3 I zero-filled


def test_build_tx_data_two_channels(fake_bridge):
    bufs = {
        TxChannel.TX2: (np.array([2, 2]), np.array([2, 2])),
        TxChannel.TX3: (np.array([3, 3]), np.array([3, 3])),
    }
    data, mask = build_tx_data(fake_bridge, bufs)
    assert len(data) == 8
    assert mask == (int(TxChannel.TX2) | int(TxChannel.TX3))
    assert list(data[2]) == [2, 2] and list(data[4]) == [3, 3]


def test_build_tx_data_length_mismatch_raises(fake_bridge):
    bufs = {
        TxChannel.TX2: (np.array([1, 2]), np.array([1, 2])),
        TxChannel.TX3: (np.array([1, 2, 3]), np.array([1, 2, 3])),
    }
    with pytest.raises(ValueError):
        build_tx_data(fake_bridge, bufs)


# --- capture: PerformRx returns the full rxInitChannelMask set ----------------


def test_returned_channel_order_full_mask():
    order = returned_channel_order(0x3FF)
    assert len(order) == 10
    assert order[:8] == [
        RxChannel.RX1,
        RxChannel.RX2,
        RxChannel.RX3,
        RxChannel.RX4,
        RxChannel.ORX1,
        RxChannel.ORX2,
        RxChannel.ORX3,
        RxChannel.ORX4,
    ]
    assert order[8] is None and order[9] is None  # internal/loopback bits


def test_extract_orx_by_adc_slot():
    order = returned_channel_order(0x3FF)
    # 20 arrays; pair k encodes its own index so we can check positions.
    raw = [np.array([k]) for k in range(20)]
    per = extract_channels(raw, order, [RxChannel.ORX2, RxChannel.ORX3], 16)
    # Only 2 physical ORx ADCs: ORx1/ORx2 -> slot 0 (order idx 4 -> arrays [8],[9]),
    # ORx3/ORx4 -> slot 1 (order idx 5 -> arrays [10],[11]). Bench- + ADI-confirmed.
    assert per[RxChannel.ORX2].i[0] == 8 and per[RxChannel.ORX2].q[0] == 9
    assert per[RxChannel.ORX3].i[0] == 10 and per[RxChannel.ORX3].q[0] == 11


def test_extract_orx1_and_orx2_share_adc0():
    order = returned_channel_order(0x3FF)
    raw = [np.array([k]) for k in range(20)]
    per = extract_channels(raw, order, [RxChannel.ORX1, RxChannel.ORX2], 16)
    # Both front-ends of ADC0 resolve to the same slot 0 (arrays [8],[9]).
    assert per[RxChannel.ORX1].i[0] == 8
    assert per[RxChannel.ORX2].i[0] == 8


def test_extract_channel_not_in_mask_raises():
    order = returned_channel_order(0x0F)  # Rx1-4 only
    raw = [np.array([k]) for k in range(8)]
    with pytest.raises(ValueError):
        extract_channels(raw, order, [RxChannel.ORX2], 16)
