from __future__ import annotations

import numpy as np
import pytest

from adrvtrx._enums import RxChannel, TxChannel
from adrvtrx.capture import channel_list
from adrvtrx.transmit import build_tx_data, pack_channel, prepare_channel_buffer


def test_channel_list_expands_mask():
    mask = int(RxChannel.RX1) | int(RxChannel.ORX1)
    assert channel_list(mask) == [RxChannel.RX1, RxChannel.ORX1]


def test_channel_list_all_rx_and_orx():
    chans = channel_list(0xFF)
    assert len(chans) == 8
    assert RxChannel.ORX4 in chans


def test_pack_channel_packed():
    i = np.array([1, -1], dtype=np.int32)
    q = np.array([2, -2], dtype=np.int32)
    packed = pack_channel(i, q, "packed")
    # high 16 bits = I, low 16 bits = Q
    assert (packed[0] >> 16) == 1
    assert (packed[0] & 0xFFFF) == 2


def test_pack_channel_interleaved():
    i = np.array([1, 3], dtype=np.int32)
    q = np.array([2, 4], dtype=np.int32)
    out = pack_channel(i, q, "interleaved")
    assert list(out) == [1, 2, 3, 4]


def test_pack_channel_bad_mode():
    with pytest.raises(ValueError):
        pack_channel(np.array([1]), np.array([1]), "nope")


def test_prepare_channel_buffer_normalizes():
    iq = np.array([0.5 + 0j, -0.5 + 0j])
    buf = prepare_channel_buffer(iq, 16, mode="interleaved")
    # normalized peak -> full scale on I
    assert buf[0] == 32767


def test_build_tx_data_orders_by_channel(fake_bridge):
    bufs = {
        TxChannel.TX3: np.array([3, 3], dtype=np.int32),
        TxChannel.TX1: np.array([1, 1], dtype=np.int32),
    }
    data, mask = build_tx_data(fake_bridge, bufs)
    assert mask == (int(TxChannel.TX1) | int(TxChannel.TX3))
    # TX1 must come before TX3 in the data list (stable channel order)
    assert list(data[0]) == [1, 1]
    assert list(data[1]) == [3, 3]
