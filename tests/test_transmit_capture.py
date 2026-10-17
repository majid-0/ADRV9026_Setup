from __future__ import annotations

import numpy as np
import pytest

from adrvtrx._enums import RxChannel, TxChannel
from adrvtrx.capture import (
    capture_arrays,
    channel_list,
    extract_channels,
    returned_channel_order,
)
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


class _DotNetLike:
    """Indexable with ``Count`` but no ``len()``, like a pythonnet collection."""

    def __init__(self, arrays):
        self._arrays = arrays
        self.Count = len(arrays)

    def __getitem__(self, k):
        return self._arrays[k]

    def __len__(self):
        raise TypeError("no len")


def test_capture_arrays_converts_only_the_kept_positions():
    raw = _DotNetLike([[1, 2], [3, 4], [5, 6], [7, 8]])
    arrays = capture_arrays(raw, keep=[2, 3])
    assert [len(a) for a in arrays] == [0, 0, 2, 2] and all(a.dtype == np.int32 for a in arrays)
    np.testing.assert_array_equal(arrays[3], [7, 8])


def test_extract_channels_takes_numpy_arrays_as_they_are():
    order = returned_channel_order(0x3FF)
    raw = [np.full(4, k, dtype=np.int32) for k in range(20)]
    out = extract_channels(raw, order, [RxChannel.ORX1], 12)[RxChannel.ORX1]
    assert out.i is raw[8] and out.q is raw[9]  # no element-by-element copy
    listed = extract_channels([list(a) for a in raw], order, [RxChannel.ORX1], 12)
    np.testing.assert_array_equal(listed[RxChannel.ORX1].i, raw[8])  # a .NET-like sequence


def test_build_tx_data_hands_a_numpy_bridge_the_arrays_and_others_lists(fake_bridge):
    from adrvtrx.client import NumpyBridge

    iq = prepare_channel_iq(0.5 * np.exp(2j * np.pi * 0.01 * np.arange(64)), 12)
    data, mask = build_tx_data(NumpyBridge(), {TxChannel.TX2: iq})
    assert mask == int(TxChannel.TX2) and len(data) == 8
    assert all(isinstance(a, np.ndarray) and a.dtype == np.int32 for a in data)
    np.testing.assert_array_equal(data[2], iq[0])
    listed, _mask = build_tx_data(fake_bridge, {TxChannel.TX2: iq})
    assert all(type(v) is int for v in listed[2]) and listed[2] == iq[0].tolist()


def test_capture_arrays_gives_int32_numpy_in_the_same_layout():
    raw = _DotNetLike([[1, 2, 3], [4, 5, 6], [-7, 8, 9], [0, 0, 1]])
    arrays = capture_arrays(raw)
    assert len(arrays) == 4
    assert all(a.dtype == np.int32 for a in arrays)
    np.testing.assert_array_equal(arrays[2], [-7, 8, 9])
    order = [RxChannel.RX1, RxChannel.ORX1]
    out = extract_channels(arrays, order, [RxChannel.ORX1], 12)
    np.testing.assert_array_equal(out[RxChannel.ORX1].i, [-7, 8, 9])


def test_a_perform_rx_replaced_on_one_radio_is_called_as_it_is_written():
    """``keep`` goes only to a ``perform_rx`` that takes it, judged on the instance (a monkeypatched
    method included), not on the radio's class."""
    from pathlib import Path

    from adrvtrx.capture import capture
    from adrvtrx.config import Config, DllConfig
    from adrvtrx.fake import FakeRadio
    from sim_bench import SimRadio

    radio = FakeRadio(Config(dll=DllConfig(install_dir=Path("C:/nonexistent"))), seed=1)
    radio._safe_hooks_installed = True
    radio.connect()
    radio.program()
    real = radio.perform_rx
    seen = []

    def plain(channel_mask, capture_time_ms, *, trig=None, timeout_ms=1000):  # no keep
        seen.append("plain")
        return real(channel_mask, capture_time_ms, trig=trig, timeout_ms=timeout_ms)

    capture(radio, int(RxChannel.ORX1), 0.01, bits=12)  # the class's own: keep accepted
    radio.perform_rx = plain
    out = capture(radio, int(RxChannel.ORX1), 0.01, bits=12).channels[RxChannel.ORX1]
    assert seen == ["plain"] and len(out.i) == 100

    sim = SimRadio()
    sim_real = sim.perform_rx
    sim.perform_rx = lambda *a, **k: (seen.append(sorted(k)), sim_real(*a, **k))[1]
    capture(sim, int(RxChannel.ORX1), 0.01, bits=12)  # a wrapper passes keep on to the stand-in
    assert seen[-1] == ["keep", "timeout_ms", "trig"]
