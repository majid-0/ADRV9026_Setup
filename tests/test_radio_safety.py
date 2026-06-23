"""Crash-safety + low-level wrapper tests using the fake .NET bridge."""

from __future__ import annotations

from adrvtrx._enums import RxChannel, TxChannel
from adrvtrx.radio import MAX_TX_ATTEN_DB


def _last_atten_item(device):
    """Pull the TxAtten struct from the most recent TxAttenSet call."""
    arr, _n = device.Tx.TxAttenSet.call_args.args
    return arr[0]


def test_safe_state_sets_max_attenuation(fake_radio):
    fake_radio.safe_state()
    item = _last_atten_item(fake_radio.device)
    assert item.txChannelMask == int(TxChannel.ALL)
    assert item.txAttenuation_mdB == round(MAX_TX_ATTEN_DB * 1000)
    assert fake_radio._tx_live is False


def test_force_safe_sets_max_attenuation(fake_radio):
    fake_radio.force_safe()
    item = _last_atten_item(fake_radio.device)
    assert item.txAttenuation_mdB == round(MAX_TX_ATTEN_DB * 1000)


def test_set_tx_atten_converts_db_to_mdb(fake_radio):
    fake_radio.set_tx_atten(TxChannel.TX2, 30.0)
    item = _last_atten_item(fake_radio.device)
    assert item.txAttenuation_mdB == 30000
    assert item.txChannelMask == int(TxChannel.TX2)


def test_set_rx_gain_sets_index_and_mask(fake_radio):
    fake_radio.set_rx_gain(RxChannel.ORX1, 195)
    arr, _n = fake_radio.device.Rx.RxGainSet.call_args.args
    assert arr[0].gainIndex == 195
    assert arr[0].rxChannelMask == int(RxChannel.ORX1)


def test_exit_forces_safe_and_disconnects(fake_radio):
    # Simulate context-manager exit; must safe TX and disconnect even on error path.
    fake_radio.__exit__(None, None, None)
    assert fake_radio.device.Tx.TxAttenSet.called
    assert fake_radio.board.Client.Disconnect.called
    assert fake_radio._connected is False


def test_perform_tx_marks_live_then_safe_clears(fake_radio):
    fake_radio.perform_tx([[1, 2, 3]], int(TxChannel.TX1), continuous=True)
    assert fake_radio._tx_live is True
    assert fake_radio._en_tx == int(TxChannel.TX1)
    fake_radio.safe_state()
    assert fake_radio._tx_live is False
    assert fake_radio._en_tx == 0


def test_status_reports_tx_off_from_hardware_readback(fake_radio):
    # (retcode, rxMask, txMask) -> nothing enabled.
    fake_radio.device.RadioCtrl.RxTxEnableGet.return_value = (0, 0, 0)
    st = fake_radio.status()
    assert st["tx_off"] is True
    assert st["tx_enabled"] == []
    assert st["enable_source"] == "hardware"
    for key in ("connected", "pll_lock", "lo1_hz", "lo2_hz", "rx_enabled", "tx_atten_db"):
        assert key in st


def test_status_decodes_enabled_channels(fake_radio):
    fake_radio.device.RadioCtrl.RxTxEnableGet.return_value = (0, 0x0F, int(TxChannel.TX2))
    st = fake_radio.status()
    assert st["tx_off"] is False
    assert "TX2" in st["tx_enabled"]
    assert "RX1" in st["rx_enabled"] and "RX4" in st["rx_enabled"]


def test_status_degrades_to_tracked_when_readback_fails(fake_radio):
    fake_radio.device.RadioCtrl.RxTxEnableGet.side_effect = RuntimeError("boom")
    fake_radio.enable_tx(int(TxChannel.TX1))  # sets the tracked mask
    st = fake_radio.status()
    assert st["enable_source"] == "tracked"
    assert "TX1" in st["tx_enabled"]


def test_enable_masks_accumulate_and_apply(fake_radio):
    fake_radio.enable_rx(0x0F)
    assert (fake_radio._en_rx, fake_radio._en_tx) == (0x0F, 0)
    fake_radio.enable_tx(int(TxChannel.TX1))
    assert (fake_radio._en_rx, fake_radio._en_tx) == (0x0F, 0x1)
    # most recent DLL call reflects both masks
    rx, tx = fake_radio.device.RadioCtrl.RxTxEnableSet.call_args.args
    assert (rx, tx) == (0x0F, 0x1)


def test_disable_tx_keeps_rx(fake_radio):
    fake_radio.enable_rx(0x0F)
    fake_radio.enable_tx(0xF)
    fake_radio.disable_tx()
    assert fake_radio._en_rx == 0x0F
    assert fake_radio._en_tx == 0


def test_safe_state_stops_tx_enable(fake_radio):
    fake_radio.enable_rx(0x0F)
    fake_radio.enable_tx(0xF)
    fake_radio.safe_state()
    rx, tx = fake_radio.device.RadioCtrl.RxTxEnableSet.call_args.args
    assert tx == 0  # TX playback stopped
