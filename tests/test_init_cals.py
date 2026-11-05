"""Radio.run_init_cals: the init cals re-run after an LO retune (fake board and DLL surface)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from adrvtrx import TxChannel
from adrvtrx._enums import INIT_CAL_BITS, RETUNE_INIT_CALS, init_cal_names
from adrvtrx.config import Config, DllConfig
from adrvtrx.fake import FakeRadio
from adrvtrx.radio import MAX_TX_ATTEN_DB

MAX_MDB = round(MAX_TX_ATTEN_DB * 1000)
RETUNE_MASK = (
    INIT_CAL_BITS["TX_LO_LEAKAGE_INTERNAL"]
    | INIT_CAL_BITS["TX_QEC_INIT"]
    | INIT_CAL_BITS["ORX_QEC_INIT"]
)


@pytest.fixture
def radio():
    r = FakeRadio(Config(dll=DllConfig(install_dir=Path("C:/nonexistent"))), seed=1)
    r._safe_hooks_installed = True
    r.connect()
    r.program()
    yield r
    r.disconnect()


def _events(r, op):
    return [c for c in r.board_model.calls if c == op]


def test_program_runs_the_init_cals_at_the_program_time_los_and_a_retune_makes_them_stale(radio):
    board = radio.board_model
    assert set(board.state["cal_lo_hz"]) == set(radio.config.init_cals.mask)
    assert board.cal_is_current("ORX_QEC_INIT") and board.cal_is_current("TX_QEC_INIT")
    radio.retune_lo("LO2", 1_870_000_000)
    assert not board.cal_is_current("ORX_QEC_INIT") and board.cal_is_current("ORX_QEC_INIT", "LO1")


def test_run_init_cals_on_the_fake_redoes_the_retune_cals_with_tx_off_and_safe(radio):
    board = radio.board_model
    radio.retune_lo("LO1", 935_000_000)
    radio.set_tx_atten(TxChannel.TX2, 20.0)
    radio.set_rx_enable(0x60)  # ORX2, ORX3 enabled
    radio.enable_tx(int(TxChannel.TX2))  # TX live: run_init_cals must stop it first
    seen = {}
    orig = board.init_cals_run

    def spy(ic):
        seen.update(
            tx=board.state["tx_mask"],
            rx=board.state["rx_mask"],
            atten=dict(board.state["tx_atten_mdb"]),
        )
        return orig(ic)

    board.init_cals_run = spy
    radio._bridge.link.Adrv9010Get(1).Cals.InitCalsRun = spy
    res = radio.run_init_cals()
    assert res["ok"] and res["error"] == "" and res["error_flag"] == 0 and res["init_err_code"] == 0
    assert res["cals"] == list(RETUNE_INIT_CALS) and res["cal_mask"] == RETUNE_MASK
    assert res["channel_mask"] == radio.config.init_cals.channel_mask == 0xF
    assert res["cals_last_run"] == [list(RETUNE_INIT_CALS)] * 4 and res["duration_us"] > 0
    assert seen["tx"] == 0 and seen["rx"] == 0  # radio off during the run
    assert all(v == MAX_MDB for v in seen["atten"].values())  # every TX at max attenuation
    assert board.state["tx_mask"] == 0 and board.state["rx_mask"] == 0x60  # Rx back, TX stays off
    assert all(v == MAX_MDB for v in board.state["tx_atten_mdb"].values())
    assert board.cal_is_current("ORX_QEC_INIT") and board.cal_is_current("TX_LO_LEAKAGE_INTERNAL")
    assert not board.cal_is_current("RX_QEC_INIT")  # not asked: still at the program-time LOs
    assert board.state["cal_runs"] == 1
    # a chosen set and channel mask
    res = radio.run_init_cals(["orx_qec_init"], channel_mask=0x4, timeout_ms=1000)
    assert (
        res["ok"]
        and res["cal_mask"] == INIT_CAL_BITS["ORX_QEC_INIT"]
        and res["channel_mask"] == 0x4
    )
    assert res["cals_last_run"] == [[], [], ["ORX_QEC_INIT"], []]


def test_a_failing_calibration_is_reported_not_raised_and_leaves_the_board_safe(radio, tmp_path):
    board = radio.board_model
    radio.retune_lo("LO2", 1_434_000_000)
    radio.set_rx_enable(0x40)
    board.init_cal_fail = 3
    res = radio.run_init_cals()
    assert not res["ok"] and res["error_flag"] == 3 and res["init_err_code"] != 0
    assert res["init_err_cal"] == INIT_CAL_BITS["TX_LO_LEAKAGE_INTERNAL"]
    assert not board.cal_is_current("ORX_QEC_INIT")  # a failed run calibrates nothing
    assert board.state["tx_mask"] == 0 and board.state["rx_mask"] == 0x40
    board.init_cal_fail = None
    assert radio.run_init_cals()["ok"] and board.cal_is_current("ORX_QEC_INIT")


def test_the_calfail_file_fails_runs_across_processes(tmp_path):
    state = tmp_path / "board.json"
    r = FakeRadio(
        Config(dll=DllConfig(install_dir=Path("C:/nonexistent"))), seed=1, state_path=state
    )
    r._safe_hooks_installed = True
    r.connect()
    r.program()
    state.with_suffix(".calfail").write_text("4")
    assert r.run_init_cals()["error_flag"] == 4
    state.with_suffix(".calfail").unlink()
    assert r.run_init_cals()["ok"]
    r.disconnect()


def test_unknown_and_external_cals_are_refused_before_the_board_is_touched(radio):
    board = radio.board_model
    radio.set_tx_atten(TxChannel.TX1, 20.0)
    n = len(board.calls)
    for bad in (
        ["NOPE"],
        ["TX_LO_LEAKAGE_EXTERNAL"],
        ["ORX_QEC_INIT", "DPD"],
        ["VSWR"],
        ["EXTERNAL_PATH_DELAY"],
        [],
    ):
        with pytest.raises(ValueError):
            radio.run_init_cals(bad)
    assert len(board.calls) == n and board.state["tx_atten_mdb"]["TX1"] == 20000


def test_a_cal_with_tx_enabled_is_refused_by_the_fake_itself(radio):
    """The fake's own guard: InitCalsRun straight on the DLL surface with TX on fails."""
    radio.enable_tx(int(TxChannel.TX1))
    ic = SimpleNamespace(calMask=RETUNE_MASK, channelMask=0xF, warmBoot=0)
    with pytest.raises(RuntimeError, match="TX enabled"):
        radio.adrv.Cals.InitCalsRun(ic)


def test_init_cal_names():
    assert init_cal_names(RETUNE_MASK) == list(RETUNE_INIT_CALS)
    assert init_cal_names(0x10000000) == ["0x10000000"] and init_cal_names(0) == []


def test_the_dll_calls_and_their_order(fake_bridge, fake_radio):
    """On the DLL surface (MagicMock bridge): TxAttenSet max, RxTxEnableSet(0, 0),
    Cals.InitCalsRun(struct), InitCalsWait(timeout, out flag), InitCalsDetailedStatusGet(out
    status), then the Rx enables back."""
    from unittest.mock import call

    fake_bridge.enum = lambda t, m: (
        INIT_CAL_BITS[m.replace("ADI_ADRV9010_", "")]
        if t == "adi_adrv9010_InitCalibrations_e"
        else f"{t}.{m}"
    )
    cals = fake_bridge.adrv.Cals
    cals.InitCalsRun.return_value = 0
    cals.InitCalsWait.return_value = (0, 0)
    cals.InitCalsDetailedStatusGet.return_value = (
        0,
        SimpleNamespace(
            initErrCode=0,
            initErrCal=0,
            calsDurationUsec=812345,
            calsLastRun=[RETUNE_MASK] * 4,
            calsSincePowerUp=[0] * 4,
        ),
    )
    fake_radio._en_rx = 0x40
    res = fake_radio.run_init_cals(timeout_ms=30000)
    assert (
        res["ok"]
        and res["duration_us"] == 812345
        and res["cals_last_run"][0] == list(RETUNE_INIT_CALS)
    )
    ic = cals.InitCalsRun.call_args.args[0]
    assert (ic.calMask, ic.channelMask, ic.warmBoot) == (RETUNE_MASK, 0xF, 0)
    assert cals.InitCalsWait.call_args == call(30000, 0)
    att = fake_bridge.device.Tx.TxAttenSet.call_args.args[0][0]
    assert att.txAttenuation_mdB == MAX_MDB and att.txChannelMask == int(TxChannel.ALL)
    en = fake_bridge.device.RadioCtrl.RxTxEnableSet.call_args_list
    assert en == [call(0, 0), call(0x40, 0)]
    # a DLL exception (e.g. a wait timeout) is reported, the cals aborted, the Rx enables restored
    cals.InitCalsWait.side_effect = RuntimeError("timeout waiting for init cals")
    res = fake_radio.run_init_cals()
    assert not res["ok"] and "timeout" in res["error"] and cals.InitCalsAbort.called
    assert fake_bridge.device.RadioCtrl.RxTxEnableSet.call_args == call(0x40, 0)
