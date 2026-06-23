"""Context-managed ADRV9026 driver: lifecycle, safe-state, program, IO wrappers.

Crash safety (concern #3): TX is forced to a safe state (max attenuation) on every
exit path -- normal ``__exit__``, exceptions, ``atexit``, and SIGINT/SIGTERM -- and
again on every startup via :meth:`force_safe`, which covers hard kills the
in-process hooks cannot. The board keeps its last register state when the socket
drops, so "leave TX safe" is the invariant that matters.

The low-level ``ref``/``out`` call idiom (pass a placeholder, read index ``[1]``)
matches the working init script, e.g. ``PllFrequencyGet(pll, 0)[1]``.
"""

from __future__ import annotations

import atexit
import signal
from types import FrameType
from typing import TYPE_CHECKING

from . import _enums
from ._enums import RxChannel, RxTrigSource, TxChannel, TxTrigSource
from .config import Config

if TYPE_CHECKING:
    from ._clr import ClrBridge

MAX_TX_ATTEN_DB = 41.95  # ADRV902x TxAtten table max (~42 dB)


class Radio:
    """High-level driver. Use as a context manager::

    with Radio(config) as radio:
        radio.program()
        ...
    """

    def __init__(self, config: Config, bridge: ClrBridge | None = None):
        self.config = config
        self._bridge = bridge
        self._connected = False
        self._tx_live = False
        self._safe_hooks_installed = False
        # Resolved on connect:
        self.link = None
        self.board = None
        self.device = None
        self.adrv = None

    # -- bridge ----------------------------------------------------------------

    @property
    def bridge(self) -> ClrBridge:
        if self._bridge is None:
            from ._clr import ClrBridge

            self._bridge = ClrBridge(self.config.dll).load()
        return self._bridge

    # -- lifecycle / context manager ------------------------------------------

    def __enter__(self) -> Radio:
        self.connect()
        self.force_safe()  # never assume the board powered up in a safe state
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.safe_state()
        self.disconnect()
        return False  # never swallow exceptions

    def connect(self) -> None:
        b = self.bridge
        self.link = b.link
        self.board = self.link.platform.board
        if not self._is_connected():
            self.board.Client.Connect(self.config.board.ip, self.config.board.port)
        if not self._is_connected():
            raise ConnectionError(
                f"failed to connect to ADS9 at {self.config.board.ip}:{self.config.board.port}"
            )
        self.device = self.board.Adrv9010Device
        self.adrv = self.link.Adrv9010Get(1)
        self._connected = True
        self._install_safe_hooks()

    def disconnect(self) -> None:
        if self._connected and self.board is not None:
            try:
                self.board.Client.Disconnect()
            finally:
                self._connected = False

    def _is_connected(self) -> bool:
        try:
            return bool(self.link.IsConnected())
        except Exception:
            return False

    # -- crash safety ----------------------------------------------------------

    def _install_safe_hooks(self) -> None:
        if self._safe_hooks_installed:
            return
        atexit.register(self._safe_on_exit)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):  # not in main thread / unsupported
                pass
        self._safe_hooks_installed = True

    def _on_signal(self, signum: int, frame: FrameType | None) -> None:
        self.safe_state()
        raise KeyboardInterrupt(f"signal {signum}: TX forced safe")

    def _safe_on_exit(self) -> None:
        try:
            if self._connected:
                self.safe_state()
        except Exception:
            pass  # best effort during interpreter shutdown

    def force_safe(self) -> None:
        """Force TX to a known-safe state regardless of prior state (run on startup)."""
        self.set_tx_atten(TxChannel.ALL, MAX_TX_ATTEN_DB)
        self._tx_live = False

    def safe_state(self) -> None:
        """Force max TX attenuation. Cheap and idempotent; called on every exit path."""
        if not self._connected:
            return
        try:
            self.set_tx_atten(TxChannel.ALL, MAX_TX_ATTEN_DB)
        except Exception:
            pass
        finally:
            self._tx_live = False

    # -- programming (mirrors the init script, driven by config) ---------------

    def program(self) -> None:
        """Reproduce the StdUseCase init sequence from config. See docs/api_notes.md."""
        cfg = self.config
        b = self.bridge
        dev = self.device

        dev.ConfigFileLoad(str(cfg.profile_path))

        init = dev.InitStructGet()
        init.clocks.deviceClock_kHz = cfg.clocks.device_clock_khz
        init.rx.rxInitChannelMask = cfg.channels.rx_init_mask
        init.tx.txInitChannelMask = cfg.channels.tx_init_mask
        init.clocks.rx12LoSelect = b.enum("adi_adrv9010_LoSel_e", _enums.LO_SEL[cfg.clocks.rx12_lo])
        init.clocks.rx34LoSelect = b.enum("adi_adrv9010_LoSel_e", _enums.LO_SEL[cfg.clocks.rx34_lo])
        init.clocks.tx12LoSelect = b.enum("adi_adrv9010_LoSel_e", _enums.LO_SEL[cfg.clocks.tx12_lo])
        init.clocks.tx34LoSelect = b.enum("adi_adrv9010_LoSel_e", _enums.LO_SEL[cfg.clocks.tx34_lo])
        init.clocks.orx12LoSelect = b.enum(
            "adi_adrv9010_OrxLoSel_e", _enums.ORX_LO_SEL[cfg.clocks.orx12_lo]
        )
        init.clocks.orx34LoSelect = b.enum(
            "adi_adrv9010_OrxLoSel_e", _enums.ORX_LO_SEL[cfg.clocks.orx34_lo]
        )
        init.clocks.rfPll1LoMode = b.enum(
            "adi_adrv9010_PllLoMode_e", _enums.PLL_LO_MODE[cfg.clocks.pll1_lo_mode]
        )
        init.clocks.rfPll2LoMode = b.enum(
            "adi_adrv9010_PllLoMode_e", _enums.PLL_LO_MODE[cfg.clocks.pll2_lo_mode]
        )
        init.clocks.extLoFreq1_kHz = cfg.clocks.ext_lo1_khz
        init.clocks.extLoFreq2_kHz = cfg.clocks.ext_lo2_khz

        post = self._build_post_mcs_init()
        dev.UtilityInitStructSet(post)
        dev.ConfigFileLoad()  # load default ARM/stream files (no-arg, order matters)
        self.board.ClockConfig(*cfg.clocks.ad9528)
        self.board.Program()

        self._apply_levels()
        self._tx_live = False

    def _build_post_mcs_init(self):
        cfg = self.config
        b = self.bridge
        post = b.Types.adi_adrv9010_PostMcsInit_t()
        rc = post.radioCtrlInit
        rc.lo1PllFreq_Hz = cfg.lo.lo1_hz
        rc.lo2PllFreq_Hz = cfg.lo.lo2_hz
        rc.auxPllFreq_Hz = cfg.lo.aux_hz
        rc.radioCtrlModeCfg.rxRadioCtrlModeCfg.rxChannelMask = cfg.channels.rx_init_mask
        rc.radioCtrlModeCfg.rxRadioCtrlModeCfg.rxEnableMode = b.enum(
            "adi_adrv9010_RxEnableMode_e", f"ADI_ADRV9010_RX_EN_{cfg.channels.rx_enable_mode}"
        )
        rc.radioCtrlModeCfg.txRadioCtrlModeCfg.txChannelMask = cfg.channels.tx_init_mask
        rc.radioCtrlModeCfg.txRadioCtrlModeCfg.txEnableMode = b.enum(
            "adi_adrv9010_TxEnableMode_e", f"ADI_ADRV9010_TX_EN_{cfg.channels.tx_enable_mode}"
        )
        rc.radioCtrlModeCfg.orxRadioCtrlModeCfg.orxEnableMode = b.enum(
            "adi_adrv9010_ORxEnableMode_e", f"ADI_ADRV9010_ORX_EN_{cfg.channels.orx_enable_mode}"
        )
        # TX -> ORx mapping (orx1Map .. orx4Map)
        for idx, mapping in enumerate(cfg.tx_to_orx, start=1):
            enum_type = f"adi_adrv9010_TxToOrx{idx}Mapping_e"
            setattr(
                rc.txToOrxMapping,
                f"orx{idx}Map",
                b.enum(enum_type, _enums.tx_to_orx_member(idx, mapping)),
            )
        # Init cals
        cal_mask = 0
        for cal in cfg.init_cals.mask:
            cal_mask |= int(b.enum("adi_adrv9010_InitCalibrations_e", _enums.init_cal_member(cal)))
        post.initCals.calMask = cal_mask
        post.initCals.channelMask = cfg.init_cals.channel_mask
        post.initCals.warmBoot = cfg.init_cals.warm_boot
        return post

    def _apply_levels(self) -> None:
        cfg = self.config
        for ch in (TxChannel.TX1, TxChannel.TX2, TxChannel.TX3, TxChannel.TX4):
            if cfg.channels.tx_init_mask & int(ch):
                self.set_tx_atten(ch, cfg.levels.tx_atten_for(ch.name.lower()))
        rx_all = [
            RxChannel.RX1,
            RxChannel.RX2,
            RxChannel.RX3,
            RxChannel.RX4,
            RxChannel.ORX1,
            RxChannel.ORX2,
            RxChannel.ORX3,
            RxChannel.ORX4,
        ]
        for ch in rx_all:
            if cfg.channels.rx_init_mask & int(ch):
                self.set_rx_gain(ch, cfg.levels.rx_gain_for(ch.name.lower()))

    # -- gain / attenuation ----------------------------------------------------

    def set_tx_atten(self, channel: TxChannel, atten_db: float) -> None:
        b = self.bridge
        arr = b.new_array("adi_adrv9010_TxAtten_t", 1)
        item = b.Types.adi_adrv9010_TxAtten_t()
        item.txAttenuation_mdB = int(round(atten_db * 1000))
        item.txChannelMask = int(channel)
        arr[0] = item
        self.device.Tx.TxAttenSet(arr, 1)

    def set_rx_gain(self, channel: RxChannel, gain_index: int) -> None:
        b = self.bridge
        arr = b.new_array("adi_adrv9010_RxGain_t", 1)
        item = b.Types.adi_adrv9010_RxGain_t()
        item.gainIndex = int(gain_index)
        item.rxChannelMask = int(channel)
        arr[0] = item
        self.device.Rx.RxGainSet(arr, 1)

    def get_rx_gain(self, channel: RxChannel) -> int:
        b = self.bridge
        ch = b.enum("adi_adrv9010_RxChannels_e", _rx_channel_member(channel))
        placeholder = b.Types.adi_adrv9010_RxGain_t()
        result = self.device.Rx.RxGainGet(ch, placeholder)
        return int(result[1].gainIndex)

    def rx_dec_power_dbfs(self, channel: RxChannel) -> float:
        """Measured DEC power for an Rx/ORx channel. Returns negative dBFS.

        ``RxDecPowerGet`` reports magnitude below full scale in mdBFS (UInt16);
        we negate to a dBFS value. Sign/scale to be confirmed on hardware.
        """
        b = self.bridge
        ch = b.enum("adi_adrv9010_RxChannels_e", _rx_channel_member(channel))
        result = self.device.Rx.RxDecPowerGet(ch, 0)
        return -(int(result[1]) / 1000.0)

    # -- LO / PLL --------------------------------------------------------------

    def set_lo(self, pll: str, freq_hz: int) -> None:
        b = self.bridge
        pll_enum = b.enum("adi_adrv9010_PllName_e", _enums.PLL_NAME[pll])
        self.adrv.RadioCtrl.PllFrequencySet(pll_enum, int(freq_hz))

    def get_lo(self, pll: str) -> int:
        b = self.bridge
        pll_enum = b.enum("adi_adrv9010_PllName_e", _enums.PLL_NAME[pll])
        return int(self.adrv.RadioCtrl.PllFrequencyGet(pll_enum, 0)[1])

    def pll_lock_status(self) -> int:
        return int(self.adrv.RadioCtrl.PllStatusGet(0)[1])

    def retune_lo(self, pll: str, freq_hz: int, *, settle_poll: int = 50) -> int:
        """Set an LO and poll lock status until stable. Returns the lock bitmask.

        Deterministic in the settled-state sense (concern #2): commanded value
        applied + relock confirmed; not a phase-continuous hop.
        """
        self.set_lo(pll, freq_hz)
        status = 0
        for _ in range(settle_poll):
            status = self.pll_lock_status()
            if status:
                break
        return status

    # -- capture / transmit (PerformRx / PerformTx) ----------------------------

    def _rx_trig(self, trig: RxTrigSource):
        return _fpga_enum(self.bridge, "adi_fpga9010_RxTollgateTrigSources_e", trig)

    def _tx_trig(self, trig: TxTrigSource):
        return _fpga_enum(self.bridge, "adi_fpga9010_TxTollgateTrigSources_e", trig)

    def perform_rx(
        self,
        channel_mask: int,
        capture_time_ms: float,
        *,
        trig: RxTrigSource = RxTrigSource.IMMEDIATE,
        timeout_ms: int = 1000,
    ):
        """Arm + trigger a snapshot capture. Returns the DLL capture result object.

        Higher-level shaping (per-channel int IQ arrays) lives in ``capture.py``;
        the exact readback container is confirmed against hardware there.
        """
        return self.board.PerformRx(
            self._rx_trig(trig), int(channel_mask), float(capture_time_ms), int(timeout_ms)
        )

    def perform_tx(
        self,
        tx_data,
        channel_mask: int,
        *,
        trig: TxTrigSource = TxTrigSource.IMMEDIATE,
        continuous: bool = True,
    ) -> None:
        """Load per-channel sample buffers and start playback (looping if continuous)."""
        self.board.PerformTx(
            self._tx_trig(trig), tx_data, int(channel_mask), 1 if continuous else 0
        )
        self._tx_live = True


# -- module helpers ------------------------------------------------------------


def _rx_channel_member(channel: RxChannel) -> str:
    return f"ADI_ADRV9010_{channel.name}"


def _fpga_enum(bridge, type_name: str, member):
    """Resolve an ``adi_fpga9010_*`` enum from ``ns.FpgaTypes``; fall back to int."""
    fpga = getattr(bridge.ns, "FpgaTypes", None)
    if fpga is None:
        return int(member)
    enum_type = getattr(fpga, type_name, None)
    if enum_type is None:
        return int(member)
    return getattr(enum_type, member.name)
