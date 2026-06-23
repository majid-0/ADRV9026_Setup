"""Channel masks, trigger sources, and enum-name maps for the ADRV9026 / ADS9.

Pure-Python constants mirrored from the TES DLL (namespace ``adrv9010_dll``,
build 6.4.0.x). Confirmed against the decompiled CHM -- see ``docs/api_notes.md``.
Importable without pythonnet so the bulk of the package is testable hardware-free.
"""

from __future__ import annotations

from enum import IntEnum, IntFlag

# --- Channel masks (match the init script: ORx1=0x10 .. ORx4=0x80) -----------


class TxChannel(IntFlag):
    TX1 = 0x01
    TX2 = 0x02
    TX3 = 0x04
    TX4 = 0x08
    ALL = 0x0F


class RxChannel(IntFlag):
    """Rx and ORx share ``adi_adrv9010_RxChannels_e``; ORx are the high nibble."""

    RX1 = 0x01
    RX2 = 0x02
    RX3 = 0x04
    RX4 = 0x08
    ORX1 = 0x10
    ORX2 = 0x20
    ORX3 = 0x40
    ORX4 = 0x80
    ALL_RX = 0x0F
    ALL_ORX = 0xF0


#: Single-bit channels in stable order (avoids composite members + ``int.bit_count``,
#: which is 3.10+; this package targets 3.9+).
RX_SINGLE: tuple[RxChannel, ...] = (
    RxChannel.RX1,
    RxChannel.RX2,
    RxChannel.RX3,
    RxChannel.RX4,
    RxChannel.ORX1,
    RxChannel.ORX2,
    RxChannel.ORX3,
    RxChannel.ORX4,
)
TX_SINGLE: tuple[TxChannel, ...] = (
    TxChannel.TX1,
    TxChannel.TX2,
    TxChannel.TX3,
    TxChannel.TX4,
)


def is_orx(channel: RxChannel) -> bool:
    """True if every set bit is an ORx channel (high nibble)."""
    return int(channel) != 0 and (int(channel) & 0x0F) == 0


# --- Capture (PerformRx) trigger sources: adi_fpga9010_RxTollgateTrigSources_e


class RxTrigSource(IntEnum):
    IMMEDIATE = 0x1
    EXTERNAL = 0x2
    TDD_SM = 0x4
    ARM_ACK = 0x8
    TX1_SOF = 0x10
    TX2_SOF = 0x20
    TX3_SOF = 0x40
    TX4_SOF = 0x80


# --- Transmit (PerformTx) trigger sources: adi_fpga9010_TxTollgateTrigSources_e


class TxTrigSource(IntEnum):
    IMMEDIATE = 0x1
    EXTERNAL = 0x2
    TDD_SM = 0x4
    ARM_ACK = 0x8


#: Map a TX channel (single bit) to the RX trigger that fires on its start-of-frame.
TX_SOF_FOR: dict[TxChannel, RxTrigSource] = {
    TxChannel.TX1: RxTrigSource.TX1_SOF,
    TxChannel.TX2: RxTrigSource.TX2_SOF,
    TxChannel.TX3: RxTrigSource.TX3_SOF,
    TxChannel.TX4: RxTrigSource.TX4_SOF,
}

#: Our trigger enums -> the DLL enum member names (Rx uses ADI_FPGA9010_*,
#: Tx uses ADI_FPGA9010_TX_*).
RX_TRIG_MEMBER: dict[RxTrigSource, str] = {
    RxTrigSource.IMMEDIATE: "ADI_FPGA9010_IMM_TRIG",
    RxTrigSource.EXTERNAL: "ADI_FPGA9010_EXT_TRIG",
    RxTrigSource.TDD_SM: "ADI_FPGA9010_TDD_SM",
    RxTrigSource.ARM_ACK: "ADI_FPGA9010_ARM_ACK",
    RxTrigSource.TX1_SOF: "ADI_FPGA9010_TX1_SOF",
    RxTrigSource.TX2_SOF: "ADI_FPGA9010_TX2_SOF",
    RxTrigSource.TX3_SOF: "ADI_FPGA9010_TX3_SOF",
    RxTrigSource.TX4_SOF: "ADI_FPGA9010_TX4_SOF",
}
TX_TRIG_MEMBER: dict[TxTrigSource, str] = {
    TxTrigSource.IMMEDIATE: "ADI_FPGA9010_TX_IMM_TRIG",
    TxTrigSource.EXTERNAL: "ADI_FPGA9010_TX_EXT_TRIG",
    TxTrigSource.TDD_SM: "ADI_FPGA9010_TX_TDD_SM",
    TxTrigSource.ARM_ACK: "ADI_FPGA9010_TX_ARM_ACK",
}

# --- PLL names: adi_adrv9010_PllName_e -> enum member name on the DLL ----------

PLL_NAME = {
    "LO1": "ADI_ADRV9010_LO1_PLL",
    "LO2": "ADI_ADRV9010_LO2_PLL",
    "AUX": "ADI_ADRV9010_AUX_PLL",
}

# --- LO-select enum member names (built by config -> DLL) ----------------------

LO_SEL = {
    "LO1": "ADI_ADRV9010_LOSEL_LO1",
    "LO2": "ADI_ADRV9010_LOSEL_LO2",
}
ORX_LO_SEL = {
    "TXLO": "ADI_ADRV9010_ORXLOSEL_TXLO",
    "AUXLO": "ADI_ADRV9010_ORXLOSEL_AUXLO",
}
PLL_LO_MODE = {
    "INTLO_NOOUTPUT": "ADI_ADRV9010_INTLO_NOOUTPUT",
    "INTLO_OUTPUT": "ADI_ADRV9010_INTLO_OUTPUT",
}


def init_cal_member(name: str) -> str:
    """``"TX_QEC_INIT"`` -> ``"ADI_ADRV9010_TX_QEC_INIT"`` (DLL enum member)."""
    return f"ADI_ADRV9010_{name}"


def tx_to_orx_member(orx_index: int, mapping: str) -> str:
    """``(1, "TX1_ORX1")`` -> ``"ADI_ADRV9010_MAP_TX1_ORX1"`` for orx1Map etc."""
    return f"ADI_ADRV9010_MAP_{mapping}"
