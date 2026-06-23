# ADRV9026 / TES DLL — API Notes (Task 0)

Source: decompiled CHM help shipped with TES build 6.4.0.x
(`AdrvTrx_TCPIP_Client_DLL.chm`, `adrv9025.chm`). Namespace is **`adrv9010_dll`**.
Doc generated 2022-06-28. These are the host-callable (pythonnet) signatures.

Object model: `AdiEvaluationSystem.Instance` → `.platform.board` is an
**`AdrvDaughterCard`** (top-level). Device sub-API via `link.Adrv9010Get(1)` →
`.Rx`, `.Tx`, `.RadioCtrl`, `.Cals`, `.Hal`, `.DataInterface`, etc.

---

## 1. ORx leveling — RESOLVED: hardware flags exist (flag-based branch)

ORx is manual-gain (no hardware AGC state machine), BUT the build exposes both a
**level readback** and **overload/clip indicators**, valid for ORx channels:

- `Adrv9010Rx.RxDecPowerGet(adi_adrv9010_RxChannels_e rxChannel, UInt16 %rxDecPower_mdBFS)`
  → measured DEC power in **milli-dBFS**. Direct level readback.
- `AdrvDaughterCard.GetEmbeddedOverloadIndicators(adi_adrv9010_RxChannels_e rxChannel)`
  (+ `GetEmbeddedOverloadIndicatorLsbI/Q`, `...LsbPlusOneI/Q`) → embedded ADC
  overload/clip indicators.
- `Adrv9010Rx.RxGainSet(array<adi_adrv9010_RxGain_t> rxGain, Byte arraySize)` /
  `RxGainGet(rxChannel, %rxGain)` — manual gain index (the lever to step).
- `Adrv9010Rx.RxGainCtrlModeSet(array<adi_adrv9010_RxAgcMode_t> gainMode, Byte n)` /
  `RxGainCtrlModeGet(...)` — gain-control mode (MGC/AGC) for Rx; ORx stays MGC.

**`adi_adrv9010_RxChannels_e` includes RX1–RX4 AND ORX1–ORX4** → all the above
accept ORx. Channel masks match the init script: ORx1=0x10 … ORx4=0x80.

→ Leveling design: read `RxDecPowerGet` (and/or overload indicators) → step
`RxGainSet` on the ORx channel → converge to target mdBFS window. IQ-from-capture
peak/clip remains as a build-independent cross-check, but is no longer the only
option. AGC target metric: peak-headroom dBFS (decide exact setpoint on hardware).

## 2. Capture (snapshot) + deterministic triggering — RESOLVED

On `AdrvDaughterCard`:
- `PerformRx(adi_fpga9010_RxTollgateTrigSources_e trigSource, UInt32 channelMask, Double captureTime_ms, UInt32 timeout_ms)`
  (overloads add a delegate, or `UInt16[] addresses, Byte[] data`).
- `CaptureRawRx(... same args ...)` → raw (unscaled) samples.
- `ScaleRxSingle(int[] iData, int[] qData, Byte np, adi_adrv9010_RxChannels_e channel)`
  and `ScaleRx(List<int[]> rxData)` → scale raw to formatted; **`np` = JESD Np**.
- `OutputRateGet(int channelMaskIndex, UInt32[] outputRates)` → per-channel sample
  rate at runtime (cross-check vs profile).

`channelMask` selects which Rx/ORx channels capture in ONE call → all captured in
the same sample-aligned snapshot.

**Trigger enum `adi_fpga9010_RxTollgateTrigSources_e`:**
`IMM_TRIG=0x1`, `EXT_TRIG=0x2`, `TDD_SM=0x4`, `ARM_ACK=0x8`,
**`TX1_SOF=0x10`, `TX2_SOF=0x20`, `TX3_SOF=0x40`, `TX4_SOF=0x80`**.
→ Use `TXn_SOF` to align ORx/Rx capture to TX start-of-frame (deterministic
TX↔ORx). Use `IMM_TRIG` for free-running snapshot.

## 3. TX playback (multi-band) — RESOLVED

On `AdrvDaughterCard`:
- `PerformTx(adi_fpga9010_TxTollgateTrigSources_e trigSource, List<int[]> txData, UInt32 channelMask)`
- `PerformTx(..., ArrayList txData, UInt32 channelMask)`
- `PerformTx(..., List<int[]> txData, UInt32 channelMask, Byte continuous)` ← **continuous/looping playback** (key for sustained multi-band TX during sweeps/captures).

`txData` = one `int[]` of interleaved/real samples per enabled TX channel;
`channelMask` selects TX paths → multi-band = multiple channel buffers in one call,
started together (deterministic relative start across TX paths).

**Tx trigger enum `adi_fpga9010_TxTollgateTrigSources_e`:**
`TX_IMM_TRIG=0x1`, `TX_EXT_TRIG=0x2`, `TX_TDD_SM=0x4`, `TX_ARM_ACK=0x8`.

DAC scaling: `Adrv9010Tx.DacFullScaleGet/Set(adi_adrv9010_TxChannels_e, adi_adrv9010_DacFullScale_e)`.

## 4. PLL retune + lock (deterministic-while-transmitting) — RESOLVED

On `Adrv9010Radioctrl` (`adrv9010.RadioCtrl`):
- `PllFrequencySet(adi_adrv9010_PllName_e pllName, UInt64 pllLoFrequency_Hz)`
  (+ `PllFrequencySet_v2(adi_adrv9010_PllConfig_t)`, range-check variants).
- `PllFrequencyGet(adi_adrv9010_PllName_e pllName, UInt64 %rfPllLoFrequency_Hz)`.
- `PllStatusGet(UInt32 %pllLockStatus)` → poll for relock after retune.

Retune pattern = `PllFrequencySet` → poll `PllStatusGet` until locked → capture.
Settled-state deterministic; NOT phase-continuous hop (as expected).

## 5. Profile fields (bit depth + sample rate) — RESOLVED

In the `.profile` JSON (e.g. `ADRV9025Init_StdUseCase102_LinkSharing.profile`):
- Sample rate: **`txInputRate_kHz`** (deframer/Tx), **`rxOutputRate_kHz`** (framer/Rx).
  StdUseCase102 = 245760 kHz (245.76 MSPS) for both.
- Bit depth: **`jesd204Np`** under `framer[]` (Rx/ORx) and `deframer[]` (Tx).
  StdUseCase102 `jesd204Np = 16` → JESD transport word = 16-bit.

→ **Quantization rule:** scale = `2^(Np-1) - 1`. For Np=16 → ×32767 (signed 16-bit),
NOT 2^11. The user's "12-bit → base 11" generalizes to `Np` read from the loaded
profile; confirm effective DAC range against `DacFullScale*` on hardware.
`captureTime_ms × rxOutputRate_kHz` gives capture sample count.

## 6. Clock / program (already in init script) — CONFIRMED

- `ClockConfig(UInt32 deviceClock_kHz, UInt32 vcxoFreq_kHz, UInt32 refAFreq_kHz, UInt32 fpgaClock_kHz)`
- `Program()`, `ProgramPhase1()/Phase2()`, `BoardPreProgram()`, `Discover()`.
- Device: `ConfigFileLoad(profile)` then `ConfigFileLoad()` (default ARM/stream),
  `InitStructGet()`, `UtilityInitStructSet(postMcsInit)`, `Cals.InitCalsRun/Wait`.

---

## Decisions this locks in for the design
- ORx leveling = **flag-based** (`RxDecPowerGet` + overload indicators), IQ-clip as backup.
- Capture = `PerformRx` w/ `channelMask` (Rx or ORx), `captureTime_ms`, `TXn_SOF` trigger for alignment.
- TX = `PerformTx` (continuous overload) with per-channel `int[]`, `channelMask` for multi-band.
- Quantize TX waveform to signed `2^(Np-1)-1` with `Np` from profile `jesd204Np`.
- Sample count from `captureTime_ms` × profile rate; verify via `OutputRateGet`.
- Retune via `PllFrequencySet` + `PllStatusGet` poll.

Open (hardware-only) confirmations: exact `RxDecPowerGet` behavior/latency on ORx,
effective DAC full-scale vs Np, and `PerformTx` int[] sample packing (real vs I/Q interleave).
