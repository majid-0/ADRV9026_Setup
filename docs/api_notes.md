# ADRV9026 / TES DLL — API Notes (Task 0)

Source: decompiled CHM help shipped with TES build 6.4.0.x
(`AdrvTrx_TCPIP_Client_DLL.chm`, `adrv9025.chm`). Namespace is **`adrv9010_dll`**.
Doc generated 2022-06-28. These are the host-callable (pythonnet) signatures.

**Update:** the two previously-open blockers (PerformRx readback, PerformTx packing)
plus channel enable/disable are now RESOLVED from working ADI sample scripts the
user supplied — see "Resolved from ADI samples" at the bottom.

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
and effective DAC full-scale vs Np.

---

## Resolved from ADI sample scripts (user-supplied)

Two ADI examples (a DPD/CLGC script and an Rx-capture script) confirm runtime
behavior the CHM didn't:

### PerformRx readback — RESOLVED
```python
rx_data = board.PerformRx(FpgaTypes...IMM_TRIG, 255, capture_ms, timeout_ms)
rx1I = rx_data[0]; rx1Q = rx_data[1]; rx2I = rx_data[2]; ...  Orx1I = rx_data[8] ...
```
Returns a **flat indexable of already-scaled int arrays**, interleaved
`[ch0_I, ch0_Q, ch1_I, ch1_Q, ...]` in ascending channel order. No `ScaleRx` call
needed. **ORx availability is profile-dependent** (link-sharing mask `0xFF` returned
only ORx1 & ORx3 → 12 arrays). `captureTime`/`timeout` are in **ms**.
→ implemented in `capture.extract_channels`.

### PerformTx packing — RESOLVED
```python
tx_data = ArrayList()
tx_data.Add(Tx1Iarray); tx_data.Add(Tx1Qarray); tx_data.Add(Tx2Iarray); ...
board.PerformTx(FpgaTypes...TX_IMM_TRIG, tx_data, 255)
```
**Separate I and Q int arrays per channel** (two arrays/channel), ascending order —
NOT packed `(I<<16|Q)` or single-array interleaved. → `transmit.build_tx_data`.

### Channel enable + TX stop — RESOLVED (was missing entirely)
`device.RadioCtrl.RxTxEnableSet(rxMask, txMask)` sets both at once. Capture enables
Rx first (`0x0F`); TX flow is **disable → PerformTx → enable** (`RxTxEnableSet(rx, 0x01)`).
Clearing the TX mask **stops** playback → used by `safe_state`/`disable_tx`.

### Full-scale code — RESOLVED
ADI uses `20*log10(sqrt(pwr)/32768)`, i.e. dBFS reference = `2**(Np-1)` (32768 for
Np=16). `clip_report` uses this reference; quantization clips at `2**(Np-1)-1`.

### Confirmed on hardware (bench, StdUseCase102_LinkSharing, rx_init=0x3FF)
- **connect + program works**: PLLs lock (status `0xF`), LO1/LO2 read back exactly.
- **PerformRx ignores its mask arg** and returns the full **programmed
  `rxInitChannelMask`** set: `0x3FF` → 20 arrays = 10 channels, in bit order
  `Rx1..Rx4, ORx1..ORx4,` + 2 internal/loopback (bits 0x100/0x200). So index a
  wanted channel by absolute position (`capture.returned_channel_order`). The
  count = popcount(rx_init_mask); which channels carry real data is profile-driven.
  ORx2 **is** present in this profile (24576 samples @ 0.1 ms / 245.76 MSPS).
- **PerformTx requires exactly 8 arrays** ("one array for each Tx ADC") = all four
  TX channels x I/Q; zero-fill undriven channels, `channelMask` selects who TXes.
- **Trigger enums** are `ADI_FPGA9010_*` (Rx) / `ADI_FPGA9010_TX_*` (Tx) members.
- **force_safe / TxAttenSet** is rejected pre-`Program()` ("Invalid Tx attenuation
  control mode") -> startup safe-state is best-effort.
- **RxDecPowerGet** read `-0.00 dBFS` on ORx2/ORx3 with no armed measurement ->
  likely needs the channel active / a measurement trigger. Use captured-IQ peak for
  ORx leveling until its arming is confirmed.

### Bonus API confirmed (for later)
- **NCO:** `Tx.TxNcoShifterSet(adi_adrv9025_TxNcoShifterCfg_t{shifterMode=MIX,
  shiftFrequency_kHz, shiftGain, txChannelMask})` and `Rx.RxNcoShifterSet` — digital
  band offset (multi-band within one channel).
- **Main-Rx hardware AGC template:** `Agc.AgcCfgSet(adi_adrv9010_AgcCfg_t[], n)` +
  `Rx.RxGainCtrlModeSet([RxAgcMode{agcMode=AGCSLOW}], n)`, with `agcPeak`
  (apd/hb2 thresholds) and `agcPower` sub-structs. Confirms AGC is a main-Rx feature
  (ORx stays manual → software leveling). Useful if main-Rx AGC is ever wanted.
- **Connection accessor varies by board:** ADS9 uses `link.platform.board.*`; older
  ADS8 samples use `link.Ads8.board.*`. We target ADS9 (`platform.board`).
- **TX vectors are integer files** ("I⟶Q signed integers, equal length") — matches
  our format; use `do_normalize=False` to preserve a fixed PeakScaling.
