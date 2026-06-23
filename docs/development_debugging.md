# Development & Debugging Guide (for a future agent)

This is the orientation a new agent needs to work on **adrvtrx** with the user.
It captures the things that are NOT obvious from the code: the vendor install
layout, how to read the (compiled) ADI docs, how the Python ↔ .NET boundary works,
and — most importantly — the **hardware behaviors discovered on the bench** that
contradict or extend the documentation. Read this first, then `docs/api_notes.md`.

---

## 0. Orientation

- **Repo (working copy):** `C:\Users\ohammi\OneDrive - aus.edu\ADRV9026_Setup`
- **GitHub:** https://github.com/majid-0/ADRV9026_Setup (remote `origin`, branch `main`).
  CI runs lint + hardware-free tests on push (`.github/workflows/ci.yml`).
- **Python env:** conda env **`myenv`** (Python 3.11). Conda lives at
  `C:\ProgramData\anaconda3` (not on PATH — call `C:\ProgramData\anaconda3\Scripts\conda.exe`).
  Package is installed **editable** (`pip install -e .[dev]`), so source edits are live —
  no reinstall after changes.
- **OFF-LIMITS:** the folder `c:\Users\ohammi\Setup Automatoin` is abandoned prior work.
  Do **not** read, copy, or reference it. The user was emphatic about this.
- **Shell:** Windows PowerShell. `git push` prints to stderr → PowerShell shows a red
  "RemoteException" wrapper even on success; check the exit/`-> main` line, not the color.
  Heredocs don't work; for multi-line commit messages write `.git\COMMIT_MSG.tmp` and
  `git commit -F`.

### Run things
```powershell
$conda = "C:\ProgramData\anaconda3\Scripts\conda.exe"
& $conda run -n myenv python -m pytest -m "not hardware" -q   # 48 mocked tests
& $conda run -n myenv python -m ruff check src tests scripts
& $conda run -n myenv python -m black src tests scripts
& $conda run -n myenv adrvtrx-program                          # connect + program + status
& $conda run -n myenv python scripts/hw_smoke.py              # bench discovery / loopback
& $conda run -n myenv python -m pytest -m hardware -v         # gated HW tests (skip if no board)
```

---

## 1. The TES installation

Root: `C:\Program Files\Analog Devices\ADRV9025 Transceiver Evaluation Software_x64_FULL`
(parameterized in `config/default.toml` `[dll].install_dir` — never hardcode in code).

| What | Path (under install root) |
|---|---|
| Host DLL (.NET) | `adrvtrx_dll.dll` (also `Resources\Adi.Adrv9025.Api\public\x64\adrvtrx_dll.dll`) |
| Profiles (JSON, `.profile` ext) | `Resources\Adi.ADRV9025.Profiles\public\*.profile` |
| Host-API docs (CHM) | `Resources\Adi.Adrv9025.Api\public\doc\AdrvTrx_TCPIP_Client_DLL.chm` |
| C-API docs (CHM) | `Resources\Adi.Adrv9025.Api\public\doc\adrv9025.chm` |
| Gain tables | `Resources\Adi.Adrv9025.GainTables\public\*.csv` |
| Board server (runs on ADS9) | `Resources\Adi.Adrv9025.Api\public\adrv9025_server.exe` (ARM/Linux) |

Naming note: the software is branded "ADRV9025" but the device is an **ADRV9026**, and
the **DLL namespace is `adrv9010_dll`**. All three names refer to the same stack.

---

## 2. Reading the vendor docs (CHM)

The authoritative host-API reference is the **CHM** files. They are compiled HTML.

- `hh.exe -decompile` **did not work** here (produced no files).
- **7-Zip works** and is installed at `C:\Program Files\7-Zip\7z.exe` (not on PATH):
  ```powershell
  & "C:\Program Files\7-Zip\7z.exe" x `
    "C:\Program Files\Analog Devices\ADRV9025 Transceiver Evaluation Software_x64_FULL\Resources\Adi.Adrv9025.Api\public\doc\AdrvTrx_TCPIP_Client_DLL.chm" `
    "-oC:\Users\ohammi\adrv9026_task0\chm\trx_tcpip" -y
  ```
- Extracted to a **scratch dir** `C:\Users\ohammi\adrv9026_task0\chm\` (NOT in the repo;
  may need re-extracting). Subdirs: `trx_tcpip` (host API — most useful), `adrv9025`
  (C API), `adrv9010_tcpip`.
- Once extracted, **Grep the HTML**. Most useful files:
  - `classadrv9010__dll_1_1_sub_classes_1_1_adrv9010_*.html` — Rx/Tx/Radioctrl/Cals/Dfe methods
  - `classadrv9010__dll_1_1_top_level_classes_1_1_adrv_daughter_card*.html` — board:
    `PerformRx`, `PerformTx`, `Program`, `ClockConfig`, capture/scale helpers
  - `namespaceadrv9010__dll_1_1_types.html`, `...fpga_types.html` — enums & struct names
- A doxygen member row shows the C++/CLI signature: `^` = handle, `%` = `ref`/`out`.
  `array<T>^` is a .NET array param.

> **Lesson learned:** the CHM gives **signatures**, not **runtime behavior**. Several
> things (PerformRx ignoring its mask, PerformTx needing 8 arrays, enable-before-capture)
> were only discovered by running on hardware or from the user's working ADI sample
> scripts. **Trust the bench over the docs for behavior.**

---

## 3. The .NET object model & calling conventions

Bootstrap (in `src/adrvtrx/_clr.py`, the **only** module that imports pythonnet):
```python
import clr; clr.AddReference(r"...\adrvtrx_dll.dll")
from adrv9010_dll import AdiEvaluationSystem, Types, FpgaTypes, Ad9528Types
link = AdiEvaluationSystem.Instance
```
Object tree:
```
AdiEvaluationSystem.Instance                      (link)
  .platform.board                                 AdrvDaughterCard  -> PerformRx/PerformTx/Program/ClockConfig/Client
    .Client.Connect(ip, port) / .Disconnect()
    .Adrv9010Device                               -> .Rx .Tx .Cals .Dfe .RadioCtrl .Hal .DataInterface
  .Adrv9010Get(1)                                 device handle (also has .RadioCtrl, .Hal, .Agc, .ExternalDpd)
```
**Calling conventions that bite you:**
- **`ref`/`out` params (`%` in docs):** pass a placeholder and read index `[1]` of the
  result. E.g. `lo = adrv.RadioCtrl.PllFrequencyGet(pll, 0); value = lo[1]`. The DLL
  returns `(retcode, outValue, ...)`.
- **Arrays:** build via `Array.CreateInstance(Types.<struct>, n)` then assign `arr[0] = ...`,
  or `System.Array[int](pylist)`. See `ClrBridge.new_array` / `int_array` / `array_list`.
- **Enums:** pass the actual .NET enum member, not an int or Python name. The member
  names are vendor-prefixed and differ from our Python enum names — e.g. our
  `RxTrigSource.IMMEDIATE` → DLL `FpgaTypes.adi_fpga9010_RxTollgateTrigSources_e.ADI_FPGA9010_IMM_TRIG`.
  Name maps live in `_enums.py` (`RX_TRIG_MEMBER`, `TX_TRIG_MEMBER`, `LO_SEL`, etc.).

---

## 4. HARDWARE FACTS — read this twice

These are confirmed on the bench (ADS9 + ADRV9026, profile `StdUseCase98_LinkSharing`).
They override anything the docs imply.

1. **The ADRV9026 has 4 Tx, 4 Rx, but only TWO ORx ADCs.** "ORx1/2/3/4" are four
   observation **front-end inputs muxed into the 2 ORx ADCs** — they are NOT four
   independent capture channels.

2. **`PerformRx(trig, mask, captureTime_ms, timeout_ms)` ignores its `mask` arg** and
   returns the full **programmed `rxInitChannelMask`** set, as a flat indexable:
   `[ch0_I, ch0_Q, ch1_I, ch1_Q, ...]`, ascending bit order. With `rx_init_mask=0x3FF`
   → 20 arrays = 10 slots. Already integer-scaled (no `ScaleRx` needed).
   - Slot order: `Rx1,Rx2,Rx3,Rx4, ORxADC0, ORxADC1,` then 4 empty (`n=0`) placeholders
     (bits ORx3/ORx4/0x100/0x200 have **no converter** → always empty).
   - **Which channels carry data is profile-driven**; the count is `popcount(rxInitChannelMask)`.
   - `captureTime`/`timeout` are in **ms**. ORx runs at the Tx rate (491.52 MSPS in UC98)
     so its arrays are longer than the main-Rx arrays (245.76 MSPS).

3. **Channels return ZEROS unless enabled.** Capture gives noise/zeros until you call
   `RxTxEnableSet`. **`RxTxEnableSet(rxChannelMask, txChannelMask)`** sets both at once
   (absolute, not additive at the DLL level):
   ```
   rxMask bit0..3 = Rx1..Rx4,  bit4=ORx1, bit5=ORx2, bit6=ORx3, bit7=ORx4
   txMask bit0..3 = Tx1..Tx4
   ```
   To observe a TX on an ORx **you must enable that ORx input bit** (e.g. ORx2=0x20,
   ORx3=0x40), not just the main Rx.

4. **`PerformTx` requires exactly 8 arrays** ("one array for each Tx ADC") = 4 channels ×
   (I array, Q array), in `[Tx1_I,Tx1_Q,...,Tx4_I,Tx4_Q]` order. Zero-fill undriven
   channels (same length); `channelMask` selects who actually transmits. Signature:
   `PerformTx(txTrig, ArrayList txData, channelMask[, continuous])`.

5. **`TxAttenSet` is rejected before `Program()`** ("Invalid Tx attenuation control mode")
   — the atten mode isn't set until programming. So startup safe-state is **best-effort**
   (an unprogrammed device isn't transmitting anyway).

6. **Bit depth = profile `jesd204Np`** (UC98 = 12 → full scale `2^11 = 2048`; UC102 = 16).
   dBFS reference and quantization read this from the loaded profile — never hardcode.
   `pll_lock_status == 15` (0xF) means all 4 PLLs locked.

7. **`RxDecPowerGet` reads ~0 / `-0.00 dBFS` unless a measurement is armed / the channel
   is active.** Don't rely on it yet for ORx leveling — use the **captured-IQ peak**
   (`gain.clip_report`) instead until its arming sequence is confirmed.

8. **Program sequence** (faithful to the user's working IronPython init), all in
   `radio.program()`: `ConfigFileLoad(profile)` → `InitStructGet()` + edit clocks/masks/LO
   → build `PostMcsInit` (LO freqs, enable modes, tx→orx map, init cals) →
   `UtilityInitStructSet(post)` → `ConfigFileLoad()` (no-arg = default ARM/stream files,
   order matters) → `ClockConfig(*ad9528)` → `Program()` → apply atten/gain.

---

## 5. How the Python API is implemented

`src/adrvtrx/` (src layout). Hardware-free modules import without pythonnet:

| Module | Role |
|---|---|
| `config.py` | Typed dataclasses + TOML loader. Mirrors the init script 1:1 (`config/default.toml`). |
| `_enums.py` | Channel masks, trigger sources, and **name maps** to DLL enum members. Pure. |
| `_clr.py` | **Only** pythonnet module. `ClrBridge.load()` + array/enum helpers. Mockable. |
| `radio.py` | `Radio` context manager: connect, `program()`, crash-safe `force_safe`/`safe_state`, `enable_rx/tx`, gain/atten/LO/PLL wrappers, `perform_rx/perform_tx`. |
| `waveform.py` | Tab-delimited `I⟶Q` load, normalize (÷peak), quantize to Np, float-rescaled save. |
| `profile.py` | Read `jesd204Np` + sample rates from a `.profile` JSON. |
| `gain.py` | `clip_report` (peak dBFS, railed count), `peak_window`, software ORx leveling loop. |
| `capture.py` | `PerformRx` → per-channel IQ by **absolute slot index** (`returned_channel_order`), with a count-mismatch guard. |
| `transmit.py` | `PerformTx` 8-array builder (zero-fill), multi-band. |
| `bands.py` | `Band` primitive + single/dual/quad orchestration. |
| `sweep.py` | 1-D + nested-grid parameter sweeps, templated filenames. |
| `experiment.py` | `session()` convenience (connect+program+safe), `verify_status`. |
| `cli.py` | `adrvtrx-program` entry point. |

Design rules to preserve:
- **All .NET access goes through `_clr.ClrBridge`** so the rest stays unit-testable.
- **No magic numbers in code** — everything device-specific is config or read from the
  profile at runtime.
- **TX is forced safe on every exit path** (`__exit__`, `atexit`, SIGINT/SIGTERM) and on
  startup. `safe_state` = max atten + clear TX enable mask.

---

## 6. Testing

- **Hardware-free (default):** the .NET boundary is faked in `tests/conftest.py`
  (`FakeBridge` — `Types` factories produce `SimpleNamespace`; `device/board/link` are
  MagicMocks). Runs anywhere incl. CI. `pytest -m "not hardware"`.
- **On-hardware:** `tests/test_hardware.py`, marked `@pytest.mark.hardware`,
  **reachability-guarded** (skips if `192.168.1.10:55556` is unreachable — safe to run
  off-bench). Module-scoped connected+programmed session, TX safe on teardown.
- When you change a DLL call, also update the fake if needed, and keep the mocked tests
  green — they are the regression net for the logic.

---

## 7. Bench workflow & common errors

**Physical:** SD card in J6; ref clock 122.88 MHz @ 7 dBm to J613; ADS9 12 V at P1, S4 on,
wait for D3 LED red→off (~3 min = booted) then CE 12 V. PC NIC `192.168.1.2/24`, board
`192.168.1.10:55556`. **Put pads in any TX→ORx loopback** and terminate unused TX.

**Order:** `adrvtrx-program` (status, no TX) → `scripts/hw_smoke.py` (discovery + loopback,
transmits) → `pytest -m hardware`.

**Discovery-first methodology:** when behavior is unknown, **dump and measure, don't
assume.** `hw_smoke.py`'s `channel_powers`/`print_table` pattern (per-slot sample count +
RMS dBFS, at idle and per-TX) is how we reverse-engineered the channel layout. Add similar
probes rather than guessing struct/index semantics.

| Error / symptom | Meaning → fix |
|---|---|
| `Invalid Tx attenuation control mode` | `TxAttenSet` before `Program()`. Make it best-effort / program first. |
| `AttributeError: ... has no attribute 'IMMEDIATE'` | Passed Python enum name, not the DLL member. Use `_enums.*_MEMBER` maps. |
| `txData must have a count of 8` | `PerformTx` needs all 4 ch × I/Q. Zero-fill (`transmit.build_tx_data`). |
| All channels read `-inf` / zeros | Channels not enabled. `RxTxEnableSet` first (incl. the ORx bit). |
| ORx3/ORx4 (`idx 6/7`) always `n=0` | Only 2 ORx ADCs exist; those slots have no converter. Expected. |
| `PerformRx returned N arrays but mask implies M` | Profile vs `rx_init_mask` mismatch — set `[channels].rx_init_mask` to match the profile. |

---

## 8. Current state & open items (update as you go)

Confirmed working on hardware: connect, **program** (PLLs lock `0xF`, LO readback exact),
PerformRx readback + per-slot decode, PerformTx (8-array), channel enable.

**DONE — TX→ORx loopback / ORx capture model.** Confirmed on the bench and against ADI's
`rxDataCapture` sample (which unpacks the readback as `...,ORx1,ORx3`): there are **2 ORx
ADCs**, and the 4 ORx inputs mux into them in fixed pairs — **ORx1/ORx2 → ADC0 (readback
slot 0), ORx3/ORx4 → ADC1 (slot 1)**. Evidence: enabling ORx1/ORx2 lights slot 0 and zeros
slot 1; TX2→ORx2 lands on slot 0; ADI names the slots ORx1/ORx3. `capture.py` now resolves
an ORx request to its ADC slot (`_ORX_ADC_INDEX`) instead of the old bit-order position,
and `capture()` enables the requested ORx **input bit** (an ORx reads zeros until enabled).
All 8 `-m hardware` tests pass, incl. `TX2→ORx2` and `TX3→ORx3`.

**Open — TX3 cleanliness.** With TX3 transmitting, every readback slot reads a uniform
~−46.7 dBFS (`hw_smoke` flags this AMBIGUOUS), unlike TX2's clean single-slot tone. Looks
like a TX3/LO spur, not the loopback tone landing on one ADC. The `TX3→ORx3` loopback test
passes (slot 1 is above the floor) but the signal isn't clean — chase the spur (LO2 now at
1.0 GHz, see [lo] in config) before trusting TX3 ORx data.

Still bench-only: `RxDecPowerGet` arming, effective DAC full-scale vs Np, crash-recovery.

---

## 9. Working style with this user

- They drive the bench and paste real output — **read the actual numbers** (sample counts,
  dBFS) before concluding; that's where the truth is.
- Be honest about what's confirmed vs assumed; mark hardware-only seams clearly in code.
- Small, focused commits; push to `origin/main` after each fix (editable install means the
  user can immediately re-run). Keep `docs/api_notes.md` (the confirmed API) and this guide
  current as new behaviors are discovered.
