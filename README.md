# adrvtrx

Python automation for the **ADRV9026** transceiver on an **ADS9** motherboard via
the Analog Devices TES DLL (`adrvtrx_dll.dll`, namespace `adrv9010_dll`). Targets
repeatable **multi-band TX + synchronized ORx capture**, crash-safe, fully
parameterized from config.

> ⚠️ **OneDrive note.** This repo lives under a OneDrive-synced folder. OneDrive can
> corrupt `.git` during sync. Mitigate: OneDrive → Settings → *Sync and back up* →
> *Manage backup* / *Choose folders*, and **exclude the `.git` folder** (or pause
> sync during heavy git operations). Push to a remote for real backup.

## What it does

- **Programs** the device from `config/default.toml`, reproducing the working
  StdUseCase init script exactly — every register/mask/cal is a config value, no
  magic numbers in code.
- **Loads waveforms** from tab-delimited `I⟶Q` files, normalizes (÷ peak) and
  quantizes to the profile's `jesd204Np` bit depth.
- **Transmits** one waveform per TX path (single → dual → "quad" = 4 paths / 2 LOs)
  with looping `PerformTx`; all paths start together.
- **Captures** one sample-aligned snapshot (`PerformRx`) over any Rx **or** ORx
  channels, aligned to TX start-of-frame (`TXn_SOF` trigger). Saves normalized
  float `I⟶Q`.
- **Levels ORx** in software (flag-based: `RxDecPowerGet` mdBFS + overload
  indicators) — works without DPD. Reports clipping (peak dBFS, railed samples).
- **Sweeps** frequency / attenuation / gain — 1-D and nested grids — with templated
  filenames and an interactive (inspect-then-proceed) mode.
- Leaves **TX safe on every exit path** (context manager + `atexit` + signals) and
  forces a safe state on startup.

See [docs/api_notes.md](docs/api_notes.md) for the confirmed DLL API surface.

## Install

```bash
pip install -e ".[dev]"     # on the control PC (Windows). Pulls pythonnet.
pre-commit install
```

`pythonnet` is Windows-only and **not** needed for the test suite — the .NET
boundary is mocked, so unit tests run anywhere.

## Use

```bash
# Connect + program + print status (LO readback, PLL lock):
adrvtrx-program --config config/default.toml
```

```python
from adrvtrx import RxChannel, TxChannel, load_tab_iq
from adrvtrx.experiment import session
from adrvtrx.bands import make_bands, run_bands

with session() as (radio, info):                 # connects, programs, leaves TX safe
    wave = load_tab_iq("band_a.txt")
    bands = make_bands([
        ("band_a", TxChannel.TX1, wave, RxChannel.ORX1, 1.0),  # capture 1 ms on ORx1
    ])
    run_bands(radio, bands, tx_bits=info.tx_bits, rx_bits=info.rx_bits, out_dir="captures")
```

Nested sweep (frequency × attenuation), each point captured and saved:

```python
from adrvtrx.sweep import run_sweep, frequency_axis, attenuation_axis, format_filename

axes = [
    frequency_axis(radio, "LO2", [1_900_000_000, 2_000_000_000, 2_100_000_000]),
    attenuation_axis(radio, TxChannel.TX1, [10, 20, 30]),
]
def action(point):
    name = format_filename("cap_{lo_hz}_{atten_db}.txt", point)
    # ... transmit + capture + save under `name` ...
run_sweep(axes, action)
```

## Develop / CI

```bash
make lint      # ruff + black --check
make test      # hardware-free unit tests (this is local "CI")
make format    # auto-fix
nox            # lint + tests across 3.9 / 3.11
make test-hw   # ONLY on the control PC with ADS9 + ADRV9026 connected
```

GitHub Actions (`.github/workflows/ci.yml`) runs lint + mocked tests on push.

## Layout

```
src/adrvtrx/
  config.py      typed config + TOML loader (mirrors the init script)
  _clr.py        the ONLY module that touches pythonnet/.NET
  radio.py       context-managed driver: connect, program, safe-state, IO wrappers
  waveform.py    tab IQ load / normalize / quantize / float save
  profile.py     read jesd204Np + sample rates from a .profile JSON
  gain.py        clip report, peak window, software ORx leveling loop
  capture.py     PerformRx snapshot -> per-channel IQ, save
  transmit.py    PerformTx multi-band buffers
  bands.py       Band primitive + single/dual/quad orchestration
  sweep.py       1-D + nested-grid sweeps
  experiment.py  session() convenience + status
  cli.py         adrvtrx-program entry point
config/default.toml   all parameters (DLL path, board, profile, clocks, cals, levels)
docs/api_notes.md     confirmed DLL API (Task 0)
```

## Hardware bring-up checklist (first run on the bench)

These confirm the seams flagged in `docs/api_notes.md` that can only be verified
live (the code marks them):

1. `adrvtrx-program` connects, programs, LO readback matches config, PLL locked.
2. `RxDecPowerGet` sign/scale on an ORx channel with a known input level.
3. `PerformRx` readback container → finish `capture.extract_channels`.
4. `PerformTx` int packing (`packed` vs `interleaved`) → confirm spectrum is right.
5. Kill the process mid-capture → next startup `force_safe` leaves TX off, reconnect
   works without a power-cycle.
