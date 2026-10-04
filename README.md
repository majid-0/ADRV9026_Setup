# adrvtrx

Python automation for the **ADRV9026** transceiver on an **ADS9** motherboard via
the Analog Devices TES DLL (`adrvtrx_dll.dll`, namespace `adrv9010_dll`). Targets
repeatable **multi-band TX + synchronized ORx capture**, crash-safe, fully
parameterized from config.

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
- **Levels ORx** in software — an **ORx AGC** (`autolevel_capture`) on the captured-IQ
  peak with a railed-sample clip veto, targeting an asymmetric dBFS band. ORx has no
  hardware AGC and its overload flags / gain readback are unusable, so the gain index
  is tracked in software. Reports clipping (peak dBFS, railed samples).
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

Declarative sweep plan (notebooks; per-block zip/grid, preview before run):

```python
from adrvtrx.sweep_plan import summarize_sweep_plan, run_planned_sweep, sweep_defaults_from_config

SWEEP = {
    "freq": {"mode": "zip", "lo1_hz": [1.0e9, 1.1e9], "lo2_hz": [0.9e9, 1.0e9]},
    "power_db": {"mode": "grid", "shared": [13, 14, 15]},
}
defaults = sweep_defaults_from_config(cfg, BANDS)
print(summarize_sweep_plan(BANDS, SWEEP, defaults))
records = run_planned_sweep(radio, BANDS, SWEEP, action, defaults=defaults, tx_bits=info.tx_bits)
```

PA operating point and DPD (single band; full definitions in
[docs/dpd_workflow_spec.md](docs/dpd_workflow_spec.md)):

```python
from adrvtrx.compression import find_compression_point
from adrvtrx.conditions import capture_point
from adrvtrx.replay import replay_conditions
from adrvtrx.linearize import linearize

# TX running: search down from a safe attenuation to 3 dB PAPR compression.
res = find_compression_point(radio, TxChannel.TX1, RxChannel.ORX1, ref, rx_bits=12, fs=fs,
                             target_compression_db=3.0, start_atten_db=20, atten_min_db=9)
# Or lock on 4 dB of gain compression at the peaks, and never let the PA clip:
res = find_compression_point(radio, TxChannel.TX1, RxChannel.ORX1, ref, rx_bits=12, fs=fs,
                             target_compression_db=4.0, comp_tol_db=0.2, lock_on="gain",
                             min_top_slope=0.08, start_atten_db=20, atten_min_db=9)
point = capture_point(radio, RxChannel.ORX1, ref, rx_bits=12, fs=fs, bw_hz=100e6)

# Offline: play your model's DPD file at every saved condition (no AGC, no rescale).
replay_conditions(radio, "captures/TX1_conditions.csv", lambda row: f"DPD/{row.name}.txt",
                  tx=TxChannel.TX1, orx=RxChannel.ORX1, tx_bits=12, rx_bits=12, fs=fs,
                  out_dir="DPD_DUT/gmp", label="dpd_gmp")

# Online: step(x, u, z, it) -> next u, or None to stop. Your model lives in step.
linearize(radio, condition, x, step, tx=TxChannel.TX1, orx=RxChannel.ORX1,
          tx_bits=12, rx_bits=12, fs=fs, n_iter=5)
```

Live DPD in one call chain (what `notebooks/dpd_linearize_loop.ipynb` does): find
the operating point, then iterative ILA with the built-in numpy GMP. No DPD
waveform with a peak above `peak_limit_dbm` is ever transmitted (full scale =
the original input peak = 10 dBm); the input backoff grows from 0.2 dB until the
peak fits, and the loop stops if 4 dB is not enough.

```python
from adrvtrx.operating_point import find_operating_point
from adrvtrx.gmp import GMP
from adrvtrx.dpd import IlaStep, iteration_table

op = find_operating_point(radio, TxChannel.TX1, RxChannel.ORX1, signal, tx_bits=12, rx_bits=12,
                          fs=fs, freq_hz=2_400_000_000, bw_mhz=100, target_compression_db=3.0,
                          start_atten_db=20, atten_min_db=9, save_dir="DPD_DUT/ila_gmp")
step = IlaStep(lambda: GMP(5, 5, 2), n_train=8192, peak_limit_dbm=9.9)
res = linearize(radio, op.condition, op.x, step, tx=TxChannel.TX1, orx=RxChannel.ORX1,
                tx_bits=12, rx_bits=12, fs=fs, n_iter=4)
for row in iteration_table(res.records, step.history):
    print(row["iteration"], row["aclr_lower_dbc"], row["aclr_upper_dbc"], row["dpd_peak_dbm"])
```

How it works, with diagrams: [docs/dpd_pass.md](docs/dpd_pass.md) (one DPD pass,
from the captured signals to the next waveform) and
[docs/linearize_notebook.md](docs/linearize_notebook.md) (the whole notebook).
`adrvtrx` imports no external model library; the GMP is the reference DPD.

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
  gain.py        clip report, peak window, software ORx AGC (autolevel_orx / verify_no_clip)
  capture.py     PerformRx snapshot -> per-channel IQ, save, autolevel_capture AGC
  transmit.py    PerformTx multi-band buffers
  bands.py       Band primitive + single/dual/quad orchestration
  sweep.py       Low-level SweepAxis + run_sweep
  sweep_plan.py  Declarative multi-band sweep plans + summarize_sweep_plan
  metrics.py     PAPR, window / gain compression, AM/AM top slope, NMSE, ACLR, corr, RMS
  compression.py TX attenuation search for a target PAPR or gain compression
  operating_point.py  transmit, lock, backoff, capture and CSV row in one call
  conditions.py  Condition / DUT CSVs, aligned IQ files, capture_point
  replay.py      Replay stored waveforms (DPD files) at saved conditions
  linearize.py   Online transmit -> capture -> step() loop
  gmp.py         Generalized memory polynomial, block least squares (numpy)
  dpd.py         Peak units, normalize_pair, peak limit, ILA step for the loop
  experiment.py  session() convenience + status
  cli.py         adrvtrx-program entry point
config/default.toml   all parameters (DLL path, board, profile, clocks, cals, levels)
docs/api_notes.md     confirmed DLL API (Task 0)
docs/dpd_workflow_spec.md   operating point, CSVs, replay and the DPD loop (spec)
docs/dpd_pass.md            one DPD pass, step by step
docs/linearize_notebook.md  the live DPD notebook, step by step
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
