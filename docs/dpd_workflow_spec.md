# PA operating point, capture, and DPD workflow — specification

Status: **implemented** on `feat/pa-operating-point-dpd`. Changes since the reviewed
draft: `waveform_for` may return a path (loaded one row at a time) or an array;
`replay_conditions` takes `file_scale` for files not stored normalized; the
capture-point helper also exposes `condition_name` and `condition_keys`.

**`feat/gain-compression-lock`:** the search can lock on the gain compression at
the peaks instead of the PAPR compression (`lock_on="gain"`), with an optional
PA clip guard on the AM/AM top slope (`min_top_slope`). Every capture row and
every search step records PAPR compression, gain compression and top slope. The
PAPR lock stays the default, so existing notebooks behave as before.

This adds a PA characterization and DPD workflow to `adrvtrx`. It comes from the
ReplicatingScalability bench scripts (`compression_search.py`, `conditions.py`,
`dpd_playback.py` and the Single/Multi Operation notebooks).

---

## 1. Workflow

1. **Find the operating point.** Start at a TX attenuation where the PA barely
   compresses. Lower the attenuation until the compression reaches the target.
   That attenuation is the *lock*, which is input backoff 0 dB. The search locks
   on either the PAPR compression on the peak window (for example 3 dB, the
   default) or the gain compression at the peaks (for example 4 dB), and can stop
   short of PA clipping (§5.2).
2. **Capture.** At the lock, or at lock + backoff, capture the ORx, align it to
   the reference, and save the files plus one CSV row per capture. This can be a
   single point or a grid (LO × signal × backoff).
3. **Study (outside `adrvtrx`).** The user's model library reads the files and
   the CSV and produces DPD waveforms.
4. **Linearize.** Either:
   - **Offline replay:** play the DPD file for each CSV row at that row's saved
     condition, capture, and log the result.
   - **Online loop:** at one saved condition, repeat transmit → capture → align
     → user `step()` → next waveform.

`adrvtrx` never imports a model library (no `dpd_kit`, no torch). The only
interfaces are files and a Python callable.

**This PR is single-band.** Dual band is covered in §9.

---

## 2. Signals, units, and files

### 2.1 Signals

| Symbol | What it is | Where it comes from |
|---|---|---|
| **x** | Original reference, one period of N samples | the signal file, after `prepare_tx` (normalize + quantize to TX codes) |
| **y** | ORx output of the PA driven by x, aligned to x, N samples | initial capture |
| **u** | Waveform actually transmitted during linearization (a DPD output), N samples | DPD file or `step()` output, after round and clip to TX codes |
| **z** | ORx output of the PA driven by u, **aligned to u**, N samples | DUT capture |

N is the same for x, u, y and z. If it isn't, it's an error.

Why z is aligned to u: u is what the DAC played, so the alignment follows the
physical path. Any time shift that a DPD model introduces stays in z and counts
as error against x. The online loop also needs (u, z) aligned for training.

### 2.2 Units

- **On disk.** Every IQ file is normalized float `I<TAB>Q`, where 1.0 is full
  scale `2**(bits-1) - 1`. TX files use `tx_bits` and ORx files use `rx_bits`.
  This is the existing `save_tab_iq_float` convention.
- **In memory, hardware side.** Captures are integer codes (float64 complex).
  References are the TX codes returned by `prepare_tx`.
- **Transmitting a stored waveform** (replay, online loop). Multiply by
  `full_scale(tx_bits)`, then `prepare_tx(..., do_normalize=False)`, which only
  rounds and clips. Nothing rescales it: no peak normalize, no power change.
- **dBFS reference.** `2**(bits-1)`, the same as `clip_report`.

Legacy note: the DPD files in `ReplicatingScalability/DPD/gmp` are integer codes
(×2048). Replay them with `replay_conditions(..., file_scale=2048)`. New DPD
files are written normalized (`file_scale=1.0`, the default).

### 2.3 File names

| File | Name |
|---|---|
| reference | `{signal_stem}_in.txt`, written once per signal |
| initial capture | `{name}_out.txt`, where `name = {TX}_{backoff}dB_{freq_MHz}MHz_{bw}BW` |
| capture CSV | `{TX}_conditions.csv` |
| DPD waveform (user-written) | anything; the replay takes a `waveform_for(row)` callable |
| DUT capture | `{name}_{label}_dut.txt`, where `label` defaults to `dpd_{model}` |
| DUT CSV | `{label}.csv` |

---

## 3. Metrics

All of these are pure numpy in `adrvtrx.metrics`. Unless stated otherwise, each
is computed on **one full aligned period**.

| Metric | Definition | Reference → measured |
|---|---|---|
| `papr_db(s)` | `10·log10(max|s|² / mean|s|²)`. NaN if `s` is empty or all zero | — |
| `window_compression_db(x, s, fs, window_s)` | `papr_db(x[w]) − papr_db(s[w])`, where `w` is `round(window_s·fs)` samples centred on `argmax|x|` and clamped inside the period. Default `window_s = 0.1 ms` (49 152 samples at 491.52 MHz). Returns `(comp, papr_in, papr_out)` | x → s |
| `gain_compression_db(x, s)` | `g0` = least-squares complex gain on the samples with `|x| < 0.2·max|x|` (small-signal gain). `a = |x|/max|x|`, `b = |s/g0|/max|x|`. **Gain compression** = `−10·log10( mean(b²) / mean(a²) )` over the samples with `a ≥ 0.975`: the power gain at the peaks relative to the small-signal gain, in dB, positive when compressed. **Top slope** = straight-line fit of the per-bin mean of `b` against the per-bin mean of `a` (40 bins of `a` on [0, 1]) over the bins centred in (0.8, 0.95): 1 for a linear PA, 0 for a flat top. Scale and phase invariant. Returns `(gain_compression_db, top_slope)`. A chain figure (TX → PA → ORx) measured on the signal, so it reads somewhat above a CW compression point; the peak region is relative to the signal's own maximum | x → s |
| `pa_clipped` | `top_slope < min_top_slope`, default `PA_CLIP_SLOPE = 0.08`. A post-inverse needs about `1/top_slope` of gain at the peaks; 0.08 sits between the clipped TX1 2.8 GHz / 100 MHz capture at 0 dB (0.03) and the lowest capture that still inverted well (0.10) | x → s |
| `nmse_db(x, s)` | `g = Σ conj(x)·s / Σ|x|²` (least-squares complex gain over **all** samples). NMSE = `10·log10( Σ|s/g − x|² / Σ|x|² )`. Invariant to scale and phase | x → s |
| `aclr_db(s, fs, bw_hz)` | One FFT of `s·hann(N)`, fftshifted, `P = |S|²`, bins `f ∈ [a, b)`. Main = `[−bw/2, bw/2)`, lower = `[−1.5bw, −0.5bw)`, upper = `[0.5bw, 1.5bw)`. Each result is `10·log10(P_adj / P_main)` in dBc. `bw_hz` is the nominal signal bandwidth (40, 100, … MHz) | s alone |
| `inband_corr(a, b)` | On an **already aligned** pair: `A = fft(a)`, `B = fft(b)`, keep bins where `|A| > 0.05·max|A|`, then `|⟨A,B⟩| / (‖A‖·‖B‖)`. 1.0 means a perfect in-band copy. Shares its core with `align.match_corr`, which also aligns first | a → b |
| `rms_dbfs(s_codes, bits)` | `10·log10( mean|s|² / (2**(bits−1))² )`, using the complex magnitude | s alone |
| `peak_dbfs`, `railed` | From `clip_report` on the **raw capture** (two periods, before alignment). Peak uses `max(|I|, |Q|)`, because that is what rails the ADC. `railed` counts samples at `2**(bits−1) − 1` | raw ORx |
| `delay_samples` | From `estimate_and_align`: where the aligned period starts inside the capture, with a fractional part. **The integer part is arbitrary** (IMMEDIATE trigger on a looping waveform). Only the fractional part carries the sub-sample path delay | template → capture |
| `delay_ns` | `delay_samples / fs · 1e9`. Same caveat as `delay_samples` | — |

Peak minus RMS is **not** PAPR. Peak is per rail and RMS is complex magnitude.

`fs` is always the ORx rate, `ProfileInfo.orx_rate_hz`. TX and ORx share
491.52 MSPS on profile 98.

---

## 4. CSV schemas

### 4.1 Capture CSV (`{TX}_conditions.csv`)

One row per capture. The original columns keep their order and meaning; four
columns are appended at the end (`lock_on`, `gain_compression_db`,
`amam_top_slope`, `pa_clipped`). They have defaults (`papr`, NaN, NaN, false),
so files written before them load as they are. The definition of `nmse_db` is
normalized by x instead of y, a difference of less than 0.01 dB.

| Column | Meaning |
|---|---|
| `name` | `{TX}_{backoff}dB_{freq_MHz}MHz_{bw}BW` |
| `signal` | source signal file name |
| `freq_hz` | LO1 |
| `bw_mhz` | nominal bandwidth, used for ACLR |
| `backoff_db` | backoff from the lock, in dB (more attenuation) |
| `locked_atten_db` | the lock: attenuation where the search converged |
| `tx_atten_db` | attenuation for this capture: `min(locked + backoff, 41.95)` |
| `orx_gain` | ORx gain index after the AGC for this capture. Replay uses it as is |
| `compression_db` | the compression the **search** locked on (`lock_on`), measured **at the lock**. The same value repeats on every backoff row |
| `converged` | whether the search converged |
| `delay_samples`, `delay_ns`, `corr` | alignment of y to x; `corr = inband_corr(x, y)` |
| `nmse_db` | `nmse_db(x, y)` |
| `papr_in_db`, `papr_out_db`, `papr_compression_db` | `papr(x)`, `papr(y)`, and their difference (full period) |
| `aclr_lower_dbc`, `aclr_upper_dbc` | `aclr_db(y)` |
| `peak_dbfs`, `rms_dbfs`, `railed` | ORx levels (§3) |
| `in_file`, `out_file` | x file and y file, relative to the CSV folder |
| `lock_on` | `papr` or `gain`: the metric `compression_db` is |
| `gain_compression_db`, `amam_top_slope` | `gain_compression_db(x, y)` for **this** capture (§3) |
| `pa_clipped` | `amam_top_slope < 0.08` (or the `min_top_slope` passed to `capture_point`) |

### 4.2 DUT CSV (`{label}.csv`)

One row per replayed condition. **x is the reference for every quality metric**,
and z is aligned to u.

| Column | Meaning |
|---|---|
| condition columns | `name` through `converged`, and `lock_on`, copied from the source capture row. `orx_gain` and `tx_atten_db` are what was applied |
| `delay_samples`, `delay_ns` | alignment of z to **u** |
| `corr` | `inband_corr(u, z)`: alignment quality against what was transmitted |
| `nmse_db` | `nmse_db(x, z)`, the original input against the DUT |
| `papr_in_db` | `papr(x)`, the **original** input |
| `papr_out_db` | `papr(z)` |
| `papr_compression_db` | `papr(x) − papr(z)`: compression left after DPD |
| `aclr_lower_dbc`, `aclr_upper_dbc` | `aclr_db(z)` |
| `peak_dbfs`, `rms_dbfs`, `railed` | ORx levels of the DUT capture. With DPD the output peak rises while the ORx gain stays fixed, so a nonzero `railed` is possible and is reported, not corrected |
| `papr_dpd_db` | `papr(u)` |
| `papr_expansion_db` | `papr(u) − papr(x)` |
| `tx_peak_dbfs` | `20·log10( max(|I_u|, |Q_u|) / 2**(tx_bits−1) )`, before clipping |
| `tx_clipped` | samples of u where `max(|I|, |Q|)` exceeds full scale before the clip |
| `ref_file` | x file (the source row's `in_file`) |
| `in_file` | u file (what was transmitted) |
| `out_file` | z file |
| `gain_compression_db`, `amam_top_slope`, `pa_clipped` | `gain_compression_db(x, z)`: the compression left in the linearized chain (about 0 dB and a slope near 1 for a good DPD) |

Improvement figures (ΔNMSE, ΔACLR) are **not** stored. The analysis joins the two
CSVs on `name`.

---

## 5. API

New modules: `metrics`, `compression`, `conditions`, `replay`, `linearize`.
All new public names are exported from `adrvtrx/__init__.py`. Hardware
functions follow the existing rules: TX must already be running where stated,
nothing reads the ORx gain back, and TX is disabled in a `finally` block.

### 5.1 `adrvtrx.metrics` (pure)

```python
papr_db(s) -> float
window_compression_db(x, s, fs, window_s=1e-4) -> tuple[float, float, float]
gain_compression_db(x, s, *, peak_frac=0.975, small_frac=0.2,
    slope_band=(0.8, 0.95), n_bins=40) -> tuple[float, float]   # (compression, top slope)
PA_CLIP_SLOPE = 0.08
nmse_db(x, s) -> float
aclr_db(s, fs, bw_hz) -> tuple[float, float]
inband_corr(a, b, energy_frac=0.05) -> float
rms_dbfs(s_codes, bits) -> float
period_metrics(x, s, *, fs, bw_hz, rx_bits, delay_samples, corr_ref=None,
    min_top_slope=PA_CLIP_SLOPE) -> dict
    # delay_samples, delay_ns, corr, nmse_db, papr_in_db, papr_out_db,
    # papr_compression_db, aclr_lower_dbc, aclr_upper_dbc, rms_dbfs,
    # gain_compression_db, amam_top_slope, pa_clipped
    # corr is inband_corr(corr_ref or x, s)
```

### 5.2 `adrvtrx.compression`

```python
@dataclass
class CompressionResult:
    converged: bool
    final_atten_db: float
    final_orx_gain: int
    compression_db: float
    peak_dbfs: float
    railed: int
    iterations: int
    reason: str
    fatal: bool = False
    at_atten_floor: bool = False
    clip_limited: bool = False   # stopped by the PA clip guard
    lock_on: str = "papr"        # what compression_db is
    papr_compression_db: float   # readings at the result (NaN if measure
    gain_compression_db: float   #   does not report them)
    top_slope: float
    history: list[dict]          # one dict per measurement (see below)

search_tx_compression(set_tx_atten, set_orx_gain, measure, *,
    target_compression_db, start_atten_db, atten_min_db, orx_gain,
    coarse_step_db=10.0, fine_step_db=0.25, comp_tol_db=0.3,
    target_dbfs=-1.0, tol_up_db=0.3, tol_down_db=0.6,
    gain_min=185, gain_max=255, db_per_index=0.5,
    atten_max_db=41.95, max_iterations=24, min_top_slope=None,
    on_step=None) -> CompressionResult
    # pure; measure() -> (peak_dbfs, railed, compression_db)
    #   or (peak_dbfs, railed, compression_db, extra: dict)

find_compression_point(radio, tx, orx, ref, *, tx_bits, rx_bits, fs,
    target_compression_db, start_atten_db, atten_min_db,
    lock_on="papr", min_top_slope=None,
    window_s=1e-4, oversample=2, <all search + AGC tolerances>,
    on_step=None) -> CompressionResult
    # sets start_atten_db, runs autolevel_capture (full-length verify), then the
    # search with measure = capture(oversample·N) -> clip_report -> align to ref
    # -> window_compression_db (peak window) and gain_compression_db (whole
    # period); compression_db is the one named by lock_on. TX must already be
    # running. Leaves the radio at the result's attenuation and gain.
```

**Search algorithm** (today's `compression_search.py`, unchanged):

- **Trusting a measurement.** The compression reading only counts when the ORx
  peak is inside `[target − tol_down, target + tol_up]` dBFS with
  `railed == 0`. It also counts when the peak is below that band at maximum ORx
  gain (the best achievable level).
- **ORx not leveled.** If the capture rails or is too hot, lower the gain (one
  index for rails only, otherwise a computed step). If it is too cold, raise the
  gain by a computed step. The attenuation does not change.
- **Converged.** Leveled and `|comp − target| ≤ comp_tol_db`.
- **Too little compression.** Lower the attenuation. The first step is
  `coarse_step_db`. Later steps are the largest multiple of `fine_step_db` that
  does not exceed the error, capped at `coarse_step_db`. The ORx gain drops by
  `round(Δatten / db_per_index)` in the same step.
- **Too much compression.** Raise the attenuation by a fine step and raise the
  ORx gain by the same amount.
- **Stops:**
  - Attenuation floor reached (`at_atten_floor`).
  - Attenuation ceiling reached.
  - ORx rails at the gain floor. The attenuation is raised one fine step and the
    result is `fatal`.
  - `max_iterations`. The radio is left at the closest trusted point.
- **PA clip guard** (only with `min_top_slope` set). A trusted reading whose
  `extra["top_slope"]` is below `min_top_slope` is PA clipping. The attenuation
  goes up by the remaining error (at least one fine step), and that attenuation
  becomes the new floor. If the target needs more drive than the floor allows,
  the search stops there with `clip_limited`, at the last unclipped point.
- The attenuation is quantized to 0.05 dB and never goes below `atten_min_db`.
- `history` rows contain `atten_db, orx_gain, peak_dbfs, railed,
  compression_db, orx_ok, action`, plus every key of `measure`'s extra dict
  (`papr_compression_db, gain_compression_db, top_slope` from
  `find_compression_point`) and `pa_clipped` when the guard is on.

### 5.3 `adrvtrx.conditions`

```python
CSV_FIELDS: tuple[str, ...]              # §4.1, legacy order, new columns last
DUT_FIELDS: tuple[str, ...]              # §4.2

@dataclass
class OperatingCondition: ...            # §4.1, to_row / from_row as today
@dataclass
class DutRecord: ...                     # §4.2

class ConditionLog:                      # CSV writer, flushes every row
    def __init__(self, path, fields=CSV_FIELDS): ...
    def append(self, record): ...        # OperatingCondition or DutRecord

load_conditions(path) -> list[OperatingCondition]
load_dut_records(path) -> list[DutRecord]
save_aligned_iq(iq_codes, path, bits)    # codes / full_scale(bits)
load_aligned_iq(path) -> np.ndarray      # normalized complex

@dataclass
class PointCapture:
    y_aligned: np.ndarray                # codes, N samples
    delay_samples: float
    clip: ClipReport                     # raw capture
    metrics: dict                        # period_metrics(ref, y_aligned, ...)

capture_point(radio, orx, ref, *, rx_bits, fs, bw_hz, oversample=2,
              metric_ref=None) -> PointCapture
    # capture oversample·N -> clip_report -> estimate_and_align(ref, ·)
    # -> period_metrics(metric_ref or ref, y_aligned, corr_ref=ref).
    # TX must already be running. Gain and attenuation are not touched.
```

### 5.4 `adrvtrx.replay` (offline linearization)

```python
replay_conditions(radio, csv_path, waveform_for, *, tx, orx, tx_bits, rx_bits,
    fs, out_dir, label="dpd", oversample=2, lo="LO1", file_scale=1.0,
    on_row=None) -> Path
    # waveform_for(row: OperatingCondition) -> path to an I<TAB>Q file holding
    #     u * file_scale, or a normalized complex ndarray (u)
    # returns the DUT CSV path
```

Return paths for large runs: a path is loaded when its row plays, while arrays
are all held in memory from the pre-flight on. An array is also written to
`out_dir` as `{name}_{label}.txt` so every DUT row points at a file.

For each row, sorted by `(freq_hz, bw_mhz, backoff_db)`:

1. **Before starting**, confirm that every x file (`row.in_file`) exists, that
   `waveform_for` succeeds for every row, and that every returned path exists.
   Otherwise raise and list everything that is missing. Length N is checked when
   the row plays.
2. If `freq_hz` changed, disable TX and call `retune_lo("LO1", freq)`.
3. Call `set_tx_atten(tx, row.tx_atten_db)` and `set_rx_gain(orx, row.orx_gain)`.
   **No AGC.**
4. `stored_tx(u, tx_bits)`: `u · full_scale(tx_bits)`, rounded and clipped,
   the same as `prepare_tx(..., do_normalize=False)`. `tx_clipped` and
   `tx_peak_dbfs` are counted before the clip.
5. `transmit_bands(radio, {tx: u_codes}, tx_bits, do_normalize=False)`.
6. `capture_point(radio, orx, ref=u_codes, metric_ref=x_codes)`, so z is aligned
   to u and the metrics are taken against x.
7. Save `{name}_{label}_dut.txt` and append a `DutRecord`.

TX is disabled in `finally`. Each row is flushed as soon as it is written.

### 5.5 `adrvtrx.linearize` (online linearization)

```python
StepFn = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray | None]
    # step(x, u, z, it) -> next u, or None to stop.
    # All three are normalized floats, length N, with z aligned to u.

@dataclass
class LinearizeResult:
    records: list[DutRecord]             # one per iteration (iteration 0 = u0)
    u: np.ndarray                        # last transmitted waveform (normalized)
    z: np.ndarray                        # last aligned DUT capture (normalized)
    reason: str                          # "n_iter" | "step returned None" | "railed" | ...

linearize(radio, condition, x, step, *, tx, orx, tx_bits, rx_bits, fs,
    n_iter=5, u0=None, stop_on_rail=True, save_dir=None, label="ila",
    on_iter=None) -> LinearizeResult
```

1. Apply `condition` once: LO, `tx_atten_db`, `orx_gain`. **No AGC.** The same
   gain is kept for every iteration so the iterations are comparable.
2. Set `u = u0` (or x if `u0` is None).
3. For `it = 0 … n_iter−1`:
   1. Transmit u (as in §5.4, step 4).
   2. `capture_point(ref=u_codes, metric_ref=x_codes)`.
   3. Build a `DutRecord`.
   4. If `railed > 0` and `stop_on_rail`, stop.
   5. `u = step(x, u, z, it)`. If it returns None, stop.
   `step` is not called after the last capture.
4. TX is disabled in `finally`. If `save_dir` is set, write
   `{name}_{label}_it{k}_u.txt`, `{name}_{label}_it{k}_z.txt` and `{label}.csv`
   (DUT columns plus `iteration`).

The example `step` in `dpd_linearize_loop.ipynb` scales like the offline study:
it peak-normalizes `z`, rotates it onto `x`, fits the post-inverse `z → u` and
predistorts `x` with its peak 0.2 dB below the training peak. Normalizing `z` by
its least-squares gain instead asks the PA for peaks above saturation, and the
fitted inverse diverges.

The DPD algorithm (ILA, DLA, one-shot fit) is entirely the user's `step`.
Nothing model-specific lives in `adrvtrx`.

---

## 6. Sample notebooks (`notebooks/`)

| Notebook | Replaces | What it does |
|---|---|---|
| `pa_operating_point.ipynb` | SingleOperationCompressionSweep | One LO and one signal. Runs `find_compression_point`, captures at the lock with `capture_point`, and writes one CSV row plus files. Plots the search history and the peak window |
| `pa_operating_sweep.ipynb` | MultiOperationCompressionSweep | Loops over signals × LOs. Runs the search per pair, then captures each backoff with an AGC before every one. Writes the capture CSV. The loop stays in the notebook so the procedure is visible |
| `dpd_replay.ipynb` | MultiOperationDpdPlayback | `replay_conditions` with a `waveform_for` that loads the user's DPD files |
| `dpd_linearize_loop.ipynb` | new | `linearize` at one condition, with an example ILA `step` written in the notebook using `dpd_kit` GMP (the path is a parameter) |

Each notebook has the usual structure: parameters, imports/config/profile,
connect/program, run, plots, then safe-state and disconnect.

---

## 7. Offline tests (`tests/`, no hardware, run in CI)

**Simulated bench (`tests/sim_bench.py`).** A fake radio that implements
`perform_tx` and `perform_rx` plus the attenuation, gain and LO setters, so the
real `transmit_bands`, `capture` and ORx AGC run unchanged against it. It
models:

- The last transmitted codes are played in a loop.
- The PA is a memoryless Rapp model, driven by `10^(−atten/20)`.
- A fixed fractional path delay.
- A random start offset inside the loop for every capture.
- The ORx gain is `(gain − 210)·0.5` dB.
- The output is quantized to `rx_bits` and clipped at the rail.

| File | Covers |
|---|---|
| `test_metrics.py` | PAPR of a constant envelope (0 dB) and of a known spike. NMSE is invariant to scale and phase and ≈ −SNR for `g·x + noise`. ACLR on synthetic band-limited noise plus a known adjacent tone. `window_compression_db` sign and magnitude on Rapp. `inband_corr` = 1 for a scaled copy. Empty and zero inputs. `gain_compression_db`: 0 dB and slope 1 for a linear PA, matches the Rapp value, slope falls with drive, a hard clipper has slope 0 and is flagged, scale and phase invariant, repeats within 0.2 dB across noisy captures |
| `test_compression_search.py` | Converges within ±tol on the Rapp model. Stops at the attenuation floor. Stops at the ceiling. Fatal when the ORx rails at the gain floor. ORx gain moves by Δatten/0.5 per step. Attenuation quantized to 0.05 dB. Returns the best point at max iterations. Gain lock converges and records every reading. The clip guard stops a hard-limiting PA at the last unclipped attenuation; without it the search clips. `find_compression_point(lock_on="gain")` on the sim bench |
| `test_conditions.py` | CSV round trip for both schemas. Loads a legacy `TX1_conditions.csv` header and row, with defaults for the new columns. New columns round-trip. IQ file round-trip scale. Missing-column error |
| `test_capture_point.py` | On the sim bench, recovers the fractional delay to within 0.02 samples. NMSE and corr are sane |
| `test_find_compression_point.py` | On the sim bench, lands within ±tol and leaves the radio at the result |
| `test_replay.py` | Pre-flight lists missing files and wrong lengths. Retunes only when the frequency changes. Applies the saved attenuation and gain without AGC. Transmits u as stored (asserts the codes). z is aligned to u and the metrics are against x. `tx_clipped` is counted. TX is disabled even if the replay raises |
| `test_linearize.py` | With an ideal inverse-Rapp `step`, NMSE improves after iteration 0. `None` stops the loop. Stops on rail. Files and CSV written. TX disabled on exception |

`make lint` (ruff + black, line length 100, py39) and `make test` must pass.

---

## 8. Moving the ReplicatingScalability code

| From | To | Changes |
|---|---|---|
| `compression_search.py` | `adrvtrx/compression.py` | Takes the ORx constants from `adrvtrx.gain` and the TX attenuation max from `radio.MAX_TX_ATTEN_DB`. Adds `find_compression_point`. `_simulate` moves to the tests |
| `conditions.py` | `adrvtrx/conditions.py` + `adrvtrx/metrics.py` | The metrics move to `metrics`. Adds `DutRecord`, `capture_point` and `load_dut_records` |
| `dpd_playback.py` | `adrvtrx/replay.py` | The hard-coded OneDrive `base` import goes away. `waveform_for` replaces the fixed `DPD/gmp` layout. Adds the new DUT columns |
| `_repair_dpd_papr.py` | not moved | One-off fix for the current GMP run |

ReplicatingScalability can switch to the library imports afterwards. Its
existing CSVs keep loading.

---

## 9. Out of scope for this PR

- **Dual band.** The plan is one CSV row per band with a `point_id` column,
  plus ACLR with `center_hz` for bands away from DC. The search rule depends on
  whether both bands share one PA. A separate PR will decide that.
- EVM, and storing AM/AM and AM/PM curves.
- Re-levelling the ORx during replay or the online loop.

---

## 10. Review checklist

1. §2.2: all files normalized, with a legacy divide-by-2048 in the replay notebook.
2. §3: metric definitions, especially the NMSE gain and the `delay_samples` caveat.
3. §4: capture CSV columns unchanged, and the DUT CSV additions.
4. §5.4 and §5.5: no AGC, and z aligned to u with metrics against x.
5. §5.5: the `step(x, u, z, it)` signature and the stop rules.
6. §6: sweep loop kept in the notebook, not a library function.
7. §9: dual band deferred.
