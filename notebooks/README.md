# Notebooks

Interactive TX + acquisition walk-throughs for the ADRV9026 / ADS9 bench. Both
have been run end-to-end on hardware.

| Notebook | What it does |
|---|---|
| [single_band_tx_capture.ipynb](single_band_tx_capture.ipynb) | Transmit one waveform on a chosen TX, capture it on a chosen ORx, plot/save. |
| [dual_band_tx_capture.ipynb](dual_band_tx_capture.ipynb) | Transmit two waveforms on two TX channels and capture both ORx in one aligned snapshot. |
| [single_band_sweep.ipynb](single_band_sweep.ipynb) | Sweep TX atten (or LO2 freq) and auto-level the ORx ("AGC") at every step before capturing. |
| [dual_band_sweep.ipynb](dual_band_sweep.ipynb) | Same sweep, two bands; auto-levels each ORx independently per step. |

**ORx "AGC":** every notebook can auto-level the ORx in software (`USE_AGC=True`) on
the captured-IQ peak with a railed-sample clip veto (`gain.autolevel_orx` /
`capture.autolevel_capture`) — `RxDecPowerGet` is range-compressed and `RxGainGet`
returns 0, so the gain index is tracked in software. It starts at the gain floor and
trims into an asymmetric band (default −1.0 +0.3/−0.6 dBFS) within the valid ORx gain
window (185–255). It stops **fatally** if the TX is too strong even at the floor (185)
and **accepts max gain** (255) if the signal is too weak to reach the band (watch the
`leveled` column / red X's). Set `USE_AGC=False` to use a manual `ORX_GAIN_INDEX`.

## Running

Use the **`myenv`** conda environment as the kernel (it has the editable
`adrvtrx` package + numpy/matplotlib):

```powershell
C:\ProgramData\anaconda3\Scripts\conda.exe run -n myenv python -m ipykernel install --user --name myenv
```

Then open a notebook, edit the **Parameters** cell (profile, TX/ORx, signal path,
levels), and *Run All*. The session opens in the "Connect, program, verify" cell
and stays open across cells; the last cell forces TX safe and disconnects — run it
when you're done.

Defaults: profile `98_linksharing`, bench wiring `TX2->ORx2` / `TX3->ORx3`. The
"Signal-path summary" cell prints the sample rate + bit depth for each path so you
know what to prepare your TX waveform at.
