# Notebooks

Interactive TX + acquisition walk-throughs for the ADRV9026 / ADS9 bench. Both
have been run end-to-end on hardware.

| Notebook | What it does |
|---|---|
| [single_band_tx_capture.ipynb](single_band_tx_capture.ipynb) | Transmit one waveform on a chosen TX, capture it on a chosen ORx, plot/save. |
| [dual_band_tx_capture.ipynb](dual_band_tx_capture.ipynb) | Transmit two waveforms on two TX channels and capture both ORx in one aligned snapshot. |

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
