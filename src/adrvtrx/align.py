"""Time-alignment + delay-estimation helpers (opt-in; nothing here is enforced).

These let a caller recover a transmitted reference from an ORx capture, time-aligned,
without any hardware trigger: capture a window LONGER than the reference (the user
picks how much; ~2x is plenty), then locate the reference inside it by correlation.

Ported from the user's standalone ``sampling.py`` with one fix (see ``estimate_delay``):
when the capture is longer than the reference we *slide the reference through it*
(valid-style search) instead of trimming both to the shorter length. Trimming chops
a long capture down to one arbitrary-phase period -> circular wrap -> the signals
decorrelate. The fix is what makes the "capture 2x then align" workflow actually work.

Pure numpy/scipy -- no hardware, unit-testable.
"""

from __future__ import annotations

import numpy as np

__all__ = ["estimate_delay", "estimate_and_align", "apply_delay", "match_corr"]


def match_corr(ref, y, fs=1.0, energy_frac=0.05):
    """Per-band match quality: time-align ``y`` to ``ref``, then correlate over only
    the reference's occupied frequency bins.

    Restricting to the reference's band rejects out-of-band energy -- receiver noise,
    and in dual-band the *other* band that sits at the LO offset inside the wideband
    ORx capture. A clean band gives ~1.0; the full-band correlation would instead be
    diluted (e.g. ~0.7 when a second equal-power band shares the capture). Returns a
    float in [0, 1]. ``energy_frac`` sets the "occupied" threshold vs the peak bin.
    """
    xa, ya, _ = estimate_and_align(ref, y, fs)
    m = min(len(xa), len(ya))
    if m == 0:
        return 0.0
    a = np.fft.fft(np.asarray(xa[:m], dtype=np.complex128))
    b = np.fft.fft(np.asarray(ya[:m], dtype=np.complex128))
    mask = np.abs(a) > energy_frac * np.abs(a).max()
    am, bm = a[mask], b[mask]
    den = (np.linalg.norm(am) * np.linalg.norm(bm)) or 1.0
    return float(np.abs(np.vdot(am, bm)) / den)


def _apply_fractional_delay_fft(x, frac_delay):
    """Apply a fractional-sample delay via FFT phase ramp. Positive=delay."""
    x = np.asarray(x)
    n = len(x)
    if n == 0 or abs(frac_delay) < 1e-12:
        return x.copy()
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(n)
    phase = np.exp(-1j * 2 * np.pi * freqs * frac_delay)
    out = np.fft.ifft(X * phase)
    if np.iscomplexobj(x):
        return out.astype(x.dtype, copy=False)
    return np.real(out).astype(x.dtype, copy=False)


def _shift_zero_fill(x, shift):
    """Integer shift with zero-fill. Positive delays (y lags); negative advances."""
    x = np.asarray(x)
    n = len(x)
    y = np.zeros_like(x)
    k = int(shift)
    if k == 0:
        return x.copy()
    if k > 0:
        if k < n:
            y[k:] = x[:-k]
    else:
        k = -k
        if k < n:
            y[:-k] = x[k:]
    return y


def _force_length(x, length):
    x = np.asarray(x)
    if len(x) == length:
        return x.copy()
    if len(x) > length:
        return x[:length].copy()
    out = np.zeros(length, dtype=x.dtype)
    out[: len(x)] = x
    return out


def apply_delay(x_ref, y_aligned, delay_samples, upsample_factor=10):
    """Apply a controllable (fractional) delay to ``y_aligned`` relative to ``x_ref``.

    Both inputs must be the same length and already aligned at delay 0. Positive
    delay => y lags x. Sub-sample delays use upsample -> integer shift -> decimate
    (resolution 1/L). Mainly used to validate the estimator against a known delay.
    """
    from scipy.signal import resample_poly

    x_ref = np.asarray(x_ref)
    y_aligned = np.asarray(y_aligned)
    n = len(x_ref)
    if len(y_aligned) != n:
        raise ValueError("x_ref and y_aligned must have the same length.")
    if n == 0:
        return x_ref.copy(), y_aligned.copy()
    delay_samples = float(delay_samples)
    if abs(delay_samples) >= n:
        raise ValueError(f"|delay_samples| must be < signal length ({n}); got {delay_samples}.")
    if abs(delay_samples) < 1e-12:
        return x_ref.copy(), y_aligned.copy()

    L = max(2, int(upsample_factor))
    delay_samples = round(delay_samples * L) / L
    shift_up = int(round(delay_samples * L))

    if shift_up == 0:
        y_out = y_aligned.copy()
    elif shift_up % L == 0:
        y_out = _shift_zero_fill(y_aligned, shift_up // L)
    else:
        y_up = resample_poly(y_aligned, L, 1)
        y_up = _shift_zero_fill(y_up, shift_up)
        y_out = _force_length(resample_poly(y_up, 1, L), n)

    if not np.iscomplexobj(x_ref):
        y_out = np.real(y_out).astype(x_ref.dtype, copy=False)
    else:
        y_out = y_out.astype(y_aligned.dtype, copy=False)
    return x_ref.copy(), y_out


def _correlate_and_peak(y, template, template_start, n_ref):
    from scipy.signal import correlate, correlation_lags

    corr = correlate(y, template, mode="full", method="fft")
    lags = correlation_lags(len(y), len(template), mode="full")
    start_if = lags - template_start
    start_eff = np.where(start_if < 0, start_if + n_ref, start_if)
    valid = (start_eff >= 0) & (start_eff <= len(y) - n_ref)
    corr_abs = np.abs(corr)
    corr_safe = np.where(valid, corr_abs, -np.inf)
    peak_idx = int(np.argmax(corr_safe))
    return peak_idx, lags


def _alignment_template(x_ref, peak_window_half):
    n_ref = len(x_ref)
    if peak_window_half is not None:
        peak_idx_ref = int(np.argmax(np.abs(x_ref)))
        lo = max(0, peak_idx_ref - peak_window_half)
        hi = min(n_ref, peak_idx_ref + peak_window_half + 1)
        return x_ref[lo:hi], lo
    return x_ref, 0


def estimate_delay(
    x_ref,
    y_meas,
    fs,
    correct_fractional=True,
    peak_window_half=None,
    upsample_factor=None,
    refine_upsample=100,
):
    """Delay of ``y_meas`` relative to ``x_ref`` in samples (positive => y lags x).

    Stage 1: integer delay via FFT cross-correlation. Stage 2 (``correct_fractional``):
    upsample a windowed segment by ``refine_upsample`` (default 100 => 0.01-sample
    resolution) and re-correlate.

    Length handling (the fix vs the original): if the capture ``y_meas`` is LONGER
    than ``x_ref`` it is kept full so the reference is located inside it; only a
    capture SHORTER than the reference is trimmed.
    """
    from scipy.signal import correlate, correlation_lags, resample_poly

    x_ref = np.asarray(x_ref)
    y_meas = np.asarray(y_meas)
    n_ref = len(x_ref)
    n_meas = len(y_meas)
    if n_ref == 0 or n_meas == 0:
        return 0.0
    if n_meas < n_ref:  # capture shorter than reference -> trim the reference
        x_ref = x_ref[:n_meas]
        n_ref = n_meas

    n_up = max(2, int(refine_upsample)) if refine_upsample else 2
    if upsample_factor is not None:
        n_up = max(2, int(upsample_factor))

    x_template, template_start_in_ref = _alignment_template(x_ref, peak_window_half)
    peak_idx, lags = _correlate_and_peak(y_meas, x_template, template_start_in_ref, n_ref)
    lag_int = lags[peak_idx]
    delay_samples = float(lag_int) - template_start_in_ref

    if correct_fractional and n_up > 1:
        win_len = min(4096, n_ref // 4)
        if win_len >= 8:
            y_start = max(0, lag_int - 2)
            y_end = min(len(y_meas), lag_int + win_len + 2)
            x_win = x_ref[0:win_len]
            y_win = y_meas[y_start:y_end]
            if len(y_win) >= win_len and len(x_win) == win_len:
                x_up = resample_poly(x_win, n_up, 1)
                y_up = resample_poly(y_win, n_up, 1)
                corr_up = correlate(y_up, x_up, mode="full", method="fft")
                lags_up = correlation_lags(len(y_up), len(x_up), mode="full")
                lag_up = lags_up[int(np.argmax(np.abs(corr_up)))]
                delay_samples = y_start + lag_up / n_up

    return float(delay_samples)


def estimate_and_align(
    x_ref,
    y_meas,
    fs,
    correct_fractional=True,
    peak_window_half=None,
    upsample_factor=None,
    refine_upsample=100,
    align_tolerance=0.01,
):
    """Time-align ``y_meas`` to ``x_ref``; returns ``(x_aligned, y_aligned, delay_samples)``.

    Delay via :func:`estimate_delay`, then integer crop + FFT fractional shift. With a
    capture longer than the reference (recommended ~2x) there is slack to crop a full
    aligned period regardless of where the (IMMEDIATE) capture started in the loop.
    """
    x_ref = np.asarray(x_ref)
    y_meas = np.asarray(y_meas)
    N_ref = len(x_ref)

    delay_samples = estimate_delay(
        x_ref,
        y_meas,
        fs,
        correct_fractional=correct_fractional,
        peak_window_half=peak_window_half,
        upsample_factor=upsample_factor,
        refine_upsample=refine_upsample,
    )

    if abs(delay_samples) < align_tolerance:
        n = min(len(x_ref), len(y_meas))
        return x_ref[:n].copy(), y_meas[:n].copy(), delay_samples

    frac = delay_samples - np.floor(delay_samples)
    lag_int_effective = (
        int(np.floor(delay_samples)) if correct_fractional else int(np.round(delay_samples))
    )
    if lag_int_effective < 0:
        lag_int_effective += N_ref
        delay_samples += N_ref

    y_proc = np.asarray(y_meas, dtype=np.complex128)
    if correct_fractional and np.abs(frac) > 1e-9:
        y_proc = _apply_fractional_delay_fft(y_proc, -frac)

    start = lag_int_effective
    end = start + N_ref
    if end > len(y_proc):
        raise ValueError(
            "y_meas too short after delay; capture a longer ORx window (e.g. >= 2x the reference)."
        )
    y_aligned = y_proc[start:end]
    if not np.iscomplexobj(x_ref):
        y_aligned = np.real(y_aligned).astype(x_ref.dtype)
    else:
        y_aligned = y_aligned.astype(y_meas.dtype)
    x_aligned = x_ref[:N_ref]
    return x_aligned, y_aligned, delay_samples
