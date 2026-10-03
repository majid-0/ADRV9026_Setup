"""Capture metrics on one aligned period. Pure numpy, no hardware.

Every quality figure takes the reference first and the measured signal second.
Definitions are in ``docs/dpd_workflow_spec.md`` section 3.

* PAPR and window compression are scale invariant.
* Gain compression and the AM/AM top slope are scale and phase invariant: the
  measured signal is divided by its small-signal gain first.
* NMSE removes one complex gain fitted by least squares over all samples and is
  normalized by the reference energy.
* ACLR uses one Hann-windowed FFT over the whole period with the channel
  centered at DC.
* dBFS figures use ``2**(bits-1)`` as full scale, like :func:`adrvtrx.gain.clip_report`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PA_CLIP_SLOPE",
    "papr_db",
    "window_compression_db",
    "gain_compression_db",
    "nmse_db",
    "aclr_db",
    "inband_corr",
    "rms_dbfs",
    "period_metrics",
]


def papr_db(s) -> float:
    """``10*log10(max|s|^2 / mean|s|^2)``. NaN for an empty or all-zero signal."""
    p = np.abs(np.asarray(s)) ** 2
    if p.size == 0:
        return float("nan")
    mean = float(p.mean())
    if mean <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(p.max() / mean))


def window_compression_db(x, s, fs: float, window_s: float = 1e-4) -> tuple[float, float, float]:
    """PAPR of ``x`` minus PAPR of ``s`` on one window centered on the peak of ``x``.

    ``x`` and ``s`` must already be aligned. The window is ``round(window_s * fs)``
    samples, clamped inside the shorter of the two. Returns
    ``(compression_db, papr_in_db, papr_out_db)``.
    """
    x = np.asarray(x)
    s = np.asarray(s)
    n = min(len(x), len(s))
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    x, s = x[:n], s[:n]
    win = min(n, max(1, int(round(window_s * float(fs)))))
    k = int(np.argmax(np.abs(x)))
    start = max(0, min(k - win // 2, n - win))
    sl = slice(start, start + win)
    pin = papr_db(x[sl])
    pout = papr_db(s[sl])
    return pin - pout, pin, pout


PA_CLIP_SLOPE = 0.08
"""AM/AM top slope below which a capture is flagged ``pa_clipped``.

The slope is ``d|out| / d|in|`` near the peak with the small-signal gain divided
out, so a post-inverse needs about ``1 / slope`` of gain there. 0.08 sits between
the clipped TX1 2.8 GHz / 100 MHz capture at 0 dB backoff (0.03) and the lowest
capture that still inverted well (0.10).
"""


def gain_compression_db(
    x,
    s,
    *,
    peak_frac: float = 0.975,
    small_frac: float = 0.2,
    slope_band: tuple[float, float] = (0.8, 0.95),
    n_bins: int = 40,
) -> tuple[float, float]:
    """Gain compression at the peaks and the AM/AM slope near the top, for aligned ``x`` -> ``s``.

    ``g0`` is the least-squares complex gain on the samples with
    ``|x| < small_frac * max|x|`` (the small-signal gain). With ``a = |x| / max|x|``
    and ``b = |s / g0| / max|x|``:

    * compression = ``-10*log10(mean(b**2) / mean(a**2))`` over the samples with
      ``a >= peak_frac``: the power gain at the peaks relative to the small-signal
      gain, in dB, positive when compressed (0 for a linear PA);
    * top slope = a straight-line fit of the mean of ``b`` against the mean of ``a``
      per bin (``n_bins`` bins of ``a`` on [0, 1]), over the bins whose centre is
      inside ``slope_band``: 1 for a linear PA, 0 for a flat top (clipping).

    The peak region is defined against this signal's own maximum, so the figure is
    the compression at the drive of its largest peaks: compare captures of the same
    waveform, not segments with different peaks.

    This is the gain compression of the whole chain that ``x`` and ``s`` span,
    measured on the signal itself; memory effects and the receiver bandwidth make
    it read slightly higher than a CW compression point. Returns
    ``(compression_db, top_slope)``; NaN where a region has no samples.
    """
    x = np.asarray(x, dtype=np.complex128)
    s = np.asarray(s, dtype=np.complex128)
    n = min(len(x), len(s))
    nan = float("nan")
    if n == 0:
        return nan, nan
    x, s = x[:n], s[:n]
    ax = np.abs(x)
    peak = float(ax.max())
    if peak <= 0.0:
        return nan, nan
    a = ax / peak
    small = a < small_frac
    ex = float(np.vdot(x[small], x[small]).real)
    if ex <= 0.0:
        return nan, nan
    g0 = np.vdot(x[small], s[small]) / ex
    if g0 == 0:
        return nan, nan
    b = np.abs(s / g0) / peak

    top = a >= peak_frac
    comp = nan
    if top.any():
        ratio = float(np.mean(b[top] ** 2) / np.mean(a[top] ** 2))
        comp = float(-10.0 * np.log10(ratio)) if ratio > 0.0 else nan

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(a, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sum_a = np.bincount(idx, weights=a, minlength=n_bins)
    sum_b = np.bincount(idx, weights=b, minlength=n_bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sel = (centres > slope_band[0]) & (centres < slope_band[1]) & (counts > 0)
    slope = nan
    if np.count_nonzero(sel) >= 2:
        slope = float(np.polyfit(sum_a[sel] / counts[sel], sum_b[sel] / counts[sel], 1)[0])
    return comp, slope


def nmse_db(x, s) -> float:
    """NMSE of ``s`` against the reference ``x`` after one least-squares complex gain.

    ``g = sum(conj(x) * s) / sum(|x|^2)`` over all samples, then
    ``10*log10(sum|s/g - x|^2 / sum|x|^2)``. Independent of the scale and phase of
    either signal. Lengths must match.
    """
    x = np.asarray(x, dtype=np.complex128)
    s = np.asarray(s, dtype=np.complex128)
    if len(x) != len(s):
        raise ValueError(f"nmse_db: lengths differ ({len(x)} vs {len(s)})")
    ex = float(np.vdot(x, x).real)
    if len(x) == 0 or ex <= 0.0:
        return float("nan")
    g = np.vdot(x, s) / ex
    if g == 0:
        return float("nan")
    err = s / g - x
    return float(10.0 * np.log10(np.vdot(err, err).real / ex))


def aclr_db(s, fs: float, bw_hz: float) -> tuple[float, float]:
    """Lower and upper adjacent-channel leakage in dBc.

    One FFT of ``s * hann(N)``. Main channel ``[-bw/2, bw/2)``, lower
    ``[-1.5bw, -0.5bw)``, upper ``[0.5bw, 1.5bw)``.
    """
    z = np.asarray(s, dtype=np.complex128)
    n = len(z)
    if n == 0:
        return float("nan"), float("nan")
    spec = np.fft.fftshift(np.fft.fft(z * np.hanning(n)))
    f = np.fft.fftshift(np.fft.fftfreq(n, 1.0 / float(fs)))
    p = np.abs(spec) ** 2

    def band(a: float, b: float) -> float:
        return float(p[(f >= a) & (f < b)].sum())

    main = band(-bw_hz / 2, bw_hz / 2)
    if main <= 0.0:
        return float("nan"), float("nan")
    lower = band(-1.5 * bw_hz, -0.5 * bw_hz)
    upper = band(0.5 * bw_hz, 1.5 * bw_hz)
    with np.errstate(divide="ignore"):
        return float(10.0 * np.log10(lower / main)), float(10.0 * np.log10(upper / main))


def inband_corr(a, b, energy_frac: float = 0.05) -> float:
    """Correlation of an aligned pair over the occupied bins of ``a``.

    Bins where ``|fft(a)| > energy_frac * max|fft(a)|``. 1.0 is a perfect copy
    up to one complex gain. Out-of-band energy in ``b`` does not lower it.
    """
    m = min(len(a), len(b))
    if m == 0:
        return 0.0
    fa = np.fft.fft(np.asarray(a[:m], dtype=np.complex128))
    fb = np.fft.fft(np.asarray(b[:m], dtype=np.complex128))
    mask = np.abs(fa) > energy_frac * np.abs(fa).max()
    am, bm = fa[mask], fb[mask]
    den = (np.linalg.norm(am) * np.linalg.norm(bm)) or 1.0
    return float(np.abs(np.vdot(am, bm)) / den)


def rms_dbfs(s_codes, bits: int) -> float:
    """``10*log10(mean|s|^2 / (2**(bits-1))^2)`` on integer-code samples."""
    s = np.asarray(s_codes)
    if s.size == 0:
        return float("-inf")
    fs_ref = float(1 << (bits - 1))
    p = float(np.mean(np.abs(s) ** 2))
    if p <= 0.0:
        return float("-inf")
    return float(10.0 * np.log10(p / fs_ref**2))


def period_metrics(
    x,
    s,
    *,
    fs: float,
    bw_hz: float,
    rx_bits: int,
    delay_samples: float,
    corr_ref=None,
    min_top_slope: float = PA_CLIP_SLOPE,
) -> dict[str, float]:
    """Every per-period CSV figure for measured ``s`` against reference ``x``.

    ``s`` is the aligned ORx period in codes. ``corr`` is taken against
    ``corr_ref`` when given (the transmitted waveform during replay), else ``x``.
    ``pa_clipped`` is ``amam_top_slope < min_top_slope``.
    Peak and rail counts come from the raw capture and are not computed here.
    """
    pin = papr_db(x)
    pout = papr_db(s)
    lower, upper = aclr_db(s, fs, bw_hz)
    gain_comp, top_slope = gain_compression_db(x, s)
    return {
        "delay_samples": float(delay_samples),
        "delay_ns": float(delay_samples) / float(fs) * 1e9,
        "corr": inband_corr(x if corr_ref is None else corr_ref, s),
        "nmse_db": nmse_db(x, s),
        "papr_in_db": pin,
        "papr_out_db": pout,
        "papr_compression_db": pin - pout,
        "aclr_lower_dbc": lower,
        "aclr_upper_dbc": upper,
        "rms_dbfs": rms_dbfs(s, rx_bits),
        "gain_compression_db": gain_comp,
        "amam_top_slope": top_slope,
        "pa_clipped": bool(np.isfinite(top_slope) and top_slope < min_top_slope),
    }
