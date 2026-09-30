"""Capture metrics on one aligned period. Pure numpy, no hardware.

Every quality figure takes the reference first and the measured signal second.
Definitions are in ``docs/dpd_workflow_spec.md`` section 3.

* PAPR and window compression are scale invariant.
* NMSE removes one complex gain fitted by least squares over all samples and is
  normalized by the reference energy.
* ACLR uses one Hann-windowed FFT over the whole period with the channel
  centered at DC.
* dBFS figures use ``2**(bits-1)`` as full scale, like :func:`adrvtrx.gain.clip_report`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "papr_db",
    "window_compression_db",
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
) -> dict[str, float]:
    """Every per-period CSV figure for measured ``s`` against reference ``x``.

    ``s`` is the aligned ORx period in codes. ``corr`` is taken against
    ``corr_ref`` when given (the transmitted waveform during replay), else ``x``.
    Peak and rail counts come from the raw capture and are not computed here.
    """
    pin = papr_db(x)
    pout = papr_db(s)
    lower, upper = aclr_db(s, fs, bw_hz)
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
    }
