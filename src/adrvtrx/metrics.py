"""Capture metrics on one aligned period. Pure numpy, no hardware.

Every quality figure takes the reference first and the measured signal second.
Definitions are in ``docs/dpd_workflow_spec.md`` section 3.

* PAPR and window compression are scale invariant.
* Gain compression and the AM/AM top slope are scale and phase invariant: the
  measured signal is divided by its small-signal gain first.
* Joint compression (several bands into one PA) recomposes the bands at their carrier offsets,
  each band's output divided by its own small-signal gain first (:func:`joint_compression`).
* NMSE removes one complex gain fitted by least squares over all samples and is
  normalized by the reference energy.
* ACLR uses one Hann-windowed FFT over the whole period with the channel
  centered at DC.
* dBFS figures use ``2**(bits-1)`` as full scale, like :func:`adrvtrx.gain.clip_report`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "PA_CLIP_SLOPE",
    "PEAK_REGION",
    "papr_db",
    "window_compression_db",
    "gain_compression_db",
    "BAND_FLOOR",
    "REFS",
    "joint_compression",
    "joint_compression_sweep",
    "joint_amam_ampm",
    "joint_envelope",
    "envelope_shares",
    "share_key",
    "joint_top_drop",
    "input_metrics",
    "composite_signal",
    "composite_pair",
    "levels_from_attens",
    "JOINT_CLIP_SLOPE",
    "JOINT_TOP_FRAC",
    "JOINT_REF_BAND",
    "SHARE_LEVELS",
    "nmse_db",
    "aclr_db",
    "inband_corr",
    "rms_dbfs",
    "period_metrics",
]


#: The peak region of :func:`gain_compression_db`: every sample at or above this fraction of the
#: signal's maximum (the probe, :mod:`adrvtrx.probe`, keeps a window around each of its peaks).
PEAK_REGION = 0.975


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
    peak_frac: float = PEAK_REGION,
    small_frac: float = 0.2,
    slope_band: tuple[float, float] = (0.8, 0.95),
    n_bins: int = 40,
    small_gain: complex | None = None,
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

    ``small_gain`` is ``g0`` given instead of fitted (:func:`joint_compression` passes 1: each
    band's output is already divided by its own small-signal gain).

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
    g0 = _small_signal_gain(x, s, a < small_frac) if small_gain is None else complex(small_gain)
    if g0 is None or g0 == 0:
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


#: ``joint_clipping`` (:func:`joint_compression`) is ``joint_top_slope`` below this: the per-band
#: clip guard's :data:`PA_CLIP_SLOPE`, for the composite.
JOINT_CLIP_SLOPE = PA_CLIP_SLOPE
#: :func:`joint_top_drop`: the top of the joint envelope (this fraction of its maximum and up) ...
JOINT_TOP_FRAC = 0.98
#: ... against each band's gain and phase where the joint envelope is in this range of its maximum.
JOINT_REF_BAND = (0.5, 0.8)
#: :func:`envelope_shares`: the fractions of the joint envelope's maximum whose share is logged.
SHARE_LEVELS = (0.8, 0.9)


def levels_from_attens(attens_db, path_gains_db=None) -> list[float]:
    """Each band's level at the PA input from its TX attenuation, dB relative to the strongest band.

    ``level_k = path_gain_k - atten_k``, shifted so the strongest band is 0 dB.
    ``path_gains_db`` is each band's gain from its DAC to the PA input at 0 dB attenuation;
    ``None`` assumes they are equal (the default until a measured calibration replaces it).
    The code scale is not in the level: :func:`joint_compression` keeps it in the inputs.
    """
    a = np.asarray(attens_db, dtype=float)
    g = np.zeros_like(a) if path_gains_db is None else np.asarray(path_gains_db, dtype=float)
    if g.shape != a.shape:
        raise ValueError(f"one path gain per band: {list(g)} for attenuations {list(a)}")
    lv = g - a
    return [float(v) for v in lv - lv.max()]


def _bands_in(xs, ys, levels_db) -> tuple[list[np.ndarray], list[np.ndarray] | None, np.ndarray]:
    """The bands as complex arrays of one length, and their levels (default 0 dB each)."""
    xs = [np.asarray(x, dtype=np.complex128) for x in xs]
    if not xs:
        raise ValueError("no band")
    if ys is not None:
        ys = [np.asarray(y, dtype=np.complex128) for y in ys]
        if len(ys) != len(xs):
            raise ValueError(f"one output per input: {len(xs)} inputs, {len(ys)} outputs")
    lengths = {len(z) for z in xs + (ys or [])}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"every band needs one period of the same length, got {sorted(lengths)}")
    lv = np.zeros(len(xs)) if levels_db is None else np.asarray(levels_db, dtype=float)
    if lv.shape != (len(xs),):
        raise ValueError(f"one level per band: {list(np.atleast_1d(lv))} for {len(xs)} bands")
    return xs, ys, lv


def _upsample_factor(offsets_hz, fs: float, oversample: float) -> int:
    """``L`` so that ``L * fs`` is ``oversample`` times the span (every band's ``fs`` included)."""
    span = (max(offsets_hz) - min(offsets_hz)) + float(fs)
    return max(1, int(np.ceil(float(oversample) * span / float(fs))))


def _placed(z: np.ndarray, upsample: int, shift_bins: int) -> np.ndarray:
    """One band in the composite: ``z`` (one period) up-sampled ``upsample`` times by exact
    periodic (FFT) interpolation and moved ``shift_bins`` bins of ``fs / len(z)`` (its carrier
    offset). The Nyquist bin of an even period is dropped (as ``cfr_joint.composite``)."""
    n = len(z)
    nh = n * upsample
    f = np.fft.fftfreq(n)
    keep = np.abs(f) < 0.5
    idx = np.round(f[keep] * n).astype(np.int64)
    spec = np.zeros(nh, dtype=np.complex128)
    spec[(idx + shift_bins) % nh] = np.fft.fft(z)[keep] * upsample
    return np.fft.ifft(spec)


def _pe_papr_db(xs: Sequence[np.ndarray]) -> float:
    """The spacing-free peak estimate: ``max(sum_k |x_k|)**2 / sum_k mean|x_k|**2`` in dB."""
    p = sum(float(np.mean(np.abs(x) ** 2)) for x in xs)
    if p <= 0.0:
        return float("nan")
    return float(10.0 * np.log10(float(np.max(sum(np.abs(x) for x in xs))) ** 2 / p))


def _scales(xs: Sequence[np.ndarray], lv: np.ndarray) -> np.ndarray:
    """Each band's factor to the PA-input scale: one common divisor (the largest peak of all the
    bands), so the codes keep their relative scale, times its level. Zeros when all are zero."""
    common = max(float(np.abs(x).max()) for x in xs)
    return 10.0 ** (lv / 20.0) / common if common > 0.0 else np.zeros(len(xs))


def _composite_pair(xs, ys, fs, freqs_hz, levels_db, oversample, small_frac, keep) -> dict:
    """The composite pair of :func:`joint_compression`, :func:`joint_amam_ampm` and
    :func:`composite_pair`.

    ``X``, ``Y`` (``Y`` None without ``ys`` or when a band that is on has no small-signal gain;
    both None when every input is zero), ``pe_papr_in_db``, and in ``bands``, per band that is
    on, its placed input and output (``keep="complex"``) or their envelopes as float32
    (``keep="envelope"``, a quarter of the memory; ``None``: nothing), with ``keep="complex"``
    also ``base``, per band its input and output at its own rate on the PA-input scale; plus the
    layout (``spacing_hz``, ``offsets_hz``, ``levels_db``, ``upsample``).
    """
    xs, ys, lv = _bands_in(xs, ys, levels_db)
    freqs = np.asarray(freqs_hz, dtype=float)
    if freqs.shape != (len(xs),):
        raise ValueError(f"one carrier per band: {list(np.atleast_1d(freqs))} for {len(xs)} bands")
    n = len(xs[0])
    offsets = freqs - freqs.mean()
    up = _upsample_factor(offsets, fs, oversample)
    peaks = [float(np.abs(x).max()) for x in xs]
    common = max(peaks)
    c: dict[str, Any] = {
        "X": None,
        "Y": None,
        "bands": {},
        "base": {},
        "pe_papr_in_db": float("nan"),
        "spacing_hz": float(freqs.max() - freqs.min()),
        "offsets_hz": [float(o) for o in offsets],
        "levels_db": [float(v) for v in lv],
        "upsample": up,
    }
    if not common > 0.0:
        return c
    on = [k for k, p in enumerate(peaks) if p > 0.0]
    scale = _scales(xs, lv)
    g0 = {}
    if ys is not None:
        g0 = {
            k: _small_signal_gain(xs[k], ys[k], np.abs(xs[k]) < small_frac * peaks[k]) for k in on
        }
    c["pe_papr_in_db"] = _pe_papr_db([xs[k] * scale[k] for k in on])

    X = np.zeros(n * up, dtype=np.complex128)
    Y = np.zeros(n * up, dtype=np.complex128) if g0 and all(g0[k] for k in on) else None
    for k in on:
        shift = int(round(offsets[k] * n / float(fs)))
        zx = _placed(xs[k] * scale[k], up, shift)
        X += zx
        if Y is None:
            continue
        zy = _placed(ys[k] * (scale[k] / g0[k]), up, shift)
        Y += zy
        if keep == "complex":
            c["bands"][k] = (zx, zy)
            c["base"][k] = (xs[k] * scale[k], ys[k] * (scale[k] / g0[k]))
        elif keep == "envelope":
            c["bands"][k] = (np.abs(zx).astype(np.float32), np.abs(zy).astype(np.float32))
    c["X"], c["Y"] = X, Y
    return c


def joint_envelope(xs, levels_db=None) -> np.ndarray:
    """The joint envelope per sample: ``sum_k |x_k(n)|`` on the PA-input scale of
    :func:`joint_compression` (one common divisor, so the codes keep their relative scale, then
    ``levels_db``), at the bands' own rate and index, so samples can be picked by it.

    It is the composite's peak over the beat phase, the peak estimator of the joint CFR
    (``cfr_joint.pe_papr``): wherever the carriers are, the composite envelope ``|X|`` of
    :func:`composite_signal` is at most ``sum_k |x_k|`` at that instant (interpolated between
    the bands' samples), and reaches it where the carriers' phases align, which a beat at the
    spacing does about once per beat period. So ``max |X|`` is close to its maximum (above it
    only by the inter-sample peaks) while ``|X|`` swings below it within each beat. Unlike
    ``|X|`` it does not depend on the spacing. One period of every band, the same length.
    """
    xs, _ys, lv = _bands_in(xs, None, levels_db)
    scale = _scales(xs, lv)
    return sum(np.abs(x) * s for x, s in zip(xs, scale))


def share_key(frac: float) -> str:
    """The column of :func:`envelope_shares`' share above ``frac``: ``joint_env_above_0p8``."""
    return f"joint_env_above_{float(frac):g}".replace(".", "p")


def envelope_shares(xs, levels_db=None, fracs=SHARE_LEVELS) -> dict[str, float]:
    """The share of samples (0 .. 1) where :func:`joint_envelope` is at or above each of
    ``fracs`` of its maximum, keyed by :func:`share_key` (``joint_env_above_0p8``,
    ``joint_env_above_0p9``).

    How dense the top of the joint envelope is: the samples a selection by the joint envelope can
    pick there (the original S20-100 signals hold 0.07 % above 0.8, their joint-CFR versions 1.7 %).
    NaN when every input is zero."""
    env = joint_envelope(xs, levels_db)
    peak = float(env.max())
    if not peak > 0.0:
        return {share_key(f): float("nan") for f in fracs}
    return {share_key(f): float(np.mean(env >= float(f) * peak)) for f in fracs}


def joint_top_drop(
    xs,
    ys,
    levels_db=None,
    *,
    top_frac: float = JOINT_TOP_FRAC,
    ref_band: tuple[float, float] = JOINT_REF_BAND,
    small_frac: float = 0.2,
) -> dict[str, list[float]]:
    """Each band's gain and phase at the top of the joint envelope against lower down: a clip
    indicator per band.

    With ``a = joint_envelope(xs, levels_db) / max`` (the bands' own rate and index) and each band's
    output divided by its own small-signal gain ``g0_k`` (as :func:`joint_compression`), a band's
    gain over a set of samples is ``10*log10(sum|y_k/g0_k|^2 / sum|x_k|^2)`` and its phase the angle
    of ``sum (y_k/g0_k) * conj(x_k)`` (only its samples at or above :data:`BAND_FLOOR` of its own
    peak). Returns per band:

    * ``gain_drop_db``: its gain where ``ref_band[0] <= a < ref_band[1]`` minus its gain where
      ``a >= top_frac`` (positive: the gain falls at the top);
    * ``phase_drop_deg``: its phase at the top minus its phase in ``ref_band``, degrees (signed).

    The reference is the middle of the envelope, not the small-signal region, so the figure is the
    extra compression at the very top: a clip shows as a drop that grows fast with drive. NaN for a
    band that is off or has no small-signal gain, or a region without samples. Like the joint
    compression it understates a real clip: each ORx averages the gain over the beat between the
    carriers, and the intermodulation of a clip falls outside the observed bands.
    """
    xs, ys, lv = _bands_in(xs, ys, levels_db)
    nan = float("nan")
    out: dict[str, list[float]] = {
        "gain_drop_db": [nan] * len(xs),
        "phase_drop_deg": [nan] * len(xs),
    }
    env = joint_envelope(xs, lv)
    peak = float(env.max())
    if ys is None or not peak > 0.0:
        return out
    a = env / peak
    top = a >= float(top_frac)
    ref = (a >= float(ref_band[0])) & (a < float(ref_band[1]))
    for k, (x, y) in enumerate(zip(xs, ys)):
        ax = np.abs(x)
        pk = float(ax.max())
        g0 = _small_signal_gain(x, y, ax < small_frac * pk) if pk > 0.0 else None
        if not g0:
            continue
        z = y / g0
        on = ax >= BAND_FLOOR * pk
        g_top, p_top = _gain_phase(x, z, top & on)
        g_ref, p_ref = _gain_phase(x, z, ref & on)
        out["gain_drop_db"][k] = float(g_ref - g_top)
        out["phase_drop_deg"][k] = float(np.rad2deg(np.angle(np.exp(1j * (p_top - p_ref)))))
    return out


def _gain_phase(x: np.ndarray, z: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    """``x -> z``'s power-ratio gain (dB) and mean phase (radians) over the samples ``m``."""
    den = float(np.sum(np.abs(x[m]) ** 2))
    num = float(np.sum(np.abs(z[m]) ** 2))
    if not (den > 0.0 and num > 0.0):
        return float("nan"), float("nan")
    return 10.0 * np.log10(num / den), float(np.angle(np.sum(z[m] * np.conj(x[m]))))


def input_metrics(xs, fs: float, freqs_hz, *, levels_db=None, oversample: float = 4.0) -> dict:
    """The joint figures of the inputs alone (a case's inputs at its levels):
    ``composite_papr_db``, the PAPR of :func:`composite_signal` at the bands' real carriers;
    ``pe_papr_db``, the spacing-free ``max(sum_k |x_k|)^2 / sum_k mean|x_k|^2``; the
    :func:`envelope_shares` (``joint_env_above_0p8``, ``joint_env_above_0p9``); ``spacing_hz``
    and ``levels_db``."""
    c = _composite_pair(xs, None, fs, freqs_hz, levels_db, oversample, 0.2, None)
    return {
        "composite_papr_db": papr_db(c["X"]) if c["X"] is not None else float("nan"),
        "pe_papr_db": c["pe_papr_in_db"],
        **envelope_shares(xs, c["levels_db"]),
        "spacing_hz": c["spacing_hz"],
        "levels_db": c["levels_db"],
    }


def composite_signal(
    xs, fs: float, freqs_hz, *, levels_db=None, oversample: float = 4.0
) -> tuple[np.ndarray, float]:
    """The composite input of :func:`joint_compression`: every band at its offset
    ``f_k - mean(f)`` on the PA-input scale (one common divisor, then ``levels_db``),
    up-sampled exactly to ``oversample`` times the span, summed. Returns ``(x_c, fs_c)``, its
    rate ``fs_c = upsample * fs``; sample ``m`` of ``x_c`` is at the bands' sample
    ``m / upsample``. ``fs`` is the bands' baseband rate (the carrier offsets need it)."""
    c = _composite_pair(xs, None, fs, freqs_hz, levels_db, oversample, 0.2, None)
    x_c = c["X"] if c["X"] is not None else np.zeros(len(xs[0]) * c["upsample"], complex)
    return x_c, float(fs) * c["upsample"]


def composite_pair(
    xs,
    ys,
    fs: float,
    freqs_hz,
    *,
    levels_db=None,
    oversample: float = 4.0,
    small_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """:func:`composite_signal` and the composite output of :func:`joint_compression` (each
    band's output divided by its own small-signal gain, placed the same way): ``(x_c, y_c,
    fs_c)``, ``y_c`` None when a band that is on has no small-signal gain."""
    c = _composite_pair(xs, ys, fs, freqs_hz, levels_db, oversample, small_frac, None)
    if ys is None:
        raise ValueError("composite_pair needs the outputs (composite_signal: the inputs only)")
    x_c = c["X"] if c["X"] is not None else np.zeros(len(xs[0]) * c["upsample"], complex)
    return x_c, c["Y"], float(fs) * c["upsample"]


def _layout(c: dict) -> dict[str, Any]:
    return {k: c[k] for k in ("spacing_hz", "offsets_hz", "levels_db", "upsample")}


def joint_compression(
    xs,
    ys,
    fs: float,
    freqs_hz,
    *,
    levels_db=None,
    oversample: float = 4.0,
    peak_frac: float = PEAK_REGION,
    small_frac: float = 0.2,
    clip_slope: float = JOINT_CLIP_SLOPE,
) -> dict[str, Any]:
    """Gain compression of the composite signal several bands make at one PA.

    ``xs`` are the bands' inputs (TX codes, or any one scale shared by every band), ``ys`` their
    aligned outputs (one period each, the length of the inputs, each at any scale and phase),
    ``fs`` the baseband rate and ``freqs_hz`` each band's RF carrier.

    **Levels at the PA input.** Every input is divided by the same number (the largest peak of
    all the bands), so the codes keep their real relative scale, then multiplied by
    ``10**(levels_db[k] / 20)`` (default 0 dB each). With TX codes the PA input of band ``k`` is
    ``codes_k * 10**((G_k - A_k) / 20)``, ``A_k`` its TX attenuation and ``G_k`` its gain from the
    DAC to the PA input: the code scale stays in ``x_k`` and the rest is
    ``levels_db[k] = G_k - A_k`` (:func:`levels_from_attens`). Only the differences matter.
    Inputs that are not at one common scale must be brought to one first.

    **The composite input** ``X``: each band at its offset ``f_k - mean(f)``, up-sampled exactly
    (periodic FFT interpolation) to ``oversample`` times the span, summed. **The composite
    output** ``Y``: each band's output divided by its own small-signal gain ``g0_k`` (the complex
    least-squares gain on its samples with ``|x_k| < small_frac * max|x_k|``, as
    :func:`gain_compression_db`), so it is in its input's units, then placed and summed the
    same way.

    The small-signal reference is per band, and ``Y``'s is 1 (``small_gain=1``, no refit):

    * each band's output carries its own path, ORx gain and phase, so only a per-band gain can
      put the outputs on the inputs' scale;
    * a refit on the composite's small samples would mix in the beat troughs, where the bands
      cancel while each is large and compressed. ``joint_refit_gain_db`` is that refit, a check
      that should stay near 0 dB.

    Each ``g0_k`` is the band's gain while it is small, whatever the other bands do then.

    Every output is band limited (each ORx sees its own band), so ``Y`` holds the gain averaged
    over the beat between the bands, not its swing within one beat period. The products of that
    swing fall between the bands. The joint figure is therefore below the instantaneous
    compression of the composite envelope.

    Returns:

    * ``joint_gain_compression_db``, ``joint_top_slope``: :func:`gain_compression_db` of
      ``X -> Y``;
    * ``joint_papr_in_db``, ``joint_papr_out_db``, ``joint_papr_compression_db``: PAPR of
      ``X``, of ``Y``, and their difference;
    * ``band_compression_at_joint_peaks_db``: per band, ``-10*log10(sum|y_k/g0_k|^2 /
      sum|x_k|^2)`` over the samples where ``|X| >= peak_frac * max|X|`` (both up-sampled);
    * ``pe_papr_in_db``: the spacing-free peak estimate ``max(sum_k |x_k|)^2 / sum_k
      mean|x_k|^2`` at the bands' own rate (above ``joint_papr_in_db`` but for the
      inter-sample peaks it misses);
    * ``joint_refit_gain_db``; ``spacing_hz`` (highest minus lowest carrier); ``offsets_hz``;
      ``levels_db``; ``upsample`` (the factor used);
    * the joint clip indicator: ``joint_clipping``, ``joint_top_slope < clip_slope``
      (:data:`JOINT_CLIP_SLOPE`, the per-band clip guard's 0.08; False when the slope is NaN), and
      per band ``band_top_gain_drop_db`` / ``band_top_phase_drop_deg`` (:func:`joint_top_drop`: its
      gain and phase at the top 2 % of the joint envelope against the 0.5 .. 0.8 range). An early
      warning that **understates** a real clip: each ORx averages the gain over the beat between
      the carriers, and the intermodulation of a clip falls outside the observed bands. For scale,
      on the S20-100 bench captures ``joint_top_slope`` was 0.15 (original signals, 3 dB per band),
      0.28 .. 0.30 (joint-CFR signals, 3 dB per band) and 0.61 (1 dB per band);
    * :func:`envelope_shares` of the inputs (``joint_env_above_0p8``, ``joint_env_above_0p9``).

    A band whose input is all zero is off: it is left out (its band figure is NaN). Output
    figures are NaN when a band that is on has no small-signal gain.
    """
    if ys is None:
        raise ValueError("joint_compression needs the outputs")
    c = _composite_pair(xs, ys, fs, freqs_hz, levels_db, oversample, small_frac, "envelope")
    nan = float("nan")
    out: dict[str, Any] = {
        "joint_gain_compression_db": nan,
        "joint_top_slope": nan,
        "joint_papr_in_db": nan,
        "joint_papr_out_db": nan,
        "joint_papr_compression_db": nan,
        "band_compression_at_joint_peaks_db": [nan] * len(c["levels_db"]),
        "pe_papr_in_db": c["pe_papr_in_db"],
        "joint_refit_gain_db": nan,
        "joint_clipping": False,
        "band_top_gain_drop_db": [nan] * len(c["levels_db"]),
        "band_top_phase_drop_deg": [nan] * len(c["levels_db"]),
        **envelope_shares(xs, c["levels_db"]),
        **_layout(c),
    }
    X, Y = c["X"], c["Y"]
    if X is None:
        return out
    out["joint_papr_in_db"] = papr_db(X)
    if Y is None:
        return out

    comp, slope = gain_compression_db(
        X, Y, peak_frac=peak_frac, small_frac=small_frac, small_gain=1.0
    )
    ax = np.abs(X)
    refit = _small_signal_gain(X, Y, ax < small_frac * ax.max())
    top = ax >= peak_frac * ax.max()
    bands = list(out["band_compression_at_joint_peaks_db"])
    for k, (ex, ey) in c["bands"].items():
        den = float(np.sum(ex[top].astype(np.float64) ** 2))
        num = float(np.sum(ey[top].astype(np.float64) ** 2))
        if den > 0.0 and num > 0.0:
            bands[k] = float(-10.0 * np.log10(num / den))
    papr_out = papr_db(Y)
    drop = joint_top_drop(xs, ys, c["levels_db"], small_frac=small_frac)
    out.update(
        {
            "joint_clipping": bool(np.isfinite(slope) and slope < clip_slope),
            "band_top_gain_drop_db": drop["gain_drop_db"],
            "band_top_phase_drop_deg": drop["phase_drop_deg"],
            "joint_gain_compression_db": comp,
            "joint_top_slope": slope,
            "joint_papr_out_db": papr_out,
            "joint_papr_compression_db": out["joint_papr_in_db"] - papr_out,
            "band_compression_at_joint_peaks_db": bands,
            "joint_refit_gain_db": (
                float(20.0 * np.log10(abs(refit))) if refit else nan  # no small sample: NaN
            ),
        }
    )
    return out


#: What :func:`joint_amam_ampm` bins each band's gain and phase on: the composite's envelope
#: (up-sampled), the band's own, and :func:`joint_envelope` (the bands' own rate and index).
REFS = ("vs_joint", "vs_own", "vs_envelope")

#: A band's own samples below this fraction of its peak carry no gain or phase to read
#: (:func:`joint_amam_ampm`, its per-band curves).
BAND_FLOOR = 0.02


def _bin_index(a: np.ndarray, bins: int) -> np.ndarray:
    """The bin of each ``a`` in ``[0, 1]`` (``bins`` equal bins), as :func:`gain_compression_db`."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    return np.clip(np.digitize(a, edges) - 1, 0, bins - 1)


def _phase_bins(idx: np.ndarray, phase: np.ndarray, bins: int, counts: np.ndarray):
    """Per bin the circular mean of ``phase`` and the RMS deviation from it (radians), and each
    sample's deviation from its bin's mean."""
    mean = np.angle(
        np.bincount(idx, weights=np.cos(phase), minlength=bins)
        + 1j * np.bincount(idx, weights=np.sin(phase), minlength=bins)
    )
    dev = np.angle(np.exp(1j * (phase - mean[idx])))
    with np.errstate(invalid="ignore", divide="ignore"):
        spread = np.sqrt(np.bincount(idx, weights=dev**2, minlength=bins) / counts)
    mean[counts == 0] = np.nan
    return mean, spread, dev


def _band_bins(zx: np.ndarray, zy: np.ndarray, idx: np.ndarray, bins: int) -> dict[str, Any]:
    """One band's gain (dB) and phase (degrees) per bin of ``idx``: the bin's power-ratio gain
    (minus the compression), the spread of each sample's gain, the circular-mean phase and its
    spread. Only the band's samples at or above :data:`BAND_FLOOR` of its peak count."""
    ax, ay = np.abs(zx), np.abs(zy)
    ok = ax >= BAND_FLOOR * ax.max()
    i = idx[ok]
    counts = np.bincount(i, minlength=bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        g_db = 20.0 * np.log10(ay[ok] / ax[ok])
        num = np.bincount(i, weights=ay[ok] ** 2, minlength=bins)
        den = np.bincount(i, weights=ax[ok] ** 2, minlength=bins)
        gain = 10.0 * np.log10(num / den)
        mean_g = np.bincount(i, weights=g_db, minlength=bins) / counts
        spread = np.sqrt(np.bincount(i, weights=(g_db - mean_g[i]) ** 2, minlength=bins) / counts)
    ph_mean, ph_spread, _dev = _phase_bins(i, np.angle(zy[ok] * np.conj(zx[ok])), bins, counts)
    gain[counts == 0] = np.nan
    return {
        "counts": counts.astype(int),
        "gain_db": gain,
        "gain_spread_db": spread,
        "phase_deg": np.rad2deg(ph_mean),
        "phase_spread_deg": np.rad2deg(ph_spread),
    }


def joint_amam_ampm(
    xs,
    ys,
    fs: float,
    freqs_hz,
    *,
    levels_db=None,
    bins: int = 40,
    oversample: float = 4.0,
    small_frac: float = 0.2,
    slope_band: tuple[float, float] = (0.8, 0.95),
    top_from: float = 0.8,
) -> dict[str, Any]:
    """Joint AM/AM and AM/PM: the composite pair of :func:`joint_compression` binned on its
    input envelope, and each band's gain and phase against the composite's envelope and its own.

    With ``a = |X| / max|X|`` in ``bins`` equal bins of [0, 1] (as :func:`gain_compression_db`)
    and ``b = |Y| / max|X|``:

    * ``amam_in``, ``amam_out``, ``amam_spread``: per bin the mean of ``a``, the mean of ``b``
      and the standard deviation of ``b`` (the AM/AM curve, 1:1 when linear);
    * ``ampm_deg``, ``ampm_spread_deg``: per bin the circular mean of ``angle(Y * conj(X))``
      and the RMS deviation from it;
    * ``counts``, ``bin_centres``;
    * ``joint_top_slope``: the line through ``(amam_in, amam_out)`` over the bins centred in
      ``slope_band`` (with 40 bins, :func:`joint_compression`'s ``joint_top_slope``);
    * ``top_amam_spread_db``, ``top_ampm_spread_deg``: the dispersion where ``a >= top_from``,
      the RMS of each sample's gain ``20*log10(b / a)`` and of its phase about its bin's mean,
      dB and degrees. A PA that follows the composite envelope alone, without memory,
      would give none; here the gain each band shows depends on every band's envelope (and is
      averaged over the beat, :func:`joint_compression`), not on ``|X|`` alone, so even a
      memoryless PA leaves some, and memory adds to it;
    * per band, arrays ``(bands, bins)``: ``band_gain_db_vs_joint``,
      ``band_gain_spread_db_vs_joint``, ``band_phase_deg_vs_joint``,
      ``band_phase_spread_deg_vs_joint``, ``band_counts_vs_joint`` (binned on ``a``), and the
      same ``_vs_own`` (binned on the band's own ``|x_k| / max|x_k|``). The gain of a bin is
      ``10*log10(sum|y_k/g0_k|^2 / sum|x_k|^2)`` (minus the compression; 0 dB when linear),
      its spread that of each sample's gain in dB; the phase is ``y_k/g0_k`` against ``x_k``.
      A band's samples below :data:`BAND_FLOOR` of its own peak are left out. Against the
      joint envelope, the curves show what the other bands do to band ``k`` where the
      composite peaks; and the same ``_vs_envelope``, at the bands' own rate and index, binned
      on :func:`joint_envelope` ``/ max`` (``envelope_peak``: its max on the PA-input
      scale): bin ``i`` holds the samples ``n`` a selection by the joint envelope would pick;
    * ``spacing_hz``, ``offsets_hz``, ``levels_db``, ``upsample``.

    Plain numpy arrays; NaN in an empty bin, and every figure NaN when the composite output
    cannot be built (as :func:`joint_compression`). It keeps every band's up-sampled input and
    output (complex): an offline analysis, not run per capture.
    """
    if ys is None:
        raise ValueError("joint_amam_ampm needs the outputs")
    c = _composite_pair(xs, ys, fs, freqs_hz, levels_db, oversample, small_frac, "complex")
    nb = len(c["levels_db"])
    edges = np.linspace(0.0, 1.0, bins + 1)
    nan_bins = np.full(bins, np.nan)
    out: dict[str, Any] = {
        "bin_centres": 0.5 * (edges[:-1] + edges[1:]),
        "counts": np.zeros(bins, dtype=int),
        "amam_in": nan_bins.copy(),
        "amam_out": nan_bins.copy(),
        "amam_spread": nan_bins.copy(),
        "ampm_deg": nan_bins.copy(),
        "ampm_spread_deg": nan_bins.copy(),
        "joint_top_slope": float("nan"),
        "top_amam_spread_db": float("nan"),
        "top_ampm_spread_deg": float("nan"),
        **{
            f"band_{q}_{ref}": np.full((nb, bins), np.nan)
            for q in ("gain_db", "gain_spread_db", "phase_deg", "phase_spread_deg")
            for ref in REFS
        },
        **{f"band_counts_{ref}": np.zeros((nb, bins), dtype=int) for ref in REFS},
        "envelope_peak": float("nan"),
        **_layout(c),
    }
    X, Y = c["X"], c["Y"]
    if X is None or Y is None:
        return out
    ax = np.abs(X)
    peak = float(ax.max())
    a, b = ax / peak, np.abs(Y) / peak
    idx = _bin_index(a, bins)
    counts = np.bincount(idx, minlength=bins).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_a = np.bincount(idx, weights=a, minlength=bins) / counts
        mean_b = np.bincount(idx, weights=b, minlength=bins) / counts
        spread_b = np.sqrt(
            np.bincount(idx, weights=(b - mean_b[idx]) ** 2, minlength=bins) / counts
        )
    ph_mean, ph_spread, dev = _phase_bins(idx, np.angle(Y * np.conj(X)), bins, counts)
    sel = (out["bin_centres"] > slope_band[0]) & (out["bin_centres"] < slope_band[1]) & (counts > 0)
    if np.count_nonzero(sel) >= 2:
        out["joint_top_slope"] = float(np.polyfit(mean_a[sel], mean_b[sel], 1)[0])
    top = a >= top_from
    if top.any():
        it = idx[top]
        with np.errstate(invalid="ignore", divide="ignore"):
            g_db = 20.0 * np.log10(
                b[top] / a[top]
            )  # each sample's gain, so a bin's width adds none
            mean_g = np.bincount(it, weights=g_db, minlength=bins) / np.bincount(it, minlength=bins)
        out["top_amam_spread_db"] = float(np.sqrt(np.mean((g_db - mean_g[it]) ** 2)))
        out["top_ampm_spread_deg"] = float(np.rad2deg(np.sqrt(np.mean(dev[top] ** 2))))
    out.update(
        {
            "counts": counts.astype(int),
            "amam_in": mean_a,
            "amam_out": mean_b,
            "amam_spread": spread_b,
            "ampm_deg": np.rad2deg(ph_mean),
            "ampm_spread_deg": np.rad2deg(ph_spread),
        }
    )
    env = sum(np.abs(bx) for bx, _by in c["base"].values())  # joint_envelope, own rate
    out["envelope_peak"] = float(env.max())
    ie = _bin_index(env / env.max(), bins)
    for k, (zx, zy) in c["bands"].items():
        ak = np.abs(zx)
        bx, by = c["base"][k]
        for ref, ix, pair in (
            ("vs_joint", idx, (zx, zy)),
            ("vs_own", _bin_index(ak / ak.max(), bins), (zx, zy)),
            ("vs_envelope", ie, (bx, by)),
        ):
            r = _band_bins(*pair, ix, bins)
            out[f"band_counts_{ref}"][k] = r["counts"]
            for q in ("gain_db", "gain_spread_db", "phase_deg", "phase_spread_deg"):
                out[f"band_{q}_{ref}"][k] = r[q]
    return out


def joint_compression_sweep(
    xs,
    ys,
    fs: float,
    spacings_hz,
    *,
    levels_db=None,
    oversample: float = 4.0,
) -> dict[str, Any]:
    """What the composite input would be with the bands ``spacings_hz`` apart.

    The bands (in the order given) are placed equally spaced around the centre, offsets
    ``(k - (N - 1) / 2) * spacing``, with the inputs scaled as in :func:`joint_compression`
    (one common scale, then ``levels_db``).

    Only the input side is a what-if: the outputs ``ys`` were measured with the bands at their
    real spacing, where the PA saw that composite, so no output figure (joint gain compression,
    output PAPR, band compression at the joint peaks) is returned for another spacing. ``ys``
    is checked against ``xs`` and not used otherwise (``None`` is accepted).

    Returns ``spacing_hz`` and ``joint_papr_in_db`` (lists, one per spacing), ``pe_papr_in_db``
    (the spacing-free bound) and ``levels_db``.
    """
    xs, _ys, lv = _bands_in(xs, ys, levels_db)
    n, nb = len(xs[0]), len(xs)
    scaled = [x * s for x, s in zip(xs, _scales(xs, lv))]
    spacings = [float(s) for s in np.atleast_1d(spacings_hz)]
    paprs = []
    for sp in spacings:
        offsets = [(k - (nb - 1) / 2.0) * sp for k in range(nb)]
        up = _upsample_factor(offsets, fs, oversample)
        X = np.zeros(n * up, dtype=np.complex128)
        for z, off in zip(scaled, offsets):
            X += _placed(z, up, int(round(off * n / float(fs))))
        paprs.append(papr_db(X))
    return {
        "spacing_hz": spacings,
        "joint_papr_in_db": paprs,
        "pe_papr_in_db": _pe_papr_db(scaled),
        "levels_db": [float(v) for v in lv],
    }


def _small_signal_gain(x: np.ndarray, s: np.ndarray, small: np.ndarray) -> complex | None:
    """The least-squares complex gain ``x -> s`` on the samples ``small`` (None: no energy)."""
    ex = float(np.vdot(x[small], x[small]).real)
    if ex <= 0.0:
        return None
    return complex(np.vdot(x[small], s[small]) / ex)


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
