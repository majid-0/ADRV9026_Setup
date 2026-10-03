from __future__ import annotations

import numpy as np
import pytest

from adrvtrx.align import match_corr
from adrvtrx.metrics import (
    PA_CLIP_SLOPE,
    aclr_db,
    gain_compression_db,
    inband_corr,
    nmse_db,
    papr_db,
    period_metrics,
    rms_dbfs,
    window_compression_db,
)
from sim_bench import BW, FS, make_signal, rapp


def test_papr_constant_envelope_is_zero():
    x = np.exp(1j * np.linspace(0, 20, 1000))
    assert papr_db(x) == pytest.approx(0.0, abs=1e-9)


def test_papr_single_spike():
    x = np.ones(100, dtype=complex)
    x[10] = 10.0
    mean = (99 + 100) / 100
    assert papr_db(x) == pytest.approx(10 * np.log10(100 / mean))


def test_papr_empty_and_zero_are_nan():
    assert np.isnan(papr_db([]))
    assert np.isnan(papr_db(np.zeros(8)))


def test_papr_is_scale_invariant():
    x = make_signal()
    assert papr_db(123.0 * x) == pytest.approx(papr_db(x))


def test_nmse_ignores_gain_and_phase():
    x = make_signal()
    assert nmse_db(x, 0.3 * np.exp(1j * 1.2) * x) < -250


def test_nmse_matches_snr():
    rng = np.random.default_rng(3)
    x = make_signal(n=1 << 16)
    px = np.mean(np.abs(x) ** 2)
    noise = rng.normal(size=x.size) + 1j * rng.normal(size=x.size)
    noise *= np.sqrt(px * 10 ** (-30 / 10) / np.mean(np.abs(noise) ** 2))
    assert nmse_db(x, 2.0 * x + 2.0 * noise) == pytest.approx(-30.0, abs=0.2)


def test_nmse_length_mismatch_raises():
    with pytest.raises(ValueError):
        nmse_db(np.ones(4), np.ones(5))


def test_nmse_zero_reference_is_nan():
    assert np.isnan(nmse_db(np.zeros(4), np.ones(4)))


def test_aclr_known_adjacent_tone():
    n = 1 << 14
    x = make_signal(n=n)
    p_main = np.sum(np.abs(np.fft.fft(x * np.hanning(n))) ** 2)
    t = np.arange(n) / FS
    tone_f = round(BW * 1.0 * n / FS) * FS / n  # on a bin inside the upper channel
    amp = np.sqrt(np.mean(np.abs(x) ** 2) * 10 ** (-40 / 10))
    s = x + amp * np.exp(2j * np.pi * tone_f * t)
    lower, upper = aclr_db(s, FS, BW)
    p_tone = np.sum(np.abs(np.fft.fft(amp * np.exp(2j * np.pi * tone_f * t) * np.hanning(n))) ** 2)
    assert upper == pytest.approx(10 * np.log10(p_tone / p_main), abs=0.05)
    assert upper == pytest.approx(-40.0, abs=0.5)
    assert lower < -100


def test_aclr_grows_with_compression():
    x = make_signal(n=1 << 14)
    lin = aclr_db(x, FS, BW)
    comp = aclr_db(rapp(1.8 * x), FS, BW)
    assert max(comp) > max(lin) + 20


def test_window_compression_on_rapp():
    x = make_signal(n=1 << 14)
    mild, _, _ = window_compression_db(x, rapp(0.5 * x), FS, 1e-3)
    hard, pin, pout = window_compression_db(x, rapp(2.0 * x), FS, 1e-3)
    assert 0 < mild < 0.5
    assert hard > 2.0
    assert hard == pytest.approx(pin - pout)


def test_window_is_centered_on_the_input_peak():
    x = np.full(1000, 0.1, dtype=complex)
    x[900] = 1.0
    y = x.copy()
    y[100] = 5.0  # outside a 100-sample window around the peak
    comp, _, _ = window_compression_db(x, y, fs=1.0, window_s=100)
    assert comp == pytest.approx(0.0, abs=1e-9)


def test_window_compression_empty():
    comp, pin, pout = window_compression_db([], [], FS)
    assert np.isnan(comp) and np.isnan(pin) and np.isnan(pout)


def test_inband_corr_scaled_copy_is_one():
    x = make_signal()
    assert inband_corr(x, (0.2 - 0.7j) * x) == pytest.approx(1.0)


def test_inband_corr_ignores_out_of_band_energy():
    x = make_signal()
    t = np.arange(x.size) / FS
    tone_f = round(0.4 * x.size) * FS / x.size  # exactly on a bin, well outside the band
    y = x + 5.0 * np.exp(2j * np.pi * tone_f * t)
    assert inband_corr(x, y) == pytest.approx(1.0, abs=1e-9)


def test_match_corr_uses_inband_corr():
    x = make_signal()
    y = np.roll(np.tile(x, 2), 100)
    assert match_corr(x, y, FS) == pytest.approx(1.0, abs=1e-4)


def test_rms_dbfs_full_scale_tone():
    s = 2048 * np.exp(1j * np.linspace(0, 50, 1000))
    assert rms_dbfs(s, 12) == pytest.approx(0.0, abs=1e-9)
    assert rms_dbfs(np.zeros(4), 12) == float("-inf")


def test_period_metrics_keys_and_corr_reference():
    x = make_signal()
    y = 100 * rapp(1.5 * x)
    m = period_metrics(x, y, fs=FS, bw_hz=BW, rx_bits=12, delay_samples=12.5)
    assert set(m) == {
        "delay_samples",
        "delay_ns",
        "corr",
        "nmse_db",
        "papr_in_db",
        "papr_out_db",
        "papr_compression_db",
        "aclr_lower_dbc",
        "aclr_upper_dbc",
        "rms_dbfs",
        "gain_compression_db",
        "amam_top_slope",
        "pa_clipped",
    }
    assert m["delay_ns"] == pytest.approx(12.5 / FS * 1e9)
    assert m["papr_compression_db"] == pytest.approx(m["papr_in_db"] - m["papr_out_db"])
    other = period_metrics(x, y, fs=FS, bw_hz=BW, rx_bits=12, delay_samples=0, corr_ref=y)
    assert other["corr"] == pytest.approx(1.0)
    assert other["nmse_db"] == m["nmse_db"]


def _clipper(x, level):
    return np.where(np.abs(x) < level, x, level * x / np.maximum(np.abs(x), 1e-300))


def test_gain_compression_linear_pa():
    x = make_signal()
    comp, slope = gain_compression_db(x, 2.5 * np.exp(0.3j) * x)
    assert comp == pytest.approx(0.0, abs=1e-9)
    assert slope == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("drive", [0.5, 1.0, 1.5, 2.0])
def test_gain_compression_matches_rapp(drive):
    """Power gain at the peak samples relative to the small-signal gain, Rapp p = 2."""
    x = make_signal()
    comp, _ = gain_compression_db(x, rapp(drive * x))
    a = np.abs(x) / np.abs(x).max()
    top = a >= 0.975
    expected = -10 * np.log10(
        np.mean(np.abs(rapp(drive * a[top])) ** 2 / drive**2) / np.mean(a[top] ** 2)
    )
    assert comp == pytest.approx(expected, abs=0.05)


def test_top_slope_falls_with_drive():
    x = make_signal()
    slopes = [gain_compression_db(x, rapp(d * x))[1] for d in (0.5, 1.0, 1.5, 2.0)]
    assert all(a > b for a, b in zip(slopes, slopes[1:]))


def test_hard_clipper_has_a_flat_top_and_is_flagged():
    x = make_signal()
    y = _clipper(x, 0.7)
    comp, slope = gain_compression_db(x, y)
    assert slope == pytest.approx(0.0, abs=1e-9)
    assert comp > 2.5
    m = period_metrics(x, y, fs=FS, bw_hz=BW, rx_bits=12, delay_samples=0)
    assert m["pa_clipped"] is True
    assert m["amam_top_slope"] < PA_CLIP_SLOPE


def test_pa_clipped_threshold_is_a_parameter():
    x = make_signal()
    y = rapp(1.5 * x)  # top slope about 0.19
    assert not period_metrics(x, y, fs=FS, bw_hz=BW, rx_bits=12, delay_samples=0)["pa_clipped"]
    m = period_metrics(x, y, fs=FS, bw_hz=BW, rx_bits=12, delay_samples=0, min_top_slope=0.3)
    assert m["pa_clipped"]


def test_gain_compression_is_scale_and_phase_invariant():
    x = make_signal()
    y = rapp(1.5 * x)
    assert gain_compression_db(x, y) == pytest.approx(gain_compression_db(x, 37j * y))
    assert gain_compression_db(x, y) == pytest.approx(gain_compression_db(5 * x, y))


def test_gain_compression_repeats_across_noisy_captures():
    """Same waveform, independent noise at -45 dB: within the 0.2 dB lock tolerance."""
    x = make_signal()
    y = rapp(1.5 * x)
    rms = np.sqrt(np.mean(np.abs(y) ** 2))
    rng = np.random.default_rng(3)
    readings = []
    for _ in range(5):
        noise = (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) / np.sqrt(2)
        readings.append(gain_compression_db(x, y + 10 ** (-45 / 20) * rms * noise)[0])
    assert max(readings) - min(readings) < 0.2


def test_gain_compression_empty_and_zero():
    assert all(np.isnan(v) for v in gain_compression_db([], []))
    assert all(np.isnan(v) for v in gain_compression_db(np.zeros(16), np.zeros(16)))
