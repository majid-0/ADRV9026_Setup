"""Joint compression of several bands into one PA (``adrvtrx.metrics.joint_compression``).

The bench here is a memoryless Rapp PA on the true composite: every band placed at its carrier
offset (exact periodic up-sampling), summed, compressed, and each band's output taken back out
of its own +-fs/2 (what its ORx sees). The geometry is the bench's: 1440 / 1860 MHz at
491.52 MSPS, 100 MHz-like signals.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from adrvtrx.metrics import (
    composite_pair,
    composite_signal,
    gain_compression_db,
    joint_amam_ampm,
    joint_compression,
    joint_compression_sweep,
    joint_envelope,
    levels_from_attens,
    papr_db,
)

N = 8192
FS = 491.52e6
FREQS = (1.44e9, 1.86e9)
UP = 8  # the composite rate of the test PA: 8 * FS covers the 420 MHz spacing


def _signal(seed: int, bw: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.fft.ifft(
        np.fft.fft(rng.normal(size=N) + 1j * rng.normal(size=N))
        * (np.abs(np.fft.fftfreq(N)) < bw / 2)
    )
    return x / np.abs(x).max()


def _rapp(v: np.ndarray, p: float = 2.0) -> np.ndarray:
    return v / (1.0 + np.abs(v) ** (2 * p)) ** (1.0 / (2 * p))


def _bins(off_hz: float) -> tuple[np.ndarray, np.ndarray, int]:
    f = np.fft.fftfreq(N)
    keep = np.abs(f) < 0.5
    return keep, np.round(f[keep] * N).astype(np.int64), int(round(off_hz * N / FS))


def _composite_pa(xs, drive: float, levels_db=(0.0, 0.0), pm_deg: float = 0.0, memory: float = 0.0):
    """Each band's output of a Rapp PA on the composite (complex gain per band, as an ORx), and
    the composite input's peak. ``pm_deg``: AM/PM of ``pm_deg * |v|^2`` degrees; ``memory``: a
    cubic term two samples late (a PA with memory)."""
    offsets = np.array(FREQS) - np.mean(FREQS)
    z = np.zeros(N * UP, complex)
    for x, lv, off in zip(xs, levels_db, offsets):
        keep, idx, sh = _bins(off)
        spec = np.zeros(N * UP, complex)
        spec[(idx + sh) % (N * UP)] = np.fft.fft(x * 10 ** (lv / 20))[keep] * UP
        z += np.fft.ifft(spec)
    v = drive * z
    w = _rapp(v) * np.exp(1j * np.deg2rad(pm_deg) * np.abs(v) ** 2)
    if memory:
        w = w + memory * np.roll(w * np.abs(w) ** 2, 2 * UP)
    w = np.fft.fft(w)
    ys = []
    for k, off in enumerate(offsets):
        keep, idx, sh = _bins(off)
        band = np.zeros(N, complex)
        band[keep] = w[(idx + sh) % (N * UP)] / UP
        ys.append(np.fft.ifft(band) * (3.0 + 1.0j) * (k + 1))  # each ORx its own gain and phase
    return ys, float(np.abs(z).max())


def _joint_cfr(xs, target_db: float, n_iter: int = 4, bw: float = 0.2):
    """The low-rate joint CFR (sum_k |x_k| clipped to the target, clipping noise filtered into each
    channel), as studies/cfr_interband/cfr_joint.joint_cfr without the peak window."""
    p = sum(np.mean(np.abs(x) ** 2) for x in xs)
    c = np.sqrt(p * 10 ** (target_db / 10))
    mask = np.abs(np.fft.fftfreq(N)) < bw / 2
    cur = list(xs)
    for _ in range(n_iter):
        s = sum(np.abs(x) for x in cur)
        g = np.where(s > c, c / np.maximum(s, 1e-30), 1.0)
        cur = [x + np.fft.ifft(np.fft.fft(x * g - x) * mask) for x in cur]
    peak = max(np.abs(x).max() for x in cur)
    return [x / peak for x in cur]  # TX codes: the strongest band at full scale


def _reference_composite(xs, fs, offsets, oversample=4.0):
    """``cfr_joint.composite`` (studies/cfr_interband; tests/reference), verbatim: the oracle."""
    n = len(xs[0])
    span = (max(offsets) - min(offsets)) + fs
    L = max(1, int(np.ceil(oversample * span / fs)))
    nh = n * L
    f = np.fft.fftfreq(n, 1 / fs)
    keep = np.abs(f) < fs / 2
    idx = np.round(f[keep] * n / fs).astype(np.int64)
    Z = np.zeros(nh, complex)
    for x, off in zip(xs, offsets):
        Z[(idx + int(round(off * n / fs))) % nh] += np.fft.fft(x)[keep] * L
    return np.fft.ifft(Z)


@pytest.fixture(scope="module")
def bands():
    return [_signal(1), _signal(2)]


# -- physics ----------------------------------------------------------------------------------


def test_a_compressive_pa_on_the_composite_compresses_jointly_and_more_with_drive(bands):
    joint, slope = [], []
    for drive in (0.3, 0.5, 0.8, 1.2):
        ys, _ = _composite_pa(bands, drive)
        r = joint_compression(bands, ys, FS, FREQS)
        joint.append(r["joint_gain_compression_db"])
        slope.append(r["joint_top_slope"])
        assert r["joint_papr_compression_db"] > 0
        assert all(v > 0 for v in r["band_compression_at_joint_peaks_db"])
        assert (
            abs(r["joint_refit_gain_db"]) < 0.5
        )  # the per-band references agree with the composite
    assert joint[0] > 0
    assert np.all(np.diff(joint) > 0.2), joint
    assert np.all(np.diff(slope) < 0), slope


def test_the_composite_compresses_more_than_either_band_at_its_own_peaks(bands):
    ys, _ = _composite_pa(bands, 0.8)
    r = joint_compression(bands, ys, FS, FREQS)
    own = [gain_compression_db(x, y)[0] for x, y in zip(bands, ys)]
    assert r["joint_gain_compression_db"] > max(own) + 0.5


def test_with_one_band_off_it_is_the_single_band_gain_compression(bands):
    y1 = _rapp(1.6 * bands[0]) * np.exp(0.3j) * 7.0
    noise = np.random.default_rng(5).normal(size=N) + 0j  # the off band's ORx sees only noise
    xs, ys = [bands[0], np.zeros(N, complex)], [y1, noise]
    single = gain_compression_db(bands[0], y1)
    exact = joint_compression(xs, ys, FS, FREQS, oversample=0.5)  # no up-sampling: same samples
    assert exact["upsample"] == 1
    # equal but for the Nyquist bin, which the composite drops
    assert exact["joint_gain_compression_db"] == pytest.approx(single[0], abs=1e-4)
    assert exact["joint_top_slope"] == pytest.approx(single[1], abs=1e-4)
    assert exact["joint_papr_in_db"] == pytest.approx(papr_db(bands[0]), abs=1e-9)
    r = joint_compression(xs, ys, FS, FREQS)  # up-sampled: the inter-sample peaks too
    assert r["joint_gain_compression_db"] == pytest.approx(single[0], abs=0.05)
    assert r["band_compression_at_joint_peaks_db"][0] == pytest.approx(single[0], abs=0.05)
    assert np.isnan(r["band_compression_at_joint_peaks_db"][1])
    assert r["pe_papr_in_db"] == pytest.approx(papr_db(bands[0]), abs=1e-9)


def test_joint_cfr_needs_more_drive_for_the_same_band_compression_and_reads_alike_at_equal_drive(
    bands,
):
    cfr = _joint_cfr(bands, 8.0)
    papr = [
        joint_compression_sweep(x, None, FS, [420e6])["joint_papr_in_db"][0] for x in (bands, cfr)
    ]
    assert papr[0] - papr[1] > 1.5  # the composite PAPR drops
    assert all(abs(papr_db(a) - papr_db(b)) < 0.4 for a, b in zip(bands, cfr))  # per band: not much

    def drive_for(xs, target=1.5):  # band 1 at ``target`` dB of its own compression
        lo, hi = 0.05, 5.0
        for _ in range(40):
            mid = np.sqrt(lo * hi)
            ys, _ = _composite_pa(xs, mid)
            lo, hi = (mid, hi) if gain_compression_db(xs[0], ys[0])[0] < target else (lo, mid)
        return mid

    d_orig, d_cfr = drive_for(bands), drive_for(cfr)
    assert 20 * np.log10(d_cfr / d_orig) > 0.5  # more drive for the same per-band compression

    ys_orig, peak_orig = _composite_pa(bands, d_orig)
    _, peak_cfr = _composite_pa(cfr, 1.0)
    ys_cfr, _ = _composite_pa(cfr, d_orig * peak_orig / peak_cfr)  # the same composite peak
    j_orig = joint_compression(bands, ys_orig, FS, FREQS)["joint_gain_compression_db"]
    j_cfr = joint_compression(cfr, ys_cfr, FS, FREQS)["joint_gain_compression_db"]
    own_orig = [gain_compression_db(x, y)[0] for x, y in zip(bands, ys_orig)]
    own_cfr = [gain_compression_db(x, y)[0] for x, y in zip(cfr, ys_cfr)]
    assert abs(j_orig - j_cfr) < 0.3
    assert min(abs(a - b) for a, b in zip(own_orig, own_cfr)) > 2 * abs(j_orig - j_cfr)


def test_the_composite_is_cfr_joint_composite(bands):
    offsets = [f - np.mean(FREQS) for f in FREQS]
    ref = _reference_composite(bands, FS, offsets)
    r = joint_compression(bands, bands, FS, FREQS)
    assert r["joint_papr_in_db"] == pytest.approx(papr_db(ref), abs=1e-9)
    assert r["upsample"] == len(ref) // N
    sweep = joint_compression_sweep(bands, bands, FS, [60e6, 420e6, 900e6])
    for sp, got in zip(sweep["spacing_hz"], sweep["joint_papr_in_db"]):
        want = papr_db(_reference_composite(bands, FS, [-sp / 2, sp / 2]))
        assert got == pytest.approx(want, abs=1e-9)
    s = sum(np.abs(x) for x in bands)
    pe = 10 * np.log10(s.max() ** 2 / sum(np.mean(np.abs(x) ** 2) for x in bands))
    assert r["pe_papr_in_db"] == pytest.approx(pe, abs=1e-9) == sweep["pe_papr_in_db"]


# ``cfr_joint.py`` is vendored verbatim from the Polynomials ``studies/cfr_interband`` study
# into ``tests/reference``; ``CFR_INTERBAND_DIR`` points the test at another copy instead.
STUDY = Path(os.environ.get("CFR_INTERBAND_DIR", Path(__file__).parent / "reference"))


@pytest.mark.skipif(
    not (STUDY / "cfr_joint.py").is_file(), reason="cfr_joint.py is not at CFR_INTERBAND_DIR"
)
def test_the_composite_matches_the_study_itself(bands):
    sys.path.insert(0, str(STUDY))
    try:
        import cfr_joint
    finally:
        sys.path.remove(str(STUDY))
    for sp in (420e6, 900e6):
        got = joint_compression(bands, bands, FS, (1e9, 1e9 + sp))
        assert got["joint_papr_in_db"] == pytest.approx(
            cfr_joint.composite_papr(bands, FS, sp), abs=1e-9
        )
        assert got["pe_papr_in_db"] == pytest.approx(cfr_joint.pe_papr(bands), abs=1e-9)


def test_the_scale_of_the_inputs_and_of_each_output_does_not_matter(bands):
    ys, _ = _composite_pa(bands, 0.8)
    a = joint_compression(bands, ys, FS, FREQS)
    b = joint_compression(
        [2047.0 * x for x in bands], [ys[0] * (0.01 - 0.3j), ys[1] * 55.0j], FS, FREQS
    )
    for k in ("joint_gain_compression_db", "joint_top_slope", "joint_papr_in_db",
              "joint_papr_out_db", "pe_papr_in_db", "joint_refit_gain_db"):  # fmt: skip
        assert b[k] == pytest.approx(a[k], abs=1e-9), k
    assert b["band_compression_at_joint_peaks_db"] == pytest.approx(
        a["band_compression_at_joint_peaks_db"], abs=1e-6  # float32 envelopes
    )


def test_levels_add_to_the_codes_and_only_their_difference_matters(bands):
    ys, _ = _composite_pa(bands, 0.8, levels_db=(0.0, -6.0))
    keys = ("joint_gain_compression_db", "joint_papr_in_db", "pe_papr_in_db")
    a = joint_compression(bands, ys, FS, FREQS, levels_db=[0.0, -6.0])
    b = joint_compression(bands, ys, FS, FREQS, levels_db=[3.0, -3.0])
    c = joint_compression([bands[0], bands[1] * 10 ** (-6 / 20)], ys, FS, FREQS)  # in the codes
    for k in keys:
        assert b[k] == pytest.approx(a[k], abs=1e-9) and c[k] == pytest.approx(a[k], abs=1e-9)


def test_levels_move_the_composite_the_expected_way(bands):
    # band 2 is 6 dB down at the PA, and compressed more than band 1 where the composite peaks
    ys, _ = _composite_pa(bands, 0.8, levels_db=(0.0, -6.0))
    rows = [
        joint_compression(bands, ys, FS, FREQS, levels_db=[0.0, lv])
        for lv in (-80.0, -12.0, -6.0, -3.0, 0.0)
    ]
    right = rows[2]
    assert (
        right["band_compression_at_joint_peaks_db"][1]
        > right["band_compression_at_joint_peaks_db"][0]
    )
    # the more band 2 is assumed to weigh, the more its compression counts
    joint = [r["joint_gain_compression_db"] for r in rows]
    assert np.all(np.diff(joint) > 0), joint
    # band 2 assumed away: band 1 alone (its PAPR up-sampled; its compression)
    alone = joint_compression([bands[0], 0 * bands[1]], ys, FS, FREQS)
    for k in ("joint_gain_compression_db", "joint_papr_in_db", "pe_papr_in_db"):
        assert rows[0][k] == pytest.approx(alone[k], abs=1e-3), k
    # the levels pick which instants are the composite's peaks
    assert rows[0]["band_compression_at_joint_peaks_db"] != pytest.approx(
        right["band_compression_at_joint_peaks_db"], abs=1e-2
    )


def test_levels_from_attenuations():
    assert levels_from_attens([6.0, 7.6]) == pytest.approx([0.0, -1.6])
    assert levels_from_attens([4.65, 6.6]) == pytest.approx([0.0, -1.95])
    assert levels_from_attens([6.0, 7.6], [0.0, 2.0]) == pytest.approx([-0.4, 0.0])
    with pytest.raises(ValueError, match="one path gain per band"):
        levels_from_attens([6.0, 7.6], [0.0])


def test_the_sweep_returns_only_the_input_side(bands):
    ys, _ = _composite_pa(bands, 0.8)
    sw = joint_compression_sweep(bands, ys, FS, [140e6, 420e6], levels_db=[0.0, -1.0])
    assert set(sw) == {"spacing_hz", "joint_papr_in_db", "pe_papr_in_db", "levels_db"}
    assert sw["spacing_hz"] == [140e6, 420e6] and len(sw["joint_papr_in_db"]) == 2
    at = joint_compression(bands, ys, FS, FREQS, levels_db=[0.0, -1.0])
    assert sw["joint_papr_in_db"][1] == pytest.approx(at["joint_papr_in_db"], abs=1e-9)
    assert sw["pe_papr_in_db"] == pytest.approx(at["pe_papr_in_db"], abs=1e-9)


def test_bad_inputs_are_refused_and_a_band_without_output_gives_nan(bands):
    with pytest.raises(ValueError, match="same length"):
        joint_compression([bands[0], bands[1][:-1]], bands, FS, FREQS)
    with pytest.raises(ValueError, match="one carrier per band"):
        joint_compression(bands, bands, FS, FREQS[:1])
    with pytest.raises(ValueError, match="one level per band"):
        joint_compression(bands, bands, FS, FREQS, levels_db=[0.0])
    r = joint_compression(bands, [bands[0], np.zeros(N, complex)], FS, FREQS)
    assert np.isfinite(r["joint_papr_in_db"]) and np.isnan(r["joint_gain_compression_db"])
    assert np.isnan(joint_compression([0 * bands[0]] * 2, bands, FS, FREQS)["joint_papr_in_db"])


# -- joint AM/AM and AM/PM --------------------------------------------------------------------


def test_a_linear_pa_has_a_straight_joint_amam_no_ampm_and_no_spread(bands):
    r = joint_amam_ampm(bands, [bands[0] * 2.0j, bands[1] * 0.5], FS, FREQS, levels_db=[0, -3])
    full = r["counts"] > 0
    assert r["amam_out"][full] == pytest.approx(r["amam_in"][full], abs=1e-9)
    assert np.nanmax(np.abs(r["ampm_deg"])) < 1e-6
    assert r["top_amam_spread_db"] < 1e-9 and r["top_ampm_spread_deg"] < 1e-6
    assert r["joint_top_slope"] == pytest.approx(1.0, abs=1e-9)
    for q in ("band_gain_db_vs_joint", "band_gain_db_vs_own", "band_phase_deg_vs_own"):
        assert np.nanmax(np.abs(r[q])) < 1e-6, q
    assert r["counts"].sum() == N * r["upsample"]


def test_the_joint_amam_compresses_and_its_top_slope_is_joint_compressions(bands):
    ys, _ = _composite_pa(bands, 0.8, pm_deg=20.0)
    r = joint_amam_ampm(bands, ys, FS, FREQS)
    j = joint_compression(bands, ys, FS, FREQS)
    assert r["joint_top_slope"] == pytest.approx(j["joint_top_slope"], abs=1e-12)
    gain = r["amam_out"] / r["amam_in"]
    assert np.all(gain[-5:] < 0.9) and np.all(gain[:5] == pytest.approx(1.0, abs=0.01))
    # AM/PM at the top (the small-signal end sits a little off 0: each band's g0 is fitted
    # while the other band rotates it too)
    assert np.all(r["ampm_deg"][-5:] > 10.0) and np.all(np.abs(r["ampm_deg"][:5]) < 3.0)
    assert np.mean(r["ampm_deg"][-5:]) - np.mean(r["ampm_deg"][:5]) > 15.0
    assert r["band_gain_db_vs_joint"].shape == (2, 40)


def test_memory_adds_to_the_spread_at_the_top(bands):
    spread = []
    for memory in (0.0, 0.1, 0.2):
        ys, _ = _composite_pa(bands, 0.8, memory=memory)
        spread.append(joint_amam_ampm(bands, ys, FS, FREQS)["top_amam_spread_db"])
    assert spread[0] > 0  # memoryless, still some: each band's gain is not a function of |X|
    assert np.all(np.diff(spread) > 0.03), spread


def test_a_weak_band_is_compressed_by_the_strong_one_where_the_composite_peaks(bands):
    ys, _ = _composite_pa(bands, 0.8, levels_db=(0.0, -10.0))
    r = joint_amam_ampm(bands, ys, FS, FREQS, levels_db=[0.0, -10.0])
    weak_joint = r["band_gain_db_vs_joint"][1, -6:]
    weak_own = r["band_gain_db_vs_own"][1, -6:]
    assert np.all(weak_joint < -1.0) and np.all(np.abs(weak_own) < 0.2)
    # each band below its floor (2 % of its peak) is left out, never where the composite peaks
    assert np.array_equal(r["band_counts_vs_joint"][0, -6:], r["band_counts_vs_joint"][1, -6:])


# -- the joint envelope per sample, and the composite itself ------------------------------------


def test_the_joint_envelope_is_the_peak_estimate_per_sample(bands):
    lv = [0.0, -4.0]
    env = joint_envelope([2047 * x for x in bands], levels_db=lv)  # codes: one scale for all
    scaled = [bands[0], bands[1] * 10 ** (-4 / 20)]
    assert env == pytest.approx(np.abs(scaled[0]) + np.abs(scaled[1]), abs=1e-12)
    pe = 10 * np.log10(env.max() ** 2 / sum(np.mean(np.abs(x) ** 2) for x in scaled))
    assert pe == pytest.approx(
        joint_compression(bands, bands, FS, FREQS, levels_db=lv)["pe_papr_in_db"]
    )
    for sp in (60e6, 420e6, 900e6):  # the composite at any spacing stays under it, sample by sample
        x_c, fs_c = composite_signal(bands, FS, (1e9, 1e9 + sp), levels_db=lv)
        up = round(fs_c / FS)
        assert len(x_c) == N * up
        assert np.all(np.abs(x_c[::up]) <= env + 1e-12)
        assert 0.9 * env.max() < np.abs(x_c).max() < 1.02 * env.max()  # reached where they align


def test_the_public_composite_is_the_reference_composite_on_the_pa_input_scale(bands):
    offsets = [f - np.mean(FREQS) for f in FREQS]
    x_c, fs_c = composite_signal([3.0 * x for x in bands], FS, FREQS)
    ref = _reference_composite(bands, FS, offsets)  # the peaks are 1: the same scale
    assert fs_c == pytest.approx(FS * len(ref) / N)
    assert x_c == pytest.approx(ref, abs=1e-12)
    xp, yp, fs_p = composite_pair(bands, [bands[0] * 2j, bands[1] * -0.3], FS, FREQS)
    assert (
        fs_p == fs_c and xp == pytest.approx(ref, abs=1e-12) and yp == pytest.approx(xp, abs=1e-12)
    )
    assert composite_pair(bands, [bands[0], 0 * bands[1]], FS, FREQS)[1] is None
    with pytest.raises(ValueError, match="needs the outputs"):
        composite_pair(bands, None, FS, FREQS)


def test_each_band_against_the_joint_envelope_at_its_own_samples(bands):
    lin = joint_amam_ampm(bands, [bands[0] * 2j, bands[1] * 0.5], FS, FREQS)
    assert np.nanmax(np.abs(lin["band_gain_db_vs_envelope"])) < 1e-9
    assert np.all(lin["band_counts_vs_envelope"].sum(axis=1) <= N)  # the bands' own samples
    assert lin["envelope_peak"] == pytest.approx(joint_envelope(bands).max())
    ys, _ = _composite_pa(bands, 0.8, levels_db=(0.0, -10.0))
    r = joint_amam_ampm(bands, ys, FS, FREQS, levels_db=[0.0, -10.0])
    full = r["band_counts_vs_envelope"][1] > 0
    weak = r["band_gain_db_vs_envelope"][1][full]
    assert np.all(weak[-3:] < -1.0)  # where the joint envelope peaks, the strong band compresses it
    assert np.all(np.abs(r["band_gain_db_vs_own"][1, -6:]) < 0.2)  # not at its own peaks
