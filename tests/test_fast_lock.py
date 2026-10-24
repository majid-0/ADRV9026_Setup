"""The fast lock (Task 6) on the fake board: the probe, one snapshot per step, the learnt ORx
step, readings outside the AGC window, the joint and pingpong moves, the full-signal confirmation.

The fake PA here is the first bench lock's (S20-100, TX2 + TX3 into one PA): about 0.3 dB of
compression per dB of a band's own attenuation and 0.06 per dB of the other band's, near a 4 dB
lock at 15 dB; ORX3's gain table can be made 0.8 dB per index, as the bench's is near 202..204.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel, multiband
from adrvtrx.capture import capture, shares_orx_adc
from adrvtrx.compression import relevel_index
from adrvtrx.config import Config, DllConfig
from adrvtrx.fake import FakeRadio
from adrvtrx.gain import OrxStep, clip_report
from adrvtrx.multiband import (
    BandSpec,
    find_multiband_operating_point,
    lock_bands,
    timing_text,
)
from adrvtrx.probe import ProbeSettings

N = 65536
#: own ~0.3, cross ~0.06 dB/dB near a 4 dB lock at 15 dB (measured on this model; noise-free)
BENCH_PA = {"shared": [["TX2", "TX3"]], "coupling": 10.0, "smoothness": 0.4, "drive": 3.0,
            "noise_codes": 0.5}  # fmt: skip
#: The probe of these signals at the fake board's 10 Msps (the bench's default is 600 / 16).
PROBE = ProbeSettings(window=300, crossfade=8)


def _signal(seed: int, n: int = N) -> np.ndarray:
    """A crest-factor-reduced band-limited signal, as test_probe's (a dozen peak events)."""
    from test_probe import cfr_signal

    return cfr_signal(seed, n)


def _radio(pa=None, seed: int = 2) -> FakeRadio:
    radio = FakeRadio(
        Config(dll=DllConfig(install_dir=Path("C:/nonexistent"))), seed=seed, pa=pa or BENCH_PA
    )
    radio._safe_hooks_installed = True
    radio.connect()
    radio.program()
    return radio


def _bands(start=(22.0, 22.0), target=4.0, floor=5.0) -> list[BandSpec]:
    return [
        BandSpec(TxChannel.TX2, RxChannel.ORX2, "LO1", 1_440_000_000, 2.0, _signal(1),
                 target_db=target, atten_start_db=start[0], atten_min_db=floor),
        BandSpec(TxChannel.TX3, RxChannel.ORX3, "LO2", 1_860_000_000, 2.0, _signal(2),
                 target_db=target, atten_start_db=start[1], atten_min_db=floor),
    ]  # fmt: skip


def _count_perform_rx(radio) -> list:
    """Every PerformRx through the radio: its capture length, ms (a spy with the same surface)."""
    calls, real = [], radio.perform_rx

    def spy(rx_init, capture_time_ms, **kw):
        calls.append(float(capture_time_ms))
        return real(rx_init, capture_time_ms, **kw)

    radio.perform_rx = spy
    return calls


# -- the probe in the lock (Task 6b B) -------------------------------------------------------------


def test_without_the_probe_every_lock_capture_is_two_full_periods():
    """The default: every capture of the lock is 2 periods of the full signal (AGC Stage A and B
    excepted: 0.1 ms, as always), and the TX plays the full signal throughout."""
    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    calls = _count_perform_rx(radio)
    search, refs = lock_bands(radio, bands, tx_bits=12, rx_bits=12, fs=fs)
    full_ms = round(2 * N / fs * 1e3, 6)
    got = [round(c, 6) for c in calls]
    assert set(got) <= {full_ms, 0.1} and got.count(full_ms) >= search.captures
    assert search.probe == {} and {h["signal"] for h in search.history} == {"full"}
    assert radio.board_model.state["tx_len"] == N and len(refs[0]) == N


def test_with_the_probe_every_lock_capture_is_two_probe_periods_and_tx_ends_on_the_full():
    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    calls = _count_perform_rx(radio)
    played = []
    real_play = multiband.play_codes
    multiband.play_codes = lambda r, b, codes, **k: (
        played.append(codes),
        real_play(r, b, codes, **k),
    )[1]
    try:
        search, refs = lock_bands(radio, bands, tx_bits=12, rx_bits=12, fs=fs, probe=PROBE)
    finally:
        multiband.play_codes = real_play
    built = PROBE.build(refs)
    assert search.probe["used"] and search.probe["n"] == built.n == len(played[0][0])
    for a, b in zip(played[0], built.codes):
        np.testing.assert_array_equal(a, b)  # the probe codes as built, crossfades included
    probe_ms = round(2 * built.n / fs * 1e3, 6)
    assert {round(c, 6) for c in calls} <= {probe_ms, 0.1}  # AGC Stage C and the search: probe
    assert {h["signal"] for h in search.history} == {"probe"}
    assert radio.board_model.state["tx_len"] == N  # swapped back to the full signal
    assert len(search.timing["swaps_s"]) == 2 and len(search.timing["agc_s"]) == 2
    assert search.probe["stats"][0]["peaks_covered"] == search.probe["stats"][0]["peaks_total"]


def test_a_probe_that_fails_its_build_check_leaves_the_lock_on_the_full_signals():
    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    search, _refs = lock_bands(radio, bands, tx_bits=12, rx_bits=12, fs=fs,
                               probe=ProbeSettings(window=100, crossfade=8))  # fmt: skip
    assert search.probe and not search.probe["used"] and not search.probe["ok"]
    assert (
        "mean power" in search.probe["reason"]
        and "the lock plays the full signals" in search.probe["reason"]
    )
    assert {h["signal"] for h in search.history} == {"full"} and search.converged


# -- F4: the ORx re-level learns its step --------------------------------------------------------


def _hot_orx3(radio, target_peak=(-0.25, -0.18)) -> tuple[int, float]:
    """TX2 + TX3 playing, ORX3 at gain 205 (0.8 dB per index on this path), TX3's attenuation
    set so the ORX3 peak sits just above the window: where a 0.5 dB/index re-level bounces."""
    bands = _bands()
    for b in bands:
        radio.retune_lo(b.lo, b.freq_hz)
    radio.set_tx_atten(TxChannel.TX2, 16.0)
    multiband.play_bands(radio, bands, tx_bits=12)
    radio.set_tx_atten(TxChannel.TX3, 16.0)
    gain = 205 + int(np.floor(-_peak(radio, 205) / 0.81))  # the peak 0 .. 0.8 dB below 0 dBFS
    for atten in np.arange(14.5, 17.55, 0.05):
        radio.set_tx_atten(TxChannel.TX3, float(atten))
        peak = _peak(radio, gain)
        if target_peak[0] <= peak <= target_peak[1]:
            return gain, peak
    raise AssertionError("no attenuation puts the ORX3 peak just above the window")


def _peak(radio, gain: int) -> float:
    radio.set_rx_gain(RxChannel.ORX3, gain)
    cap = capture(radio, int(RxChannel.ORX3), 2 * N / radio.board_model.rate_hz * 1e3, bits=12)
    c = cap.channels[RxChannel.ORX3]
    return clip_report(c.i, c.q, 12).peak_dbfs


def _relevel(radio, gain: int, rule, captures: int = 8) -> list[tuple[int, float]]:
    """Capture and re-level until the peak is in the window (-1.6 .. -0.7 dBFS)."""
    seen = []
    for _ in range(captures):
        peak = _peak(radio, gain)
        seen.append((gain, round(peak, 2)))
        if -1.6 <= peak <= -0.7:
            break
        gain += rule(gain, peak)
    return seen


BOUNCE_PA = {**BENCH_PA, "bands": {"TX3": {"orx_db_per_index": 0.81}}}  # the bench ORX3


def test_a_fixed_half_db_relevel_bounces_on_a_0p8_db_per_index_orx():
    """The first bench lock's TX3: -0.25 dBFS at one gain, -1.87 two indices down, never in
    (from about -0.18 dBFS and hotter, two indices of 0.81 dB fall below the window)."""
    radio = _radio(BOUNCE_PA)
    gain, _peak0 = _hot_orx3(radio)

    def fixed(_g, peak):
        return relevel_index(peak, 0, target_dbfs=-1.0, band_hi=-0.7, db_per_index=0.5)

    seen = _relevel(radio, gain, fixed)
    assert len(seen) == 8 and {g for g, _p in seen} == {gain, gain - 2}  # the bounce
    assert all(not -1.6 <= p <= -0.7 for _g, p in seen)


def test_the_learnt_relevel_settles_a_0p8_db_per_index_orx_in_at_most_3_captures():
    radio = _radio(BOUNCE_PA)
    gain, _peak0 = _hot_orx3(radio)
    step = OrxStep(clip_dbfs=0.0)  # the bench's -0.25 dBFS read unclipped then (12-bit, UC98)

    def learnt(g, peak):
        return step.relevel(g, peak, 0, target_dbfs=-1.0, band_lo=-1.6, band_hi=-0.7)

    seen = _relevel(radio, gain, learnt)
    assert len(seen) <= 3 and -1.6 <= seen[-1][1] <= -0.7, seen
    assert step.db_per_index == pytest.approx(0.8, abs=0.05)


def test_the_learnt_step_takes_the_closest_capture_when_one_index_jumps_the_window():
    step = OrxStep(clip_dbfs=0.0)  # -0.2 dBFS is no clip here: the step rule alone
    kw = {"target_dbfs": -1.0, "band_lo": -1.3, "band_hi": -0.8}  # a 0.5 dB window, 1.5 dB/index
    assert step.relevel(200, -0.2, 0, **kw) == -2  # 0.5 dB / index assumed
    assert step.relevel(198, -3.2, 0, **kw) == 1  # overshot: learnt 1.5, then 1-index steps
    assert step.db_per_index == pytest.approx(1.5)
    step.observe(199, -1.7, 0, band_lo=-1.3, band_hi=-0.8)
    step.observe(200, -0.2, 0, band_lo=-1.3, band_hi=-0.8)
    assert step.jumped and step.closest(-1.0) == (199, -1.7)


def test_the_single_band_search_shares_the_learnt_relevel():
    from adrvtrx.compression import search_tx_compression

    peaks = {}

    def measure():
        g = state["g"]
        peak = -1.0 + 0.8 * (g - 203)  # 0.8 dB per index; -1.0 dBFS at 203
        peaks[g] = peak
        return min(peak, 0.0), 0, 3.0 if -1.6 <= peak <= -0.7 else 2.0

    state = {"g": 204}
    res = search_tx_compression(
        lambda db: None, lambda g: state.__setitem__("g", g), measure,
        target_compression_db=3.0, start_atten_db=20.0, atten_min_db=5.0, orx_gain=204,
    )  # fmt: skip
    assert res.converged and len(res.history) <= 3  # -0.2 (hot), then in by the learnt step


# -- F5 / F6 on the fake board: one snapshot per step ------------------------------------------


@pytest.mark.parametrize("start", [(22.0, 22.0), (22.0, 20.0)])
def test_joint_locks_the_bench_pa_from_7_db_away_in_at_most_5_snapshots(start):
    radio, bands = _radio(), _bands(start=start)
    fs = radio.board_model.rate_hz
    calls = _count_perform_rx(radio)
    after_agc = []
    real_agc = multiband.autolevel_for

    def agc(*a, **k):
        out = real_agc(*a, **k)
        after_agc.append(len(calls))
        return out

    multiband.autolevel_for, saved = agc, multiband.autolevel_for
    try:
        search, _refs = lock_bands(radio, bands, tx_bits=12, rx_bits=12, fs=fs, probe=PROBE)
    finally:
        multiband.autolevel_for = saved
    assert search.converged, search.reason
    assert search.snapshots <= 5, search.snapshots
    assert search.snapshot and search.captures == search.snapshots
    assert len(calls) - after_agc[-1] == search.captures  # one PerformRx per snapshot
    assert len({h["step"] for h in search.history}) == search.steps
    assert search.steps == search.snapshots or any(
        h["action"].startswith("ORx gain") for h in search.history
    )


def test_pingpong_locks_the_bench_pa_from_7_db_away_in_at_most_12_snapshots():
    radio, bands = _radio(), _bands()
    search, _refs = lock_bands(
        radio, bands, tx_bits=12, rx_bits=12, fs=radio.board_model.rate_hz, probe=PROBE,
        method="pingpong",
    )  # fmt: skip
    assert search.converged, search.reason
    assert search.snapshots <= 12, search.snapshots


def test_without_a_two_orx_snapshot_each_band_is_captured_alone():
    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    search, _refs = lock_bands(
        radio, bands, tx_bits=12, rx_bits=12, fs=fs, probe=PROBE, snapshot=False
    )
    assert search.converged and not search.snapshot
    assert search.captures == len(search.history)  # one capture per band read
    assert search.snapshots < search.captures <= 2 * search.snapshots
    assert shares_orx_adc([RxChannel.ORX1, RxChannel.ORX2])
    assert not shares_orx_adc([RxChannel.ORX2, RxChannel.ORX3])


# -- F3 / F7: the full-signal confirmation and the timing --------------------------------------


def test_a_case_on_the_probe_is_confirmed_by_its_0_db_capture(tmp_path):
    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    writer = multiband.CaseWriter(tmp_path, bands, tx_bits=12, rx_bits=12, fs=fs)
    res = find_multiband_operating_point(
        radio, bands, tx_bits=12, rx_bits=12, fs=fs, backoffs=(0, 1), probe=PROBE,
        writer=writer,
    )  # fmt: skip
    s = res.search
    assert s.converged and s.confirm["converged_on"] in ("probe", "fallback"), s.reason
    gc0 = {c.name: c.metrics["gain_compression_db"] for c in res.captures[0]}
    assert s.confirm["gc_full_db"] == pytest.approx(gc0, abs=1e-3)  # the 0 dB capture IS it
    assert set(s.confirm["offset_db"]) == {"TX2", "TX3"}
    assert all(abs(v) < 0.5 for v in s.confirm["offset_db"].values())
    assert all(abs(g - 4.0) <= 0.1 for g in gc0.values())
    assert s.timing["total_s"] > 0 and s.timing["confirm_s"] and s.timing["captures_s"]
    rows = list(__import__("csv").DictReader(open(writer.dir / "conditions.csv")))
    assert {r["lock_signal"] for r in rows} == {"probe"} and all(r["probe_offset_db"] for r in rows)
    text = timing_text(s)
    assert "lock timing" in text and "probe -> full GC offset" in text
    doc = __import__("json").loads(writer.write_case_json().read_text())
    assert doc["lock_timing"]["total_s"] == s.timing["total_s"]
    assert doc["lock"][0]["probe_offset_db"] == s.confirm["offset_db"]["TX2"]


@pytest.mark.parametrize("confirm_steps", [0, 1, 2])
def test_the_fallback_runs_when_the_full_gc_is_off_and_stops_after_confirm_steps(
    confirm_steps, monkeypatch
):
    """The full signal reads 0.5 dB more compression than the probe on TX2 (an injected offset
    on every full capture): the search continues on full captures, at most confirm_steps."""
    real_case, real_reading = multiband.capture_case, multiband._reading

    def shifted_case(*a, **k):
        caps = real_case(*a, **k)
        for c in caps:
            if c.name == "TX2":
                c.metrics["gain_compression_db"] += 0.5
        return caps

    def shifted_reading(cap, fs, window_s):
        peak, railed, gc, extra = real_reading(cap, fs, window_s)
        full = len(cap.x_aligned) == N
        return peak, railed, gc + (0.5 if full and cap is not None and _is_tx2(cap) else 0), extra

    def _is_tx2(cap):
        return cap.raw.channel == RxChannel.ORX2

    monkeypatch.setattr(multiband, "capture_case", shifted_case)
    monkeypatch.setattr(multiband, "_reading", shifted_reading)
    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    res = find_multiband_operating_point(
        radio, bands, tx_bits=12, rx_bits=12, fs=fs, backoffs=(0,), probe=PROBE,
        confirm_steps=confirm_steps,
    )  # fmt: skip
    s = res.search
    assert s.confirm["offset_db"]["TX2"] > 0.35
    full = [h for h in s.history if h["signal"] == "full"]
    assert s.confirm["fallback_steps"] <= confirm_steps
    assert len({h["step"] for h in full}) <= confirm_steps + (1 if confirm_steps else 0)
    if confirm_steps == 0:
        assert not s.converged and not full and "no fallback steps" in s.reason
    else:
        assert full and all(h["signal"] == "full" for h in full)
        assert s.confirm["converged_on"] in ("fallback", "")
        assert s.converged == (s.confirm["converged_on"] == "fallback")


def test_the_fallback_starts_from_the_prior_not_the_probes_slopes():
    """Task 6b B: on the bench the probe-learnt J moved TX2 1 dB for a 0.15 dB error. The
    correction steps after the 0 dB confirmation start J from SLOPE_PRIOR and learn only from
    full-signal readings: a probe search that ended with J ten times too shallow does not
    make the first full-signal move ten times too large."""
    import json

    from adrvtrx.multiband import SLOPE_PRIOR, BandLock, MultibandSearch, confirm_lock, prior_slopes

    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    for b in bands:
        radio.retune_lo(b.lo, b.freq_hz)
        radio.set_tx_atten(b.tx, 17.0)
    refs = multiband.play_bands(radio, bands, tx_bits=12)
    nan = float("nan")
    locks = [BandLock(b.name, 17.0, 210, 4.0, nan, 0.4, -1.0, 0, 5.0, "converged") for b in bands]
    shallow = (np.diag([-0.03, -0.03])).tolist()  # what a bad probe search could leave
    search = MultibandSearch(True, "every band within tolerance", "joint", locks, 3, 3,
                             slopes=shallow, probe={"used": True})  # fmt: skip
    caps = confirm_lock(radio, bands, refs, search, rx_bits=12, fs=fs, confirm_steps=1)
    full = [h for h in search.history if h.get("signal") == "full"]
    first = [h for h in full if h["step"] == min(r["step"] for r in full)]
    assert first, search.reason
    np.testing.assert_allclose(json.loads(first[0]["slopes"]), prior_slopes(2, SLOPE_PRIOR))
    expected = np.linalg.solve(prior_slopes(2), [h["error_db"] for h in first])
    for h, e in zip(first, expected):
        assert h["move_db"] == pytest.approx(float(np.clip(e, -4, 4)), abs=0.05)
    assert search.confirm["fallback_steps"] == 1 and len(caps) == 2


@pytest.mark.parametrize(
    "profile, rates",
    [("ADRV9025Init_StdUseCase98_LinkSharing.profile", (245.76e6, 491.52e6)),
     ("ADRV9025Init_StdUseCase102_LinkSharing.profile", (245.76e6,)),
     ("ADRV9025Init_StdUseCase61_LinkSharing.profile", (368.64e6,))],
)  # fmt: skip
def test_a_requested_0p01_ms_capture_is_a_valid_length_at_every_rate(profile, rates):
    """On the bench 0.01 ms failed on UC98 ("the number of bytes has to be divisible by 4"):
    4915.2 samples at 491.52 Msps. The length is rounded up to whole 4-sample words at every
    rate of the profile; a valid length is left as it is."""
    from adrvtrx.capture import valid_capture_ms
    from adrvtrx.profile import profile_rates_hz

    tes = Path(r"C:\Program Files\Analog Devices\ADRV9025 Transceiver Evaluation Software_x64_FULL")
    path = tes / "Resources" / "Adi.Adrv9025.Profiles" / "public" / profile
    if path.is_file():
        assert profile_rates_hz(path) == [int(r) for r in rates]
    ms = valid_capture_ms(0.01, rates)
    assert 0.01 <= ms < 0.0101
    for r in rates:
        samples = ms * 1e-3 * r
        assert abs(samples - round(samples)) < 1e-6 and round(samples) % 4 == 0, (r, samples)
    two_probe_periods = 2 * 24576 / rates[-1] * 1e3
    assert valid_capture_ms(two_probe_periods, (245.76e6, 491.52e6)) == pytest.approx(
        two_probe_periods, rel=1e-12
    )
    assert valid_capture_ms(0.01, ()) == 0.01


def test_capture_rounds_the_length_with_the_radios_profile(monkeypatch):
    from adrvtrx import capture as capture_module

    radio = _radio()
    monkeypatch.setattr(capture_module, "profile_rates", lambda _r: (245_760_000, 491_520_000))
    calls = _count_perform_rx(radio)
    out = capture_module.capture(radio, int(RxChannel.ORX2), 0.01, bits=12)
    assert calls == [pytest.approx(out.capture_time_ms)] and out.capture_time_ms > 0.01
    assert round(out.capture_time_ms * 1e-3 * 491.52e6) % 4 == 0


def test_the_fidelity_report_of_a_case(tmp_path):
    from adrvtrx.probe import fidelity_report

    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    res = find_multiband_operating_point(radio, bands, tx_bits=12, rx_bits=12, fs=fs,
                                         backoffs=(0, 2), probe=PROBE)  # fmt: skip
    outputs = {k: [c.y_aligned for c in caps] for k, caps in res.captures.items()}
    rep = fidelity_report(res.refs, outputs, [b.name for b in bands], PROBE)
    assert set(rep["backoffs"]) == {"0", "2"} and set(rep["backoffs"]["0"]) == {"TX2", "TX3"}
    assert rep["probe"]["ok"] and rep["worst_offset_db"] < 0.1
    full = rep["backoffs"]["0"]["TX2"]["full_db"]
    assert full == pytest.approx(res.captures[0][0].metrics["gain_compression_db"], abs=1e-3)
