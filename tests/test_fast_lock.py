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
    make_probes,
    probe_cut,
    probe_length,
    timing_text,
)

N = 16384
#: own ~0.3, cross ~0.06 dB/dB near a 4 dB lock at 15 dB (measured on this model; noise-free)
BENCH_PA = {"shared": [["TX2", "TX3"]], "coupling": 10.0, "smoothness": 0.4, "drive": 3.0,
            "noise_codes": 0.5}  # fmt: skip


def _signal(seed: int, n: int = N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.fft.ifft(
        np.fft.fft(rng.normal(size=n) + 1j * rng.normal(size=n))
        * (np.abs(np.fft.fftfreq(n)) < 0.15)
    )
    return x / np.abs(x).max()


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


# -- F1: the probe ------------------------------------------------------------------------------


def test_the_probe_has_the_full_signals_codes_and_peak():
    ref = multiband.tx_codes(_bands(), tx_bits=12)[0]
    w, info = probe_cut(ref, 500)
    assert len(w) == 500 and np.abs(w).max() == np.abs(ref).max()  # the same peak drive
    start = info["start"]
    np.testing.assert_array_equal(w, ref[(start + np.arange(500)) % len(ref)])  # cut, as is
    peak_at = (int(np.argmax(np.abs(ref))) - start) % len(ref)
    assert 250 - 25 <= peak_at <= 250 + 25  # centred on the peak, moved by at most n / 20
    assert info["jump_rms"] <= info["jump_centred_rms"]  # the wrap jump is no worse than centred
    jump = abs(w[0] - w[-1]) / np.sqrt(np.mean(np.abs(ref) ** 2))
    assert info["jump_rms"] == pytest.approx(jump)
    assert info["papr_full_db"] == pytest.approx(
        10 * np.log10(np.max(np.abs(ref) ** 2) / np.mean(np.abs(ref) ** 2)), abs=1e-3
    )


@pytest.mark.parametrize("seed", range(6))
def test_the_chosen_wrap_jump_is_never_larger_than_the_centred_cut(seed):
    ref = _signal(seed, 4096) * 2047
    _w, info = probe_cut(ref, 400)
    assert info["jump_rms"] <= info["jump_centred_rms"] and abs(info["shift"]) <= 20


def test_both_probes_have_one_length_rounded_up_to_the_tx_block():
    refs = multiband.tx_codes(_bands(), tx_bits=12)
    probes, info = make_probes(refs, 10e6, 0.05, block=64)
    assert info["n"] == 512 and {len(p) for p in probes} == {512}  # 500 up to a block of 64
    assert probe_length(0.05, 491.52e6) == 24576  # 0.1 ms of capture = 49,152 samples
    assert probe_length(0.05, 491.52e6, block=1000) == 25000
    assert probe_length(0.01, 10e6, block=7) == 105
    for bad in (0.001, 0.6):
        with pytest.raises(ValueError, match="probe_ms"):
            probe_length(bad, 10e6)
    assert [b["n"] for b in info["bands"]] == [512, 512]


def test_probe_ms_none_captures_two_full_periods_as_before():
    """Without the probe, every capture of the lock is 2 periods of the full signal (AGC Stage A
    and B excepted: 0.1 ms, as always), and the TX plays the full signal throughout."""
    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    calls = _count_perform_rx(radio)
    search, refs = lock_bands(radio, bands, tx_bits=12, rx_bits=12, fs=fs, probe_ms=None)
    full_ms = round(2 * N / fs * 1e3, 6)
    got = [round(c, 6) for c in calls]
    assert set(got) <= {full_ms, 0.1} and got.count(full_ms) >= search.captures
    assert search.probe == {} and {h["signal"] for h in search.history} == {"full"}
    assert radio.board_model.state["tx_len"] == N and len(refs[0]) == N


def test_with_the_probe_every_lock_capture_is_two_probe_periods_and_tx_ends_on_the_full():
    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    calls = _count_perform_rx(radio)
    search, refs = lock_bands(radio, bands, tx_bits=12, rx_bits=12, fs=fs, probe_ms=0.5)
    probe_ms = round(2 * 5000 / fs * 1e3, 6)
    assert {round(c, 6) for c in calls} <= {probe_ms, 0.1}  # AGC Stage C and the search: probe
    assert {h["signal"] for h in search.history} == {"probe"}
    assert radio.board_model.state["tx_len"] == N  # swapped back to the full signal
    assert len(search.timing["swaps_s"]) == 2 and len(search.timing["agc_s"]) == 2
    assert search.probe["n"] == 5000 and len(search.probe["bands"]) == 2


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
    step = OrxStep()

    def learnt(g, peak):
        return step.relevel(g, peak, 0, target_dbfs=-1.0, band_lo=-1.6, band_hi=-0.7)

    seen = _relevel(radio, gain, learnt)
    assert len(seen) <= 3 and -1.6 <= seen[-1][1] <= -0.7, seen
    assert step.db_per_index == pytest.approx(0.8, abs=0.05)


def test_the_learnt_step_takes_the_closest_capture_when_one_index_jumps_the_window():
    step = OrxStep()
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
        search, _refs = lock_bands(radio, bands, tx_bits=12, rx_bits=12, fs=fs, probe_ms=0.5)
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
        radio, bands, tx_bits=12, rx_bits=12, fs=radio.board_model.rate_hz, probe_ms=0.5,
        method="pingpong",
    )  # fmt: skip
    assert search.converged, search.reason
    assert search.snapshots <= 12, search.snapshots


def test_without_a_two_orx_snapshot_each_band_is_captured_alone():
    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    search, _refs = lock_bands(
        radio, bands, tx_bits=12, rx_bits=12, fs=fs, probe_ms=0.5, snapshot=False
    )
    assert search.converged and not search.snapshot
    assert search.captures == 2 * search.snapshots
    assert shares_orx_adc([RxChannel.ORX1, RxChannel.ORX2])
    assert not shares_orx_adc([RxChannel.ORX2, RxChannel.ORX3])


# -- F3 / F7: the full-signal confirmation and the timing --------------------------------------


def test_a_case_on_the_probe_is_confirmed_by_its_0_db_capture(tmp_path):
    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    writer = multiband.CaseWriter(tmp_path, bands, tx_bits=12, rx_bits=12, fs=fs)
    res = find_multiband_operating_point(
        radio, bands, tx_bits=12, rx_bits=12, fs=fs, backoffs=(0, 1), probe_ms=0.5,
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
        radio, bands, tx_bits=12, rx_bits=12, fs=fs, backoffs=(0,), probe_ms=0.5,
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
