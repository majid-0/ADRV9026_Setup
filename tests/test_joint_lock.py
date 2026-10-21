"""The joint lock (``lock_on="joint"``), the joint clip guard, the joint figures of every capture.

The fake PAs: the first bench lock's shared PA (test_fast_lock's: each band compressed against its
own envelope, its headroom cut by the other band's mean power), and one that compresses every band
on the joint envelope ``|w_1| + |w_2|`` (``joint_envelope``; ``smoothness`` 20 clips the composite
hard). The carriers are 20 MHz apart (1440 / 1460 MHz at the fake board's 10 Msps), so the composite
is 12 times up-sampled, not 172 times as at 420 MHz: the joint figure is the same definition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel, multiband
from adrvtrx.config import Config, DllConfig
from adrvtrx.fake import FakeRadio
from adrvtrx.metrics import (
    envelope_shares,
    input_metrics,
    joint_compression,
    joint_top_drop,
    share_key,
)
from adrvtrx.multiband import (
    JOINT_CLIP_KEYS,
    JOINT_SHARE_KEYS,
    JOINT_SLOPE_PRIOR,
    BandSpec,
    CaseWriter,
    capture_case,
    find_multiband_operating_point,
    joint_levels,
    lock_bands,
    lock_summary,
    resolve_joint_clip,
    search_multiband_compression,
    timing_text,
)
from adrvtrx.probe import ProbeSettings

#: test_fast_lock's: own ~0.3, cross ~0.06 dB/dB per band; the joint slope reads 0.34 dB/dB
BENCH_PA = {"shared": [["TX2", "TX3"]], "coupling": 10.0, "smoothness": 0.4, "drive": 3.0,
            "noise_codes": 0.5}  # fmt: skip
#: every band compressed on the joint envelope, the composite clipped hard: its joint top slope
#: falls through 0.2 between 11.5 and 10 dB of attenuation, where the bands read about 1 dB each
HARD_CLIP_PA = {"shared": [["TX2", "TX3"]], "joint_envelope": True, "smoothness": 20.0,
                "drive": 3.0, "noise_codes": 0.5}  # fmt: skip
#: a softer joint-envelope PA (joint 4.2 dB with about 2.8 dB per band)
ENVELOPE_PA = {**HARD_CLIP_PA, "smoothness": 1.0}
PROBE = ProbeSettings(window=300, crossfade=8)
TARGET = 4.2


def _signal(seed: int, n: int) -> np.ndarray:
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


def _bands(start=(22.0, 23.0), floor=(5.0, 5.0), n=16384, target=4.0) -> list[BandSpec]:
    return [
        BandSpec(TxChannel.TX2, RxChannel.ORX2, "LO1", 1_440_000_000, 2.0, _signal(1, n),
                 target_db=target, atten_start_db=start[0], atten_min_db=floor[0]),
        BandSpec(TxChannel.TX3, RxChannel.ORX3, "LO2", 1_460_000_000, 2.0, _signal(2, n),
                 target_db=target, atten_start_db=start[1], atten_min_db=floor[1]),
    ]  # fmt: skip


def _lock(pa=None, bands=None, **kw):
    radio, bands = _radio(pa), bands or _bands()
    search, refs = lock_bands(radio, bands, tx_bits=12, rx_bits=12, fs=radio.board_model.rate_hz,
                              **kw)  # fmt: skip
    return radio, bands, search, refs


def _steps(search) -> dict[int, dict[str, dict]]:
    """Each step's last row per band (the reading that drove the step)."""
    out: dict[int, dict[str, dict]] = {}
    for h in search.history:
        out.setdefault(h["step"], {})[h["band"]] = h
    return out


def _joint_now(radio, bands, refs, attens) -> dict:
    """The joint figures of a fresh full capture at ``attens`` (TX playing ``refs``)."""
    caps = capture_case(radio, bands, refs, attens, rx_bits=12, fs=radio.board_model.rate_hz)
    return caps[0].metrics


# -- the joint lock ----------------------------------------------------------------------------


@pytest.mark.parametrize("pa", [BENCH_PA, ENVELOPE_PA], ids=["bench_pa", "envelope_pa"])
def test_the_joint_lock_reaches_its_target_from_10_db_away_in_at_most_7_snapshots(pa):
    radio, bands, s, refs = _lock(pa, lock_on="joint", joint_target_db=TARGET)
    assert s.converged and s.lock_on == "joint", s.reason
    assert s.snapshots <= 7, s.snapshots
    assert abs(s.joint["achieved_db"] - TARGET) <= 0.1
    assert s.joint["target_db"] == TARGET and s.joint["levels_db"] == [0.0, -1.0]
    assert all(bl.status == "converged" for bl in s.bands)
    now = _joint_now(radio, bands, refs, [bl.atten_db for bl in s.bands])  # a fresh capture agrees
    assert abs(now["joint_gain_compression_db"] - TARGET) <= 0.15
    rows = [h for h in s.history if h["counts"]]
    assert all(np.isfinite(h["joint_gain_compression_db"]) for h in rows)
    assert rows[0]["slopes"] == f"{JOINT_SLOPE_PRIOR:.4f}"  # the secant starts from the prior
    assert s.joint["slope_db_per_db"] != JOINT_SLOPE_PRIOR  # ... and learns
    assert "joint lock: target 4.2 dB" in timing_text(s)


def test_every_step_keeps_the_level_difference():
    _radio_, _bands_, s, _refs = _lock(lock_on="joint", joint_target_db=TARGET)
    steps = _steps(s)
    assert len(steps) == s.steps >= 3
    for rows in steps.values():  # the start attenuations' difference: TX3 1 dB below TX2
        assert rows["TX3"]["atten_db"] - rows["TX2"]["atten_db"] == pytest.approx(1.0, abs=1e-9)
        assert rows["TX2"]["common_atten_db"] == rows["TX2"]["atten_db"]
    assert s.bands[1].atten_db - s.bands[0].atten_db == pytest.approx(1.0, abs=1e-9)

    _radio_, _bands_, s, _refs = _lock(
        lock_on="joint", joint_target_db=TARGET, joint_levels_db=[-2.53, 0.0]
    )  # given levels, on the 0.05 dB grid: TX2 2.55 dB below TX3
    assert s.converged and s.joint["levels_db"] == [-2.55, 0.0]
    first = _steps(s)[0]
    assert (first["TX2"]["atten_db"], first["TX3"]["atten_db"]) == (25.55, 23.0)  # none below start
    for rows in _steps(s).values():
        assert rows["TX2"]["atten_db"] - rows["TX3"]["atten_db"] == pytest.approx(2.55, abs=1e-9)


def test_the_start_is_the_common_offset_that_keeps_every_band_at_or_above_its_start():
    _radio_, _bands_, s, _refs = _lock(
        lock_on="joint", joint_target_db=TARGET, joint_levels_db=[0.0, -3.0], max_steps=1
    )
    first = _steps(s)[0]  # TX2 22, TX3 would be 22 + 3 = 25 >= its 23 start
    assert (first["TX2"]["atten_db"], first["TX3"]["atten_db"]) == (22.0, 25.0)


def test_a_band_at_its_floor_stops_the_joint_lock_and_says_which_and_where():
    bands = _bands(floor=(5.0, 16.0))  # TX3 may not go below 16 dB: TX2 then sits at 15
    _radio_, _bands_, s, _refs = _lock(bands=bands, lock_on="joint", joint_target_db=TARGET)
    assert not s.converged
    assert "TX3 at its attenuation limit 16 dB" in s.reason and "joint gain compression" in s.reason
    assert f"below the {TARGET:g} dB joint target" in s.reason
    assert [bl.atten_db for bl in s.bands] == [15.0, 16.0]
    assert s.bands[1].status == "at limit"
    assert all(h["atten_db"] >= 16.0 for h in s.history if h["band"] == "TX3")
    assert s.joint["achieved_db"] < TARGET - 0.1


def test_the_joint_lock_refuses_what_it_cannot_do():
    def search(**kw):
        args = {"targets_db": [4, 4], "start_attens_db": [20, 20], "atten_mins_db": [10, 10],
                "orx_gains": [210, 210]}  # fmt: skip
        return search_multiband_compression(lambda *a: None, lambda *a: None, lambda i: [], **{
            **args, **kw})  # fmt: skip

    with pytest.raises(ValueError, match="lock_on must be one of"):
        search(lock_on="composite")
    with pytest.raises(ValueError, match="need joint="):
        search(lock_on="joint", joint_target_db=4.2)
    with pytest.raises(ValueError, match="joint_target_db > 0"):
        search(lock_on="joint", joint=lambda e, a: {})
    with pytest.raises(ValueError, match="need joint="):
        search(joint_clip_slope=0.08)
    with pytest.raises(ValueError, match="one finite level per band"):
        search(lock_on="joint", joint_target_db=4.2, joint=lambda e, a: {}, joint_levels_db=[0])
    with pytest.raises(ValueError, match="joint_slope_prior"):
        search(lock_on="joint", joint_target_db=4.2, joint=lambda e, a: {}, joint_slope_prior=3)


def test_joint_settings():
    assert resolve_joint_clip("per_band") is None and resolve_joint_clip("joint") == 0.08
    assert resolve_joint_clip("joint", None) is None and resolve_joint_clip("per_band", 0.1) == 0.1
    with pytest.raises(ValueError, match="0 .. 1"):
        resolve_joint_clip("joint", 1.5)
    with pytest.raises(ValueError, match="'default'"):
        resolve_joint_clip("joint", "auto")
    assert joint_levels(None, [6.0, 7.6]).tolist() == pytest.approx([0.0, -1.6])
    assert joint_levels("start", [8.0, 6.0]).tolist() == pytest.approx([-2.0, 0.0])
    assert joint_levels([1.0, -0.62], [0, 0]).tolist() == pytest.approx([0.0, -1.6])
    with pytest.raises(ValueError, match="'start'"):
        joint_levels("auto", [0, 0])


# -- the probe -------------------------------------------------------------------------------


@pytest.mark.parametrize("pa", [BENCH_PA, ENVELOPE_PA], ids=["bench_pa", "envelope_pa"])
def test_the_joint_compression_on_the_probe_tracks_the_full_signal(pa, tmp_path):
    """The probe keeps the same indices in every band: its joint compression is the full
    signal's within 0.1 dB at the lock (0.01 / 0.03 dB on these PAs), so the 0 dB capture
    confirms the probe's lock without full-signal steps."""
    radio, bands = _radio(pa), _bands(n=65536)
    fs = radio.board_model.rate_hz
    writer = CaseWriter(tmp_path, bands, tx_bits=12, rx_bits=12, fs=fs)
    res = find_multiband_operating_point(
        radio, bands, tx_bits=12, rx_bits=12, fs=fs, backoffs=(0, 2), probe=PROBE, writer=writer,
        lock_on="joint", joint_target_db=TARGET,
    )  # fmt: skip
    s = res.search
    assert s.probe["used"] and {h["signal"] for h in s.history} == {"probe"}
    assert s.converged and s.confirm["converged_on"] == "probe", s.reason
    full = res.captures[0][0].metrics["joint_gain_compression_db"]
    assert s.confirm["joint_gc_full_db"] == pytest.approx(full, abs=1e-3)
    assert abs(s.confirm["joint_offset_db"]) < 0.1
    assert abs(full - TARGET) <= 0.1
    assert "joint GC: full" in timing_text(s)
    doc = json.loads(writer.write_case_json().read_text())  # the joint result is in case.json
    assert doc["lock_on"] == "joint" and doc["lock_joint"]["target_db"] == TARGET
    assert doc["lock"][0]["joint_lock_db"] == pytest.approx(full, abs=1e-3)
    assert doc["lock"][0]["joint_lock_levels_db"] == "0/-1"


def test_a_joint_lock_off_on_the_full_signal_takes_joint_steps_from_the_prior(monkeypatch):
    """The full signal reads 0.5 dB more joint compression than the probe (injected into the 0 dB
    capture): the correction runs joint steps on full captures, from the prior slope, the level
    difference kept."""
    real_case = multiband.capture_case

    def shifted(*a, **k):
        caps = real_case(*a, **k)
        for c in caps:
            c.metrics["joint_gain_compression_db"] += 0.5
        return caps

    monkeypatch.setattr(multiband, "capture_case", shifted)
    radio, bands = _radio(), _bands(n=65536)
    res = find_multiband_operating_point(
        radio, bands, tx_bits=12, rx_bits=12, fs=radio.board_model.rate_hz, backoffs=(0,),
        probe=PROBE, lock_on="joint", joint_target_db=TARGET, confirm_steps=2,
    )  # fmt: skip
    s = res.search
    assert s.confirm["joint_offset_db"] > 0.4
    full = [h for h in s.history if h["signal"] == "full" and h["counts"]]
    assert full and full[0]["slopes"] == f"{JOINT_SLOPE_PRIOR:.4f}"
    err = full[0]["error_db"]
    assert full[0]["move_db"] == pytest.approx(-err / JOINT_SLOPE_PRIOR, abs=0.05)
    for rows in _steps(s).values():
        assert rows["TX3"]["atten_db"] - rows["TX2"]["atten_db"] == pytest.approx(1.0, abs=1e-9)
    assert 1 <= s.confirm["fallback_steps"] <= 2
    assert s.joint["probe_achieved_db"] is not None


# -- the joint clip guard ----------------------------------------------------------------------


def test_the_joint_clip_guard_backs_every_band_off_on_a_pa_that_clips_the_composite():
    """Joint mode: the target needs more drive than the hard-clipping PA takes with a joint top
    slope of 0.15: the guard backs both bands off together and the lock stops clip-limited."""
    _radio_, _bands_, s, _refs = _lock(
        HARD_CLIP_PA, lock_on="joint", joint_target_db=6.0, joint_clip_slope=0.15,
        min_top_slope=None, max_steps=12,
    )  # fmt: skip
    assert not s.converged and s.reason.startswith("clip-limited: joint clip guard"), s.reason
    assert "6 dB joint target needs more drive" in s.reason
    assert all(bl.status == "clip-limited" for bl in s.bands)
    assert s.joint["clip_guard"]["joint_top_slope"] < 0.15 and s.joint["clip_slope"] == 0.15
    fired = [h for h in s.history if h["action"].startswith("joint clip guard")]
    assert fired and all(h["joint_top_slope"] < 0.15 and h["move_db"] > 0 for h in fired)
    steps = _steps(s)
    for st in {h["step"] for h in fired}:  # every band backed off by the same amount
        assert steps[st]["TX2"]["move_db"] == steps[st]["TX3"]["move_db"]
    assert s.bands[0].atten_db >= max(h["atten_db"] for h in fired if h["band"] == "TX2") - 1e-9


def test_the_joint_clip_guard_in_per_band_mode_holds_the_bands_clip_limited():
    _radio_, _bands_, s, _refs = _lock(
        HARD_CLIP_PA, joint_clip_slope=0.15, min_top_slope=None, max_steps=12
    )
    assert s.lock_on == "per_band" and not s.converged
    assert "held by the joint clip guard" in s.reason, s.reason
    assert all(bl.status == "clip-limited" for bl in s.bands)
    assert all(bl.compression_db < 4.0 - 0.1 for bl in s.bands)
    fired = [h for h in s.history if h["action"].startswith("joint clip guard")]
    assert fired and s.joint["clip_guard"]["joint_top_slope"] < 0.15


def test_the_joint_clip_guard_does_not_fire_on_the_bench_pa_at_the_usual_targets():
    _radio_, _bands_, s, _refs = _lock(lock_on="joint", joint_target_db=TARGET)  # default 0.08
    assert s.converged and s.joint["clip_slope"] == 0.08 and "clip_guard" not in s.joint
    assert all(h["joint_top_slope"] > 0.3 for h in s.history if h["counts"])
    _radio_, _bands_, s, _refs = _lock(joint_clip_slope=0.08)
    assert s.converged and "clip_guard" not in s.joint
    assert not any(h["action"].startswith("joint clip guard") for h in s.history)


def test_per_band_mode_is_unchanged_with_the_guard_off(monkeypatch):
    """The default (no joint reading at all) and an explicit ``joint_clip_slope=None`` lock the
    same way, reading nothing joint, on the PA that clips the composite."""

    def no_joint(*_a, **_k):
        raise AssertionError("per_band mode with the guard off reads no joint figure")

    monkeypatch.setattr(multiband, "joint_compression", no_joint)
    runs = [_lock(HARD_CLIP_PA)[2], _lock(HARD_CLIP_PA, joint_clip_slope=None)[2]]
    for s in runs:
        assert s.lock_on == "per_band" and s.joint == {} and s.converged
        assert not any("joint_gain_compression_db" in h for h in s.history)
    keys = ("step", "band", "atten_db", "orx_gain", "gain_compression_db", "move_db", "action")
    assert [[h[k] for k in keys] for h in runs[0].history] == [
        [h[k] for k in keys] for h in runs[1].history
    ]


# -- the joint figures of every capture, and the case's inputs -------------------------------------


def test_every_capture_carries_the_envelope_shares_and_the_joint_clip_indicator(tmp_path):
    radio, bands = _radio(), _bands()
    fs = radio.board_model.rate_hz
    writer = CaseWriter(tmp_path, bands, tx_bits=12, rx_bits=12, fs=fs)
    res = find_multiband_operating_point(radio, bands, tx_bits=12, rx_bits=12, fs=fs,
                                         backoffs=(0, 3), writer=writer)  # fmt: skip
    import csv

    rows = list(csv.DictReader(open(res.case_dir / "conditions.csv", encoding="utf-8")))
    for k in (*JOINT_SHARE_KEYS, *JOINT_CLIP_KEYS, "lock_on"):
        assert k in rows[0], k
    assert {r["lock_on"] for r in rows} == {"per_band"} and "joint_target_db" not in rows[0]
    for r in rows:
        assert 0.0 < float(r["joint_env_above_0p8"]) < 0.05
        assert float(r["joint_env_above_0p9"]) <= float(r["joint_env_above_0p8"])
        assert r["joint_clipping"] == "False" and np.isfinite(float(r["joint_top_gain_drop_db"]))
    drop = {(r["backoff_db"], r["band"]): float(r["joint_top_gain_drop_db"]) for r in rows}
    assert all(drop[("0", b)] > drop[("3", b)] for b in ("TX2", "TX3"))  # less drive, less drop
    doc = json.loads(writer.write_case_json().read_text())
    inp = doc["inputs"]
    assert inp["attens_db"] == [bl.atten_db for bl in res.search.bands]
    want = input_metrics(res.refs, fs, [b.freq_hz for b in bands],
                         levels_db=multiband.levels_from_attens(inp["attens_db"]))  # fmt: skip
    for k in ("composite_papr_db", "pe_papr_db", "joint_env_above_0p8", "joint_env_above_0p9"):
        assert inp[k] == pytest.approx(want[k]) and np.isfinite(inp[k]), k
    assert inp["pe_papr_db"] >= inp["composite_papr_db"] - 0.3  # the spacing-free bound
    assert doc["lock_on"] == "per_band" and doc["lock_joint"] == {}
    assert "lock_on" not in lock_summary(res.search)[0]  # a per-band lock's summary as before


def test_the_input_metrics_of_a_case_written_without_its_inputs_are_left_out(tmp_path):
    radio, bands = _radio(), _bands()
    s, _refs = lock_bands(radio, bands, tx_bits=12, rx_bits=12, fs=radio.board_model.rate_hz)
    w = CaseWriter(tmp_path, bands, tx_bits=12, rx_bits=12, fs=radio.board_model.rate_hz)
    doc = json.loads(w.write_case_json(search=s).read_text())
    assert "inputs" not in doc and doc["lock_on"] == "per_band"


# -- the metrics --------------------------------------------------------------------------------


N = 8192
FS = 491.52e6
FREQS = (1.44e9, 1.86e9)


def _band(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.fft.ifft(np.fft.fft(rng.normal(size=N) + 1j * rng.normal(size=N))
                    * (np.abs(np.fft.fftfreq(N)) < 0.1))  # fmt: skip
    return x / np.abs(x).max()


def _envelope_pa(xs, drive: float, p: float = 2.0):
    """Every band compressed on the joint envelope (as the fake's ``joint_envelope``), each ORx
    with its own gain and phase."""
    w = [drive * x for x in xs]
    env = sum(np.abs(v) for v in w)
    g = 1.0 / (1.0 + env ** (2 * p)) ** (1.0 / (2 * p))
    return [v * g * (2.0 - 1.0j) * (k + 1) for k, v in enumerate(w)]


def test_the_envelope_shares_count_the_top_of_the_joint_envelope():
    xs = [_band(1), _band(2)]
    env = np.abs(xs[0]) + np.abs(xs[1])
    got = envelope_shares(xs)
    assert set(got) == {"joint_env_above_0p8", "joint_env_above_0p9"} == set(JOINT_SHARE_KEYS)
    assert got["joint_env_above_0p8"] == pytest.approx(np.mean(env >= 0.8 * env.max()))
    assert 0 < got["joint_env_above_0p9"] < got["joint_env_above_0p8"] < 0.1
    assert share_key(0.95) == "joint_env_above_0p95"
    assert all(np.isnan(v) for v in envelope_shares([0 * xs[0], 0 * xs[1]]).values())


def test_the_top_drop_is_zero_on_a_linear_pa_and_grows_into_a_clip():
    xs = [_band(1), _band(2)]
    lin = joint_top_drop(xs, [xs[0] * 3j, xs[1] * -0.2])
    assert np.allclose(lin["gain_drop_db"], 0, atol=1e-9)
    assert np.allclose(lin["phase_drop_deg"], 0, atol=1e-6)
    drops = [joint_top_drop(xs, _envelope_pa(xs, d))["gain_drop_db"] for d in (0.3, 0.8, 1.5)]
    assert all(np.all(np.diff([d[k] for d in drops]) > 0.3) for k in (0, 1)), drops
    hard = _envelope_pa(xs, 1.5, p=20.0)
    j = joint_compression(xs, hard, FS, FREQS)
    soft = joint_compression(xs, _envelope_pa(xs, 0.3), FS, FREQS)
    assert soft["joint_clipping"] is False and soft["joint_top_slope"] > 0.5
    assert j["joint_top_slope"] < soft["joint_top_slope"] and min(j["band_top_gain_drop_db"]) > 2
    assert j["joint_clipping"] == (j["joint_top_slope"] < 0.08)
    assert joint_compression(xs, hard, FS, FREQS, clip_slope=1.0)["joint_clipping"] is True
    assert set(JOINT_SHARE_KEYS) <= set(j)


def test_the_input_metrics_are_the_composite_and_the_spacing_free_peak():
    xs = [_band(1), _band(2)]
    got = input_metrics(xs, FS, FREQS, levels_db=[0.0, -2.0])
    j = joint_compression(xs, xs, FS, FREQS, levels_db=[0.0, -2.0])
    assert got["composite_papr_db"] == pytest.approx(j["joint_papr_in_db"])
    assert got["pe_papr_db"] == pytest.approx(j["pe_papr_in_db"])
    assert got["joint_env_above_0p8"] == pytest.approx(j["joint_env_above_0p8"])
    assert got["spacing_hz"] == pytest.approx(420e6) and got["levels_db"] == [0.0, -2.0]
