"""The ORx saturating below full scale, and the empty capture after it (bench, UC102, 2026-10-21).

Overdriven, the ORx pinned its peak near -0.4 dBFS with no railed code, so the old railed-only
clip check read a clipped capture as in band; the confirm capture's AGC started from a gain
predicted from the lock, ~2 dB hot, inside that clip; and a capture of ORX2 alone came back all
zeros (peak -inf), which crashed the AGC's gain step (``OverflowError``). The fake board
reproduces the clip and the zeros (``PaModel.orx_clip_dbfs``, ``orx_clip_zeros``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import (
    EmptyCaptureError,
    aligned_capture,
    autolevel_capture,
    capture,
    orx_capture,
    orx_clip_for,
)
from adrvtrx.compression import search_tx_compression
from adrvtrx.config import Config, DllConfig, orx_clip_for_profile
from adrvtrx.fake import FakeRadio
from adrvtrx.gain import (
    EMPTY_RETAKES,
    EMPTY_STEP_IDX,
    ORX_CEILING_DEFAULT_DBFS,
    ORX_CLIP_DBFS,
    ORX_CLIP_MARGIN_DB,
    AgcError,
    OrxStep,
    autolevel_orx,
    check_orx_window,
    orx_clipped,
    relevel_index,
    verify_no_clip,
)
from adrvtrx.multiband import search_multiband_compression
from adrvtrx.transmit import transmit_bands

ORX = RxChannel.ORX1
BITS = 12
PINNED = -0.4  # the bench's clip level, dBFS
UC102 = "ADRV9025Init_StdUseCase102_LinkSharing.profile"


def _config(**kw) -> Config:
    return Config(dll=DllConfig(install_dir=Path("C:/nonexistent")), **kw)


def _radio(**pa) -> FakeRadio:
    """TX1 -> ORX1 playing a noise-like signal at 15 dB: the ORX1 peak is -16.6 dBFS at gain 185,
    -1.6 at 215, -1.1 at 216, -0.6 at 217 (at the clip threshold, -0.65) and pinned at -0.4 dBFS
    (no railed code) from 218 up."""
    radio = FakeRadio(_config(), seed=1, pa={"orx_clip_dbfs": PINNED, **pa})
    radio._safe_hooks_installed = True
    radio.connect()
    radio.program()
    rng = np.random.default_rng(3)
    x = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    radio.set_tx_atten(TxChannel.TX1, 15.0)
    transmit_bands(radio, {TxChannel.TX1: 0.9 * x / np.max(np.abs(x))}, BITS)
    return radio


def _spy(radio) -> list:
    """Every PerformRx through the radio (its capture length, ms)."""
    calls, real = [], radio.perform_rx

    def spy(rx_init, capture_time_ms, **kw):
        calls.append(float(capture_time_ms))
        return real(rx_init, capture_time_ms, **kw)

    radio.perform_rx = spy
    return calls


def _gains(radio) -> list[int]:
    """Every ORx gain set on the radio, in order."""
    seen, real = [], radio.set_rx_gain
    radio.set_rx_gain = lambda ch, g: (seen.append(int(g)), real(ch, g))[1]
    return seen


def _peak(radio, gain: int) -> tuple[float, int]:
    radio.set_rx_gain(ORX, gain)
    rep = capture(radio, int(ORX), 0.8192, bits=BITS).channels[ORX].clip()
    return rep.peak_dbfs, rep.railed_samples


# -- the clip threshold: the profile's ceiling minus the margin ----------------------------------


def test_the_clip_threshold_is_the_profile_ceiling_minus_the_margin():
    assert ORX_CEILING_DEFAULT_DBFS == -0.45 and ORX_CLIP_MARGIN_DB == 0.2
    assert ORX_CLIP_DBFS == -0.65  # a profile not measured yet
    cfg = _config()
    assert cfg.orx_ceiling_dbfs(UC102) == -0.35 and cfg.orx_clip_dbfs(UC102) == -0.55
    assert cfg.orx_clip_dbfs("ADRV9025Init_StdUseCase98_LinkSharing.profile") == -0.65
    assert cfg.orx_clip_dbfs(rf"C:\profiles\{UC102.lower()}") == -0.55  # by file name
    measured = Config.from_dict(
        {"dll": {"install_dir": "C:/x"}, "profile": {"name": UC102},
         "orx_ceiling": {UC102: -0.3, "default": -0.4}}
    )  # fmt: skip
    assert measured.orx_clip_dbfs() == -0.5 and measured.orx_clip_dbfs("other.profile") == -0.6
    with pytest.raises(ValueError, match="orx_ceiling"):
        Config.from_dict({"dll": {"install_dir": "C:/x"}, "orx_ceiling": {UC102: 0.5}})
    assert orx_clip_for_profile(UC102) == -0.55 and orx_clip_for_profile("x.profile") == -0.65


def test_the_radio_gives_its_profiles_clip_threshold():
    radio = _radio()
    assert orx_clip_for(radio) == -0.65  # the fake runs 95C: not measured, the default
    radio.config.profile_name = UC102
    assert orx_clip_for(radio) == -0.55
    assert orx_clip_for(object()) == ORX_CLIP_DBFS  # a stand-in with no config


def test_the_minus_1_window_is_accepted_and_one_reaching_the_clip_is_refused():
    check_orx_window(-1.0, 0.3)  # top -0.7, below -0.65 (default) ...
    check_orx_window(-1.0, 0.3, _config().orx_clip_dbfs(UC102))  # ... and -0.55 (UC102)
    with pytest.raises(ValueError, match="ORx clip"):
        check_orx_window(-0.8, 0.3)  # top -0.5
    with pytest.raises(ValueError, match="ORx clip"):
        check_orx_window(-0.8, 0.25, -0.55)  # top -0.55: reaching it is refused too
    with pytest.raises(ValueError, match="ORx clip"):
        autolevel_orx(lambda g: None, lambda: (-10.0, 0), target_dbfs=-0.5)
    with pytest.raises(ValueError, match="ORx clip"):
        search_tx_compression(lambda d: None, lambda g: None, lambda: (-2.0, 0, 3.0),
                              target_compression_db=3.0, start_atten_db=20.0, atten_min_db=5.0,
                              orx_gain=220, target_dbfs=-0.65, tol_up_db=0.0)  # fmt: skip
    with pytest.raises(ValueError, match="ORx clip"):
        search_multiband_compression(lambda i, d: None, lambda i, g: None, lambda idx: [],
                                     targets_db=[3.0], start_attens_db=[20.0], atten_mins_db=[5.0],
                                     orx_gains=[220], target_dbfs=-0.5)  # fmt: skip


# -- the fake board reproduces the bench ---------------------------------------------------------


def test_the_fake_orx_pins_below_full_scale_with_no_railed_code():
    radio = _radio()
    for gain in (230, 255):
        peak, railed = _peak(radio, gain)
        assert peak == pytest.approx(PINNED, abs=0.01) and railed == 0
        assert orx_clipped(peak, railed)  # the railed count alone misses it
    peak, railed = _peak(radio, 216)
    assert -1.6 <= peak <= -0.7 and not orx_clipped(peak, railed)


def test_the_fake_orx_driven_past_its_clip_returns_zeros_then_data():
    radio = _radio(orx_clip_zeros=1)
    radio.set_rx_gain(ORX, 240)
    first, second = (capture(radio, int(ORX), 0.8192, bits=BITS).channels[ORX] for _ in "ab")
    assert len(first.i) == 8192 and not np.any(first.i) and not np.any(first.q)
    assert first.clip().peak_dbfs == -math.inf
    assert second.clip().peak_dbfs == pytest.approx(PINNED, abs=0.01)


def test_the_fake_default_has_no_orx_clip():
    radio = _radio(orx_clip_dbfs=None)
    peak, railed = _peak(radio, 255)
    assert peak > PINNED or railed > 0


# -- a peak pinned at the ceiling is clipped: the AGC backs off ---------------------------------


def test_the_agc_never_accepts_a_pinned_peak_and_backs_off():
    radio = _radio()
    seen = _gains(radio)
    res = autolevel_capture(radio, ORX, bits=BITS, orx_rate_hz=10e6, start_gain=226,
                            verify_capture_ms=0.8192)  # fmt: skip
    assert seen[0] == 222  # 4 below the prediction, still pinned: it backs off
    assert res.converged and res.final_gain_index == 216 == seen[-1]
    assert -1.6 <= res.final_peak_dbfs <= -0.7 and res.railed == 0


def test_a_predicted_gain_start_approaches_from_below():
    radio = _radio()
    calls = _spy(radio)
    seen = _gains(radio)
    res = autolevel_capture(
        radio, ORX, bits=BITS, orx_rate_hz=10e6, start_gain=219, verify_capture_ms=0.8192
    )  # 1.5 dB hot: the prediction is pinned
    assert seen[0] == 215 and _peak(_radio(), 215)[0] < -1.5  # the first capture: from below
    assert res.converged and -1.6 <= res.final_peak_dbfs <= -0.7 and len(calls) > 1


def test_a_pinned_peak_at_the_gain_floor_is_fatal():
    # The ORx pins (no railed code) even at the minimum gain: the "clips at minimum gain" path.
    res = autolevel_orx(lambda g: None, lambda: (PINNED, 0))
    assert res.fatal and "clips at minimum gain" in res.reason
    vr = verify_no_clip(lambda g: None, lambda: (PINNED, 0), 186)
    assert vr.fatal and "minimum gain" in vr.reason and vr.final_gain_index == 185
    res = autolevel_orx(lambda g: None, lambda: (PINNED, 0), start_gain=189)  # starts at 185
    assert res.fatal and "clips at minimum gain" in res.reason


def test_a_pinned_capture_teaches_no_step_and_is_never_the_closest():
    step = OrxStep()
    kw = {"band_lo": -1.6, "band_hi": -0.7}
    step.observe(230, PINNED, 0, **kw)
    step.observe(228, -3.4, 0, **kw)  # 3 dB below the pinned peak in 2 indices: not a slope
    assert step.db_per_index == 0.5
    assert step.closest(-1.0) == (228, -3.4)
    assert relevel_index(PINNED, 0, target_dbfs=-1.0, band_hi=-0.7, db_per_index=0.5) == -1


def test_the_multiband_search_does_not_count_a_pinned_reading():
    gains = {"g": 224}

    def measure(idx):
        g = gains["g"]
        peak = min(PINNED, -1.0 + 0.5 * (g - 220))  # at the clip from 221 up, pinned from 222
        return [(peak, 0, 4.0 if peak < ORX_CLIP_DBFS else 2.0, {"top_slope": 1.0})]

    s = search_multiband_compression(
        lambda i, d: None, lambda i, g: gains.__setitem__("g", g), measure,
        targets_db=[4.0], start_attens_db=[20.0], atten_mins_db=[5.0], orx_gains=[224],
        max_steps=1, min_top_slope=None,
    )  # fmt: skip
    first = s.history[0]
    assert first["peak_dbfs"] == PINNED and not first["counts"] and not first["leveled"]
    assert first["action"].startswith("ORx gain -")
    counted = [h for h in s.history if h["counts"]]
    assert counted and all(h["peak_dbfs"] < ORX_CLIP_DBFS for h in counted)


# -- one empty capture: the gain goes down, the retake succeeds ----------------------------------


def test_one_empty_capture_lowers_the_gain_and_the_retake_succeeds():
    radio = _radio(orx_clip_zeros=1)
    seen = _gains(radio)
    res = autolevel_capture(radio, ORX, bits=BITS, orx_rate_hz=10e6, start_gain=225,
                            verify_capture_ms=0.8192)  # fmt: skip
    assert seen[:2] == [221, 221 - EMPTY_STEP_IDX]  # zeros at 221: 2 indices down, retaken
    assert res.converged and -1.6 <= res.final_peak_dbfs <= -0.7


def test_the_agc_stages_lower_the_gain_on_an_empty_reading():
    readings = iter([(-math.inf, 0), (-1.0, 0)])
    seen = []
    res = autolevel_orx(seen.append, lambda: next(readings), start_gain=224)
    assert res.converged and seen == [220, 218] and res.final_gain_index == 218
    readings = iter([(-math.inf, 0), (-1.0, 0)])
    seen = []
    vr = verify_no_clip(seen.append, lambda: next(readings), 220)
    assert vr.converged and seen == [220, 218]


def test_an_empty_capture_at_a_fixed_gain_is_retaken_at_that_gain():
    radio = _radio(orx_clip_zeros=1)
    radio.set_rx_gain(ORX, 240)
    seen = _gains(radio)
    ref = radio.board_model.waves["TX1"]
    calls = _spy(radio)
    cap = aligned_capture(radio, ORX, ref, bits=BITS, fs=10e6)
    assert len(calls) == 2 and not seen  # retaken, the gain untouched
    assert cap.clip.peak_dbfs == pytest.approx(PINNED, abs=0.01)


def test_the_single_band_search_lowers_the_gain_on_an_empty_reading():
    readings = iter([(-1.0, 0, 1.0), (-math.inf, 0, math.nan), (-1.0, 0, 3.0)])
    gains = []
    res = search_tx_compression(lambda d: None, gains.append, lambda: next(readings),
                                target_compression_db=3.0, start_atten_db=20.0,
                                atten_min_db=5.0, orx_gain=230, max_iterations=3)  # fmt: skip
    empty = res.history[1]
    assert empty["action"] == "ORx returned no data (all zeros); ORx gain -2, retake"
    assert res.history[2]["orx_gain"] == empty["orx_gain"] - EMPTY_STEP_IDX
    assert res.converged


def test_the_multiband_search_lowers_the_gain_on_an_empty_reading():
    gains = {"g": 220}
    readings = iter([-math.inf, -1.0, -1.0])

    def measure(idx):
        return [(next(readings), 0, 4.0, {"top_slope": 1.0})]

    s = search_multiband_compression(
        lambda i, d: None, lambda i, g: gains.__setitem__("g", g), measure,
        targets_db=[4.0], start_attens_db=[20.0], atten_mins_db=[5.0], orx_gains=[220],
        max_steps=1, min_top_slope=None, names=["TX1"],
    )  # fmt: skip
    assert s.history[0]["action"] == "ORx returned no data (all zeros); ORx gain -2, retake"
    assert s.history[1]["orx_gain"] == 218 and s.history[1]["counts"]
    assert s.converged


# -- persistent empty captures: the clear AgcError, never OverflowError --------------------------


def test_persistent_empty_captures_raise_the_clear_agc_error():
    radio = _radio(orx_clip_zeros=10)
    calls = _spy(radio)
    seen = _gains(radio)
    msg = r"ORX1: ORx returned no data \(all zeros\) after 2 retakes at gain 237 .*overdriven"
    with pytest.raises(AgcError, match=msg):
        autolevel_capture(radio, ORX, bits=BITS, orx_rate_hz=10e6, start_gain=245)
    assert seen == [241, 239, 237] and len(calls) == EMPTY_RETAKES + 1
    assert not radio.is_connected()  # the bench left safe


def test_an_empty_reading_at_the_floor_is_retaken_there_then_fatal():
    seen, calls = [], []
    empty = lambda: (calls.append(1), (-math.inf, 0))[1]  # noqa: E731
    res = autolevel_orx(seen.append, empty)  # Stage A: at the floor already
    assert res.fatal and "no data (all zeros) after 2 retakes at gain 185" in res.reason
    assert seen == [185] and len(calls) == 3


def test_the_agc_stages_report_persistent_no_data_as_fatal():
    empty = lambda: (-math.inf, 0)  # noqa: E731
    for res in (
        autolevel_orx(lambda g: None, empty),  # Stage A
        autolevel_orx(lambda g: None, empty, start_gain=220),  # Stage B
        verify_no_clip(lambda g: None, empty, 220),  # Stage C
    ):
        assert res.fatal and "returned no data (all zeros) after 2 retakes" in res.reason


def test_a_fixed_gain_capture_raises_after_its_retakes():
    radio = _radio(orx_clip_zeros=10)
    radio.set_rx_gain(ORX, 240)
    with pytest.raises(EmptyCaptureError, match="ORX1 returned no data .* after 1 retakes") as e:
        orx_capture(radio, int(ORX), 0.1, bits=BITS, retakes=1, wait_s=0.0)
    assert isinstance(e.value, AgcError) and e.value.channels == (ORX,)


def test_the_searches_raise_the_clear_error_on_persistent_empty_readings():
    with pytest.raises(AgcError, match="ORx returned no data"):
        search_tx_compression(lambda d: None, lambda g: None, lambda: (-math.inf, 0, math.nan),
                              target_compression_db=3.0, start_atten_db=20.0,
                              atten_min_db=5.0, orx_gain=200)  # fmt: skip
    with pytest.raises(AgcError, match="TX1: ORx returned no data"):
        search_multiband_compression(
            lambda i, d: None, lambda i, g: None, lambda idx: [(-math.inf, 0, math.nan, {})],
            targets_db=[4.0], start_attens_db=[20.0], atten_mins_db=[5.0], orx_gains=[220],
            names=["TX1"],
        )  # fmt: skip


def test_relevel_index_with_minus_inf_does_not_overflow():
    kw = {"target_dbfs": -1.0, "band_hi": -0.7, "db_per_index": 0.5}
    assert relevel_index(-math.inf, 0, **kw) == -EMPTY_STEP_IDX
    assert relevel_index(math.nan, 0, **kw) == -EMPTY_STEP_IDX
