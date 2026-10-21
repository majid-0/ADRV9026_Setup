"""The ORx clip below full scale and the empty capture after it (bench, UC102, 2026-10-21).

Overdriven, the ORx pinned its peak near -0.4 dBFS with no railed code, so the old railed-only
clip check read a clipped capture as in band; and a capture of ORX2 alone came back all zeros
(peak -inf), which crashed the AGC's gain step (``OverflowError``). The fake board reproduces
both (``PaModel.orx_clip_dbfs``, ``orx_clip_zeros``).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import aligned_capture, autolevel_capture, capture, orx_capture
from adrvtrx.compression import search_tx_compression
from adrvtrx.config import Config, DllConfig
from adrvtrx.fake import FakeRadio
from adrvtrx.gain import (
    ORX_CLIP_DBFS,
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


def _radio(**pa) -> FakeRadio:
    """TX1 -> ORX1 playing a noise-like signal at 15 dB: the ORX1 peak is -16.6 dBFS at gain 185,
    -1.6 at 215 and pinned at -0.4 dBFS (no railed code) from about 218 up."""
    radio = FakeRadio(
        Config(dll=DllConfig(install_dir=Path("C:/nonexistent"))),
        seed=1,
        pa={"orx_clip_dbfs": PINNED, **pa},
    )
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


def _peak(radio, gain: int) -> tuple[float, int]:
    radio.set_rx_gain(ORX, gain)
    rep = capture(radio, int(ORX), 0.8192, bits=BITS).channels[ORX].clip()
    return rep.peak_dbfs, rep.railed_samples


# -- the fake board reproduces the bench ---------------------------------------------------------


def test_the_fake_orx_pins_below_full_scale_with_no_railed_code():
    radio = _radio()
    for gain in (230, 255):
        peak, railed = _peak(radio, gain)
        assert peak == pytest.approx(PINNED, abs=0.01) and railed == 0
        assert orx_clipped(peak, railed)  # the railed count alone misses it
    peak, railed = _peak(radio, 205)
    assert peak < ORX_CLIP_DBFS and not orx_clipped(peak, railed)


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


# -- (a) a pinned peak is never in band: the AGC backs off --------------------------------------


def test_the_agc_never_accepts_a_pinned_peak_and_backs_off():
    radio = _radio()
    seen = []
    real = radio.set_rx_gain
    radio.set_rx_gain = lambda ch, g: (seen.append(int(g)), real(ch, g))[1]
    res = autolevel_capture(radio, ORX, bits=BITS, orx_rate_hz=10e6, start_gain=245,
                            verify_capture_ms=0.8192)  # fmt: skip
    assert res.converged and res.final_gain_index < 218
    assert -2.6 <= res.final_peak_dbfs <= -1.7 and res.railed == 0
    assert seen[0] == 245 and res.final_gain_index == seen[-1]


def test_a_pinned_peak_at_the_gain_floor_is_fatal():
    # The ORx pins (no railed code) even at the minimum gain: the "clips at minimum gain" path.
    res = autolevel_orx(lambda g: None, lambda: (PINNED, 0))
    assert res.fatal and "clips at minimum gain" in res.reason
    vr = verify_no_clip(lambda g: None, lambda: (PINNED, 0), 186)
    assert vr.fatal and "minimum gain" in vr.reason and vr.final_gain_index == 185
    res = autolevel_orx(lambda g: None, lambda: (PINNED, 0), start_gain=185)
    assert res.fatal and "clips at minimum gain" in res.reason


def test_a_pinned_capture_teaches_no_step_and_is_never_the_closest():
    step = OrxStep()
    kw = {"band_lo": -2.6, "band_hi": -1.7}
    step.observe(230, PINNED, 0, **kw)
    step.observe(228, -3.4, 0, **kw)  # 3 dB below the pinned peak in 2 indices: not a slope
    assert step.db_per_index == 0.5
    assert step.closest(-2.0) == (228, -3.4)
    assert relevel_index(PINNED, 0, target_dbfs=-2.0, band_hi=-1.7, db_per_index=0.5) == -3


def test_a_window_that_reaches_the_clip_is_rejected():
    with pytest.raises(ValueError, match="ORx clip"):
        check_orx_window(-1.0, 0.3)  # the old default: accepted up to -0.7 dBFS
    check_orx_window(-2.0, 0.3)
    with pytest.raises(ValueError, match="ORx clip"):
        autolevel_orx(lambda g: None, lambda: (-10.0, 0), target_dbfs=-1.0)
    with pytest.raises(ValueError, match="ORx clip"):
        search_tx_compression(lambda d: None, lambda g: None, lambda: (-2.0, 0, 3.0),
                              target_compression_db=3.0, start_atten_db=20.0, atten_min_db=5.0,
                              orx_gain=220, target_dbfs=-1.5, tol_up_db=0.0)  # fmt: skip
    with pytest.raises(ValueError, match="ORx clip"):
        search_multiband_compression(lambda i, d: None, lambda i, g: None, lambda idx: [],
                                     targets_db=[3.0], start_attens_db=[20.0], atten_mins_db=[5.0],
                                     orx_gains=[220], target_dbfs=-1.0)  # fmt: skip


def test_the_multiband_search_does_not_count_a_pinned_reading():
    gains = {"g": 230}

    def measure(idx):
        g = gains["g"]
        peak = min(PINNED, -2.0 + 0.5 * (g - 220))  # pinned from 223 up
        return [(peak, 0, 4.0 if peak <= ORX_CLIP_DBFS else 2.0, {"top_slope": 1.0})]

    s = search_multiband_compression(
        lambda i, d: None, lambda i, g: gains.__setitem__("g", g), measure,
        targets_db=[4.0], start_attens_db=[20.0], atten_mins_db=[5.0], orx_gains=[230],
        max_steps=1, min_top_slope=None,
    )  # fmt: skip
    first = s.history[0]
    assert first["peak_dbfs"] == PINNED and not first["counts"] and not first["leveled"]
    assert first["action"].startswith("ORx gain -")
    assert any(h["counts"] for h in s.history)  # re-leveled below the clip, then it counts


# -- (b) one empty capture is retaken and the AGC succeeds ---------------------------------------


def test_one_empty_capture_is_retaken_and_the_agc_succeeds():
    radio = _radio(orx_clip_zeros=1)
    calls = _spy(radio)
    res = autolevel_capture(radio, ORX, bits=BITS, orx_rate_hz=10e6, start_gain=245,
                            verify_capture_ms=0.8192)  # fmt: skip
    assert res.converged and -2.6 <= res.final_peak_dbfs <= -1.7
    assert len(calls) > res.fine_iters + res.verify_steps + 1  # each overdriven capture: twice


def test_an_aligned_capture_retakes_an_empty_capture():
    radio = _radio(orx_clip_zeros=1)
    radio.set_rx_gain(ORX, 240)
    ref = radio.board_model.waves["TX1"]
    calls = _spy(radio)
    cap = aligned_capture(radio, ORX, ref, bits=BITS, fs=10e6)
    assert len(calls) == 2 and cap.clip.peak_dbfs == pytest.approx(PINNED, abs=0.01)


# -- (c) persistent empty captures raise the clear error, not OverflowError ----------------------


def test_persistent_empty_captures_raise_the_clear_agc_error():
    radio = _radio(orx_clip_zeros=10)
    calls = _spy(radio)
    with pytest.raises(AgcError, match=r"ORX1 returned no data \(all zeros\) 3 times"):
        autolevel_capture(radio, ORX, bits=BITS, orx_rate_hz=10e6, start_gain=245)
    assert len(calls) == 3
    assert not radio.is_connected()  # the bench left safe


def test_orx_capture_raises_after_its_retakes():
    radio = _radio(orx_clip_zeros=10)
    radio.set_rx_gain(ORX, 240)
    with pytest.raises(AgcError, match="likely driven into clipping"):
        orx_capture(radio, int(ORX), 0.1, bits=BITS, retakes=1, wait_s=0.0)


# -- (d) a peak of -inf never reaches a gain step ------------------------------------------------


def test_relevel_index_with_minus_inf_does_not_overflow():
    kw = {"target_dbfs": -2.0, "band_hi": -1.7, "db_per_index": 0.5}
    assert relevel_index(-math.inf, 0, **kw) == -1
    assert relevel_index(math.nan, 0, **kw) == -1
    step = OrxStep()
    assert step.relevel(220, -math.inf, 0, target_dbfs=-2.0, band_lo=-2.6, band_hi=-1.7) == -1


def test_the_agc_stages_report_no_data_as_fatal():
    empty = lambda: (-math.inf, 0)  # noqa: E731
    for res in (
        autolevel_orx(lambda g: None, empty),  # Stage A
        autolevel_orx(lambda g: None, empty, start_gain=220),  # Stage B
        verify_no_clip(lambda g: None, empty, 220),  # Stage C
    ):
        assert res.fatal and "no data" in res.reason
    readings = iter([(-10.0, 0, 1.0), (-math.inf, 0, math.nan)])
    res = search_tx_compression(lambda d: None, lambda g: None, lambda: next(readings),
                                target_compression_db=3.0, start_atten_db=20.0,
                                atten_min_db=5.0, orx_gain=200)  # fmt: skip
    assert res.fatal and "no data" in res.reason
