"""The fake board's shared PA: several TX into one PA (dual band)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import capture
from adrvtrx.config import Config, DllConfig
from adrvtrx.fake import FakeRadio, PaModel
from adrvtrx.metrics import gain_compression_db
from adrvtrx.radio import MAX_TX_ATTEN_DB
from adrvtrx.transmit import transmit_bands

TX1, TX4 = TxChannel.TX1, TxChannel.TX4
ORX1, ORX4 = RxChannel.ORX1, RxChannel.ORX4
N = 8192


def _signal(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N) + 1j * rng.normal(size=N)
    return x / np.abs(x).max()


def _radio(pa: dict) -> FakeRadio:
    radio = FakeRadio(Config(dll=DllConfig(install_dir=Path("C:/nonexistent"))), seed=3, pa=pa)
    radio._safe_hooks_installed = True
    radio.connect()
    radio.program()
    radio.retune_lo("LO1", 1_440_000_000)
    radio.retune_lo("LO2", 1_860_000_000)
    return radio


def _play(radio, atten1: float, atten4: float, x1=None, x4=None) -> None:
    radio.set_tx_atten(TX1, atten1)
    radio.set_tx_atten(TX4, atten4)
    transmit_bands(
        radio, {TX1: _signal(1) if x1 is None else x1, TX4: _signal(2) if x4 is None else x4}, 12
    )


def _compression(radio, tx: TxChannel) -> float:
    board = radio.board_model
    y = np.roll(board.pa_output(tx), -board.pa.delay_samples)
    x = board.waves[tx.name]
    return gain_compression_db(x, y)[0]


SHARED = {"shared": [["TX1", "TX4"]], "coupling": 4.0, "noise_codes": 0.0}


def test_one_band_compresses_more_when_the_other_band_drives_harder():
    radio = _radio(SHARED)
    _play(radio, 12.0, 16.0)
    quiet = _compression(radio, TX1)
    _play(radio, 12.0, 12.0)
    loud = _compression(radio, TX1)
    assert 0.15 < (loud - quiet) / 4.0 < 0.5  # dB of TX1 compression per dB of TX4 (coupling 4)
    _play(radio, 16.0, 12.0)
    assert _compression(radio, TX1) < quiet - 1.0  # its own attenuation still matters most

    own = _radio({"noise_codes": 0.0})  # without the shared PA, TX4 does not matter
    _play(own, 12.0, 30.0)
    a = _compression(own, TX1)
    _play(own, 12.0, 12.0)
    assert _compression(own, TX1) == pytest.approx(a, abs=1e-9)


def test_a_band_alone_in_a_shared_pa_behaves_like_its_own_pa():
    shared, own = _radio(SHARED), _radio({"noise_codes": 0.0, "coupling": 0.5})
    for radio in (shared, own):
        radio.set_tx_atten(TX1, 12.0)
        transmit_bands(radio, {TX1: _signal(1)}, 12)  # TX4 not enabled: it does not feed the PA
    np.testing.assert_allclose(
        shared.board_model.pa_output(TX1), own.board_model.pa_output(TX1), rtol=1e-12
    )


def test_a_band_beside_a_member_at_full_attenuation_compresses_like_its_own_pa():
    """The other member plays (so the shared model runs) but at 41.95 dB its power is nil: a band's
    own power must not use up its own headroom."""
    shared, own = _radio(SHARED), _radio({"noise_codes": 0.0, "coupling": 4.0})
    _play(shared, 12.0, MAX_TX_ATTEN_DB)
    own.set_tx_atten(TX1, 12.0)
    transmit_bands(own, {TX1: _signal(1)}, 12)
    assert shared.board_model.state["tx_mask"] == int(TX1) | int(TX4)
    assert _compression(shared, TX1) == pytest.approx(_compression(own, TX1), abs=0.01)
    assert _compression(own, TX1) > 1.0  # compressed: a lost headroom would show


def test_coupling_zero_decouples_the_bands():
    radio = _radio({**SHARED, "coupling": 0.0})
    _play(radio, 12.0, 30.0)
    a = _compression(radio, TX1)
    _play(radio, 12.0, 12.0)
    assert _compression(radio, TX1) == pytest.approx(a, abs=1e-9)


def test_a_spec_file_may_carry_comment_keys(tmp_path):
    spec = tmp_path / "preset.json"
    spec.write_text('{"_about": "fitted to ...", "shared": [["TX1", "TX4"]], "coupling": 0.3}')
    pa = PaModel.from_spec(str(spec))
    assert pa.shared == [["TX1", "TX4"]] and pa.coupling == 0.3


def test_per_band_overrides_and_their_validation():
    radio = _radio({**SHARED, "bands": {"TX4": {"smoothness": 0.8, "drive": 3.0}}})
    assert radio.board_model.pa.for_tx("TX4").smoothness == 0.8
    assert radio.board_model.pa.for_tx("TX1").smoothness == 2.0
    with pytest.raises(ValueError, match="unknown fake PA parameter"):
        PaModel(bands={"TX4": {"not_a_parameter": 1}})
    with pytest.raises(ValueError, match="shared PA group"):
        PaModel(shared=[["TX1", "TX9"]])


def _tone_power_db(cap, f_hz: float, fs: float) -> float:
    z = cap.iq
    spec = np.abs(np.fft.fft(z * np.hanning(len(z)))) ** 2
    k = int(round(f_hz / fs * len(z))) % len(z)
    return 10 * np.log10(spec[k - 3 : k + 4].sum() + 1e-30)


def _tone(k: int) -> np.ndarray:
    return 0.5 * np.exp(2j * np.pi * k / N * np.arange(N))  # k whole cycles per period


def test_the_other_band_shows_in_the_orx_window_only_when_asked_and_inside():
    fs = 10e6  # the fake's default ORx rate (no profile file): window +-5 MHz
    cases = (
        (False, 1_442_000_000, False),  # 2 MHz above LO1, not asked
        (True, 1_442_000_000, True),  # inside the window
        (True, 1_460_000_000, False),  # 20 MHz above LO1: outside
    )
    for other_in_window, lo2, expect in cases:
        radio = _radio({**SHARED, "other_in_window": other_in_window})
        radio.retune_lo("LO2", lo2)
        _play(radio, 20.0, 20.0, x1=_tone(-1638), x4=_tone(410))  # -2.0 MHz and +0.5 MHz
        radio.set_rx_gain(ORX1, 200)
        cap = capture(radio, int(ORX1), 0.8192, bits=12).channels[ORX1]
        at = _tone_power_db(cap, (lo2 - 1_440_000_000) + 410 / N * fs, fs)
        quiet = _tone_power_db(cap, -3.3e6, fs)
        own = _tone_power_db(cap, -1638 / N * fs, fs)
        assert own > quiet + 40  # band 1's own tone is always there
        assert bool(at > quiet + 20) is expect, (other_in_window, lo2, at, quiet)


def test_skew_delays_one_tx_before_the_pa():
    x = _signal(1)
    decoupled = {**SHARED, "coupling": 0.0}  # so TX1 does not see TX4's (skewed) power change
    plain, skewed = _radio(decoupled), _radio({**decoupled, "tx_skew_samples": {"TX4": 3}})
    for radio in (plain, skewed):
        _play(radio, 25.0, 25.0, x1=x, x4=x)
    a = plain.board_model.pa_output(TX4)
    b = skewed.board_model.pa_output(TX4)
    np.testing.assert_allclose(np.roll(a, 3), b, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        plain.board_model.pa_output(TX1), skewed.board_model.pa_output(TX1), rtol=1e-6
    )


def test_one_snapshot_starts_every_channel_together_so_skew_shows():
    from adrvtrx.align import estimate_and_align

    for skew in (0, 3):
        radio = _radio({**SHARED, "coupling": 0.0, "tx_skew_samples": {"TX4": skew}})
        _play(radio, 25.0, 25.0)
        radio.set_rx_gain(ORX1, 215)
        radio.set_rx_gain(ORX4, 215)
        joint = capture(
            radio, int(ORX1) | int(ORX4), 2 * N / 10e3, bits=12
        )  # two periods at 10 Msps
        board = radio.board_model
        delays = [
            estimate_and_align(board.waves[tx], joint.channels[orx].iq, 10e6)[2]
            for tx, orx in (("TX1", ORX1), ("TX4", ORX4))
        ]
        assert delays[1] - delays[0] == pytest.approx(skew, abs=0.05)


def test_tx_iq_image():
    x = _signal(1)
    radio = _radio({"noise_codes": 0.0, "tx_image_db": -20.0})
    radio.set_tx_atten(TX1, 30.0)  # nearly linear: the output is about the drive times x
    transmit_bands(radio, {TX1: x}, 12)
    board = radio.board_model
    y = np.roll(board.pa_output(TX1), -board.pa.delay_samples)
    xq = board.waves["TX1"]
    coef = np.linalg.lstsq(np.c_[xq, np.conj(xq)], y, rcond=None)[0]
    assert 20 * np.log10(abs(coef[1] / coef[0])) == pytest.approx(-20.0, abs=0.5)
