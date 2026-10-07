"""The fake board's PA model: the simple default, the rich memory PA, drift."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.align import estimate_and_align
from adrvtrx.capture import capture
from adrvtrx.compression import find_compression_point
from adrvtrx.config import Config, DllConfig, lo_for_tx
from adrvtrx.fake import FakeRadio, PaModel
from adrvtrx.gmp import GMP
from adrvtrx.metrics import nmse_db
from adrvtrx.transmit import transmit_bands
from adrvtrx.waveform import prepare_tx

FS = 491.52e6
TX, ORX = TxChannel.TX1, RxChannel.ORX1


@pytest.fixture
def cfg(tmp_path) -> Config:
    """Profile 98's datapath: 491.52 MSPS, 12 bits (the fake reads it for its rate)."""
    profile = tmp_path / "uc98.profile"
    profile.write_text(
        json.dumps(
            {
                "framer": [{"jesd204Np": 12, "rxOutputRate_kHz": 491520}],
                "deframer": [{"jesd204Np": 12, "txInputRate_kHz": 491520}],
            }
        )
    )
    return Config(dll=DllConfig(install_dir=Path("C:/nonexistent")), profile_name=str(profile))


def signal(bw_hz: float, n: int = 16384, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n, 1 / FS)
    spec = (rng.normal(size=n) + 1j * rng.normal(size=n)) * (np.abs(f) < 0.45 * bw_hz)
    x = np.fft.ifft(spec)
    return x / np.abs(x).max()


def radio_at(cfg: Config, freq_hz: float, pa="rich", **kw) -> FakeRadio:
    radio = FakeRadio(cfg, seed=3, pa=pa, **kw)
    radio._safe_hooks_installed = True
    radio.connect()
    radio.program()
    radio.retune_lo(lo_for_tx(cfg.clocks, TX), int(freq_hz))
    return radio


def ref_codes(x: np.ndarray) -> np.ndarray:
    i, q = prepare_tx(x, 12)
    return i + 1j * q


def lock(radio: FakeRadio, x: np.ndarray):
    transmit_bands(radio, {TX: x}, 12)
    return find_compression_point(
        radio,
        TX,
        ORX,
        ref_codes(x),
        rx_bits=12,
        fs=FS,
        target_compression_db=3.5,
        start_atten_db=15.0,
        atten_min_db=7.0,
        lock_on="gain",
    )


def aligned_capture(radio: FakeRadio, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(x normalized, ORx capture aligned to it and divided by the linear gain)."""
    ref = ref_codes(x)
    cap = capture(radio, int(ORX), 2 * len(x) / FS * 1e3, bits=12).channels[ORX]
    _x, y, _delay = estimate_and_align(ref, cap.iq, FS)
    xn = ref / 2047.0
    return xn, y * np.vdot(xn, xn) / np.vdot(xn, y)


def test_the_default_model_is_the_simple_rapp():
    x = signal(100e6, 1024) * 0.9
    v = x * 6.0 * 10 ** (-12.0 / 20)
    expected = np.roll(v / (1 + np.abs(v) ** 4) ** 0.25, 5)
    np.testing.assert_allclose(PaModel().output(x, 12.0, 2.4e9), expected)
    assert PaModel.from_spec(None) == PaModel.from_spec("simple") == PaModel()


def test_gain_lock_converges_at_frequency_dependent_attenuations(cfg):
    x = signal(100e6)
    low, high = lock(radio_at(cfg, 1.6e9), x), lock(radio_at(cfg, 2.8e9), x)
    for res in (low, high):
        assert res.converged and not res.at_atten_floor
        assert abs(res.compression_db - 3.5) <= 0.3
        assert 7.0 < res.final_atten_db < 15.0
    assert low.final_atten_db - high.final_atten_db >= 1.5


def test_gmp_inverse_fit_on_a_capture_beats_no_dpd_and_needs_memory(cfg):
    results = {}
    for bw in (40e6, 100e6):
        radio = radio_at(cfg, 2.2e9)
        x = signal(bw)
        res = lock(radio, x)
        assert res.converged
        xn, z = aligned_capture(radio, x)
        half = len(xn) // 2
        none = nmse_db(xn[half:], z[half:])
        gmp = GMP(5, 3, 2).fit(z[:half], xn[:half]).predict(z[half:])
        memoryless = GMP(5, 1, 0).fit(z[:half], xn[:half]).predict(z[half:])
        results[bw] = (none, nmse_db(xn[half:], memoryless), nmse_db(xn[half:], gmp))
    for none, memoryless, gmp in results.values():
        assert np.isfinite(gmp) and gmp < none - 15.0  # a post-inverse clearly helps
        assert gmp < memoryless - 10.0  # the PA has memory
    assert results[100e6][1] > results[40e6][1] + 3.0  # memory hurts the wider signal more


def _rms_db(y: np.ndarray) -> float:
    return 10 * np.log10(np.mean(np.abs(y) ** 2))


def _small_signal_output(radio: FakeRadio) -> np.ndarray:
    radio.set_tx_atten(TX, 30.0)  # far below compression: output follows the gain
    return radio.board_model.pa_output(TX)


def test_drift_file_shifts_the_gain_on_demand(cfg, tmp_path):
    drift = tmp_path / "drift_db.txt"
    radio = radio_at(cfg, 2.2e9, pa={"preset": "rich", "drift_file": str(drift)})
    transmit_bands(radio, {TX: signal(40e6, 4096)}, 12)
    before = _rms_db(_small_signal_output(radio))
    drift.write_text("-1.5\n")
    after = _rms_db(_small_signal_output(radio))
    assert after - before == pytest.approx(-1.5, abs=0.05)
    # and the ORx captures follow
    radio.set_rx_gain(ORX, 240)
    radio.set_rx_enable(int(ORX))
    drift.write_text("0")
    level0 = _rms_db(capture(radio, int(ORX), 0.05, bits=12).channels[ORX].iq)
    drift.write_text("-1.5")
    level1 = _rms_db(capture(radio, int(ORX), 0.05, bits=12).channels[ORX].iq)
    assert level1 - level0 == pytest.approx(-1.5, abs=0.2)


def test_drift_rate_and_step_follow_the_clock(cfg):
    pa = {
        "preset": "rich",
        "drift_db_per_hour": 2.0,
        "drift_step_db": 0.5,
        "drift_step_after_s": 60,
    }
    radio = radio_at(cfg, 2.2e9, pa=pa)
    board = radio.board_model
    transmit_bands(radio, {TX: signal(40e6, 4096)}, 12)
    now = board.created
    board.clock = lambda: now
    start = _rms_db(_small_signal_output(radio))
    board.clock = lambda: now + 30  # before the step: 2 dB/h for 30 s
    assert _rms_db(_small_signal_output(radio)) - start == pytest.approx(2 * 30 / 3600, abs=0.01)
    board.clock = lambda: now + 1800  # half an hour: 1 dB, plus the 0.5 dB step
    assert _rms_db(_small_signal_output(radio)) - start == pytest.approx(1.5, abs=0.02)


def test_the_model_is_deterministic_with_a_seed(cfg):
    caps = []
    for _ in range(2):
        radio = radio_at(cfg, 2.0e9, pa={"preset": "rich", "noise_codes": 2.0})
        transmit_bands(radio, {TX: signal(40e6, 2048)}, 12)
        radio.set_rx_enable(int(ORX))
        caps.append(capture(radio, int(ORX), 0.01, bits=12).channels[ORX].iq)
    np.testing.assert_array_equal(caps[0], caps[1])


def test_pa_parameters_load_from_env_json_and_toml(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("ADRVTRX_FAKE_PA", "rich")
    assert FakeRadio(cfg).board_model.pa == PaModel.rich()
    as_json = tmp_path / "pa.json"
    as_json.write_text(json.dumps({"preset": "rich", "noise_codes": 3.0}))
    monkeypatch.setenv("ADRVTRX_FAKE_PA", str(as_json))
    assert FakeRadio(cfg).board_model.pa == PaModel.rich(noise_codes=3.0)
    as_toml = tmp_path / "pa.toml"
    as_toml.write_text("drive = 4.0\npre_taps = [1.0, 0.1]\nseed = 5\n")
    assert PaModel.from_spec(as_toml) == PaModel(drive=4.0, pre_taps=[1.0, 0.1], seed=5)
    with pytest.raises(ValueError, match="unknown fake PA parameter"):
        PaModel.from_spec({"drvie": 1.0})
