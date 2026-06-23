from __future__ import annotations

import numpy as np
import pytest

from adrvtrx import waveform as wf


def test_full_scale():
    assert wf.full_scale(16) == 32767
    assert wf.full_scale(12) == 2047
    with pytest.raises(ValueError):
        wf.full_scale(1)


def test_samples_for_duration():
    # 1 ms at 245.76 MSPS
    assert wf.samples_for_duration(1.0, 245760) == 245760


def test_load_round_trip(tmp_path):
    iq = np.array([0.5 + 0.5j, -0.25 + 0.1j, 0.0 + 0.0j])
    path = tmp_path / "wave.txt"
    out = np.column_stack([iq.real, iq.imag])
    np.savetxt(path, out, delimiter="\t")
    loaded = wf.load_tab_iq(path)
    np.testing.assert_allclose(loaded, iq)


def test_load_rejects_wrong_columns(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1\t2\t3\n")
    with pytest.raises(ValueError):
        wf.load_tab_iq(path)


def test_normalize_unit_peak():
    iq = np.array([1 + 1j, 2 + 0j, 0 - 3j])
    out = wf.normalize(iq)
    assert np.isclose(np.max(np.abs(out)), 1.0)


def test_normalize_zero_signal():
    iq = np.zeros(4, dtype=complex)
    out = wf.normalize(iq)
    assert np.all(out == 0)


def test_quantize_full_scale_maps_to_max():
    iq = np.array([1 + 0j, 0 + 1j, -1 + 0j])
    i_int, q_int = wf.quantize(iq, 16)
    assert i_int[0] == 32767
    assert q_int[1] == 32767
    assert i_int[2] == -32767


def test_quantize_clips_out_of_range():
    iq = np.array([2.0 + 0j])  # beyond unit scale -> must clip, not overflow
    i_int, q_int = wf.quantize(iq, 16)
    assert i_int[0] == 32767


def test_prepare_tx_normalizes_then_quantizes():
    iq = np.array([0.5 + 0j, 0.25 + 0j])  # peak 0.5 -> normalizes to 1.0
    i_int, _ = wf.prepare_tx(iq, 16)
    assert i_int[0] == 32767


def test_save_float_is_normalized(tmp_path):
    i_int = np.array([32767, -32767, 0], dtype=np.int32)
    q_int = np.array([0, 16384, 0], dtype=np.int32)
    path = tmp_path / "cap.txt"
    wf.save_tab_iq_float(i_int, q_int, path, 16)
    data = np.loadtxt(path, delimiter="\t")
    assert np.isclose(data[0, 0], 1.0)
    assert np.isclose(data[1, 0], -1.0)
    assert abs(data[1, 1] - 0.5) < 0.01
