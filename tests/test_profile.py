from __future__ import annotations

import pytest

from adrvtrx.profile import read_profile


def test_read_profile_fields(sample_profile):
    info = read_profile(sample_profile)
    assert info.rx_bits == 16
    assert info.tx_bits == 16
    assert info.rx_rate_khz == 245760
    assert info.tx_rate_khz == 245760
    assert info.rx_rate_hz == 245_760_000


def test_read_profile_missing_field(tmp_path):
    path = tmp_path / "bad.profile"
    path.write_text('{"framer": [{"jesd204Np": 0}]}')
    with pytest.raises(ValueError):
        read_profile(path)


def test_read_real_profile_if_present():
    """If the TES install is present, the real StdUseCase102 profile must parse."""
    from adrvtrx.config import load_config

    cfg = load_config()
    if not cfg.profile_path.exists():
        pytest.skip("TES install not present on this machine")
    info = read_profile(cfg.profile_path)
    assert info.rx_bits in (12, 16)
    assert info.tx_rate_khz > 0
