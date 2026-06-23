from __future__ import annotations

from pathlib import Path

import pytest

from adrvtrx.config import Config, load_config


def test_load_bundled_default():
    cfg = load_config()
    assert cfg.board.ip == "192.168.1.10"
    assert cfg.board.port == 55556
    assert cfg.channels.rx_init_mask == 0x3FF
    assert cfg.channels.tx_init_mask == 0xF
    assert cfg.lo.lo1_hz == 1_800_000_000
    assert cfg.lo.lo2_hz == 2_000_000_000
    assert "TX_QEC_INIT" in cfg.init_cals.mask
    assert cfg.tx_to_orx == ["TX1_ORX1", "TX2_ORX2", "TX3_ORX3", "TX4_ORX4"]


def test_profile_path_resolves_under_install_dir():
    cfg = load_config()
    p = cfg.profile_path
    assert p.name == "ADRV9025Init_StdUseCase98_LinkSharing.profile"
    assert "Adi.ADRV9025.Profiles" in str(p)


def test_levels_defaults_and_overrides():
    cfg = Config.from_dict(
        {
            "dll": {"install_dir": "C:/x"},
            "levels": {
                "tx_atten_db": {"default": 30, "tx1": 20},
                "rx_gain_index": {"default": 195},
            },
        }
    )
    assert cfg.levels.tx_atten_for("tx1") == 20.0
    assert cfg.levels.tx_atten_for("tx2") == 30.0
    assert cfg.levels.rx_gain_for("orx1") == 195


def test_missing_install_dir_raises():
    with pytest.raises(ValueError):
        Config.from_dict({"board": {"ip": "10.0.0.1"}})


def test_absolute_profile_name_is_respected():
    cfg = Config.from_dict(
        {"dll": {"install_dir": "C:/x"}, "profile": {"name": "C:/abs/my.profile"}}
    )
    assert cfg.profile_path == Path("C:/abs/my.profile")
