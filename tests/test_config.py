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
    assert cfg.lo.lo1_hz == 1_100_000_000
    assert cfg.lo.lo2_hz == 900_000_000
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


def test_server_section_defaults_and_overrides(tmp_path):
    cfg = load_config()
    assert cfg.server.port == 55600 and cfg.server.port != cfg.board.port
    assert cfg.server.heartbeat_timeout_s == 10
    assert cfg.server.idle_timeout_s == 1800
    assert cfg.server.timeout_for("program") == 600
    assert cfg.server.timeout_for("startup") == 600  # falls back to program
    assert cfg.server.timeout_for("perform_rx") == 60

    cfg = Config.from_dict(
        {
            "dll": {"install_dir": "C:/x"},
            "server": {
                "port": 50001,
                "state_dir": str(tmp_path),
                "call_timeout_s": {"perform_rx": 5},
                "not_a_key": 1,
            },
        }
    )
    assert cfg.server.port == 50001
    assert cfg.server.timeout_for("perform_rx") == 5
    assert cfg.server.timeout_for("program") == 600  # defaults kept
    assert cfg.server.authkey_path == tmp_path / "server.key"
    assert cfg.server.log_path == tmp_path / "logs"
