from __future__ import annotations

import pytest

from adrvtrx.sweep_plan import (
    flatten_point,
    iter_sweep_points,
    summarize_sweep_plan,
)


def _dual_bands():
    return [
        {"name": "band1", "tx": "TX2", "orx": "ORX2", "signal": "/a/Signal1.txt"},
        {"name": "band2", "tx": "TX3", "orx": "ORX3", "signal": "/a/Signal2.txt"},
    ]


def _defaults(**overrides):
    base = {
        "lo1_hz": 1_100_000_000,
        "lo2_hz": 900_000_000,
        "power_db": {"band1": 30.0, "band2": 30.0},
        "signals": {"band1": "/a/Signal1.txt", "band2": "/a/Signal2.txt"},
    }
    base.update(overrides)
    return base


def test_zip_freq_grid_power_grid_signals():
    """2 zip LO pairs x 3 grid power x 2 grid band1 signals = 12 points."""
    sweep = {
        "freq": {
            "mode": "zip",
            "lo1_hz": [1.0e9, 1.1e9],
            "lo2_hz": [0.9e9, 1.0e9],
        },
        "power_db": {"mode": "grid", "shared": [13, 14, 15]},
        "signals": {
            "mode": "grid",
            "band1": ["/a/Signal1a.txt", "/a/Signal1b.txt"],
            "band2": "/a/Signal2.txt",
        },
    }
    summary = summarize_sweep_plan(_dual_bands(), sweep, _defaults(lo1_hz=0, lo2_hz=0))
    assert summary.n_points == 12
    assert len(summary.blocks) == 3
    assert summary.blocks[0]["size"] == 2
    assert summary.blocks[1]["size"] == 3
    assert summary.blocks[2]["size"] == 2

    points = list(iter_sweep_points(_dual_bands(), sweep, _defaults(lo1_hz=0, lo2_hz=0)))
    assert len(points) == 12
    assert points[0]["lo1_hz"] == 1_000_000_000
    assert points[0]["lo2_hz"] == 900_000_000
    assert points[0]["power_db"] == {"band1": 13.0, "band2": 13.0}
    assert points[0]["signals"]["band1"].endswith("Signal1a.txt")


def test_zip_power_shared_separate_block():
    sweep = {
        "freq": {
            "mode": "zip",
            "lo1_hz": [1.0e9, 1.1e9],
            "lo2_hz": [0.9e9, 1.0e9],
        },
        "power_db": {"mode": "zip", "shared": [20, 10]},
    }
    assert summarize_sweep_plan(_dual_bands(), sweep, _defaults()).n_points == 4


def test_zip_mismatched_lengths_raises():
    sweep = {
        "freq": {
            "mode": "zip",
            "lo1_hz": [1.0e9, 1.1e9],
            "lo2_hz": [0.9e9],
        },
    }
    with pytest.raises(ValueError, match="equal"):
        list(iter_sweep_points(_dual_bands(), sweep, _defaults()))


def test_single_band_power_only():
    bands = [{"name": "band1", "tx": "TX2", "orx": "ORX2", "signal": "/a/S1.txt"}]
    sweep = {"power_db": {"mode": "grid", "shared": [20, 10, 5]}}
    defaults = _defaults(power_db={"band1": 30.0})
    assert summarize_sweep_plan(bands, sweep, defaults).n_points == 3


def test_empty_sweep_one_point_at_defaults():
    bands = [{"name": "band1", "tx": "TX2", "orx": "ORX2", "signal": "/a/S1.txt"}]
    defaults = {"lo1_hz": 1_100_000_000, "lo2_hz": 0, "power_db": {"band1": 30.0}}
    pts = list(iter_sweep_points(bands, {}, defaults))
    assert len(pts) == 1
    assert pts[0]["lo1_hz"] == 1_100_000_000
    assert pts[0]["signals"]["band1"] == "/a/S1.txt"


def test_summarize_format_includes_total():
    text = str(
        summarize_sweep_plan(
            _dual_bands(), {"power_db": {"mode": "grid", "shared": [13, 14]}}, _defaults()
        )
    )
    assert "Total points: 2" in text
    assert "power_db" in text


def test_flatten_point_keys():
    pt = {
        "lo1_hz": 1e9,
        "lo2_hz": None,
        "power_db": {"band1": 13.0},
        "signals": {"band1": "/x/Sig.txt"},
    }
    flat = flatten_point(pt)
    assert flat["lo1_hz"] == 1e9
    assert flat["band1_power_db"] == 13.0
    assert flat["band1_signal"] == "/x/Sig.txt"
    assert "lo2_hz" not in flat


def test_sweep_defaults_from_config():
    from adrvtrx import TxChannel
    from adrvtrx.config import Config, DllConfig, LevelsConfig, LoConfig
    from adrvtrx.sweep_plan import sweep_defaults_from_config

    cfg = Config(
        dll=DllConfig(install_dir="C:/dummy"),
        lo=LoConfig(lo1_hz=1_000_000_000, lo2_hz=800_000_000),
        levels=LevelsConfig(tx_atten_db={"default": 30.0, "tx2": 12.0}),
    )
    bands = [{"name": "band1", "tx": TxChannel.TX2, "orx": "ORX2", "signal": "/x/S1.txt"}]
    d = sweep_defaults_from_config(cfg, bands)
    assert d["lo1_hz"] == 1_000_000_000
    assert d["power_db"]["band1"] == 12.0
    assert d["signals"]["band1"] == "/x/S1.txt"


def test_missing_power_default_raises():
    bands = [{"name": "band1", "tx": "TX2", "orx": "ORX2"}]
    with pytest.raises(ValueError, match="default TX attenuation"):
        list(iter_sweep_points(bands, {}))
