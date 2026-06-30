"""Typed configuration for the ADRV9026 / ADS9 setup, loaded from TOML.

Mirrors the working init script one-to-one (see ``config/default.toml``). Loading
is pure-Python (no DLL), so config is fully unit-testable.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on 3.9/3.10 only
    import tomli as tomllib


@dataclass
class DllConfig:
    install_dir: Path
    dll_relpath: str = "adrvtrx_dll.dll"
    namespace: str = "adrv9010_dll"  # "" -> auto-detect

    @property
    def dll_path(self) -> Path:
        return self.install_dir / self.dll_relpath

    @property
    def profiles_dir(self) -> Path:
        return self.install_dir / "Resources" / "Adi.ADRV9025.Profiles" / "public"


@dataclass
class BoardConfig:
    ip: str = "192.168.1.10"
    port: int = 55556


@dataclass
class ClocksConfig:
    device_clock_khz: int = 245760
    ad9528: list[int] = field(default_factory=lambda: [245760, 122880, 122880, 245760])
    rx12_lo: str = "LO1"
    rx34_lo: str = "LO1"
    tx12_lo: str = "LO2"
    tx34_lo: str = "LO2"
    orx12_lo: str = "TXLO"
    orx34_lo: str = "TXLO"
    pll1_lo_mode: str = "INTLO_NOOUTPUT"
    pll2_lo_mode: str = "INTLO_NOOUTPUT"
    ext_lo1_khz: int = 0
    ext_lo2_khz: int = 0


@dataclass
class ChannelsConfig:
    rx_init_mask: int = 0x3FF
    tx_init_mask: int = 0xF
    rx_enable_mode: str = "SPI_MODE"
    tx_enable_mode: str = "SPI_MODE"
    orx_enable_mode: str = "SPI_MODE"


@dataclass
class LoConfig:
    lo1_hz: int = 1_800_000_000
    lo2_hz: int = 2_000_000_000
    aux_hz: int = 0


@dataclass
class InitCalsConfig:
    channel_mask: int = 0xF
    warm_boot: int = 0
    mask: list[str] = field(default_factory=list)


@dataclass
class LevelsConfig:
    tx_atten_db: dict[str, float] = field(default_factory=lambda: {"default": 30.0})
    rx_gain_index: dict[str, int] = field(default_factory=lambda: {"default": 195})

    def tx_atten_for(self, channel_name: str) -> float:
        return float(self.tx_atten_db.get(channel_name, self.tx_atten_db["default"]))

    def rx_gain_for(self, channel_name: str) -> int:
        return int(self.rx_gain_index.get(channel_name, self.rx_gain_index["default"]))


@dataclass
class OrxAgcConfig:
    target_dbfs: float = -1.0
    tol_up_db: float = 0.3
    tol_down_db: float = 0.6
    coarse_ms: float = 0.1
    max_iterations: int = 16
    gain_min: int = 185
    gain_max: int = 255
    db_per_index: float = 0.50


@dataclass
class Config:
    dll: DllConfig
    board: BoardConfig = field(default_factory=BoardConfig)
    clocks: ClocksConfig = field(default_factory=ClocksConfig)
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    lo: LoConfig = field(default_factory=LoConfig)
    tx_to_orx: list[str] = field(
        default_factory=lambda: ["TX1_ORX1", "TX2_ORX2", "TX3_ORX3", "TX4_ORX4"]
    )
    init_cals: InitCalsConfig = field(default_factory=InitCalsConfig)
    levels: LevelsConfig = field(default_factory=LevelsConfig)
    orx_agc: OrxAgcConfig = field(default_factory=OrxAgcConfig)
    profile_name: str = "ADRV9025Init_StdUseCase102_LinkSharing.profile"

    @property
    def profile_path(self) -> Path:
        """Resolve the profile to an absolute path under the install profiles dir."""
        p = Path(os.path.expandvars(self.profile_name))
        return p if p.is_absolute() else self.dll.profiles_dir / p

    # -- loading ---------------------------------------------------------------

    @classmethod
    def from_toml(cls, path: str | os.PathLike[str]) -> Config:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        dll_raw = data.get("dll", {})
        if "install_dir" not in dll_raw:
            raise ValueError("config [dll] section must set install_dir")
        dll = DllConfig(
            install_dir=Path(os.path.expandvars(dll_raw["install_dir"])),
            dll_relpath=dll_raw.get("dll_relpath", "adrvtrx_dll.dll"),
            namespace=dll_raw.get("namespace", "adrv9010_dll"),
        )
        profile = data.get("profile", {})
        return cls(
            dll=dll,
            board=BoardConfig(**data.get("board", {})),
            clocks=ClocksConfig(**data.get("clocks", {})),
            channels=ChannelsConfig(**data.get("channels", {})),
            lo=LoConfig(**data.get("lo", {})),
            tx_to_orx=data.get("tx_to_orx", {}).get(
                "map", ["TX1_ORX1", "TX2_ORX2", "TX3_ORX3", "TX4_ORX4"]
            ),
            init_cals=InitCalsConfig(**data.get("init_cals", {})),
            levels=_levels_from(data.get("levels", {})),
            orx_agc=_orx_agc_from(data.get("orx_agc", {})),
            profile_name=profile.get("name", cls.profile_name),
        )


def _orx_agc_from(raw: dict[str, Any]) -> OrxAgcConfig:
    """Build OrxAgcConfig, ignoring unknown keys (e.g. a retired ``tolerance_db``).

    Keeps the loader forward/backward compatible with TOML files that carry extra
    or missing ``[orx_agc]`` keys.
    """
    known = {f.name for f in fields(OrxAgcConfig)}
    return OrxAgcConfig(**{k: v for k, v in raw.items() if k in known})


def _levels_from(raw: dict[str, Any]) -> LevelsConfig:
    levels = LevelsConfig()
    if "tx_atten_db" in raw:
        levels.tx_atten_db = {k: float(v) for k, v in raw["tx_atten_db"].items()}
    if "rx_gain_index" in raw:
        levels.rx_gain_index = {k: int(v) for k, v in raw["rx_gain_index"].items()}
    return levels


def lo_for_tx(clocks: ClocksConfig, tx_channel) -> str:
    """Which LO ('LO1'/'LO2') a TX channel uses, per the per-pair clock select.

    TX1/TX2 follow ``tx12_lo``; TX3/TX4 follow ``tx34_lo``. Use this to retune the
    right LO for a given TX (e.g. a frequency sweep of TX2 should retune its LO).
    """
    return clocks.tx12_lo if int(tx_channel) & 0x03 else clocks.tx34_lo


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load config from ``path``, or from the ``ADRVTRX_CONFIG`` env var, or the
    bundled ``config/default.toml``."""
    if path is None:
        path = os.environ.get("ADRVTRX_CONFIG")
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "default.toml"
    return Config.from_toml(path)
