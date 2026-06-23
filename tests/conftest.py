"""Shared fixtures: a fake .NET bridge so hardware-facing code is testable.

The fake mimics just enough of :class:`adrvtrx._clr.ClrBridge` for the driver's
gain/atten/leveling paths -- ``Types`` factories produce plain ``SimpleNamespace``
objects (settable + inspectable), and ``device``/``board``/``link`` are MagicMocks
that record calls.
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class _TypesFactory:
    """``Types.<struct_name>()`` -> a fresh SimpleNamespace (inspectable struct)."""

    def __getattr__(self, name):
        return lambda: types.SimpleNamespace()


class FakeBridge:
    def __init__(self):
        self.Types = _TypesFactory()
        self.ns = types.SimpleNamespace(FpgaTypes=None)
        self.link = MagicMock(name="link")
        self.board = self.link.platform.board
        self.device = self.board.Adrv9010Device
        self.adrv = self.link.Adrv9010Get.return_value

    # mirror ClrBridge's helper surface
    def enum(self, enum_type_name, member):
        return f"{enum_type_name}.{member}"

    def new_array(self, type_name, length):
        return [None] * length

    def int_array(self, values):
        return list(values)

    def array_list(self, items=()):
        return list(items)


@pytest.fixture
def fake_bridge() -> FakeBridge:
    return FakeBridge()


@pytest.fixture
def fake_radio(fake_bridge):
    """A :class:`Radio` wired to the fake bridge and marked connected (no sockets)."""
    from adrvtrx.config import Config, DllConfig
    from adrvtrx.radio import Radio

    cfg = Config(dll=DllConfig(install_dir=Path("C:/nonexistent")))
    radio = Radio(cfg, bridge=fake_bridge)
    radio.link = fake_bridge.link
    radio.board = fake_bridge.board
    radio.device = fake_bridge.device
    radio.adrv = fake_bridge.adrv
    radio._connected = True
    return radio


@pytest.fixture
def sample_profile(tmp_path) -> Path:
    """A minimal but valid .profile JSON with the fields read_profile needs."""
    content = """
    {
      "framer": [
        {"jesd204M": 8, "jesd204Np": 16, "rxOutputRate_kHz": 245760},
        {"jesd204M": 0, "jesd204Np": 0}
      ],
      "deframer": [
        {"jesd204M": 8, "jesd204Np": 16, "txInputRate_kHz": 245760}
      ]
    }
    """
    path = tmp_path / "Test_StdUseCase.profile"
    path.write_text(content)
    return path
