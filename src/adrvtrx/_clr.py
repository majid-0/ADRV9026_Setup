"""The single module that touches the .NET runtime (pythonnet / clr).

Everything else imports symbols from here, so the rest of the package is testable
without pythonnet by injecting a fake ``ClrBridge`` (see ``tests/conftest.py``).

Mirrors the init script's bootstrap::

    clr.AddReferenceToFileAndPath(".../adrvtrx_dll.dll")
    from adrv9010_dll import AdiEvaluationSystem, Types, Ad9528Types
    link = AdiEvaluationSystem.Instance
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from .config import DllConfig

#: Namespaces to try when ``DllConfig.namespace`` is blank (auto-detect).
_CANDIDATE_NAMESPACES = ("adrv9010_dll", "adrv9025_dll", "adrvtrx_dll")


class ClrBridge:
    """Loads ``adrvtrx_dll.dll`` and exposes its key types.

    Attributes:
        link: ``AdiEvaluationSystem.Instance`` singleton.
        Types: the ``Types`` namespace (enums + structs).
        Ad9528Types: the ``Ad9528Types`` namespace.
        Array: ``System.Array`` (for ``Array.CreateInstance``).
        ns: the resolved DLL namespace module.
    """

    def __init__(self, dll: DllConfig):
        self._dll = dll
        self.ns = None
        self.Types = None
        self.Ad9528Types = None
        self.Array = None
        self.link = None

    def load(self) -> ClrBridge:
        dll_path = self._dll.dll_path
        if not dll_path.exists():
            raise FileNotFoundError(f"adrvtrx DLL not found at {dll_path}")

        # Dependent assemblies (Newtonsoft.Json, etc.) live next to the DLL.
        install_dir = str(self._dll.install_dir)
        if install_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = install_dir + os.pathsep + os.environ.get("PATH", "")
        if install_dir not in sys.path:
            sys.path.append(install_dir)

        import clr  # pythonnet; imported lazily so non-hardware code never needs it

        clr.AddReference(str(dll_path))

        ns_name = self._dll.namespace or ""
        candidates = (ns_name,) if ns_name else _CANDIDATE_NAMESPACES
        last_err: Exception | None = None
        for name in candidates:
            if not name:
                continue
            try:
                self.ns = importlib.import_module(name)
                break
            except ImportError as exc:  # pragma: no cover - hardware path
                last_err = exc
        if self.ns is None:
            raise ImportError(f"could not import DLL namespace (tried {candidates}): {last_err}")

        import System  # noqa: N813  (provided by pythonnet)

        self.System = System
        self.Array = System.Array
        self.Types = self.ns.Types
        self.Ad9528Types = getattr(self.ns, "Ad9528Types", None)
        self.link = self.ns.AdiEvaluationSystem.Instance
        return self

    # -- small helpers used by higher layers ----------------------------------

    def enum(self, enum_type_name: str, member: str):
        """Resolve ``Types.<enum_type_name>.<member>`` to a .NET enum value."""
        enum_type = getattr(self.Types, enum_type_name)
        return getattr(enum_type, member)

    def new_array(self, type_name: str, length: int):
        """``Array.CreateInstance(Types.<type_name>, length)`` (as in the script)."""
        return self.Array.CreateInstance(getattr(self.Types, type_name), length)

    def int_array(self, values):
        """Build a ``System.Int32[]`` from a Python sequence (for PerformTx buffers)."""
        from System import Int32  # type: ignore

        arr = self.Array.CreateInstance(Int32, len(values))
        for idx, v in enumerate(values):
            arr[idx] = int(v)
        return arr

    def array_list(self, items=()):
        """Build a ``System.Collections.ArrayList`` (the PerformTx txData overload)."""
        from System.Collections import ArrayList  # type: ignore

        al = ArrayList()
        for item in items:
            al.Add(item)
        return al


def vendor_profile_dir(dll: DllConfig) -> Path:
    return dll.profiles_dir
