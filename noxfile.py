"""Local CI entrypoints. Run `nox` to lint + test without hardware.

The test session runs only the hardware-free unit tests (the .NET boundary is
mocked), so it is safe on any machine, including Linux CI runners.
"""

from __future__ import annotations

import nox

PYTHON_VERSIONS = ["3.9", "3.11"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    session.install("-e", ".[dev]")
    session.run("pytest", "-m", "not hardware", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    session.install("ruff>=0.4", "black>=24.0")
    session.run("ruff", "check", "src", "tests")
    session.run("black", "--check", "src", "tests")
