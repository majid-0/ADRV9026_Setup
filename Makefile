# Convenience targets (work in Git Bash / make on Windows, or any POSIX shell).
.PHONY: install lint format test test-hw ci

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

# Hardware-free unit tests (the .NET boundary is mocked) -- this is "CI".
test:
	pytest -m "not hardware"

# Run ONLY on the control PC with the ADS9 + ADRV9026 connected and the hardware
# server stopped: these tests connect to the board directly and program it. A plain
# `pytest` deselects them; this target opts in (tests/test_hardware.py).
test-hw:
	ADRVTRX_ALLOW_HARDWARE_TESTS=1 pytest -m hardware -o addopts="" -v

ci: lint test
