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

# Run ONLY on the control PC with the ADS9 + ADRV9026 connected.
test-hw:
	pytest -m hardware

ci: lint test
