"""Run notebooks/dpd_linearize_loop.ipynb end to end on the simulated bench.

The notebook's code cells run in order with ``Radio`` replaced by the sim bench,
the profile read replaced by one at the sim rate, and a small synthetic signal
file in a temporary folder. Parameters are overridden right after the
parameters cell. Skipped when matplotlib is not installed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import adrvtrx.experiment
import adrvtrx.profile
import adrvtrx.radio
from adrvtrx.profile import ProfileInfo
from sim_bench import BW, FS, SimRadio, make_signal

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "dpd_linearize_loop.ipynb"
PEAK_LIMIT_DBM = 9.9

SIM_PARAMS = {
    "SIGNAL_PATH": "sig.txt",
    "BW_MHZ": int(BW / 1e6),
    "LO1_HZ": 2_000_000_000,
    "TX_ATTEN_START_DB": 20.0,
    "TX_ATTEN_MIN_DB": 5.0,
    "SAVE_DIR": "run",
    "LABEL": "ila",
}


class _BenchRadio(SimRadio):
    """The sim bench with the connection calls the notebook makes."""

    def __init__(self, cfg):
        super().__init__(seed=1)

    def connect(self):
        pass

    def force_safe(self):
        self.safe_state()

    def program(self):
        pass

    def get_lo(self, pll):
        return self.lo_hz.get(pll, 0)

    def get_tx_atten(self, channel):
        return self.atten


@pytest.fixture
def bench(tmp_path, monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    plt.switch_backend("Agg")
    monkeypatch.setattr(plt, "show", lambda *a, **k: plt.close("all"))
    khz = int(FS / 1e3)
    info = ProfileInfo(tx_bits=12, rx_bits=12, tx_rate_khz=khz, rx_rate_khz=khz, orx_rate_khz=khz)
    monkeypatch.setattr(adrvtrx.radio, "Radio", _BenchRadio)
    monkeypatch.setattr(adrvtrx.profile, "read_profile", lambda path: info)
    monkeypatch.setattr(adrvtrx.experiment, "verify_status", lambda radio: {"sim": True})
    monkeypatch.chdir(tmp_path)
    x = make_signal(seed=3)
    np.savetxt("sig.txt", np.column_stack((x.real, x.imag)), delimiter="\t", fmt="%.9g")
    return tmp_path


def run_notebook(**params) -> dict:
    cells = [c for c in json.loads(NOTEBOOK.read_text())["cells"] if c["cell_type"] == "code"]
    ns: dict = {"__name__": "__main__"}
    for i, cell in enumerate(cells):
        exec(compile("".join(cell["source"]), f"{NOTEBOOK.name}[{i}]", "exec"), ns)
        if i == 0:
            ns.update(SIM_PARAMS)
            ns.update(params)
    return ns


def _check_loop(ns) -> None:
    table = ns["table"]
    assert ns["result"].reason == "n_iter"
    assert len(table) == ns["N_ITER"] == 4
    worst = [r["aclr_worst_dbc"] for r in table]
    assert worst[-1] <= worst[0] - 20.0, worst
    assert all(b <= a + 0.3 for a, b in zip(worst, worst[1:])), worst
    assert table[-1]["nmse_db"] <= table[0]["nmse_db"] - 20.0
    assert all(r["dpd_peak_dbm"] <= PEAK_LIMIT_DBM for r in table[1:])
    assert all(r["tx_clipped"] == 0 for r in table)
    with open(Path(ns["SAVE_DIR"]) / "ila_steps.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["iteration"]) for r in rows] == [0, 1, 2, 3]
    assert {"aclr_lower_dbc", "aclr_upper_dbc", "margin_db", "dpd_peak_dbm"} <= set(rows[0])
    assert (Path(ns["SAVE_DIR"]) / "ila.csv").is_file()
    assert ns["radio"].disconnected and not ns["radio"].tx_on


def test_lock_on_papr_then_loop(bench):
    ns = run_notebook()
    op = ns["op"]
    assert op.search.converged and op.condition.lock_on == "papr"
    assert abs(op.condition.compression_db - ns["TARGET_COMPRESSION_DB"]) <= ns["COMP_TOL_DB"]
    assert (bench / "run" / "TX1_conditions.csv").is_file()
    _check_loop(ns)


def test_lock_on_gain_with_clip_guard_then_loop(bench):
    ns = run_notebook(
        LOCK_ON="gain", TARGET_COMPRESSION_DB=4.0, COMP_TOL_DB=0.2, MIN_TOP_SLOPE=0.08
    )
    op = ns["op"]
    assert op.condition.lock_on == "gain"
    assert abs(op.search.gain_compression_db - 4.0) <= 0.2
    _check_loop(ns)


def test_saved_condition_then_loop(bench):
    first = run_notebook(N_ITER=1, SAVE_DIR="lock")
    name = first["condition"].name
    ns = run_notebook(
        CONDITION_SOURCE="csv", CONDITIONS_CSV="lock/TX1_conditions.csv", CONDITION_NAME=name
    )
    assert "op" not in ns
    assert ns["condition"].name == name
    assert ns["condition"].tx_atten_db == first["condition"].tx_atten_db
    _check_loop(ns)
