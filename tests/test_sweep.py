from __future__ import annotations

from adrvtrx.sweep import SweepAxis, format_filename, run_sweep, sweep_points


def test_sweep_points_1d():
    axis = SweepAxis("lo_hz", lambda v: None, [1, 2, 3])
    pts = list(sweep_points([axis]))
    assert pts == [{"lo_hz": 1}, {"lo_hz": 2}, {"lo_hz": 3}]


def test_sweep_points_nested_grid():
    a = SweepAxis("lo_hz", lambda v: None, [10, 20])
    b = SweepAxis("atten_db", lambda v: None, [1, 2])
    pts = list(sweep_points([a, b]))
    assert len(pts) == 4
    assert {"lo_hz": 10, "atten_db": 2} in pts
    assert {"lo_hz": 20, "atten_db": 1} in pts


def test_run_sweep_applies_setters_in_order():
    applied = []
    a = SweepAxis("freq", lambda v: applied.append(("freq", v)), [100, 200])
    b = SweepAxis("att", lambda v: applied.append(("att", v)), [5])
    results = run_sweep([a, b], action=lambda pt: dict(pt))
    assert len(results) == 2
    # both setters fire for each point
    assert ("freq", 100) in applied and ("att", 5) in applied


def test_run_sweep_interactive_early_stop():
    a = SweepAxis("freq", lambda v: None, [1, 2, 3, 4])
    seen = []

    def action(pt):
        seen.append(pt["freq"])
        return pt["freq"]

    # stop after we observe value 2
    run_sweep([a], action=action, interactive=lambda pt, res: res < 2)
    assert seen == [1, 2]  # stopped right after 2


def test_format_filename():
    name = format_filename("cap_{lo_hz}_{atten_db}.txt", {"lo_hz": 2000, "atten_db": 30})
    assert name == "cap_2000_30.txt"
