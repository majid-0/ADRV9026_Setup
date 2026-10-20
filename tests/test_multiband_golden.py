"""The multi-band search, pinned: its full output on noise-free synthetic benches.

``tests/data/multiband_search_golden.json`` holds, per scenario, the result, every history
row, every ``on_step`` call and the exact sequence of ``set_atten`` / ``set_gain`` /
``measure`` calls (each a snapshot of the bands it names). It was recorded again for the fast
lock (Task 6: the joint method, one snapshot per step, the learnt ORx step, readings outside the
AGC window), so a refactor that changes nothing passes; a change of behaviour must regenerate it
on purpose:

    python tests/test_multiband_golden.py --regenerate

The benches are noise-free (no random stream), so the file does not depend on the numpy
version. Floats compare to 1e-9.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from adrvtrx.multiband import search_multiband_compression
from test_multiband import LinearPa, TwoBandPa

GOLDEN = Path(__file__).parent / "data" / "multiband_search_golden.json"


def _scenarios() -> dict[str, tuple]:
    two, coupled, own = TwoBandPa, [[1.0, 0.6], [0.6, 1.0]], [[0.5, 0.0], [0.0, 0.5]]
    out: dict[str, tuple] = {}
    for m in ("joint", "pingpong"):
        for cross in ((0.0, 0.0), (0.3, 0.3), (0.0, 0.4), (0.6, 0.6)):
            out[f"{m} cross {cross}"] = (two, {"cross": cross, "noise": 0.0}, {"method": m})
        out[f"{m} at limit"] = (
            two,
            {"locks": (14.0, 9.0), "cross": (0.0, 0.0), "noise": 0.0},
            {"method": m, "atten_mins_db": [10.0, 12.0]},
        )
        out[f"{m} clip guard"] = (
            two,
            {"locks": (12.0, 14.0), "cross": (0.0, 0.0), "noise": 0.0, "clip_below": {0: 13.0}},
            {"method": m},
        )
        out[f"{m} relevel"] = (two, {"noise": 0.0}, {"method": m, "orx_gains": [240, 180]})
        out[f"{m} max steps"] = (two, {"noise": 0.0}, {"method": m, "max_steps": 3})
        out[f"{m} never leveled"] = (
            two,
            {"noise": 0.0, "orx_peak": {0: 0.5}},
            {"method": m, "max_steps": 4, "gain_min": 100},
        )
        out[f"{m} fatal"] = (two, {"noise": 0.0, "orx_peak": {1: 0.5}}, {"method": m})
        out[f"{m} nan every 3rd"] = (two, {"noise": 0.0, "comp_nan": {1: 3}}, {"method": m})
        out[f"{m} nan always"] = (
            two,
            {"noise": 0.0, "comp_nan": {1: 1}},
            {"method": m, "max_steps": 5},
        )
        out[f"{m} railed in band"] = (two, {"noise": 0.0, "railed_first": {0: 1}}, {"method": m})
        out[f"{m} linear coupled"] = (
            LinearPa,
            {"slopes": coupled, "locks": (16.0, 13.5)},
            {"method": m},
        )
        out[f"{m} linear own"] = (LinearPa, {"slopes": own, "locks": (13.0, 17.5)}, {"method": m})
    return out


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    if isinstance(value, float):
        return "nan" if math.isnan(value) else float(f"{value:.10g}")
    return value


def run(kind, pa_kw: dict, search_kw: dict) -> dict:
    """One scenario: the search's result, its on_step calls and every callback, in order.

    ``on_step`` is kept as the history positions of the rows it got (each must be a history
    row, unchanged since).
    """
    pa = kind(**pa_kw)
    calls: list = []
    steps: list = []  # (the rows, a copy of them when on_step got them)
    set_atten, set_gain, measure = pa.a.__setitem__, pa.g.__setitem__, pa.measure

    def atten(i, db):
        calls.append(["atten", i, db])
        set_atten(i, db)

    def gain(i, g):
        calls.append(["gain", i, g])
        set_gain(i, g)

    def read(idx):
        calls.append(["measure", list(idx)])
        return [measure(i) for i in idx]

    args = {
        "targets_db": [4.0, 4.0],
        "start_attens_db": [20.0, 20.0],
        "atten_mins_db": [10.0, 10.0],
        "orx_gains": [210, 210],
        "names": ["TX1", "TX4"],
        "on_step": lambda rows: steps.append((rows, copy.deepcopy(rows))),
    }
    if kind is LinearPa:
        args.update(
            atten_mins_db=[5.0, 5.0], orx_gains=[500, 500], db_per_index=pa.dbpi, gain_min=0,
            gain_max=1000,
        )  # fmt: skip
    args.update(search_kw)
    result = search_multiband_compression(atten, gain, read, **args)
    where = {id(row): k for k, row in enumerate(result.history)}
    on_step = []
    for rows, then in steps:
        assert all(id(row) in where for row in rows) and rows == then  # history rows, unchanged
        on_step.append([where[id(row)] for row in rows])
    out = asdict(result)
    columns = list(out["history"][0]) if out["history"] else []
    assert all(list(row) == columns for row in out["history"])
    out["history"] = {"columns": columns, "rows": [list(row.values()) for row in out["history"]]}
    return _plain({"result": out, "on_step": on_step, "calls": calls})


def _same(got, want, where: str) -> None:
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(
            want
        ), f"{where}: keys {set(got) ^ set(want)}"
        for k in want:
            _same(got[k], want[k], f"{where}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), f"{where}: length"
        for k, (g, w) in enumerate(zip(got, want)):
            _same(g, w, f"{where}[{k}]")
    elif isinstance(want, float) and not isinstance(want, bool):
        assert isinstance(got, (int, float)), f"{where}: {got!r} != {want!r}"
        assert got == pytest.approx(want, rel=1e-9, abs=1e-9), f"{where}: {got!r} != {want!r}"
    else:
        assert got == want, f"{where}: {got!r} != {want!r}"


@pytest.mark.parametrize("name", sorted(_scenarios()))
def test_the_search_does_what_it_did_when_pinned(name):
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    kind, pa_kw, search_kw = _scenarios()[name]
    _same(run(kind, pa_kw, search_kw), golden[name], name)


if __name__ == "__main__":
    if "--regenerate" not in sys.argv:
        sys.exit("python tests/test_multiband_golden.py --regenerate  (overwrites the golden file)")
    data = {name: run(*spec) for name, spec in sorted(_scenarios().items())}
    with open(GOLDEN, "w", encoding="utf-8", newline="\n") as fh:  # LF on every system
        fh.write(json.dumps(data, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {GOLDEN} ({GOLDEN.stat().st_size / 1e3:.0f} kB, {len(data)} scenarios)")
