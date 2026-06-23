"""Parameter sweeps: 1-D and nested grids over the capture primitive.

A sweep is a set of named axes (each a parameter setter + value list). The engine
walks the Cartesian product (nested grid) -- a single axis is just the 1-D case --
applies every setter, runs an ``action`` per point, and names outputs from a
template carrying the swept values. ``interactive`` lets a callback inspect each
point (e.g. clip report) and decide whether to keep going.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SweepAxis:
    name: str
    setter: Callable[[Any], None]
    values: Sequence[Any]


def sweep_points(axes: Sequence[SweepAxis]) -> Iterator[dict[str, Any]]:
    """Yield one ``{axis_name: value}`` dict per grid point (Cartesian product)."""
    names = [a.name for a in axes]
    for combo in itertools.product(*[a.values for a in axes]):
        yield dict(zip(names, combo))


def format_filename(template: str, point: dict[str, Any]) -> str:
    """Fill ``{axis}`` placeholders in a filename template from a sweep point."""
    return template.format(**point)


def run_sweep(
    axes: Sequence[SweepAxis],
    action: Callable[[dict[str, Any]], Any],
    *,
    interactive: Callable[[dict[str, Any], Any], bool] | None = None,
) -> list[Any]:
    """Walk the grid: set every axis, run ``action(point)``, collect results.

    If ``interactive`` is given it is called as ``interactive(point, result)`` after
    each point; returning ``False`` stops the sweep early (e.g. user aborts after
    inspecting a clip report).
    """
    setters = {a.name: a.setter for a in axes}
    results: list[Any] = []
    for point in sweep_points(axes):
        for name, value in point.items():
            setters[name](value)
        result = action(point)
        results.append(result)
        if interactive is not None and not interactive(point, result):
            break
    return results


# -- ready-made axis factories -------------------------------------------------


def frequency_axis(radio, pll: str, freqs_hz: Sequence[int], *, name: str = "lo_hz") -> SweepAxis:
    """Sweep an LO; uses lock-checked retune so each point is settled before capture."""
    return SweepAxis(name, lambda hz: radio.retune_lo(pll, int(hz)), list(freqs_hz))


def attenuation_axis(radio, channel, attens_db: Sequence[float], *, name: str = "atten_db"):
    return SweepAxis(name, lambda db: radio.set_tx_atten(channel, float(db)), list(attens_db))


def gain_axis(radio, channel, gains: Sequence[int], *, name: str = "gain_index") -> SweepAxis:
    return SweepAxis(name, lambda g: radio.set_rx_gain(channel, int(g)), list(gains))
