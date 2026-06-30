"""Declarative multi-band sweep plans with per-block zip or grid combine.

Each top-level block in a sweep dict (``freq``, ``power_db``, ``signals``) declares
``mode: "zip" | "grid"``. Blocks multiply against each other; inside a block, ``zip``
pairs values by index and ``grid`` takes the Cartesian product of its lists.

Use :func:`summarize_sweep_plan` before running hardware to see point count and samples.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_BLOCK_NAMES = frozenset({"freq", "power_db", "signals"})
_FREQ_KEYS = frozenset({"lo1_hz", "lo2_hz"})


def as_list(value: Any) -> list[Any]:
    """Scalar -> single-element list; sequences (not str) pass through as list."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _band_names(bands: Sequence[Mapping[str, Any]]) -> list[str]:
    return [str(b["name"]) for b in bands]


def _resolve_defaults(
    bands: Sequence[Mapping[str, Any]],
    defaults: Mapping[str, Any] | None,
) -> dict[str, Any]:
    names = _band_names(bands)
    base: dict[str, Any] = {
        "lo1_hz": None,
        "lo2_hz": None,
        "power_db": {},
        "signals": {},
    }
    if defaults:
        if defaults.get("lo1_hz") is not None:
            base["lo1_hz"] = int(defaults["lo1_hz"])
        if defaults.get("lo2_hz") is not None:
            base["lo2_hz"] = int(defaults["lo2_hz"])
        for key in ("power_db", "signals"):
            if key in defaults:
                raw = defaults[key]
                if isinstance(raw, Mapping):
                    base[key] = dict(raw)
                else:
                    raise TypeError(f"defaults[{key!r}] must be a mapping")
    for b in bands:
        nm = b["name"]
        if nm not in base["power_db"] and "power_db" in b:
            base["power_db"][nm] = float(b["power_db"])
        if nm not in base["signals"] and "signal" in b:
            base["signals"][nm] = str(b["signal"])
    for nm in names:
        if nm not in base["power_db"]:
            raise ValueError(
                f"no default TX attenuation for band {nm!r}; "
                "pass defaults or sweep_defaults_from_config(cfg, bands)"
            )
        base["signals"].setdefault(nm, "")
    return base


def sweep_defaults_from_config(
    cfg,
    bands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build sweep defaults from ``Config`` LOs, ``[levels].tx_atten_db``, and band signals."""
    power_db: dict[str, float] = {}
    signals: dict[str, str] = {}
    for b in bands:
        nm = str(b["name"])
        tx = b["tx"]
        tx_key = tx.name.lower() if hasattr(tx, "name") else str(tx).lower()
        power_db[nm] = cfg.levels.tx_atten_for(tx_key)
        if "signal" in b:
            signals[nm] = str(b["signal"])
    return {
        "lo1_hz": int(cfg.lo.lo1_hz),
        "lo2_hz": int(cfg.lo.lo2_hz),
        "power_db": power_db,
        "signals": signals,
    }


def _block_entries(block: Mapping[str, Any]) -> dict[str, list[Any]]:
    return {k: as_list(v) for k, v in block.items() if k != "mode"}


def _block_size(block: Mapping[str, Any]) -> int:
    mode = block.get("mode", "grid")
    entries = _block_entries(block)
    if not entries:
        return 1
    sizes = [len(v) for v in entries.values()]
    if mode == "zip":
        if len(set(sizes)) != 1:
            keys = ", ".join(f"{k}={len(v)}" for k, v in entries.items())
            raise ValueError(f"zip block needs equal-length lists, got {keys}")
        return sizes[0]
    return int(np_product(sizes))


def np_product(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out *= v
    return out


def _partial_points(block_name: str, block: Mapping[str, Any]) -> list[dict[str, Any]]:
    mode = block.get("mode", "grid")
    entries = _block_entries(block)
    if not entries:
        return [{}]

    keys = list(entries.keys())
    lists = [entries[k] for k in keys]

    combos: list[tuple[Any, ...]]
    if mode == "zip":
        n = len(lists[0])
        if any(len(lst) != n for lst in lists):
            detail = ", ".join(f"{k}={len(entries[k])}" for k in keys)
            raise ValueError(f"zip block {block_name!r}: equal lengths required ({detail})")
        combos = [tuple(lst[i] for lst in lists) for i in range(n)]
    else:
        combos = list(itertools.product(*lists))

    partials: list[dict[str, Any]] = []
    for combo in combos:
        chunk: dict[str, Any] = {}
        for k, v in zip(keys, combo):
            if block_name == "freq":
                if k not in _FREQ_KEYS:
                    raise ValueError(f"unknown freq key {k!r}; use lo1_hz and/or lo2_hz")
                chunk[k] = int(v)
            elif block_name == "power_db":
                if k == "shared":
                    chunk.setdefault("power_db", {})["__shared__"] = float(v)
                else:
                    chunk.setdefault("power_db", {})[k] = float(v)
            elif block_name == "signals":
                chunk.setdefault("signals", {})[k] = str(v)
        partials.append(chunk)
    return partials


def _merge_partials(partials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in partials:
        for k, v in p.items():
            if k in ("power_db", "signals") and isinstance(v, dict):
                out.setdefault(k, {}).update(v)
            else:
                out[k] = v
    return out


def _finalize_point(
    merged: dict[str, Any], defaults: dict[str, Any], band_names: list[str]
) -> dict[str, Any]:
    lo1 = merged.get("lo1_hz", defaults["lo1_hz"])
    lo2 = merged.get("lo2_hz", defaults["lo2_hz"])
    merged_power = merged.get("power_db", {})
    shared = merged_power.get("__shared__")
    sig_raw = dict(defaults["signals"])
    sig_raw.update(merged.get("signals", {}))

    power: dict[str, float] = {}
    for nm in band_names:
        if nm in merged_power:
            power[nm] = float(merged_power[nm])
        elif shared is not None:
            power[nm] = float(shared)
        else:
            power[nm] = float(defaults["power_db"][nm])

    signals = {nm: str(sig_raw[nm]) for nm in band_names}
    return {
        "lo1_hz": int(lo1) if lo1 is not None else None,
        "lo2_hz": int(lo2) if lo2 is not None else None,
        "power_db": power,
        "signals": signals,
    }


def iter_sweep_points(
    bands: Sequence[Mapping[str, Any]],
    sweep: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield fully resolved sweep points (LO, per-band power, per-band signal paths)."""
    names = _band_names(bands)
    resolved_defaults = _resolve_defaults(bands, defaults)

    unknown = set(sweep) - _BLOCK_NAMES
    if unknown:
        raise ValueError(f"unknown sweep block(s) {sorted(unknown)}; use {_BLOCK_NAMES}")

    block_partials: list[list[dict[str, Any]]] = []
    for name in ("freq", "power_db", "signals"):
        if name not in sweep:
            continue
        block = sweep[name]
        if not isinstance(block, Mapping):
            raise TypeError(f"sweep[{name!r}] must be a mapping")
        block_partials.append(_partial_points(name, block))

    if not block_partials:
        yield _finalize_point({}, resolved_defaults, names)
        return

    for combo in itertools.product(*block_partials):
        merged = _merge_partials(combo)
        yield _finalize_point(merged, resolved_defaults, names)


def flatten_point(point: Mapping[str, Any]) -> dict[str, Any]:
    """Flat dict for filenames, tables, and plot keys."""
    flat: dict[str, Any] = {}
    if point.get("lo1_hz") is not None:
        flat["lo1_hz"] = point["lo1_hz"]
    if point.get("lo2_hz") is not None:
        flat["lo2_hz"] = point["lo2_hz"]
    for nm, db in point["power_db"].items():
        flat[f"{nm}_power_db"] = db
    for nm, path in point["signals"].items():
        flat[f"{nm}_signal"] = path
    return flat


def format_point_label(point: Mapping[str, Any]) -> str:
    """Compact underscore label from a flat point."""
    flat = flatten_point(point)
    parts = []
    for k, v in flat.items():
        if k.endswith("_signal"):
            parts.append(f"{k}_{Path(str(v)).stem}")
        elif "hz" in k:
            parts.append(f"{k}_{int(v) // 1_000_000}M")
        else:
            parts.append(f"{k}_{v:g}")
    return "_".join(parts)


@dataclass
class SweepPlanSummary:
    band_names: list[str]
    blocks: list[dict[str, Any]] = field(default_factory=list)
    n_points: int = 0
    sample_points: list[dict[str, Any]] = field(default_factory=list)

    def format(self, *, max_samples: int = 5) -> str:
        lines = ["Sweep plan summary", "=" * 40]
        if not self.blocks:
            lines.append("  (no sweep blocks — 1 point at defaults)")
        else:
            lines.append("Blocks (sizes multiply):")
            for b in self.blocks:
                mode = b["mode"]
                size = b["size"]
                axes = ", ".join(f"{k}[{n}]" for k, n in b["axes"].items())
                lines.append(f"  {b['name']}: {mode} -> {size}  ({axes})")
        lines.append(f"Total points: {self.n_points}")
        if self.n_points == 0:
            return "\n".join(lines)
        show = self.sample_points[:max_samples]
        if self.n_points > len(show):
            lines.append(f"First {len(show)} point(s):")
        else:
            lines.append("Points:")
        for i, pt in enumerate(show):
            lines.append(f"  [{i}] {_format_point_line(pt)}")
        if self.n_points > max_samples:
            lines.append(f"  ... ({self.n_points - max_samples} more)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


def _format_point_line(point: Mapping[str, Any]) -> str:
    bits = []
    if point.get("lo1_hz") is not None:
        bits.append(f"LO1={point['lo1_hz']/1e6:.3f} MHz")
    if point.get("lo2_hz") is not None:
        bits.append(f"LO2={point['lo2_hz']/1e6:.3f} MHz")
    for nm, db in point["power_db"].items():
        bits.append(f"{nm}={db:g} dB")
    for nm, path in point["signals"].items():
        bits.append(f"{nm}={Path(path).name}")
    return " | ".join(bits)


def summarize_sweep_plan(
    bands: Sequence[Mapping[str, Any]],
    sweep: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
    *,
    max_samples: int = 5,
) -> SweepPlanSummary:
    """Return point count, per-block sizes, and sample rows for the chosen sweep."""
    names = _band_names(bands)
    block_info: list[dict[str, Any]] = []
    for name in ("freq", "power_db", "signals"):
        if name not in sweep:
            continue
        block = sweep[name]
        entries = _block_entries(block)
        size = _block_size(block)
        block_info.append(
            {
                "name": name,
                "mode": block.get("mode", "grid"),
                "size": size,
                "axes": {k: len(v) for k, v in entries.items()},
            }
        )

    points = list(iter_sweep_points(bands, sweep, defaults))
    return SweepPlanSummary(
        band_names=names,
        blocks=block_info,
        n_points=len(points),
        sample_points=points[:max_samples],
    )


def apply_sweep_point(
    radio,
    bands: Sequence[Mapping[str, Any]],
    point: Mapping[str, Any],
    *,
    tx_bits: int,
    wave_cache: dict[str, Any] | None = None,
    do_normalize: bool = True,
) -> dict[Any, Any]:
    """Set LOs, TX atten, and reload TX buffers when a band's signal path changes.

    Returns ``{TxChannel: waveform}`` for the active bands (cached by signal path).
    """
    from .transmit import transmit_bands
    from .waveform import load_tab_iq

    cache = wave_cache if wave_cache is not None else {}
    if point.get("lo1_hz") is not None:
        radio.retune_lo("LO1", int(point["lo1_hz"]))
    if point.get("lo2_hz") is not None:
        radio.retune_lo("LO2", int(point["lo2_hz"]))

    channel_to_iq: dict[Any, Any] = {}
    for b in bands:
        nm = b["name"]
        tx = b["tx"]
        radio.set_tx_atten(tx, float(point["power_db"][nm]))
        path = str(point["signals"][nm])
        if path not in cache:
            cache[path] = load_tab_iq(path)
        channel_to_iq[tx] = cache[path]

    lengths = {len(w) for w in channel_to_iq.values()}
    if len(lengths) != 1:
        raise ValueError(f"all TX waveforms must be equal length, got {sorted(lengths)}")

    transmit_bands(radio, channel_to_iq, tx_bits, continuous=True, do_normalize=do_normalize)
    return channel_to_iq


def run_planned_sweep(
    radio,
    bands: Sequence[Mapping[str, Any]],
    sweep: Mapping[str, Any],
    action: Callable[[dict[str, Any], dict[str, Any]], Any],
    *,
    defaults: Mapping[str, Any] | None = None,
    tx_bits: int,
    do_normalize: bool = True,
    interactive: Callable[[dict[str, Any], Any], bool] | None = None,
) -> list[Any]:
    """Walk :func:`iter_sweep_points`, apply each point, run ``action(point, waves)``."""
    wave_cache: dict[str, Any] = {}
    results: list[Any] = []
    for point in iter_sweep_points(bands, sweep, defaults):
        waves = apply_sweep_point(
            radio,
            bands,
            point,
            tx_bits=tx_bits,
            wave_cache=wave_cache,
            do_normalize=do_normalize,
        )
        result = action(point, waves)
        results.append(result)
        if interactive is not None and not interactive(point, result):
            break
    return results


def max_power_db(
    bands: Sequence[Mapping[str, Any]],
    sweep: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
) -> float:
    """Highest TX attenuation (dB) across all points — use as a safe startup level."""
    best = 0.0
    for point in iter_sweep_points(bands, sweep, defaults):
        for db in point["power_db"].values():
            best = max(best, float(db))
    return best
