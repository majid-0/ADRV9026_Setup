"""Bench bring-up + channel-mapping discovery for the ADRV9026 / ADS9.

Empirically maps the PerformRx readback for the loaded profile: it dumps the power
of all returned channels at idle, then transmits a tone on each wired TX (TX2, TX3)
and reports which returned-array index lights up. That reveals the true TX->ORx
index mapping without assuming bit order (link-sharing profiles reorder/!populate
some slots).

Run inside myenv with the board connected + booted:
    conda run -n myenv python scripts/hw_smoke.py
"""

from __future__ import annotations

import numpy as np

from adrvtrx import RxChannel, TxChannel
from adrvtrx._enums import is_orx
from adrvtrx.capture import returned_channel_order
from adrvtrx.config import load_config
from adrvtrx.experiment import session, verify_status
from adrvtrx.transmit import transmit_bands

PAIRS = [(TxChannel.TX2, RxChannel.ORX2), (TxChannel.TX3, RxChannel.ORX3)]
TONE_HZ = 5_000_000
TONE_SAMPLES = 4096
CAPTURE_MS = 0.1
PROBE_TX_ATTEN_DB = 10.0  # lower than the 30 dB default for a clean loopback signal


def make_tone(n: int, freq_hz: float, rate_hz: float) -> np.ndarray:
    t = np.arange(n) / rate_hz
    return 0.5 * np.exp(2j * np.pi * freq_hz * t)


def _len(arr) -> int:
    try:
        return len(arr)
    except TypeError:
        return int(getattr(arr, "Length", 0) or getattr(arr, "Count", 0))


def channel_powers(raw, n_bits: int):
    """Return [(idx, n_samples, rms_dbfs)] for each channel in a PerformRx result."""
    fs = float(1 << (n_bits - 1))
    n_arrays = _len(raw)
    rows = []
    for k in range(n_arrays // 2):
        i = np.fromiter(raw[2 * k], dtype=np.int64)
        q = np.fromiter(raw[2 * k + 1], dtype=np.int64)
        if i.size == 0:
            rows.append((k, 0, float("-inf")))
            continue
        rms = np.sqrt(np.mean(i.astype(float) ** 2 + q.astype(float) ** 2))
        dbfs = 20 * np.log10(rms / fs) if rms > 0 else float("-inf")
        rows.append((k, i.size, dbfs))
    return rows


ORX_FLOOR_MARGIN_DB = 10.0  # a slot must beat the idle floor by this to count as carrying the input


def orx_readback_slots(rows, order):
    """Readback indices of the physical ORx ADC slots that actually carry data."""
    return [
        idx
        for idx, n, _d in rows
        if idx < len(order) and order[idx] and is_orx(order[idx]) and n > 0
    ]


def resolve_selected_orx(rows, order, selected, floor_dbfs):
    """Label the physical ORx slots in terms of the ORx INPUT that was selected.

    You enable ORx1/2/3/4; its data lands on one of the 2 ORx ADC slots. The slot
    that rises above the idle floor is the one carrying ``selected`` -- so we name
    it after the input you picked (e.g. ``ORX2``). Returns ``(labels, note)`` and
    refuses to guess when zero slots (input not landing) or several slots (a spur
    lighting everything) are hot.
    """
    by_idx = {idx: d for idx, _n, d in rows}
    slots = orx_readback_slots(rows, order)
    labels = dict.fromkeys(slots, "ORx(off)")
    hot = [
        idx
        for idx in slots
        if np.isfinite(by_idx[idx]) and by_idx[idx] > floor_dbfs + ORX_FLOOR_MARGIN_DB
    ]
    if len(hot) == 1:
        labels[hot[0]] = selected.name
        return labels, f"{selected.name} -> readback idx {hot[0]} at {by_idx[hot[0]]:+.1f} dBFS"
    if not hot:
        return labels, (
            f"{selected.name}: NOT LANDING -- no ORx slot beat the idle floor by "
            f"{ORX_FLOOR_MARGIN_DB:.0f} dB (check cable / TX level)"
        )
    for idx in hot:
        labels[idx] = f"{selected.name}?"
    idxs = ", ".join(str(i) for i in hot)
    return labels, (
        f"{selected.name}: AMBIGUOUS -- idx {idxs} all hot at ~{by_idx[hot[0]]:+.1f} dBFS "
        f"(looks like a spur, not a clean tone)"
    )


def print_table(rows, order, header, orx_label=None):
    orx_label = orx_label or {}
    print(f"    {header}")
    print(f"    {'idx':>3}  {'channel':<9}  {'n':>7}  rms_dBFS")
    for idx, n, dbfs in rows:
        ch = order[idx] if idx < len(order) and order[idx] else None
        if ch is not None and is_orx(ch):
            name = orx_label.get(idx) or ("ORx(none)" if n == 0 else "ORx")
        elif ch is not None:
            name = ch.name
        else:
            name = "-"
        d = f"{dbfs:+.1f}" if np.isfinite(dbfs) else "  -inf"
        print(f"    {idx:>3}  {name:<9}  {n:>7}  {d}")


def main() -> None:
    cfg = load_config()
    print(f"Connecting to {cfg.board.ip}:{cfg.board.port} and programming ...")
    with session(cfg) as (radio, info):
        order = returned_channel_order(cfg.channels.rx_init_mask)
        print(f"Profile: {cfg.profile_path.name}")
        print(
            f"  Tx {info.tx_bits}-bit @ {info.tx_rate_khz/1000:.3f} MSPS | "
            f"Rx {info.rx_bits}-bit @ {info.rx_rate_khz/1000:.3f} MSPS | "
            f"ORx @ {info.orx_rate_khz/1000:.3f} MSPS"
        )
        for k, v in verify_status(radio).items():
            print(f"  {k}: {v}")

        # Enable the main-Rx datapath; link-sharing carries ORx on the same framer.
        # Without this, PerformRx returns zero-filled buffers for every channel.
        radio.enable_rx(0x0F)

        # [1] Idle baseline -- power of every returned channel, no TX.
        print("\n[1] Idle channel powers (rx_init_mask = " f"0x{cfg.channels.rx_init_mask:X})")
        raw = radio.perform_rx(cfg.channels.rx_init_mask, CAPTURE_MS, timeout_ms=1000)
        idle_rows = channel_powers(raw, info.rx_bits)
        idle_slots = orx_readback_slots(idle_rows, order)
        print_table(idle_rows, order, "baseline, no TX:", dict.fromkeys(idle_slots, "ORx(idle)"))
        idle_floor = max(
            (d for i, _n, d in idle_rows if i in idle_slots and np.isfinite(d)), default=-100.0
        )

        # [2] Per-input probe: select an ORx INPUT (ORx1/2/3/4), transmit its TX,
        # and the readback names that input on whichever slot carries it -- so the
        # feedback is in your terms, not ADC indices.
        print("\n[2] ORx probe -- select an ORx input, transmit, read it back by name")
        tone = make_tone(TONE_SAMPLES, TONE_HZ, info.tx_rate_hz)
        for tx, orx in PAIRS:
            radio.set_tx_atten(tx, PROBE_TX_ATTEN_DB)
            radio.rx_tx_enable(0x0F | int(orx), 0)  # main Rx + this ORx input
            transmit_bands(radio, {tx: tone}, info.tx_bits, continuous=True)
            raw = radio.perform_rx(cfg.channels.rx_init_mask, CAPTURE_MS, timeout_ms=1000)
            rows = channel_powers(raw, info.rx_bits)
            labels, note = resolve_selected_orx(rows, order, orx, idle_floor)
            print_table(
                rows,
                order,
                f"{tx.name} -> {orx.name} selected (rxMask 0x{0x0F | int(orx):X}):",
                labels,
            )
            print(f"      -> {note}")
            radio.disable_tx()

        print("\nDone. Each ORx input you select is labeled by name on the slot it lands on.")


if __name__ == "__main__":
    main()
