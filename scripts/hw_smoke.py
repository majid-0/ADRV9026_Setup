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


def print_table(rows, order, header):
    print(f"    {header}")
    print(f"    {'idx':>3}  {'assumed':<6}  {'n':>7}  rms_dBFS")
    for idx, n, dbfs in rows:
        name = order[idx].name if idx < len(order) and order[idx] else "-"
        d = f"{dbfs:+.1f}" if np.isfinite(dbfs) else "  -inf"
        print(f"    {idx:>3}  {name:<6}  {n:>7}  {d}")


def main() -> None:
    cfg = load_config()
    print(f"Connecting to {cfg.board.ip}:{cfg.board.port} and programming ...")
    with session(cfg) as (radio, info):
        order = returned_channel_order(cfg.channels.rx_init_mask)
        print(f"Profile: {cfg.profile_path.name}")
        print(
            f"  Tx {info.tx_bits}-bit @ {info.tx_rate_khz/1000:.3f} MSPS | "
            f"Rx {info.rx_bits}-bit @ {info.rx_rate_khz/1000:.3f} MSPS"
        )
        for k, v in verify_status(radio).items():
            print(f"  {k}: {v}")

        # [1] Idle baseline -- power of every returned channel, no TX.
        print("\n[1] Idle channel powers (rx_init_mask = " f"0x{cfg.channels.rx_init_mask:X})")
        raw = radio.perform_rx(cfg.channels.rx_init_mask, CAPTURE_MS, timeout_ms=1000)
        print_table(channel_powers(raw, info.rx_bits), order, "baseline, no TX:")

        # [2] Per-TX probe: which index lights up when TXn transmits a tone?
        print("\n[2] TX probe -- transmit a tone on each TX, see which index responds")
        tone = make_tone(TONE_SAMPLES, TONE_HZ, info.tx_rate_hz)
        for tx, expected in PAIRS:
            radio.set_tx_atten(tx, PROBE_TX_ATTEN_DB)
            transmit_bands(radio, {tx: tone}, info.tx_bits, continuous=True)
            raw = radio.perform_rx(cfg.channels.rx_init_mask, CAPTURE_MS, timeout_ms=1000)
            rows = channel_powers(raw, info.rx_bits)
            live = [r for r in rows if np.isfinite(r[2])]
            hot = max(live, key=lambda r: r[2]) if live else None
            print_table(rows, order, f"{tx.name} transmitting (expected {expected.name}):")
            if hot:
                hot_name = order[hot[0]].name if order[hot[0]] else f"index {hot[0]}"
                print(f"      -> strongest: idx {hot[0]} ({hot_name}) at {hot[2]:+.1f} dBFS")
            radio.disable_tx()

        print("\nDone. Use the 'strongest idx' per TX to confirm the real mapping.")


if __name__ == "__main__":
    main()
