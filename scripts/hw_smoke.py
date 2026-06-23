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

        # Enable the main-Rx datapath; link-sharing carries ORx on the same framer.
        # Without this, PerformRx returns zero-filled buffers for every channel.
        radio.enable_rx(0x0F)

        # [1] Idle baseline -- power of every returned channel, no TX.
        print("\n[1] Idle channel powers (rx_init_mask = " f"0x{cfg.channels.rx_init_mask:X})")
        raw = radio.perform_rx(cfg.channels.rx_init_mask, CAPTURE_MS, timeout_ms=1000)
        print_table(channel_powers(raw, info.rx_bits), order, "baseline, no TX:")

        # [2] Per-TX probe. The ADRV9026 has only 2 ORx ADCs (data idx 4 & 5);
        # ORx1/2/3/4 are front-end inputs muxed into them. Enable the ORx input for
        # this TX (rxMask bit: ORx2=0x20, ORx3=0x40) so its front-end powers up, then
        # see which of idx 4/5 carries the tone.
        print("\n[2] TX probe -- enable the ORx input + transmit, see which ORx ADC responds")
        tone = make_tone(TONE_SAMPLES, TONE_HZ, info.tx_rate_hz)
        for tx, orx in PAIRS:
            radio.set_tx_atten(tx, PROBE_TX_ATTEN_DB)
            radio.rx_tx_enable(0x0F | int(orx), 0)  # main Rx + this ORx front-end
            transmit_bands(radio, {tx: tone}, info.tx_bits, continuous=True)
            raw = radio.perform_rx(cfg.channels.rx_init_mask, CAPTURE_MS, timeout_ms=1000)
            rows = channel_powers(raw, info.rx_bits)
            print_table(
                rows,
                order,
                f"{tx.name} -> {orx.name} input enabled (rxMask 0x{0x0F | int(orx):X}):",
            )
            orx_slots = [r for r in rows if r[0] in (4, 5) and np.isfinite(r[2])]
            hot = max(orx_slots, key=lambda r: r[2]) if orx_slots else None
            if hot:
                print(f"      -> {tx.name} energy on ORx ADC idx {hot[0]} at {hot[2]:+.1f} dBFS")
            radio.disable_tx()

        print("\nDone. The ORx ADC idx that lights up per TX is the real data slot.")


if __name__ == "__main__":
    main()
