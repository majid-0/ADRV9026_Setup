"""Bench bring-up smoke test for the ADRV9026 / ADS9 (run on the control PC).

Discovers the things that can only be confirmed live (docs/api_notes.md open items)
and exercises the TX->ORx loopback on the channels you have wired:
    TX2 -> ORx2  and  TX3 -> ORx3.

Run inside myenv with the board connected and booted:
    conda run -n myenv python scripts/hw_smoke.py

It prints a report and writes captures to ./captures/ for you to eyeball. It does
NOT assert -- it's for discovery. The pytest hardware tests (tests/test_hardware.py)
do the pass/fail checks once the layout is known.
"""

from __future__ import annotations

import numpy as np

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import capture
from adrvtrx.config import load_config
from adrvtrx.experiment import session, verify_status
from adrvtrx.gain import clip_report
from adrvtrx.transmit import transmit_bands

# Channels you have wired (TX2->ORx2, TX3->ORx3).
PAIRS = [(TxChannel.TX2, RxChannel.ORX2), (TxChannel.TX3, RxChannel.ORX3)]
TONE_HZ = 5_000_000
TONE_SAMPLES = 4096
CAPTURE_MS = 0.1


def make_tone(n: int, freq_hz: float, rate_hz: float) -> np.ndarray:
    t = np.arange(n) / rate_hz
    return 0.5 * np.exp(2j * np.pi * freq_hz * t)  # 0.5 -> headroom before quantize


def main() -> None:
    cfg = load_config()
    print(f"Connecting to {cfg.board.ip}:{cfg.board.port} and programming ...")
    with session(cfg) as (radio, info):
        print(f"Profile: {cfg.profile_path.name}")
        print(
            f"  Tx {info.tx_bits}-bit @ {info.tx_rate_khz/1000:.3f} MSPS | "
            f"Rx {info.rx_bits}-bit @ {info.rx_rate_khz/1000:.3f} MSPS"
        )
        for k, v in verify_status(radio).items():
            print(f"  {k}: {v}")

        # 1) Discover the PerformRx channel layout for THIS profile.
        print("\n[1] PerformRx layout discovery (mask 0xFF, immediate trigger)")
        raw = radio.perform_rx(0xFF, CAPTURE_MS, timeout_ms=1000)
        n_arrays = _count(raw)
        print(f"    returned {n_arrays} arrays -> {n_arrays // 2} channels present")
        if n_arrays:
            print(
                f"    samples per array: {_len(raw[0])} "
                f"(expected ~{int(CAPTURE_MS * 1e-3 * info.rx_rate_khz * 1e3)})"
            )

        # 2) ORx level readback (confirms RxDecPowerGet sign/scale).
        print("\n[2] RxDecPowerGet on ORx2 / ORx3")
        for orx in (RxChannel.ORX2, RxChannel.ORX3):
            try:
                print(f"    {orx.name}: {radio.rx_dec_power_dbfs(orx):+.2f} dBFS")
            except Exception as exc:  # noqa: BLE001 - discovery, report and continue
                print(f"    {orx.name}: read failed -> {exc}")

        # 3) TX -> ORx loopback on each wired pair.
        print("\n[3] TX->ORx loopback")
        tone = make_tone(TONE_SAMPLES, TONE_HZ, info.tx_rate_hz)
        for tx, orx in PAIRS:
            print(f"    {tx.name} -> {orx.name}: transmit tone, capture ORx")
            transmit_bands(radio, {tx: tone}, info.tx_bits, continuous=True)
            try:
                result = capture(radio, int(orx), CAPTURE_MS, bits=info.rx_bits)
                cap = result.channels[orx]
                rep = clip_report(cap.i, cap.q, info.rx_bits)
                print(
                    f"      captured {len(cap.i)} samples | peak {rep.peak_dbfs:+.2f} dBFS "
                    f"| railed {rep.railed_samples}"
                )
                paths = result.save_all("captures", prefix=f"{tx.name}_{orx.name}")
                print(f"      saved: {', '.join(str(p) for p in paths)}")
            except RuntimeError as exc:
                print(f"      capture mapping issue: {exc}")
            radio.disable_tx()

        print("\nDone. TX will be left safe on exit. Inspect ./captures/*.txt")


def _count(raw) -> int:
    try:
        return len(raw)
    except TypeError:
        n = 0
        try:
            while True:
                raw[n]
                n += 1
        except Exception:  # noqa: BLE001
            return n


def _len(arr) -> int:
    try:
        return len(arr)
    except TypeError:
        return int(arr.Length)


if __name__ == "__main__":
    main()
