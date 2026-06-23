"""AGC test -- SINGLE band.

Define a test signal, TX level(s) and a frequency grid, then transmit and run the
ORx auto-level ("AGC") at every grid point. Prints a clean table of the settled
ORx gain + achieved level per point, and suggests a static gain for the signal.

    conda run -n myenv python scripts/agc_test_single.py

The sweep is the Cartesian product of TX_ATTENS_DB x FREQS_HZ. Leave TX_ATTENS_DB
a single value to sweep frequency only; LO2 drives both TX and ORx (link-sharing),
so retuning it moves the whole loopback together.
"""

from __future__ import annotations

import statistics

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import capture
from adrvtrx.config import load_config
from adrvtrx.experiment import session
from adrvtrx.gain import ORX_GAIN_MAX, ORX_GAIN_MIN, autolevel_orx, clip_report
from adrvtrx.sweep import attenuation_axis, frequency_axis, run_sweep
from adrvtrx.transmit import transmit_bands
from adrvtrx.waveform import load_tab_iq

# ============================ user parameters ================================
PROFILE = "ADRV9025Init_StdUseCase98_LinkSharing.profile"

TX_CHANNEL = TxChannel.TX2
ORX_CHANNEL = RxChannel.ORX2
SIGNAL_PATH = "C:/Users/ohammi/OneDrive - aus.edu/DualBand/input_100/Signal1.txt"

# Test grid -------------------------------------------------------------------
TX_ATTENS_DB = [15.0]  # TX level(s); dB, 0..41.95
FREQS_HZ = [0.90e9, 1.00e9, 1.10e9]  # LO2 (TX+ORx) frequencies

# ORx AGC ---------------------------------------------------------------------
ORX_TARGET_DBFS = -15.0  # aim point for captured peak
ORX_TOLERANCE_DB = 2.0
CAPTURE_MS = 0.1
# =============================================================================


def main() -> None:
    cfg = load_config()
    cfg.profile_name = PROFILE
    with session(cfg) as (radio, info):
        wave = load_tab_iq(SIGNAL_PATH)
        print(f"Profile {cfg.profile_path.name}")
        print(
            f"  Tx {info.tx_bits}b @ {info.tx_rate_khz/1e3:.3f} MSPS | "
            f"ORx @ {info.orx_rate_khz/1e3:.3f} MSPS"
        )
        print(f"  Signal {SIGNAL_PATH} ({len(wave)} samples)")
        print(
            f"  {TX_CHANNEL.name}->{ORX_CHANNEL.name} | target "
            f"{ORX_TARGET_DBFS:+.1f}+/-{ORX_TOLERANCE_DB:.0f} dBFS | "
            f"gain window {ORX_GAIN_MIN}-{ORX_GAIN_MAX}"
        )

        transmit_bands(radio, {TX_CHANNEL: wave}, info.tx_bits, continuous=True, do_normalize=True)

        def peak_dbfs():
            cap = capture(radio, int(ORX_CHANNEL), CAPTURE_MS, bits=info.rx_bits).channels[
                ORX_CHANNEL
            ]
            return clip_report(cap.i, cap.q, info.rx_bits).peak_dbfs

        def action(point):
            lr = autolevel_orx(
                lambda g: radio.set_rx_gain(ORX_CHANNEL, g),
                peak_dbfs,
                target_dbfs=ORX_TARGET_DBFS,
                tolerance_db=ORX_TOLERANCE_DB,
            )
            cap = capture(radio, int(ORX_CHANNEL), CAPTURE_MS, bits=info.rx_bits).channels[
                ORX_CHANNEL
            ]
            rep = clip_report(cap.i, cap.q, info.rx_bits)
            return {
                **point,
                "gain": lr.final_gain_index,
                "peak": rep.peak_dbfs,
                "ok": lr.converged,
                "railed": rep.railed_samples,
                "note": lr.reason,
            }

        axes = [
            attenuation_axis(radio, TX_CHANNEL, TX_ATTENS_DB, name="atten_db"),
            frequency_axis(radio, "LO2", [int(f) for f in FREQS_HZ], name="lo2_hz"),
        ]
        records = run_sweep(axes, action)
        radio.disable_tx()
        _print_table(records)


def _print_table(records) -> None:
    print(
        f"\n  {'atten':>6} {'LO2_MHz':>9} {'orx_gain':>9} "
        f"{'peak_dBFS':>10} {'ok':>5} {'railed':>7}  note"
    )
    for r in records:
        note = "" if r["ok"] else r["note"]
        print(
            f"  {r['atten_db']:>6.1f} {r['lo2_hz']/1e6:>9.1f} {r['gain']:>9} "
            f"{r['peak']:>10.1f} {str(r['ok']):>5} {r['railed']:>7}  {note}"
        )
    ok = [r for r in records if r["ok"]]
    if ok:
        gains = [r["gain"] for r in ok]
        peaks = [r["peak"] for r in ok]
        med = int(statistics.median(gains))
        print(
            f"\n  converged {len(ok)}/{len(records)} | gain {min(gains)}-{max(gains)} "
            f"(median {med}) | peak {min(peaks):+.1f}..{max(peaks):+.1f} dBFS"
        )
        print(f"  -> suggested static ORx gain for this signal/grid: {med}")
    else:
        print("\n  no point converged -- adjust ORX_TARGET_DBFS or TX level (see notes)")


if __name__ == "__main__":
    main()
