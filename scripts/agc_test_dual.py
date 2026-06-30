"""AGC test -- DUAL band.

Transmit two bands at fixed TX levels, sweep a frequency grid, and run the ORx
auto-level ("AGC") for EACH band at every grid point. Prints a clean per-band
table and a suggested static gain per band.

    conda run -n myenv python scripts/agc_test_dual.py

LO2 drives both TX and ORx, so the frequency grid moves both bands' loopback
together. Each band's ORx is leveled independently (different ADCs).
"""

from __future__ import annotations

import statistics

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import capture
from adrvtrx.config import load_config
from adrvtrx.experiment import session
from adrvtrx.gain import ORX_GAIN_MAX, ORX_GAIN_MIN, autolevel_orx, clip_report
from adrvtrx.sweep import frequency_axis, run_sweep
from adrvtrx.transmit import transmit_bands
from adrvtrx.waveform import load_tab_iq

# ============================ user parameters ================================
PROFILE = "ADRV9025Init_StdUseCase98_LinkSharing.profile"

_DIR = "C:/Users/ohammi/OneDrive - aus.edu/DualBand/input_100"
BANDS = [
    {
        "name": "band1",
        "tx": TxChannel.TX2,
        "orx": RxChannel.ORX2,
        "signal": f"{_DIR}/Signal1.txt",
        "atten_db": 15.0,
    },
    {
        "name": "band2",
        "tx": TxChannel.TX3,
        "orx": RxChannel.ORX3,
        "signal": f"{_DIR}/Signal2.txt",
        "atten_db": 15.0,
    },
]

# Test grid -------------------------------------------------------------------
FREQS_HZ = [0.90e9, 1.00e9, 1.10e9]  # LO2 (TX+ORx) frequencies

# ORx AGC ---------------------------------------------------------------------
ORX_TARGET_DBFS = -1.0
ORX_TOL_UP_DB = 0.3  # accept up to target + this (toward the rail)
ORX_TOL_DOWN_DB = 0.6  # accept down to target - this (toward the floor)
CAPTURE_MS = 0.1
# =============================================================================


def main() -> None:
    cfg = load_config()
    cfg.profile_name = PROFILE
    with session(cfg) as (radio, info):
        for b in BANDS:
            b["wave"] = load_tab_iq(b["signal"])
        assert len({len(b["wave"]) for b in BANDS}) == 1, "waveforms must be equal length"

        print(f"Profile {cfg.profile_path.name}")
        print(
            f"  Tx {info.tx_bits}b @ {info.tx_rate_khz/1e3:.3f} MSPS | "
            f"ORx @ {info.orx_rate_khz/1e3:.3f} MSPS"
        )
        for b in BANDS:
            print(
                f"  {b['name']}: {b['tx'].name}->{b['orx'].name} @ "
                f"{b['atten_db']:.1f} dB atten, {len(b['wave'])} samples"
            )
        print(
            f"  target {ORX_TARGET_DBFS:+.1f} (+{ORX_TOL_UP_DB:.1f}/-{ORX_TOL_DOWN_DB:.1f}) dBFS | "
            f"gain window {ORX_GAIN_MIN}-{ORX_GAIN_MAX}"
        )

        for b in BANDS:
            radio.set_tx_atten(b["tx"], b["atten_db"])
        transmit_bands(
            radio,
            {b["tx"]: b["wave"] for b in BANDS},
            info.tx_bits,
            continuous=True,
            do_normalize=True,
        )

        cap_mask = 0
        for b in BANDS:
            cap_mask |= int(b["orx"])

        def measure(ch):
            cap = capture(radio, int(ch), CAPTURE_MS, bits=info.rx_bits).channels[ch]
            rep = clip_report(cap.i, cap.q, info.rx_bits)
            return rep.peak_dbfs, rep.railed_samples

        def action(point):
            row = dict(point)
            for b in BANDS:
                lr = autolevel_orx(
                    lambda g, ch=b["orx"]: radio.set_rx_gain(ch, g),
                    lambda ch=b["orx"]: measure(ch),
                    target_dbfs=ORX_TARGET_DBFS,
                    tol_up_db=ORX_TOL_UP_DB,
                    tol_down_db=ORX_TOL_DOWN_DB,
                )
                row[f"{b['name']}_gain"] = lr.final_gain_index
                row[f"{b['name']}_ok"] = lr.converged
                row[f"{b['name']}_note"] = lr.reason
            res = capture(radio, cap_mask, CAPTURE_MS, bits=info.rx_bits)  # aligned snapshot
            for b in BANDS:
                cap = res.channels[b["orx"]]
                rep = clip_report(cap.i, cap.q, info.rx_bits)
                row[f"{b['name']}_peak"] = rep.peak_dbfs
                row[f"{b['name']}_railed"] = rep.railed_samples
            return row

        axis = frequency_axis(radio, "LO2", [int(f) for f in FREQS_HZ], name="lo2_hz")
        records = run_sweep([axis], action)
        radio.disable_tx()
        _print_table(records)


def _print_table(records) -> None:
    hdr = f"\n  {'LO2_MHz':>9}"
    for b in BANDS:
        hdr += f" | {b['name']:>5} {'gain':>5}{'peak':>8}{'ok':>6}{'rail':>5}"
    print(hdr)
    for r in records:
        line = f"  {r['lo2_hz']/1e6:>9.1f}"
        for b in BANDS:
            n = b["name"]
            line += (
                f" | {'':>5} {r[f'{n}_gain']:>5}{r[f'{n}_peak']:>8.1f}"
                f"{str(r[f'{n}_ok']):>6}{r[f'{n}_railed']:>5}"
            )
        print(line)
    for b in BANDS:
        n = b["name"]
        ok = [r for r in records if r[f"{n}_ok"]]
        if ok:
            gains = [r[f"{n}_gain"] for r in ok]
            peaks = [r[f"{n}_peak"] for r in ok]
            med = int(statistics.median(gains))
            print(
                f"  {n}: converged {len(ok)}/{len(records)} | gain {min(gains)}-{max(gains)} "
                f"(median {med}) | peak {min(peaks):+.1f}..{max(peaks):+.1f} dBFS "
                f"-> static gain {med}"
            )
        else:
            note = next((r[f"{n}_note"] for r in records), "")
            print(f"  {n}: no point converged -- {note}")


if __name__ == "__main__":
    main()
