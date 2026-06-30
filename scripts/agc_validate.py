"""Live bench validation of the software ORx AGC (end-to-end, on hardware).

Reproduces the exact bench point the AGC was characterized at and runs the full
A/B/C auto-level via ``autolevel_capture``, then prints the AgcResult and an explicit
PASS/FAIL verdict so the result can be eyeballed.

Bench point (TX2->ORx2, UC98 link-sharing, Signal1.txt, Tx2 atten 3 dB, LO1 1800 MHz):
the gain table is clean & monotonic and the AGC should settle near gain ~250
(peak ~= -0.86 dBFS, in the -1.0 (+0.3/-0.6) band) with railed == 0; gain 255 rails.

    conda run -n myenv python scripts/agc_validate.py
"""

from __future__ import annotations

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import AgcResult, autolevel_capture
from adrvtrx.config import load_config
from adrvtrx.experiment import session
from adrvtrx.gain import AgcError
from adrvtrx.transmit import transmit_bands
from adrvtrx.waveform import load_tab_iq

# ============================ bench operating point ==========================
PROFILE = "ADRV9025Init_StdUseCase98_LinkSharing.profile"
SIGNAL_PATH = "C:/Users/ohammi/OneDrive - aus.edu/DualBand/input_100/Signal1.txt"

TX_CHANNEL = TxChannel.TX2
ORX_CHANNEL = RxChannel.ORX2
TX_ATTEN_DB = 3.0
LO1_HZ = 1_800_000_000

ORX_TARGET_DBFS = -1.0
ORX_TOL_UP_DB = 0.3
ORX_TOL_DOWN_DB = 0.6

# Expectation for the PASS/FAIL check (operator-calibrated ground truth).
EXPECTED_GAIN = 250
GAIN_TOLERANCE = 3
# =============================================================================


def _verdict(res: AgcResult) -> bool:
    band_lo = ORX_TARGET_DBFS - ORX_TOL_DOWN_DB
    band_hi = ORX_TARGET_DBFS + ORX_TOL_UP_DB
    gain_ok = abs(res.final_gain_index - EXPECTED_GAIN) <= GAIN_TOLERANCE
    clip_ok = res.railed == 0
    band_ok = band_lo <= res.final_peak_dbfs <= band_hi
    print(
        f"\n  EXPECT: settled gain {EXPECTED_GAIN} +/-{GAIN_TOLERANCE} | railed == 0 | "
        f"final peak in [{band_lo:+.1f}, {band_hi:+.1f}] dBFS"
    )
    print(
        f"  GOT   : gain {res.final_gain_index} ({'ok' if gain_ok else 'FAIL'}) | "
        f"railed {res.railed} ({'ok' if clip_ok else 'FAIL'}) | "
        f"peak {res.final_peak_dbfs:+.2f} dBFS ({'ok' if band_ok else 'FAIL'})"
    )
    return gain_ok and clip_ok and band_ok


def main() -> None:
    cfg = load_config()
    cfg.profile_name = PROFILE
    with session(cfg) as (radio, info):
        radio.set_lo("LO1", LO1_HZ)
        radio.set_tx_atten(TX_CHANNEL, TX_ATTEN_DB)
        wave = load_tab_iq(SIGNAL_PATH)

        print(f"Profile {cfg.profile_path.name}")
        print(
            f"  Tx {info.tx_bits}b @ {info.tx_rate_khz/1e3:.3f} MSPS | "
            f"ORx @ {info.orx_rate_khz/1e3:.3f} MSPS"
        )
        print(
            f"  {TX_CHANNEL.name}->{ORX_CHANNEL.name} | atten {TX_ATTEN_DB} dB | "
            f"LO1 {LO1_HZ/1e6:.1f} MHz | target {ORX_TARGET_DBFS:+.1f} "
            f"(+{ORX_TOL_UP_DB:.1f}/-{ORX_TOL_DOWN_DB:.1f}) dBFS"
        )

        transmit_bands(radio, {TX_CHANNEL: wave}, info.tx_bits, continuous=True, do_normalize=True)
        print(f"  LO1 readback {radio.get_lo('LO1')/1e6:.3f} MHz | TX {len(wave)} samples")

        # Verify on the FULL waveform duration (Stage C). ORx runs at the Tx rate
        # in UC98, so this many ms yields ~len(wave) ORx samples.
        verify_ms = len(wave) / info.tx_rate_hz * 1e3

        try:
            res = autolevel_capture(
                radio,
                ORX_CHANNEL,
                bits=info.rx_bits,
                orx_rate_hz=info.orx_rate_hz,
                target_dbfs=ORX_TARGET_DBFS,
                tol_up_db=ORX_TOL_UP_DB,
                tol_down_db=ORX_TOL_DOWN_DB,
                verify_capture_ms=verify_ms,
            )
        except AgcError as exc:
            # autolevel_capture already safe-stated + disconnected the radio.
            print(f"\n  AGC FATAL: {exc}")
            print("\n  RESULT: FAIL (AGC reported a fatal condition)")
            return

        radio.disable_tx()

        print("\n  AgcResult:")
        print(f"    final_gain_index : {res.final_gain_index}")
        print(f"    converged        : {res.converged}")
        print(f"    at_max_gain      : {res.at_max_gain}")
        print(f"    reason           : {res.reason}")
        print(f"    coarse_peak_dbfs : {res.coarse_peak_dbfs:+.2f}")
        print(f"    final_peak_dbfs  : {res.final_peak_dbfs:+.2f}")
        print(f"    railed           : {res.railed}")
        print(f"    fine_iters       : {res.fine_iters}")
        print(f"    verify_steps     : {res.verify_steps}")

        passed = _verdict(res)
        print(f"\n  RESULT: {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
