"""ORx gain-table sweep diagnostic.

Walk the ORx gain index across the usable window at a fixed bench point and print,
per gain, the captured peak (dBFS), the per-index slope, the railed-sample count,
and the (unreliable) RxDecPowerGet readback. This is the measurement that
characterizes the gain table the software AGC relies on: it should be clean and
MONOTONIC at ~0.5 dB/index with ``railed == 0`` up to ~250, then rail hard near 255
(the ``railed`` count -- not peak dBFS, which compresses near full scale -- is the
true clip detector). Use it to re-confirm the table after any bench change.

    conda run -n myenv python scripts/orx_gain_sweep.py
"""

from __future__ import annotations

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import capture
from adrvtrx.config import load_config
from adrvtrx.experiment import session
from adrvtrx.gain import clip_report
from adrvtrx.transmit import transmit_bands
from adrvtrx.waveform import load_tab_iq

PROFILE = "ADRV9025Init_StdUseCase98_LinkSharing.profile"
SIGNAL_PATH = "C:/Users/ohammi/OneDrive - aus.edu/DualBand/input_100/Signal1.txt"
TX_CHANNEL = TxChannel.TX2
ORX_CHANNEL = RxChannel.ORX2
TX_ATTEN_DB = 3.0
LO1_HZ = 1_800_000_000
GAINS = list(range(185, 256, 5))  # 185, 190, ..., 250, 255
CAPTURE_MS = 0.1


def main() -> None:
    cfg = load_config()
    cfg.profile_name = PROFILE
    with session(cfg) as (radio, info):
        radio.set_lo("LO1", LO1_HZ)
        radio.set_tx_atten(TX_CHANNEL, TX_ATTEN_DB)
        wave = load_tab_iq(SIGNAL_PATH)
        transmit_bands(radio, {TX_CHANNEL: wave}, info.tx_bits, continuous=True, do_normalize=True)

        print(
            f"\n  {TX_CHANNEL.name}->{ORX_CHANNEL.name} | atten {TX_ATTEN_DB} dB | "
            f"LO1 {radio.get_lo('LO1')/1e6:.1f} MHz | Rx {info.rx_bits}b"
        )
        print(f"  {'gain':>5} {'peak_dBFS':>10} {'d/idx':>6} {'railed':>7} {'dec_dBFS':>9}")
        prev_peak = None
        for g in GAINS:
            radio.set_rx_gain(ORX_CHANNEL, g)
            cap = capture(radio, int(ORX_CHANNEL), CAPTURE_MS, bits=info.rx_bits).channels[
                ORX_CHANNEL
            ]
            rep = clip_report(cap.i, cap.q, info.rx_bits)
            dec = radio.rx_dec_power_dbfs(ORX_CHANNEL)
            slope = "" if prev_peak is None else f"{(rep.peak_dbfs - prev_peak) / 5:.2f}"
            prev_peak = rep.peak_dbfs
            print(f"  {g:>5} {rep.peak_dbfs:>10.2f} {slope:>6} {rep.railed_samples:>7} {dec:>9.2f}")
        radio.disable_tx()


if __name__ == "__main__":
    main()
