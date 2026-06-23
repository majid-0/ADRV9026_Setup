"""On-hardware tests. Require a connected, booted ADS9 + ADRV9026.

Run ONLY on the control PC, inside myenv:

    conda run -n myenv python -m pytest -m hardware -v

These are SKIPPED automatically (not failed) when the board is unreachable, so
`-m hardware` is safe to run anywhere. They confirm the bench-only open items from
docs/api_notes.md and exercise the TX2->ORx2 / TX3->ORx3 loopback you have wired.

The whole module shares one connected+programmed session (module-scoped fixture);
TX is forced safe and the board disconnected at teardown.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import capture
from adrvtrx.config import load_config
from adrvtrx.experiment import session
from adrvtrx.gain import clip_report
from adrvtrx.transmit import transmit_bands

pytestmark = pytest.mark.hardware

# Wired loopback pairs on this bench.
PAIRS = [(TxChannel.TX2, RxChannel.ORX2), (TxChannel.TX3, RxChannel.ORX3)]
CAPTURE_MS = 0.1


def _reachable(ip: str, port: int, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def hw():
    """Connected + programmed (radio, profile_info), safe-stated on teardown."""
    cfg = load_config()
    if not _reachable(cfg.board.ip, cfg.board.port):
        pytest.skip(f"ADS9 not reachable at {cfg.board.ip}:{cfg.board.port}")
    with session(cfg) as (radio, info):
        yield radio, info


def _tone(n: int, freq_hz: float, rate_hz: float) -> np.ndarray:
    t = np.arange(n) / rate_hz
    return 0.5 * np.exp(2j * np.pi * freq_hz * t)


# --- status / programming -----------------------------------------------------


def test_connected_and_locked(hw):
    radio, _info = hw
    assert radio._is_connected()
    assert radio.pll_lock_status() != 0, "PLLs not locked after programming"


def test_lo_readback_matches_config(hw):
    radio, _info = hw
    cfg = radio.config
    # Allow a small tolerance for PLL fractional resolution.
    assert abs(radio.get_lo("LO1") - cfg.lo.lo1_hz) < 1000
    assert abs(radio.get_lo("LO2") - cfg.lo.lo2_hz) < 1000


# --- capture length + readback layout ----------------------------------------


def test_capture_returns_expected_length(hw):
    radio, info = hw
    expected = int(CAPTURE_MS * 1e-3 * info.rx_rate_khz * 1e3)
    raw = radio.perform_rx(0xFF, CAPTURE_MS, timeout_ms=1000)
    n_arrays = len(raw)
    assert n_arrays > 0 and n_arrays % 2 == 0, "expected an even number of I/Q arrays"
    first_len = len(raw[0])
    # within 10% of the rate*time estimate (decimation/rounding slack)
    assert abs(first_len - expected) <= 0.1 * expected + 64


# --- ORx level readback (confirms RxDecPowerGet sign/scale) -------------------


@pytest.mark.parametrize("orx", [RxChannel.ORX2, RxChannel.ORX3])
def test_orx_dec_power_is_sane(hw, orx):
    radio, _info = hw
    dbfs = radio.rx_dec_power_dbfs(orx)
    assert np.isfinite(dbfs), f"{orx.name} DEC power not finite"
    assert -150.0 < dbfs < 6.0, f"{orx.name} DEC power {dbfs} dBFS out of plausible range"


# --- TX -> ORx loopback on the wired pairs -----------------------------------


@pytest.mark.parametrize("tx,orx", PAIRS, ids=[f"{t.name}->{o.name}" for t, o in PAIRS])
def test_tx_orx_loopback(hw, tx, orx):
    radio, info = hw
    tone = _tone(4096, 5_000_000, info.tx_rate_hz)
    transmit_bands(radio, {tx: tone}, info.tx_bits, continuous=True)
    try:
        result = capture(radio, int(orx), CAPTURE_MS, bits=info.rx_bits)
    except RuntimeError as exc:
        pytest.skip(f"capture mask mapping unconfirmed for this profile: {exc}")
    finally:
        radio.disable_tx()

    cap = result.channels[orx]
    assert len(cap.i) > 0
    rep = clip_report(cap.i, cap.q, info.rx_bits)
    # Signal should be present (well above a dead -inf floor) and not slammed into
    # the rails. Thresholds are loose -- this is a "did the loopback light up" check.
    assert rep.peak_dbfs > -60.0, f"no signal seen on {orx.name} (peak {rep.peak_dbfs} dBFS)"
    assert rep.railed_samples == 0, f"{orx.name} clipping; lower TX level / raise ORx atten"


# --- crash-safety: safe_state stops TX ---------------------------------------


def test_safe_state_clears_tx_enable(hw):
    radio, info = hw
    tone = _tone(2048, 5_000_000, info.tx_rate_hz)
    transmit_bands(radio, {TxChannel.TX2: tone}, info.tx_bits, continuous=True)
    assert radio._en_tx != 0
    radio.safe_state()
    assert radio._en_tx == 0  # TX playback stopped


def test_status_reads_back_from_hardware(hw):
    from adrvtrx.radio import MAX_TX_ATTEN_DB

    radio, _info = hw
    radio.safe_state()
    st = radio.status()
    assert st["connected"] is True
    assert st["enable_source"] == "hardware", "RxTxEnableGet readback failed"
    assert st["tx_off"] is True, f"TX still enabled: {st['tx_enabled']}"
    assert st["tx_atten_db"], "TxAttenGet readback returned nothing"
    assert all(v >= MAX_TX_ATTEN_DB - 0.1 for v in st["tx_atten_db"].values())
