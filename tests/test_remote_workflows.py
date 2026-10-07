"""Existing modules, unchanged, on a RemoteRadio: same results as on the radio in-process.

The server's backend is the simulated bench (``sim_bench.SimRadio``) with the
same seed as the in-process one, so every capture is bit-identical when the
call sequence is. Each workflow runs once in-process and once through
``hardware()``; the outputs must match exactly.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from adrvtrx import RxChannel, TxChannel
from adrvtrx.capture import autolevel_capture, capture
from adrvtrx.client import hardware
from adrvtrx.compression import find_compression_point
from adrvtrx.conditions import load_conditions
from adrvtrx.config import load_config
from adrvtrx.linearize import linearize
from adrvtrx.operating_point import find_operating_point
from adrvtrx.replay import replay_conditions
from adrvtrx.sweep import attenuation_axis, frequency_axis, run_sweep
from adrvtrx.transmit import transmit_bands
from hw_helpers import running_server, write_config
from sim_bench import BITS, BW, FS, SimRadio, make_signal, reference_codes

TX, ORX = TxChannel.TX1, RxChannel.ORX1
SEED = 7


class SimBackend(SimRadio):
    """SimRadio plus the lifecycle the server calls (connect, force_safe, program)."""

    def __init__(self, _config=None):
        super().__init__(seed=SEED)

    def connect(self) -> None:
        pass

    def force_safe(self) -> None:
        self.safe_state()

    def program(self) -> None:
        pass


def local_radio() -> SimRadio:
    radio = SimRadio(seed=SEED)
    radio.safe_state()  # what the server's start-up does
    return radio


@pytest.fixture
def remote(tmp_path):
    """Run ``fn(radio)`` on a RemoteRadio served from a fresh SimBackend."""
    cfg = load_config(write_config(tmp_path))

    def run(fn):
        with running_server(cfg, SimBackend) as srv:
            with hardware("workflow", config=cfg) as radio:
                assert radio.config.channels == srv.radio.config.channels
                return fn(radio)

    return run


def same(a, b) -> None:
    """Equal, NaN included (dataclasses and dicts compared through JSON)."""
    if hasattr(a, "__dataclass_fields__"):
        a, b = asdict(a), asdict(b)
    assert json.dumps(a, default=_plain, sort_keys=True) == json.dumps(
        b, default=_plain, sort_keys=True
    )


def _plain(value):
    if isinstance(value, np.ndarray):
        return (
            value.tolist()
            if not np.iscomplexobj(value)
            else [value.real.tolist(), value.imag.tolist()]
        )
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def test_transmit_and_capture(remote):
    x = make_signal()

    def work(radio):
        radio.set_tx_atten(TX, 14.0)
        radio.set_rx_gain(ORX, 214)
        mask = transmit_bands(radio, {TX: x}, BITS)
        cap = capture(radio, int(ORX), 0.8192, bits=BITS).channels[ORX]
        return mask, cap.i, cap.q

    mask_l, i_l, q_l = work(local_radio())
    mask_r, i_r, q_r = remote(work)
    assert mask_l == mask_r == int(TX)
    np.testing.assert_array_equal(i_l, i_r)
    np.testing.assert_array_equal(q_l, q_r)
    assert np.abs(i_r).max() > 100


def test_orx_agc(remote):
    x = make_signal()

    def work(radio):
        radio.set_tx_atten(TX, 14.0)
        transmit_bands(radio, {TX: x}, BITS)
        return autolevel_capture(radio, ORX, bits=BITS, orx_rate_hz=FS, verify_capture_ms=0.82)

    local, remote_res = work(local_radio()), remote(work)
    same(local, remote_res)
    assert remote_res.converged


def test_compression_search(remote):
    x = make_signal()

    def work(radio):
        transmit_bands(radio, {TX: x}, BITS)
        return find_compression_point(
            radio,
            TX,
            ORX,
            reference_codes(x),
            rx_bits=BITS,
            fs=FS,
            target_compression_db=3.0,
            start_atten_db=20.0,
            atten_min_db=5.0,
            coarse_step_db=5.0,
            fine_step_db=0.2,
            comp_tol_db=0.1,
        )

    local, remote_res = work(local_radio()), remote(work)
    same(local, remote_res)
    assert remote_res.converged


def test_operating_point(remote, tmp_path):
    x = make_signal()

    def work(save_dir):
        def run(radio):
            op = find_operating_point(
                radio,
                TX,
                ORX,
                2.5 * x,
                tx_bits=BITS,
                rx_bits=BITS,
                fs=FS,
                freq_hz=2_000_000_000,
                bw_mhz=int(BW / 1e6),
                target_compression_db=3.0,
                start_atten_db=20.0,
                atten_min_db=5.0,
                coarse_step_db=5.0,
                fine_step_db=0.2,
                comp_tol_db=0.1,
                backoff_db=2,
                save_dir=save_dir,
            )
            return op.condition

        return run

    local = work(tmp_path / "local")(local_radio())
    remote_res = remote(work(tmp_path / "remote"))
    same(local, remote_res)
    for name in (local.in_file, local.out_file, "TX1_conditions.csv"):
        assert (tmp_path / "local" / name).read_text() == (tmp_path / "remote" / name).read_text()


def _captures_csv(tmp_path: Path) -> Path:
    """A two-condition capture CSV recorded in-process (the replay input)."""
    radio = local_radio()
    x = make_signal()
    for backoff in (0, 2):
        find_operating_point(
            radio,
            TX,
            ORX,
            2.5 * x,
            tx_bits=BITS,
            rx_bits=BITS,
            fs=FS,
            freq_hz=2_000_000_000,
            bw_mhz=int(BW / 1e6),
            target_compression_db=3.0,
            start_atten_db=20.0,
            atten_min_db=5.0,
            coarse_step_db=5.0,
            fine_step_db=0.2,
            comp_tol_db=0.1,
            backoff_db=backoff,
            save_dir=tmp_path / "caps",
        )
    return tmp_path / "caps" / "TX1_conditions.csv"


def test_replay(remote, tmp_path):
    csv_path = _captures_csv(tmp_path)
    ref = csv_path.parent / load_conditions(csv_path)[0].in_file

    def work(out_dir):
        def run(radio):
            return replay_conditions(
                radio,
                csv_path,
                lambda row: ref,
                tx=TX,
                orx=ORX,
                tx_bits=BITS,
                rx_bits=BITS,
                fs=FS,
                out_dir=out_dir,
                label="again",
            )

        return run

    local = work(tmp_path / "local")(local_radio())
    remote_res = remote(work(tmp_path / "remote"))
    assert local.read_text() == remote_res.read_text()
    names = sorted(p.name for p in (tmp_path / "local").iterdir())
    assert names == sorted(p.name for p in (tmp_path / "remote").iterdir())
    for name in names:
        assert (tmp_path / "local" / name).read_text() == (tmp_path / "remote" / name).read_text()


def test_linearize(remote, tmp_path):
    csv_path = _captures_csv(tmp_path)
    condition = load_conditions(csv_path)[0]
    x = reference_codes(make_signal()) / 2047.0

    def step(x, u, z, it):
        return 0.9 * u  # any deterministic update

    def work(radio):
        res = linearize(
            radio, condition, x, step, tx=TX, orx=ORX, tx_bits=BITS, rx_bits=BITS, fs=FS, n_iter=3
        )
        return [asdict(r) for r in res.records], res.reason

    local, remote_res = work(local_radio()), remote(work)
    same(local, remote_res)
    assert remote_res[1] == "n_iter" and len(remote_res[0]) == 3


def test_sweep(remote):
    x = make_signal()

    def work(radio):
        transmit_bands(radio, {TX: x}, BITS)
        axes = [
            frequency_axis(radio, "LO1", [1_900_000_000, 2_100_000_000]),
            attenuation_axis(radio, TX, [14.0, 18.0]),
        ]
        return run_sweep(
            axes,
            lambda point: capture(radio, int(ORX), 0.1, bits=BITS).channels[ORX].clip().peak_dbfs,
        )

    local, remote_res = work(local_radio()), remote(work)
    assert local == remote_res and len(remote_res) == 4
