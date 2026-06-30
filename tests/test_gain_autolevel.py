"""Unit tests for the software ORx AGC (hardware-free, callback-driven).

The AGC levels on the captured-IQ peak with a ``railed``-sample clip veto and an
asymmetric accept band; see the frozen spec in ``src/adrvtrx/gain.py``. ``FakeOrx``
maps a gain index -> ``(peak_dbfs, railed)`` like the bench gain table: clean and
monotonic at ~0.5 dB/index, with a hard ADC rail near the top.
"""

from __future__ import annotations

from adrvtrx.gain import (
    ORX_GAIN_MAX,
    ORX_GAIN_MIN,
    autolevel_orx,
    verify_no_clip,
)

TARGET = -1.0
TOL_UP = 0.3
TOL_DOWN = 0.6
BAND_LO = TARGET - TOL_DOWN  # -1.6
BAND_HI = TARGET + TOL_UP  # -0.7


class FakeOrx:
    """Gain index -> (peak_dbfs, railed), modelling the measured ORx table.

    ``peak = ref_dbfs + (gain - ref_gain) * db_per_index`` (monotonic ~0.5 dB/index)
    while ``gain < rail_gain``; at/above ``rail_gain`` the ADC rails: ``railed`` jumps
    to a large count and the peak saturates near 0 dBFS (peak compresses at the rail,
    so ``railed`` is the true clip detector). Records every applied gain in ``history``.
    """

    def __init__(
        self,
        *,
        ref_gain: int = 250,
        ref_dbfs: float = -0.86,
        db_per_index: float = 0.5,
        rail_gain: int = 251,
        gain_min: int = ORX_GAIN_MIN,
        gain_max: int = ORX_GAIN_MAX,
    ):
        self.ref_gain = ref_gain
        self.ref_dbfs = ref_dbfs
        self.db_per_index = db_per_index
        self.rail_gain = rail_gain
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.gain = gain_min
        self.history: list[int] = []

    def set_gain(self, g: int) -> None:
        self.gain = int(min(max(g, self.gain_min), self.gain_max))
        self.history.append(self.gain)

    def measure(self) -> tuple[float, int]:
        g = self.gain
        if g >= self.rail_gain:
            return 0.0, 343  # saturated peak + many railed samples
        return self.ref_dbfs + (g - self.ref_gain) * self.db_per_index, 0


def _level(orx: FakeOrx):
    return autolevel_orx(
        orx.set_gain,
        orx.measure,
        target_dbfs=TARGET,
        tol_up_db=TOL_UP,
        tol_down_db=TOL_DOWN,
    )


# --- Stage A: FATAL floor conditions -----------------------------------------


def test_fatal_when_clipping_at_minimum_gain():
    # The rail starts at the floor: even gain 185 rails -> TX is too strong.
    orx = FakeOrx(rail_gain=ORX_GAIN_MIN)
    res = _level(orx)
    assert res.fatal
    assert not res.converged
    assert res.final_gain_index == ORX_GAIN_MIN
    assert "minimum gain" in res.reason and "reduce TX power" in res.reason


def test_fatal_when_already_in_band_at_minimum_gain():
    # At the floor the signal is already inside the band -> can only get hotter.
    orx = FakeOrx(ref_gain=ORX_GAIN_MIN, ref_dbfs=-1.0, rail_gain=ORX_GAIN_MAX + 1)
    res = _level(orx)
    assert res.fatal
    assert not res.converged
    assert res.final_gain_index == ORX_GAIN_MIN
    assert "already at/above band" in res.reason


# --- Stages A+B: coarse jump + fine trim -------------------------------------


def test_coarse_jump_then_converges_in_band():
    # gain 250 -> -0.86 dBFS (in band), 251+ rails. One jump should land at 250.
    orx = FakeOrx(ref_gain=250, ref_dbfs=-0.86, rail_gain=251)
    res = _level(orx)
    assert res.converged
    assert not res.fatal
    assert res.railed == 0
    assert abs(res.final_gain_index - 250) <= 2
    assert BAND_LO <= res.final_dbfs <= BAND_HI


def test_railed_overshoot_forces_step_down():
    # The coarse jump lands on a railing gain (250 rails); the clip veto must back
    # off one index to the clean in-band gain 249 (peak -1.36 dBFS).
    orx = FakeOrx(ref_gain=250, ref_dbfs=-0.86, rail_gain=250)
    res = _level(orx)
    assert res.converged
    assert res.railed == 0
    assert res.final_gain_index == 249
    assert 250 in orx.history  # visited the railing gain, then stepped down
    assert BAND_LO <= res.final_dbfs <= BAND_HI


def test_accept_at_max_gain_when_signal_below_target():
    # Even at gain 255 the signal is only -5 dBFS (below band) and never rails ->
    # accept 255 as the best achievable, NOT an error.
    orx = FakeOrx(ref_gain=ORX_GAIN_MAX, ref_dbfs=-5.0, rail_gain=ORX_GAIN_MAX + 1)
    res = _level(orx)
    assert not res.fatal
    assert res.at_max_gain
    assert res.final_gain_index == ORX_GAIN_MAX
    assert not res.converged
    assert "max gain reached" in res.reason


def test_never_commands_outside_the_gain_window():
    orx = FakeOrx(ref_gain=250, ref_dbfs=-0.86, rail_gain=251)
    _level(orx)
    assert all(ORX_GAIN_MIN <= g <= ORX_GAIN_MAX for g in orx.history)


def test_asymmetric_band_rejects_hot_accepts_cold_edge():
    # Above +tol_up is too hot (step down); within [-1.6, -0.7] is accepted.
    # ref so gain 250 -> -0.4 dBFS (hot), 249 -> -0.9 dBFS (in band, cold side).
    orx = FakeOrx(ref_gain=250, ref_dbfs=-0.4, rail_gain=256)
    res = _level(orx)
    assert res.converged
    assert res.final_dbfs <= BAND_HI  # not left above the rail-side tolerance
    assert BAND_LO <= res.final_dbfs


# --- Stage C: verify on the full signal --------------------------------------


class FakeFull:
    """Full-signal model for verify_no_clip: rails at/above ``rail_gain``."""

    def __init__(self, *, ref_gain=250, ref_dbfs=-0.86, db_per_index=0.5, rail_gain=251):
        self.ref_gain = ref_gain
        self.ref_dbfs = ref_dbfs
        self.db_per_index = db_per_index
        self.rail_gain = rail_gain
        self.gain = ref_gain

    def set_gain(self, g):
        self.gain = int(g)

    def measure(self):
        g = self.gain
        if g >= self.rail_gain:
            return 0.0, 343
        return self.ref_dbfs + (g - self.ref_gain) * self.db_per_index, 0


def test_verify_backs_off_on_full_signal_clip():
    # Settled at 250 on short captures, but the full signal rails there; back off to 249.
    full = FakeFull(ref_gain=250, ref_dbfs=-0.86, rail_gain=250)
    vr = verify_no_clip(full.set_gain, full.measure, 250, target_dbfs=TARGET, tol_up_db=TOL_UP)
    assert not vr.fatal
    assert vr.converged
    assert vr.railed == 0
    assert vr.final_gain_index == 249
    assert vr.iterations == 1


def test_verify_backs_off_when_hot_above_tolerance():
    # No rail, but peak above target+tol_up -> still step down into band.
    full = FakeFull(ref_gain=250, ref_dbfs=-0.3, rail_gain=256)
    vr = verify_no_clip(full.set_gain, full.measure, 250, target_dbfs=TARGET, tol_up_db=TOL_UP)
    assert vr.converged
    assert vr.railed == 0
    assert vr.final_gain_index < 250
    assert vr.final_dbfs <= BAND_HI


def test_verify_no_action_when_clean_and_in_band():
    full = FakeFull(ref_gain=250, ref_dbfs=-0.86, rail_gain=251)
    vr = verify_no_clip(full.set_gain, full.measure, 250, target_dbfs=TARGET, tol_up_db=TOL_UP)
    assert vr.converged
    assert vr.iterations == 0
    assert vr.final_gain_index == 250


def test_verify_fatal_when_rails_down_to_minimum():
    # The full signal rails at every gain -> bottoms out at gain_min and is FATAL.
    full = FakeFull(rail_gain=0)  # rails everywhere
    vr = verify_no_clip(full.set_gain, full.measure, 190, target_dbfs=TARGET, tol_up_db=TOL_UP)
    assert vr.fatal
    assert vr.final_gain_index == ORX_GAIN_MIN
    assert "minimum gain" in vr.reason
