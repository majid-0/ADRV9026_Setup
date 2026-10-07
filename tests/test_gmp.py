from __future__ import annotations

import numpy as np
import pytest

from adrvtrx.gmp import GMP, peak_block


def _signal(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n) + 1j * rng.normal(size=n)
    return u / np.abs(u).max()


@pytest.mark.parametrize("K,N,M", [(1, 1, 0), (5, 3, 1), (5, 5, 2), (3, 2, 3), (7, 4, 0)])
def test_coefficient_count_matches_dpd_kit_unified(K, N, M):
    model = GMP(K, N, M)
    assert model.n_coeffs == N * (K * (1 + 2 * M) + 1)
    assert model.basis(_signal(50)).shape == (50, model.n_coeffs)
    assert model.memory == N + M
    assert model.lookahead == M


@pytest.mark.parametrize("K,N,M", [(5, 3, 1), (5, 5, 2), (3, 2, 3), (2, 1, 2)])
def test_causal_drops_the_leading_terms_that_read_the_future(K, N, M):
    model = GMP(K, N, M, causal=True)
    lead_kept = K * sum(max(N - m, 0) for m in range(1, M + 1))
    assert model.n_coeffs == N * (K + 1) + K * N * M + lead_kept
    assert model.basis(_signal(50)).shape == (50, model.n_coeffs)
    assert model.lookahead == 0


def test_causal_model_reads_no_future_sample():
    u = _signal()
    n0 = 1500
    v = u.copy()
    v[n0 + 1 :] = _signal(len(u) - n0 - 1, seed=7)
    rng = np.random.default_rng(8)
    for causal, same in ((True, True), (False, False)):
        model = GMP(5, 3, 2, causal=causal)
        model.coef = rng.normal(size=model.n_coeffs) + 1j * rng.normal(size=model.n_coeffs)
        a, b = model.predict(u), model.predict(v)
        assert np.allclose(a[: n0 + 1], b[: n0 + 1]) is same


@pytest.mark.parametrize("causal", [False, True])
def test_fit_recovers_the_coefficients_of_a_gmp(causal):
    rng = np.random.default_rng(3)
    u = _signal(4000, seed=1)
    truth = GMP(5, 3, 2, causal=causal)
    w = (rng.normal(size=truth.n_coeffs) + 1j * rng.normal(size=truth.n_coeffs)) * 0.1
    target = truth.basis(u) @ w
    model = GMP(5, 3, 2, causal=causal).fit(u, target, block=(1000, 3000))
    np.testing.assert_allclose(model.coef, w, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(model.predict(u), target, atol=1e-9)


def test_column_scaling_gives_the_plain_least_squares_answer():
    u = _signal(2000, seed=2)
    target = u * (1 - 0.3 * np.abs(u) ** 2) + 0.05 * np.roll(u, 1)
    model = GMP(4, 2, 1).fit(u, target, block=(200, 1800))
    A = model.basis(u, 200, 1800)
    w, *_ = np.linalg.lstsq(A, target[200:1800], rcond=None)
    np.testing.assert_allclose(model.coef, w, rtol=1e-6, atol=1e-9)


def test_ridge_shrinks_the_scaled_coefficients():
    u = _signal(2000, seed=4)
    target = u * (1 - 0.3 * np.abs(u) ** 2) + 0.01 * _signal(2000, seed=9)
    plain = GMP(5, 3, 1).fit(u, target)
    ridged = GMP(5, 3, 1).fit(u, target, ridge=1e-2)
    norms = np.linalg.norm(plain.basis(u), axis=0)
    assert np.linalg.norm(ridged.coef * norms) < np.linalg.norm(plain.coef * norms)
    err = [np.linalg.norm(m.predict(u) - target) for m in (plain, ridged)]
    assert err[0] < err[1]


def test_block_rows_equal_full_signal_rows():
    u = _signal(500)
    model = GMP(3, 4, 2)
    np.testing.assert_array_equal(model.basis(u, 100, 220), model.basis(u)[100:220])
    np.testing.assert_array_equal(model.basis(u, 0, 7), model.basis(u)[:7])
    np.testing.assert_array_equal(model.basis(u, 493, 500), model.basis(u)[493:])


def test_chunked_predict_equals_one_chunk():
    u = _signal(5000, seed=5)
    model = GMP(5, 5, 2).fit(u, u * (1 - 0.2 * np.abs(u) ** 2))
    full = model.basis(u) @ model.coef
    np.testing.assert_allclose(model.predict(u, chunk=777), full, atol=1e-12)
    np.testing.assert_allclose(model.predict(u), full, atol=1e-12)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        GMP(3, 2, 1).predict(_signal(10))


def test_bad_arguments():
    with pytest.raises(ValueError):
        GMP(3, 0, 1)
    with pytest.raises(ValueError):
        GMP(3, 2, 1).fit(_signal(10), _signal(11))
    with pytest.raises(ValueError):
        GMP(3, 2, 1).basis(_signal(10), 5, 20)


def test_peak_block_is_centred_on_the_peak():
    x = np.zeros(1000, complex)
    x[500] = 1.0
    assert peak_block(x, 100) == (450, 550)
    x[:] = 0
    x[10] = 1.0
    assert peak_block(x, 100) == (0, 100)
    x[:] = 0
    x[995] = 1.0
    assert peak_block(x, 100) == (900, 1000)
    assert peak_block(x, 5000) == (0, 1000)
    with pytest.raises(ValueError):
        peak_block(x, 0)
