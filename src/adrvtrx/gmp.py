"""Generalized memory polynomial (GMP), fitted by block least squares. Pure numpy.

This is the reference DPD model for the sample loop (``adrvtrx.dpd``). It has the
same basis and coefficient count as ``dpd_kit``'s ``GMP.unified(K, N, M)``,
``N * (K * (1 + 2M) + 1)`` complex coefficients:

* aligned  ``u(n-l) |u(n-l)|^k``     k = 0..K, l = 0..N-1
* lagging  ``u(n-l) |u(n-l-m)|^k``   k = 1..K, l = 0..N-1, m = 1..M
* leading  ``u(n-l) |u(n-l+m)|^k``   k = 1..K, l = 0..N-1, m = 1..M

The leading terms with ``l < m`` read a future sample. ``causal=True`` drops
them. Samples before the start or after the end of the signal read as zero.

The fit is plain least squares on one block of samples (usually the block
around the signal peak, :func:`peak_block`). The basis columns are scaled to
unit norm before the solve, which only improves the conditioning; an optional
ridge term is added on the scaled columns.
"""

from __future__ import annotations

import numpy as np

__all__ = ["GMP", "peak_block"]


def peak_block(x, n: int) -> tuple[int, int]:
    """``(start, stop)`` of ``n`` samples centred on ``argmax|x|``, clamped inside ``x``.

    The whole signal when ``n`` is at least its length.
    """
    x = np.asarray(x)
    length = len(x)
    if n <= 0:
        raise ValueError(f"block length must be positive, got {n}")
    if n >= length:
        return 0, length
    k = int(np.argmax(np.abs(x)))
    start = max(0, min(k - n // 2, length - n))
    return start, start + n


def _shift(u: np.ndarray, s: int) -> np.ndarray:
    """``u[n - s]`` with zeros outside the signal (``s < 0`` looks ahead)."""
    out = np.zeros_like(u)
    if s > 0:
        out[s:] = u[:-s]
    elif s < 0:
        out[:s] = u[-s:]
    else:
        out[:] = u
    return out


class GMP:
    """Generalized memory polynomial: nonlinear order ``K``, memory ``N``, cross memory ``M``.

    ``fit`` solves the coefficients on one block of samples; ``predict`` runs the
    model on a whole signal in chunks. ``coef`` is ``None`` until ``fit`` runs.
    """

    def __init__(self, K: int, N: int, M: int, causal: bool = False):
        if K < 0 or N < 1 or M < 0:
            raise ValueError(f"need K >= 0, N >= 1, M >= 0; got K={K}, N={N}, M={M}")
        self.K = int(K)
        self.N = int(N)
        self.M = int(M)
        self.causal = bool(causal)
        self.coef: np.ndarray | None = None

    def __repr__(self) -> str:
        causal = ", causal=True" if self.causal else ""
        return f"GMP(K={self.K}, N={self.N}, M={self.M}{causal})"

    @property
    def n_coeffs(self) -> int:
        """Number of complex coefficients."""
        K, N, M = self.K, self.N, self.M
        if self.causal:
            lead = K * sum(max(N - m, 0) for m in range(1, M + 1))
        else:
            lead = K * N * M
        return N * (K + 1) + K * N * M + lead

    @property
    def memory(self) -> int:
        """Samples a prediction reads up to and including the current one: ``N + M``."""
        return self.N + self.M

    @property
    def lookahead(self) -> int:
        """Future samples a prediction reads: ``M``, or 0 when causal."""
        return 0 if self.causal else self.M

    def basis(self, u, start: int = 0, stop: int | None = None) -> np.ndarray:
        """Regressor matrix for samples ``start:stop`` of ``u``, one column per coefficient.

        The shifts read across the slice edges, so a block gives the same rows as
        the full signal would.
        """
        u = np.asarray(u, dtype=np.complex128)
        stop = len(u) if stop is None else int(stop)
        start = int(start)
        if not 0 <= start <= stop <= len(u):
            raise ValueError(f"block ({start}, {stop}) is outside a signal of {len(u)} samples")
        K, N, M = self.K, self.N, self.M
        pad = N + M + 1
        lo, hi = max(start - pad, 0), min(stop + pad, len(u))
        seg = u[lo:hi]
        a, b = start - lo, stop - lo
        mag = np.abs(seg)
        cols = []
        for d in range(N):
            ul = _shift(seg, d)[a:b]
            al = _shift(mag, d)[a:b]
            pw = ul.copy()
            cols.append(ul)
            for _k in range(1, K + 1):
                pw = pw * al
                cols.append(pw)
            for m in range(1, M + 1):
                lag = _shift(mag, d + m)[a:b]
                lead = _shift(mag, d - m)[a:b]
                use_lead = not self.causal or d >= m
                lag_k = np.ones_like(lag)
                lead_k = np.ones_like(lead)
                for _k in range(1, K + 1):
                    lag_k = lag_k * lag
                    cols.append(ul * lag_k)
                    if use_lead:
                        lead_k = lead_k * lead
                        cols.append(ul * lead_k)
        return np.stack(cols, axis=1)

    def fit(self, u, target, block: tuple[int, int] | None = None, ridge: float = 0.0) -> GMP:
        """Least-squares fit of ``target`` from ``u`` on the samples ``block = (start, stop)``.

        ``block`` defaults to the whole signal. Each basis column is divided by its
        norm before the solve and the coefficients are scaled back after it.
        ``ridge`` adds ``ridge * ||w||^2`` on the scaled coefficients (0 = plain
        least squares). Returns ``self``.
        """
        u = np.asarray(u, dtype=np.complex128)
        target = np.asarray(target, dtype=np.complex128)
        if len(u) != len(target):
            raise ValueError(f"u has {len(u)} samples, target {len(target)}")
        start, stop = (0, len(u)) if block is None else (int(block[0]), int(block[1]))
        A = self.basis(u, start, stop)
        t = target[start:stop]
        norms = np.linalg.norm(A, axis=0)
        norms[norms == 0] = 1.0
        A = A / norms
        if ridge > 0:
            n = A.shape[1]
            A = np.vstack((A, np.sqrt(ridge) * np.eye(n)))
            t = np.concatenate((t, np.zeros(n, dtype=np.complex128)))
        w, *_ = np.linalg.lstsq(A, t, rcond=None)
        self.coef = w / norms
        return self

    def predict(self, u, chunk: int = 65536) -> np.ndarray:
        """Model output for every sample of ``u``, ``chunk`` samples at a time."""
        if self.coef is None:
            raise RuntimeError("GMP.predict before fit")
        u = np.asarray(u, dtype=np.complex128)
        out = np.empty(len(u), dtype=np.complex128)
        for start in range(0, len(u), int(chunk)):
            stop = min(start + int(chunk), len(u))
            out[start:stop] = self.basis(u, start, stop) @ self.coef
        return out
