"""Joint (multi-band) crest factor reduction for any set of carriers.

The low-rate CFR of Fehri, Boumaiza & Sich (TMTT 62(12), 2014): the combined peak is
estimated as S(n) = sum_k |x_k(n)| at the carriers' own rate. Wherever S(n) > C, every
carrier is scaled by C/S(n) (eq. 11/14), the gain is smoothed by a Hann peak window, and
the clipping noise is filtered back into each carrier's channel. Clip and filter are
repeated n_iter times. The combined PAPR is checked on the true composite at a chosen
carrier spacing.

Library:
    from cfr_joint import load_iq, joint_cfr, composite_papr, pe_papr, papr_db
    ys = joint_cfr([x1, x2], fs, target_db=9.0, bws=[80e6, 80e6])

Command line:
    python3 cfr_joint.py SIG1.txt SIG2.txt --fs 368.64e6 --bw 80e6 --spacing 900e6 \
        --targets 6:orig:1 --plot psd.png --csv papr.csv

Files: two columns (I Q) per line, or .npy complex arrays. All carriers must share one
sample rate and length (they are processed sample by sample at that rate).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

__all__ = ["load_iq", "papr_db", "occupied_bw", "channel_mask", "window_gain", "joint_cfr",
           "pe_papr", "composite", "composite_papr", "error_pct", "psd"]


# ── basics ────────────────────────────────────────────────────────────


def load_iq(path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(complex)
    a = np.loadtxt(path)
    return a[:, 0] + 1j * a[:, 1]


def papr_db(x) -> float:
    p = np.abs(x) ** 2
    return float(10 * np.log10(p.max() / p.mean()))


def occupied_bw(x, fs: float, frac: float = 0.999) -> float:
    """Bandwidth (Hz) holding ``frac`` of the power, symmetric about 0 Hz."""
    X = np.abs(np.fft.fft(x)) ** 2
    f = np.fft.fftfreq(len(x), 1 / fs)
    order = np.argsort(np.abs(f))
    c = np.cumsum(X[order]) / X.sum()
    return float(2 * np.abs(f[order][np.searchsorted(c, frac)]))


def channel_mask(n: int, fs: float, bw: float) -> np.ndarray:
    """Ideal in-channel mask |f| <= bw/2 (FFT order)."""
    return (np.abs(np.fft.fftfreq(n, 1 / fs)) <= bw / 2).astype(float)


def window_gain(g, win_len: int):
    """Hann peak windowing of a gain g(n) <= 1 by max-hold: the attenuation 1-g is spread
    over ``win_len`` samples, so every peak still ends at <= C but the gain is smooth."""
    if win_len is None or win_len <= 1:
        return g
    M = win_len // 2
    w = 0.5 * (1 + np.cos(np.pi * np.arange(-M, M + 1) / (M + 1)))
    d = 1.0 - g
    dw = d.copy()
    for k, wk in zip(range(-M, M + 1), w):
        if k:
            np.maximum(dw, wk * np.roll(d, k), out=dw)
    return 1.0 - dw


# ── CFR ───────────────────────────────────────────────────────────────


def joint_cfr(xs: Sequence[np.ndarray], fs: float, target_db: float,
              bws: Optional[Sequence[float]] = None, n_iter: int = 4,
              win_s: float = 9 / 92.16e6, weights: Optional[Sequence[float]] = None,
              filter_noise: bool = True) -> list:
    """Reduce the combined PAPR of carriers ``xs`` to ``target_db``.

    Parameters
    ----------
    xs : carriers at their baseband rate ``fs`` (same length).
    target_db : combined PAPR target, relative to the total power of all carriers.
    bws : channel bandwidth per carrier for the noise filter (default: 99.9 % occupied).
    n_iter : clip + filter passes (the paper uses 4).
    win_s : peak window length in seconds (default 98 ns, the value that matched the paper).
    weights : per-carrier weights G_k in S = sum G_k |x_k| (eq. 12, unequal power); default 1.
    filter_noise : False returns the first clipping pass, unfiltered.
    """
    xs = [np.asarray(x, complex) for x in xs]
    n = len(xs[0])
    G = np.ones(len(xs)) if weights is None else np.asarray(weights, float)
    P = sum(np.mean(np.abs(x) ** 2) for x in xs)
    C = np.sqrt(P * 10 ** (target_db / 10))
    win = int(round(win_s * fs)) | 1
    if bws is None:
        bws = [occupied_bw(x, fs) for x in xs]
    masks = [channel_mask(n, fs, b) for b in bws]
    cur = xs
    for _ in range(n_iter if filter_noise else 1):
        S = sum(g * np.abs(x) for g, x in zip(G, cur))
        gain = window_gain(np.where(S > C, C / np.maximum(S, 1e-30), 1.0), win)
        clipped = [x * gain for x in cur]
        if not filter_noise:
            return clipped
        cur = [x + np.fft.ifft(np.fft.fft(c - x) * m) for x, c, m in zip(cur, clipped, masks)]
    return cur


# ── measurements ──────────────────────────────────────────────────────


def pe_papr(xs, weights=None) -> float:
    """Combined PAPR from the peak estimate sum_k |x_k| (spacing-free upper bound)."""
    G = np.ones(len(xs)) if weights is None else np.asarray(weights, float)
    S = sum(g * np.abs(x) for g, x in zip(G, xs))
    P = sum(np.mean(np.abs(x) ** 2) for x in xs)
    return float(10 * np.log10(S.max() ** 2 / P))


def composite(xs, fs: float, offsets: Sequence[float], oversample: float = 4.0) -> np.ndarray:
    """True composite envelope: carriers placed at ``offsets`` (Hz, relative to a common
    centre) by exact periodic up-sampling to >= oversample x the occupied span."""
    n = len(xs[0])
    span = (max(offsets) - min(offsets)) + fs
    L = max(1, int(np.ceil(oversample * span / fs)))
    nh = n * L
    f = np.fft.fftfreq(n, 1 / fs)
    keep = np.abs(f) < fs / 2
    idx = np.round(f[keep] * n / fs).astype(np.int64)
    Z = np.zeros(nh, complex)
    for x, off in zip(xs, offsets):
        Z[(idx + int(round(off * n / fs))) % nh] += np.fft.fft(x)[keep] * L
    return np.fft.ifft(Z)


def composite_papr(xs, fs: float, spacing: float, oversample: float = 4.0) -> float:
    """Combined PAPR with the carriers equally spaced by ``spacing`` Hz around the centre."""
    k = len(xs)
    offsets = [(i - (k - 1) / 2) * spacing for i in range(k)]
    return papr_db(composite(xs, fs, offsets, oversample))


def error_pct(ref, y) -> float:
    """Waveform error |y - g ref| / |g ref| in % after a complex-gain fit (an NMSE, not a
    demodulated EVM; close to RMS EVM for fully loaded OFDM with in-channel noise)."""
    g = np.vdot(ref, y) / np.vdot(ref, ref)
    return float(100 * np.linalg.norm(y - g * ref) / np.linalg.norm(g * ref))


def psd(x, fs: float, nfft: int = 4096):
    """Welch PSD (Hann, 50 % overlap), shifted; returns (f_Hz, dB/Hz)."""
    win = np.hanning(nfft)
    step = nfft // 2
    segs = [x[i:i + nfft] * win for i in range(0, len(x) - nfft + 1, step)]
    P = np.mean([np.abs(np.fft.fft(s)) ** 2 for s in segs], axis=0) / (fs * np.sum(win ** 2))
    return np.fft.fftshift(np.fft.fftfreq(nfft, 1 / fs)), np.fft.fftshift(10 * np.log10(P + 1e-300))


# ── command line ──────────────────────────────────────────────────────


def _targets(spec: str, orig: float) -> list:
    """'6:orig:1' -> 6, 7, ... below orig; '9,8.5' -> [9, 8.5]."""
    if ":" in spec:
        a, b, s = spec.split(":")
        hi = orig if b == "orig" else float(b)
        out = list(np.arange(float(a), hi - 1e-9, float(s)))
        return [round(t, 3) for t in out]
    return [float(t) for t in spec.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("signals", nargs="+", help="carrier files (I Q text or .npy), one per band")
    ap.add_argument("--fs", type=float, required=True, help="sample rate of the files (Hz)")
    ap.add_argument("--bw", type=float, nargs="+", help="channel bandwidth(s) in Hz (default: 99.9%% occupied)")
    ap.add_argument("--spacing", type=float, required=True, help="carrier spacing (Hz) for the composite PAPR")
    ap.add_argument("--targets", default="6:orig:1", help="'start:orig:step' or comma list (dB)")
    ap.add_argument("--n-iter", type=int, default=4)
    ap.add_argument("--win-ns", type=float, default=9 / 92.16e6 * 1e9, help="peak window length (ns)")
    ap.add_argument("--plot", help="PNG with each band's PSD per target")
    ap.add_argument("--csv", help="CSV with the PAPR table")
    ap.add_argument("--save-dir", help="write the CFR'd carriers here as I Q text, one folder per target")
    a = ap.parse_args(argv)

    xs = [load_iq(p) for p in a.signals]
    n = min(len(x) for x in xs)
    xs = [x[:n] for x in xs]
    bws = a.bw * len(xs) if a.bw and len(a.bw) == 1 else (a.bw or [occupied_bw(x, a.fs) for x in xs])
    orig = composite_papr(xs, a.fs, a.spacing)
    targets = _targets(a.targets, orig)
    names = [Path(p).stem for p in a.signals]

    rows = [dict(label="original", target_db=np.nan, composite_db=orig, pe_db=pe_papr(xs),
                 **{f"papr_{nm}": papr_db(x) for nm, x in zip(names, xs)},
                 **{f"err_pct_{nm}": 0.0 for nm in names})]
    outs = {"original": xs}
    for t in targets:
        ys = joint_cfr(xs, a.fs, t, bws, n_iter=a.n_iter, win_s=a.win_ns * 1e-9)
        lab = f"target {t:g} dB"
        outs[lab] = ys
        rows.append(dict(label=lab, target_db=t, composite_db=composite_papr(ys, a.fs, a.spacing),
                         pe_db=pe_papr(ys), **{f"papr_{nm}": papr_db(y) for nm, y in zip(names, ys)},
                         **{f"err_pct_{nm}": error_pct(x, y) for nm, x, y in zip(names, xs, ys)}))
        if a.save_dir:
            d = Path(a.save_dir) / f"target_{t:g}dB"
            d.mkdir(parents=True, exist_ok=True)
            for nm, y in zip(names, ys):
                np.savetxt(d / f"{nm}.txt", np.column_stack([y.real, y.imag]))

    print(f"fs {a.fs/1e6:g} Msps, {n} samples, channel BW {[round(b/1e6, 2) for b in bws]} MHz, "
          f"spacing {a.spacing/1e6:g} MHz, {a.n_iter} passes, window {int(round(a.win_ns*1e-9*a.fs))|1} samples\n")
    head = ["", "combined PAPR (dB)", "peak estimate (dB)"] + [f"PAPR {nm} (dB)" for nm in names] \
        + [f"error {nm} (%)" for nm in names]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for r in rows:
        cells = [r["label"], f"{r['composite_db']:.2f}", f"{r['pe_db']:.2f}"] \
            + [f"{r[f'papr_{nm}']:.2f}" for nm in names] + [f"{r[f'err_pct_{nm}']:.2f}" for nm in names]
        print("| " + " | ".join(cells) + " |")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    if a.plot:
        _plot(outs, rows, names, a.fs, bws, a.spacing, a.plot)


def _plot(outs, rows, names, fs, bws, spacing, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = len(names)
    fig, ax = plt.subplots(2, k, figsize=(6.5 * k, 8), sharex="col", squeeze=False)
    cmap = plt.get_cmap("viridis")
    labels = list(outs)
    xs = outs["original"]
    for j, nm in enumerate(names):
        for i, lab in enumerate(labels):
            r = rows[i]
            col = "k" if lab == "original" else cmap((i - 1) / max(1, len(labels) - 2))
            txt = (f"original, {r['composite_db']:.2f} dB" if lab == "original"
                   else f"{r['composite_db']:.2f} dB (target {r['target_db']:g}), err {r[f'err_pct_{nm}']:.1f} %")
            f, p = psd(outs[lab][j], fs)
            ax[0, j].plot(f / 1e6, p, color=col, lw=1.6 if lab == "original" else 0.9, label=txt,
                          zorder=3 if lab == "original" else 2)
            if lab != "original":
                y = outs[lab][j]
                g = np.vdot(xs[j], y) / np.vdot(xs[j], xs[j])
                f, pe = psd(y - g * xs[j], fs)
                ax[1, j].plot(f / 1e6, pe, color=col, lw=0.9, label=f"target {r['target_db']:g} dB")
        f, p0 = psd(xs[j], fs)
        ax[1, j].plot(f / 1e6, p0, color="k", lw=1.2, alpha=0.35, label="original signal")
        for a_ in ax[:, j]:
            a_.axvspan(-bws[j] / 2e6, bws[j] / 2e6, color="0.9", zorder=0)
            a_.grid(alpha=0.3)
        ax[0, j].set_title(f"{nm}: output PSD per combined PAPR")
        ax[1, j].set_title(f"{nm}: CFR distortion (output − gain·original)")
        ax[1, j].set_xlabel("frequency offset from carrier (MHz)")
        ax[0, j].set_ylabel("PSD (dB/Hz)")
        ax[1, j].set_ylabel("PSD (dB/Hz)")
        ax[0, j].legend(fontsize=7, loc="lower center")
        ax[1, j].legend(fontsize=7, loc="lower center")
        top = p0.max()
        for a_ in ax[:, j]:
            a_.set_ylim(top - 110, top + 5)
    fig.suptitle(f"Joint CFR (Fehri et al. 2014), carrier spacing {spacing/1e6:g} MHz; "
                 f"combined PAPR measured on the true composite; grey band = channel", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\nplot: {path}")


if __name__ == "__main__":
    main()
