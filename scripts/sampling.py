"""Time-alignment / delay-estimation helpers.

The implementation now lives in the package at ``adrvtrx.align`` (one source of
truth, unit-tested, importable without putting scripts/ on sys.path). This shim
keeps ``import sampling`` / ``from sampling import ...`` working.

It also carries the fix over the original: when the capture is longer than the
reference, ``estimate_delay`` slides the reference through it instead of trimming
both to the shorter length (trimming chopped a long capture to one arbitrary-phase
period -> circular wrap -> decorrelation). Capture >= ~2x the reference and these
recover the sent signal time-aligned.
"""

from adrvtrx.align import apply_delay, estimate_and_align, estimate_delay

__all__ = ["estimate_delay", "estimate_and_align", "apply_delay"]
