"""Quick read-only status check -- "is my TX/ORx off?".

Run:
    conda run -n myenv python scripts/status.py

Connects and prints the live status WITHOUT programming or changing TX state, so
it's safe to run any time (e.g. to confirm TX is off after a run). Pass --safe to
also force TX to the safe state (max attenuation + TX disabled) before reading.
"""

from __future__ import annotations

import sys

from adrvtrx.config import load_config
from adrvtrx.radio import Radio


def main() -> None:
    radio = Radio(load_config())
    radio.connect()  # read-only: does NOT program or alter TX state
    try:
        if "--safe" in sys.argv:
            radio.safe_state()
            print("(forced safe state)\n")
        radio.print_status()
    finally:
        radio.disconnect()


if __name__ == "__main__":
    main()
