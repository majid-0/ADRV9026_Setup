"""Command-line entry point: connect, program, print status.

adrvtrx-program --config config/default.toml
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .experiment import session, verify_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="adrvtrx-program", description=__doc__)
    parser.add_argument("--config", help="path to TOML config (default: bundled default.toml)")
    parser.add_argument("--no-program", action="store_true", help="connect only, do not program")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    print(f"Connecting to {cfg.board.ip}:{cfg.board.port} ...")
    with session(cfg, program=not args.no_program) as (radio, info):
        print(f"Profile: {cfg.profile_path.name}")
        print(
            f"  Tx {info.tx_bits}-bit @ {info.tx_rate_khz/1000:.3f} MSPS, "
            f"Rx {info.rx_bits}-bit @ {info.rx_rate_khz/1000:.3f} MSPS"
        )
        for key, value in verify_status(radio).items():
            print(f"  {key}: {value}")
    print("Done (TX left safe, disconnected).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
