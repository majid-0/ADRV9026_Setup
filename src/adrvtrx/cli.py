"""Command-line entry points.

``adrvtrx-program``: connect, program, print status (owns the board directly)::

    adrvtrx-program --config config/default.toml

``adrvtrx``: bench jobs that go through the hardware server (docs/hw_server.md)::

    adrvtrx replay --conditions captures/TX1_conditions.csv \\
                   --dpd input=input --dpd gmp=DPD/gmp/{name}.txt --out DPD_DUT/run1
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

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


# -- adrvtrx (jobs through the hardware server) ----------------------------------

SUMMARY_FIELDS = (
    "state",
    "label",
    "file",
    "nmse_db",
    "aclr_lower_dbc",
    "aclr_upper_dbc",
    "gain_compression_db",
    "rms_dbfs",
    "peak_dbfs",
    "time",
)


def tool_main(argv: list[str] | None = None) -> int:
    """``adrvtrx``: bench jobs that use the board through ``adrvtrx-server``."""
    parser = argparse.ArgumentParser(
        prog="adrvtrx", description="Bench jobs through the hardware server (adrvtrx-server)."
    )
    sub = parser.add_subparsers(dest="cmd", metavar="{replay}")
    sub.required = True
    p = sub.add_parser(
        "replay",
        help="play waveforms (DPD files or the input) at saved conditions and score them",
        description="For each state of a conditions CSV, play every --dpd waveform back to "
        "back at the state's saved LO, TX attenuation and ORx gain (no AGC, no rescale), "
        "capture and score against the state's input.",
    )
    p.add_argument("--conditions", required=True, help="capture CSV, e.g. TX1_conditions.csv")
    p.add_argument("--states", nargs="+", metavar="NAME", help="rows to replay (default: all)")
    p.add_argument(
        "--dpd",
        action="append",
        required=True,
        metavar="LABEL=TEMPLATE",
        help="waveform per state: a path with {name} (relative to the current directory), "
        "or 'input' for the state's own input; repeat for several labels",
    )
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--no-wait", action="store_true", help="fail at once if the board is busy")
    p.add_argument("--timeout", type=float, help="seconds to wait for the board")
    p.add_argument("--name", help="job name shown in adrvtrx-server status")
    p.add_argument("--tx", help="TX channel (default: from the state name, e.g. TX1_...)")
    p.add_argument("--orx", help="ORx channel (default: mapped to the TX in [tx_to_orx])")
    p.add_argument("--lo", help="LO to retune (default: the TX's LO in [clocks])")
    p.add_argument(
        "--file-scale",
        type=float,
        default=1.0,
        help="TEMPLATE files hold u * file_scale (2048 for files in TX codes x 2**11)",
    )
    p.add_argument("--oversample", type=int, default=2, help="captured periods per state")
    p.add_argument("--config", help="TOML config ([server] port and key)")
    args = parser.parse_args(argv)
    return _replay_command(args)


def _parse_labels(specs: list[str]) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    for spec in specs:
        label, sep, template = spec.partition("=")
        label, template = label.strip(), template.strip()
        if not sep or not label or not template:
            raise ValueError(f"--dpd needs LABEL=TEMPLATE, got {spec!r}")
        if any(c in label for c in "/\\:{}"):
            raise ValueError(f"--dpd label {label!r} must be a plain name (it names files)")
        if label in (name for name, _t in labels):
            raise ValueError(f"--dpd label {label!r} given twice")
        labels.append((label, template))
    return labels


def _replay_plan(args) -> tuple[list, list[str]]:
    """``[(row, ref_path, [(label, source, file_scale)])]`` and every problem found."""
    from .conditions import load_conditions

    csv_path = Path(args.conditions)
    problems: list[str] = []
    try:
        labels = _parse_labels(args.dpd)
    except ValueError as exc:
        return [], [str(exc)]
    if not csv_path.is_file():
        return [], [f"conditions CSV not found: {csv_path}"]
    rows = load_conditions(csv_path)
    if args.states:
        by_name = {row.name: row for row in rows}
        problems += [
            f"state {n!r} is not in {csv_path.name}" for n in args.states if n not in by_name
        ]
        rows = [by_name[n] for n in args.states if n in by_name]
    plan = []
    for row in rows:
        ref = csv_path.parent / row.in_file
        if not ref.is_file():
            problems.append(f"{row.name}: reference {ref}")
        sources = []
        for label, template in labels:
            if template == "input":
                src, scale = ref, 1.0
            else:
                src, scale = Path(template.replace("{name}", row.name)), args.file_scale
                if not src.is_file():
                    problems.append(f"{row.name} [{label}]: {src}")
            sources.append((label, src, scale))
        plan.append((row, ref, sources))
    if not plan and not problems:
        problems.append("no states to replay")
    return plan, problems


def _replay_command(args) -> int:
    from ._enums import RxChannel, TxChannel
    from .capture import orx_for_tx
    from .client import BoardBusy, ServerUnavailable, hardware, server_config
    from .conditions import DUT_FIELDS, ConditionLog, load_aligned_iq
    from .config import lo_for_tx
    from .profile import read_profile
    from .replay import replay_row
    from .waveform import full_scale

    plan, problems = _replay_plan(args)
    if problems:  # before the job asks for the board
        print("adrvtrx replay cannot start:", file=sys.stderr)
        for problem in problems:
            print(f"  missing or invalid: {problem}", file=sys.stderr)
        return 2

    client_cfg = load_config(args.config)
    try:
        cfg = server_config(client_cfg)  # what the board was programmed with
    except ServerUnavailable as exc:
        print(f"adrvtrx replay: {exc}", file=sys.stderr)
        return 1
    info = read_profile(cfg.profile_path)

    def channels(row):
        tx = TxChannel[args.tx] if args.tx else TxChannel[row.name.split("_", 1)[0]]
        orx = RxChannel[args.orx] if args.orx else orx_for_tx(tx, cfg.tx_to_orx)
        if orx is None:
            raise ValueError(f"no ORx mapped to {tx.name} in [tx_to_orx]; pass --orx")
        return tx, orx, args.lo or lo_for_tx(cfg.clocks, tx)

    try:
        routes = {row.name: channels(row) for row, _ref, _s in plan}
    except (KeyError, ValueError) as exc:
        print(f"adrvtrx replay: cannot tell TX / ORx: {exc} (use --tx / --orx)", file=sys.stderr)
        return 2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = [label for label, _src, _scale in plan[0][2]]
    name = args.name or f"replay {Path(args.conditions).name}"
    summary_path = out / "summary.csv"
    try:
        radio = hardware(name, wait=not args.no_wait, timeout=args.timeout, config=client_cfg)
    except (BoardBusy, ServerUnavailable) as exc:
        print(f"adrvtrx replay: {exc}", file=sys.stderr)
        return 1
    logs = {label: ConditionLog(out / f"{label}.csv", DUT_FIELDS) for label in labels}
    with radio, open(summary_path, "w", newline="") as fh:
        summary = csv.DictWriter(fh, fieldnames=list(SUMMARY_FIELDS))
        summary.writeheader()
        current_freq = None
        try:
            for row, ref, sources in plan:
                tx, orx, lo = routes[row.name]
                x_codes = load_aligned_iq(ref) * float(full_scale(info.tx_bits))
                for label, src, scale in sources:
                    record = replay_row(
                        radio,
                        row,
                        x_codes,
                        src,
                        tx=tx,
                        orx=orx,
                        tx_bits=info.tx_bits,
                        rx_bits=info.rx_bits,
                        fs=info.orx_rate_hz,
                        out_dir=out,
                        label=label,
                        oversample=args.oversample,
                        lo=lo,
                        file_scale=scale,
                        retune=row.freq_hz != current_freq,
                    )
                    current_freq = row.freq_hz
                    logs[label].append(record)
                    summary.writerow(
                        {
                            "state": row.name,
                            "label": label,
                            "file": str(src),
                            "nmse_db": f"{record.nmse_db:.2f}",
                            "aclr_lower_dbc": f"{record.aclr_lower_dbc:.2f}",
                            "aclr_upper_dbc": f"{record.aclr_upper_dbc:.2f}",
                            "gain_compression_db": f"{record.gain_compression_db:.2f}",
                            "rms_dbfs": f"{record.rms_dbfs:.2f}",
                            "peak_dbfs": f"{record.peak_dbfs:.2f}",
                            "time": datetime.now().isoformat(timespec="seconds"),
                        }
                    )
                    fh.flush()
                    print(
                        f"{row.name} [{label}]: NMSE {record.nmse_db:.2f} dB, ACLR "
                        f"{record.aclr_lower_dbc:.2f} / {record.aclr_upper_dbc:.2f} dBc",
                        flush=True,
                    )
        finally:
            try:
                radio.disable_tx()
            except Exception:  # noqa: BLE001 - releasing the job forces safe anyway
                pass
            for log in logs.values():
                log.close()
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
