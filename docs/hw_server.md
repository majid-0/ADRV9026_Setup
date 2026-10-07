# Hardware server — specification

Status: **implemented** on `feat/hw-server`. Not yet run against the board: the
hardware acceptance test (§10) is still to do.

## 1. Purpose

Every bench task used to be its own script that repeated the same ~60 lines:
load the config, read the profile, build a `Radio`, connect, force safe,
program, and `try/finally` back to the safe state and disconnect. Several
processes (scripts, Jupyter kernels) loaded the DLL and connected, so they raced
for the hardware. A force-killed process once left TX enabled and the reconnect
was refused.

The hardware server fixes this:

- **One** long-lived process owns the board. It is the only process that loads
  the DLL and connects to the ADS9.
- Every other process uses the board through it, one job at a time, in a FIFO
  queue.
- Every way a process can end has a path that stops TX (§6).

It is general: any number of TX / RX / ORx channels, any masks, profiles and
LOs. It forwards every public `Radio` method, so nothing in it is specific to
one experiment.

## 2. Architecture

```
 script / notebook A ──┐
 script / notebook B ──┤   TCP 127.0.0.1:55600          adrvtrx-server run
 adrvtrx replay ...  ──┼── multiprocessing.connection ─► supervisor (no DLL)
 adrvtrx-server status ┤   + auth key (HMAC)              │  spawns, pings,
         safe / kick / ┘                                  │  kills, restarts
               stop                                       ▼
                                                    server process
                                       ┌──────────────────────────────────────┐
                                       │ accept loop → one thread / connection │
                                       │ lease manager: owner + FIFO queue     │
                                       │ monitor thread: heartbeats            │
                                       │ hardware thread (the only DLL caller) │
                                       └──────────────────┬───────────────────┘
                                                          ▼
                                           Radio → adrvtrx_dll.dll (pythonnet)
                                                          ▼
                                           ADS9 192.168.1.10:55556 → ADRV9026
```

- **Supervisor.** `adrvtrx-server run` starts a small parent process that
  never loads the DLL. It spawns the server as a child process and watches it
  (§6, watchdog).
- **Server.** On start: bind the port (a second server stops here, before it
  touches the board), connect, `force_safe`, `program`. It then serves clients.
- **Hardware thread.** Every DLL call runs on this one thread, in order. Two
  jobs can never interleave hardware calls.
- **Control.** `status`, `safe`, `kick`, `stop` and the supervisor's `ping` are
  answered by their own connection thread. They never wait behind queued
  hardware calls. `status` reads the board live when the hardware thread is
  idle and serves the cached state when it is busy.
- **Transport.** TCP bound to `127.0.0.1` only, with the standard library's
  `multiprocessing.connection` and an auth key (HMAC challenge on connect). No
  new dependencies. Messages are pickled; only processes that can read the key
  file can connect.
- **Numpy boundary.** .NET objects cannot cross processes. Clients send numpy
  arrays to `perform_tx`; `Radio.perform_tx` converts them to .NET arrays
  itself. `perform_rx` results come back as a list of `int32` numpy arrays
  `[ch0_I, ch0_Q, ...]`, the same layout `capture.extract_channels` already
  reads.

## 3. Client API

```python
from adrvtrx.client import hardware

with hardware("TX1 sweep") as radio:          # waits in the queue for the board
    radio.set_tx_atten(TxChannel.TX1, 20.0)
    transmit_bands(radio, {TxChannel.TX1: x}, info.tx_bits)
    res = capture(radio, int(RxChannel.ORX1), 0.5, bits=info.rx_bits)
# leaving the block: lease released, server forces TX safe, next job starts
```

`hardware(name, wait=True, timeout=None, *, config=None)` returns a
`RemoteRadio` that holds the board until it is released:

| Argument | Meaning |
|---|---|
| `name` | Job name shown in `status` and in the log |
| `wait` | `True`: wait in the FIFO queue. `False`: raise `BoardBusy` at once if another job holds the board ("owned by NAME, pid P, host H, since T") |
| `timeout` | Seconds to wait in the queue before `BoardBusy`. `None` = no limit |
| `config` | Config path or `Config`; only `[server]` is used to find the port and key. Default: `load_config()` |

`RemoteRadio`:

- Has every public `Radio` method with the same signature (derived from the
  `Radio` class, so a new `Radio` method is forwarded with no change here).
  Arguments are checked against the `Radio` signature before sending.
- `config` is the server's `Config` (the one the board was programmed with).
- `bridge` builds numpy arrays instead of .NET arrays, so
  `transmit.transmit_bands` runs unchanged.
- `print_status()` prints locally from a remote `status()`.
- `release()` (same as leaving the `with` block) ends the job. For a notebook
  that keeps the board across cells: `radio = hardware("nb")` … `radio.release()`.
- Two lifecycle methods are not passed through literally: `disconnect()` only
  forces TX safe (the server keeps its board connection for the next job), and
  `connect()` is a no-op while the server is connected.

`transmit`, `capture`, `compression`, `replay`, `operating_point`,
`linearize`, `sweep`, `sweep_plan` and `bands` take a `RemoteRadio` wherever
they take a `Radio`. The one change they needed: `capture.capture` used
`radio._en_tx` to keep TX enabled while it set the Rx enables; it now calls the
new `Radio.set_rx_enable(mask)`, which sets the Rx mask and keeps TX.

Errors:

| Exception | When |
|---|---|
| `ServerUnavailable` (`ConnectionError`) | No server on the port, or no key file |
| `BoardBusy` (`RuntimeError`) | `wait=False` and the board is owned, or the queue `timeout` passed |
| `LeaseRevoked` (`RuntimeError`) | The job was ended by `kick`, `safe`, a heartbeat timeout, or server stop. Every later call raises it |
| `ServerConnectionLost` (`ConnectionError`) | The connection dropped during the job (server killed or restarted). The job must be run again |
| `RemoteError` (`RuntimeError`) | A hardware call raised in the server. The remote traceback is in the message. Built-in exception types (`ValueError`, `KeyError`, ...) are raised as that type |

A background thread sends a heartbeat every `heartbeat_timeout_s / 4` while
the job holds the board.

## 4. Queue and lease semantics

- A **lease** is one job's hold on the board, from acquire to release. At most
  one lease exists. Only the lease owner's calls reach the hardware thread.
- Waiting jobs form a FIFO queue. A waiting client is told its position once
  (printed to stderr).
- On every release the server runs `safe_state()` (max attenuation on
  `TxChannel.ALL`, TX enable mask cleared) **before** the next job gets the
  board. Every job starts from the safe state.
- A revoked lease (kick, safe, heartbeat timeout, server stop) is ended the
  same way. Calls of the revoked job already queued but not started are
  rejected, not run.
- A client that disconnects while waiting is removed from the queue.
- Server start-up counts as busy: jobs queue until programming has finished.

## 5. Limits

- One board per server. The port is fixed per config; run two configs on two
  ports for two boards.
- A DLL call cannot be interrupted. A call stuck inside the DLL is ended only
  by the watchdog killing the server process.
- The heartbeat proves the client process is alive, not that its script is
  making progress. A script blocked in Python (for example on `input()`) still
  heartbeats; use `kick`.
- After a server restart, running jobs get `ServerConnectionLost`; nothing is
  replayed. The board is re-programmed from the config.
- Large `perform_tx` buffers are pickled over localhost; the .NET conversion
  inside `Radio` is the same cost as before.
- The ADS9 may refuse a new connection after a hard kill (seen on 2026-10-06).
  The watchdog retries and then reports the TX state as unknown. **A hardware
  PA-supply switch remains the last-resort emergency stop.**

## 6. Stop paths

Every path ends in `safe_state()` on all TX channels.

| Event | Path | Bound |
|---|---|---|
| Job ends, raises, or Ctrl+C in the client | `with` exit → release → `safe_state` → next job | immediate |
| Client force-killed | the OS closes its socket → server sees the drop → `safe_state` → next job | immediate (after a call in progress) |
| Client alive but stuck (frozen process, debugger pause, sleep) | no heartbeat for `heartbeat_timeout_s` (10 s) while TX is enabled → lease revoked → `safe_state` | 10 s |
| `adrvtrx-server kick` | current lease revoked → `safe_state`; that client's next call raises `LeaseRevoked`. Replaces STOP files | immediate |
| `adrvtrx-server safe` | skips the queue: `safe_state` runs next on the hardware thread (ahead of queued calls); the current lease is revoked so its next call cannot re-enable TX | immediate (after a call in progress) |
| `adrvtrx-server safe --direct` | server unreachable: this process connects to the board itself and runs `force_safe`. Refused while a live server answers, unless `--force` | connect time |
| `adrvtrx-server stop`, Ctrl+C or SIGTERM to the server, supervisor gone, unhandled server error | lease revoked → `safe_state` → `disconnect` → exit | immediate |
| Server force-killed or crashed | **watchdog**: the supervisor sees the child exit, runs `force_safe` in a fresh process with a fresh board connection, then restarts the server | seconds |
| Server stuck inside a DLL call | **watchdog**: the server reports the running call and its timeout in every ping; past `call_timeout_s` for that method (or no ping answer for `ping_timeout_s`) the supervisor kills it, runs `force_safe` in a fresh process and restarts it | call timeout |
| Server start | always `force_safe` before `program` | — |

Watchdog details:

- Restarts are limited to `restart_limit` within `restart_window_s`, with
  exponential backoff from `restart_backoff_s`. Past the limit the supervisor
  exits with an error.
- The fresh-process `force_safe` retries the board connection
  `connect_retries` times. It reads the enable mask back to confirm TX is off.
  If it cannot connect or confirm, the supervisor prints and logs
  `TX STATE UNKNOWN - switch off the PA supply`.
- If the server never became ready (bad config, board off), the supervisor
  does not restart it: it runs `force_safe` once and exits with an error.
- The server watches its supervisor through a pipe on its stdin. If the
  supervisor dies, the server stops itself safely (no unwatched server).
- Exit codes of the server process: `0` stopped on request, `3` port in use
  (another server), `4` bad config or backend, `5` start-up (connect / program)
  failed, anything else is a crash.

## 7. Observability

`adrvtrx-server status` (or `--json`):

```
server   pid 1234, 127.0.0.1:55600, up 0:12:34, restarts 0, supervised
state    ready (backend adrvtrx.radio:Radio)
owner    "TX1 sweep" pid 5678 on BENCH-PC since 10:01:02 (3m 4s), last heard 0.4 s ago
queue    1 waiting: "notebook B" pid 4321
hardware idle
TX       live: TX1 (mask 0x1)
LO       LO1 2400000000 Hz, LO2 900000000 Hz
atten    TX1 15.00, TX2 41.95, TX3 41.95, TX4 41.95 dB
gains    ORX1 214
PLL      0xF (all locked) [live]
error    -
```

Board values are the last commanded ones (LO, attenuation, gain, enables),
refreshed from a live `Radio.status()` read when the hardware thread is idle.

Log: one JSON object per line in `<log_dir>/server-YYYY-MM-DD.jsonl`, written
by the server and the supervisor. Every command is logged with time, client
(name, pid, host), method, a short argument summary (arrays as shape and
dtype), duration and result or error. Lease events (acquire, release, revoke),
start, stop, crashes, watchdog actions and restarts are logged too. Heartbeats
and pings are not.

## 8. Configuration

`config/default.toml`, section `[server]`:

| Key | Default | Meaning |
|---|---|---|
| `port` | `55600` | TCP port on `127.0.0.1` (the ADS9 is on 55556) |
| `state_dir` | `""` | Holds the auth key `server.key`. Blank: `%LOCALAPPDATA%\adrvtrx` on Windows, `$XDG_STATE_HOME/adrvtrx` or `~/.local/state/adrvtrx` elsewhere |
| `log_dir` | `""` | Blank: `<state_dir>/logs` |
| `heartbeat_timeout_s` | `10` | Revoke a silent lease after this long while TX is enabled |
| `call_timeout_s` | `{ default = 60, program = 600 }` | Per-method limit before the watchdog treats the server as stuck. `startup` (connect + force_safe + program) falls back to `program` |
| `ping_interval_s` | `1` | Supervisor ping period |
| `ping_timeout_s` | `15` | No ping answer this long: server is stuck |
| `start_timeout_s` | `60` | Time for a new server process to answer its first ping |
| `restart_limit` | `3` | Restarts allowed within `restart_window_s` |
| `restart_window_s` | `600` | |
| `restart_backoff_s` | `2` | First restart delay; doubles per restart in the window |
| `force_safe_timeout_s` | `120` | Limit for the fresh-process `force_safe` |
| `connect_retries` | `3` | Board connection attempts in the fresh-process `force_safe` |
| `connect_retry_delay_s` | `5` | Delay between those attempts |

The auth key (32 random bytes) is created by the first `adrvtrx-server run`.
On POSIX it is written with mode `0600`.

## 9. CLI

```
adrvtrx-server run    [--config PATH] [--backend real|fake|MODULE:ATTR] [--no-program]
adrvtrx-server status [--config PATH] [--json]
adrvtrx-server safe   [--config PATH] [--direct [--force] [--backend ...]]
adrvtrx-server kick   [--config PATH]
adrvtrx-server stop   [--config PATH]
```

- `--backend real` (default) is `adrvtrx.radio:Radio`. `--backend fake` is
  `adrvtrx.fake:FakeRadio`, the simulated board of §9.1, for dry runs and
  tests.
- When `ADRVTRX_FORBID_HARDWARE` is set (the test suite sets it), `Radio`
  refuses to load the real DLL and the server refuses `--backend real`.

### 9.1 The fake backend

`FakeRadio` is the real `Radio` over a simulated DLL (no pythonnet, no board):
every `Radio` code path runs, only the bottom layer is simulated. It keeps the
register state (connection, programmed flag, enables, attenuation, gains,
LOs), rejects `TxAttenSet` before programming like the device, requires the
eight bridge-built arrays in `PerformTx`, and returns the full
`rxInitChannelMask` set from `PerformRx`, with the TX waveform looped back on
the ORx mapped to it. Its sample rate and bit widths come from the config's
profile (10 MSPS / 12 bits if the profile is not on the machine).

| Environment variable | Effect |
|---|---|
| `ADRVTRX_FAKE_STATE=<file.json>` | Register state shared across processes (the board keeps its registers when a client dies); every register write is logged to `<file>.events.jsonl`; creating `<file>.refuse` makes `Connect` fail |
| `ADRVTRX_FAKE_DELAYS='{"PerformRx": 30}'` | Seconds per DLL call, to simulate slow or stuck calls |
| `ADRVTRX_FAKE_PA=rich` or `=<file.json / .toml>` | The PA model below; a file holds parameters, `preset = "rich"` starts from the rich values |

PA model (`adrvtrx.fake.PaModel`), applied to one period of the looping TX
waveform (circular), at the TX's LO frequency `f`, with `df = (f − f0_hz)` in GHz:

1. `v = x / full_scale · drive · 10^((G − atten)/20)`, with
   `G = gain_db + gain_slope_db_per_ghz·df + gain_curve_db_per_ghz2·df² + drift`.
2. `w = pre_taps ∗ v` (FIR: linear memory before the nonlinearity).
3. Rapp: `g = w / (1 + (|w|/sat)^(2p))^(1/2p)`,
   `sat = 10^((sat_db + sat_slope_db_per_ghz·df + drift_sat_fraction·drift)/20)`,
   `p = smoothness`; then AM/PM `g ·= exp(j·am_pm_deg·π/180·|g/sat|²)`.
4. Nonlinear memory: `g += g · Σₘ nl_memory[m−1]·|g[n−m]/sat|²` (bounded,
   taken after compression).
5. `y = post_taps ∗ g`, delayed by `delay_samples`. Memory taps after the
   first are scaled by `1 + memory_slope_per_ghz·df`.
6. ORx: `y · orx_level · full_scale · 10^((gain_index − 210)·0.5/20)` plus
   complex noise of `noise_codes` per I and Q, rounded and clipped at the rail.

Drift in dB: `drift_db_per_hour` × hours since the fake board was created,
plus `drift_step_db` once `drift_step_after_s` seconds have passed, plus the
number in `drift_file` (read at every capture: drift on demand). `seed` (or
`FakeRadio(seed=...)`) makes captures repeatable.

| Parameter | Default (simple) | `rich` |
|---|---|---|
| `drive` | 6.0 | 5.35 |
| `smoothness` | 2.0 | 1.6 |
| `f0_hz` | 2.2e9 | 2.2e9 |
| `gain_db`, `gain_slope_db_per_ghz`, `gain_curve_db_per_ghz2` | 0, 0, 0 | 0, −1.5, −0.5 |
| `sat_db`, `sat_slope_db_per_ghz` | 0, 0 | 0, 0.6 |
| `am_pm_deg` | 0 | 12 |
| `pre_taps` | [1] | [1, 0.22, −0.08] |
| `post_taps` | [1] | [1, −0.12, 0.04] |
| `nl_memory` | [] | [0.15, 0.08, 0.04] |
| `memory_slope_per_ghz` | 0 | 0.35 |
| `delay_samples`, `orx_level`, `noise_codes` | 5, 0.73, 0.5 | same |
| `drift_db_per_hour`, `drift_step_db`, `drift_step_after_s`, `drift_file`, `drift_sat_fraction` | 0, 0, 0, "", 0.5 | same |
| `seed` | none | none |

The simple default is the memoryless Rapp the server tests were written
against. On the rich model at 491.52 MSPS, a 3.5 dB gain-compression lock
(`find_compression_point(..., lock_on="gain")`, start 15 dB, floor 7 dB)
converges at about 12.0 / 11.0 / 9.3 dB at 1.6 / 2.2 / 2.8 GHz for a
100 MHz signal; a GMP(5, 3, 2) post-inverse fit reaches about −45 dB NMSE
(−20 dB without one); a memoryless fit stays near −28 dB at 100 MHz and
−35 dB at 40 MHz.

Generic replay of saved conditions:

```
adrvtrx replay --conditions captures/TX1_conditions.csv \
               [--states NAME ...] --dpd LABEL=TEMPLATE [--dpd ...] --out DIR \
               [--no-wait] [--timeout S] [--name JOB] [--tx TX1] [--orx ORX1] \
               [--lo LO1] [--file-scale 1.0] [--oversample 2] [--config PATH]
```

- `TEMPLATE` is a path with `{name}` for the state name, relative to the
  current directory, or the word `input` to replay the state's own input (no
  DPD). Example: `--dpd input=input --dpd gmp=DPD/gmp/{name}.txt`.
- `--states` picks rows by `name` (default: all rows).
- `--tx` defaults to the TX in the state name (`TX1_...`); `--orx` to the ORx
  mapped to that TX in `[tx_to_orx]`; `--lo` to that TX's LO in `[clocks]`.
- All files are checked before the job asks for the board; missing files are
  listed and nothing is transmitted.
- For each state, all labels play back to back at the state's saved LO, TX
  attenuation and ORx gain (no AGC, no rescale; `replay.replay_row`).
- Output in `--out`: per label `{label}.csv` (the DUT CSV of
  `replay_conditions`) and `{name}_{label}_dut.txt`, plus one `summary.csv`:
  `state, label, file, nmse_db, aclr_lower_dbc, aclr_upper_dbc,
  gain_compression_db, rms_dbfs, peak_dbfs, time`.

## 10. Hardware acceptance test

Run this once on the bench before relying on the server. Keep the PA supply
switch in reach throughout. Use a dummy load or attenuator on the PA output and
watch the spectrum (analyzer, or ORx in a second terminal through `status`).
Close every notebook and script that builds its own `Radio`.

1. **Dry run, no board.** `adrvtrx-server run --backend fake`. In a second
   terminal: `adrvtrx-server status`, then `adrvtrx-server stop`. Expect a clean
   exit.
2. **Start.** `adrvtrx-server run`. Expect connect, force safe, program, then
   `ready`. `status`: PLL all locked, TX off, attenuation 41.95 dB on all TX.
3. **Second server.** In another terminal, `adrvtrx-server run`. Expect an
   immediate refusal; the first server is unaffected (`status` unchanged).
4. **One job.** Run a short script: `with hardware("acceptance") as radio:`
   set TX1 attenuation to 41.95 dB, transmit a low-level tone, capture ORx,
   `input()` to pause. `status` shows the owner and `TX live: TX1`. Press Enter:
   TX goes off, `status` shows no owner.
5. **Queue.** Start the script twice. The second prints its queue position and
   starts when the first ends. With `wait=False` the second fails at once with
   the owner's name, pid, host and start time.
6. **Client killed.** While the script holds TX on, `taskkill /F /PID <pid>`.
   The carrier disappears within a second; `status` shows TX off.
7. **Kick and safe.** While TX is on: `adrvtrx-server kick` (TX off; the
   script's next call raises `LeaseRevoked`). Repeat with `adrvtrx-server safe`.
8. **Heartbeat.** While TX is on, freeze the script (Resource Monitor →
   Suspend process, or pause it in a debugger). TX goes off after about 10 s.
9. **Watchdog.** With TX on at 41.95 dB attenuation, kill the server process
   (not the supervisor): `taskkill /F /PID <server pid from status>`. Expect the
   supervisor to report the crash, run `force_safe` in a fresh process (carrier
   gone), and restart the server (`status`: new pid, `restarts 1`). On
   2026-10-06 a reconnect after a hard kill was refused. If the supervisor
   prints `TX STATE UNKNOWN`, switch off the PA supply and note what the ADS9
   did.
10. **Stop.** `adrvtrx-server stop`: TX off, disconnected, supervisor exits.
    Restart and press Ctrl+C in the server console: same result.
11. **Direct safe.** With no server running: `adrvtrx-server safe --direct`
    connects, forces safe, disconnects. With a server running it refuses.
12. **Replay.** On a saved conditions CSV:
    `adrvtrx replay --conditions <csv> --states <one state> --dpd input=input --out <dir>`.
    Check `summary.csv` against the original capture's NMSE and ACLR.

## 11. Tests (`tests/`, no hardware, CI)

All server tests use the fake backend; `ADRVTRX_FORBID_HARDWARE` is set for
the whole suite, so the real `Radio` cannot load the DLL even by mistake.

| Test | What it checks |
|---|---|
| forwarding | `RemoteRadio` has exactly the public `Radio` methods; each one returns what an in-process `FakeRadio` returns |
| numpy boundary | `perform_tx` numpy → .NET conversion inside `Radio`; `perform_rx` comes back as `int32` arrays |
| queue | FIFO order, `wait=False`, queue timeout, waiting client disconnects |
| client killed | a client process killed with TX on → `safe_state`, next job proceeds |
| heartbeat | silent client with TX on → revoked after the timeout |
| kick / safe | lease revoked, TX safe, the job's next call raises `LeaseRevoked` |
| watchdog | server process killed → fresh-process `force_safe`, restart; stuck call → killed, safe, restarted |
| single instance | a second `run` is refused without touching the board |
| safe --direct | forces safe through the fake backend; refused while a server answers |
| programming identity | `program_count` / `program_id` change on a client re-program, a new server and a watchdog restart |
| fake PA | default = simple Rapp; rich model: gain lock converges at two LOs with different attenuations, GMP fit beats no DPD and needs memory, drift (file, rate, step), seed, parameter files |
| existing modules | transmit, capture, compression, replay, operating point, linearize and sweep give the same results on a `RemoteRadio` as in-process |
| replay CLI | end to end on the fake: summary, per-label CSVs, missing files reported before the board is taken |
