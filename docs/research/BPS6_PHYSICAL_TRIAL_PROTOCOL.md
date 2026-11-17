# BPS-6 — reproducible physical measurement qualification (research)

**Status:** CPU-trial program and exact-front workload generator only. This
repository neither reads privileged energy counters nor claims measured energy.
Physical qualification is governed exclusively by `Memorithm/RemoteOps`.

Sources:
- `crates/flat-algebraic-attention/src/boolean_physical_protocol.rs`
- `crates/flat-algebraic-attention/examples/bps6_physical_trial.rs`
- companion BPS-5 synthetic correctness corpus:
  `docs/research/BPS5_HELDOUT_COST_BENCHMARK.md`

## Hypothesis, objective and the negative control

The hypothesis to falsify is that BPS-3's event-driven execution uses less
**wall time** or less **package/DRAM energy under specified hardware loads**
than its *exactly equivalent* dense Boolean recurrence at low event density.
This is a **same-model** comparison; it does not prove superiority over
FP32 LSTMs or minLSTM. Both sparse and dense return identical 64-bit output
traces, and BPS-6 rejects a corpus that violates exactness.

Four separately timed implementation modes:

- `dense`: evaluate the identity or event transition at **every sample**.
- `sparse`: detect rising edges on **every sample** and execute recurrence
  only when a front occurs; still writes a full per-sample output.
- `event_scan`: build the list of event transitions, scan them and
  reconstruct all outputs; retains input inspection and output costs.
- `output_copy`: allocate/copy a *precomputed* correct output vector.
  **Negative control only**; not a drop-in semantic implementation. Never
  subtract its energy uncritically from actual methods.

The 64-bit states, forget and write gates, shared rotation, initial memory
and corpus stay frozen across the methods. All result lengths and full
sequences are verified **before** the timed loop, including a full
temporal scan consistency test.

## Exact front density, not input occupancy

Each sampled word is either zero or one-hot. A rational accumulator selects
exactly `floor(steps * requested_percent / 100)` eventful timesteps.
The selected bit position is `tick mod 64`. Therefore two adjacent
eventful samples always use *different* bit positions, so **100% front-step
density is possible** even with no quiet sample between events.
The baseline is zero and the generator's independent literal front counter
must match the formula.

Fixed densities: 0%, 1%, 10%, 50%, 100%.
Normal experiment: 4096 samples, 128 rounds per phase, three ABBA-like
passages alternating `output_copy, dense, sparse, event_scan` with its
reverse. Total: 5 densities × 4 methods × 3 passages = **60 measured phases**.
Smoke mode: `--smoke`, 100 samples, one round, 0/50/100 densities,
one passage, **12 phases**.

The hash printed for each corpus is a lightweight fingerprint, NOT a
cryptographic SHA-256. The trusted remote observer must record the **exact
source commit SHA**, compiler/toolchain version and host identity separately.

## Timing and RAPL interpretation

The Rust binary:
- records monotonic `Instant` duration for each trial including allocations,
  all output writes and a full result-consumption checksum;
- emits `BPS6_MARKER=Rxx_Dyyy_MODE_BEGIN` then
  `BPS6_MARKER=Rxx_Dyyy_MODE_END` around each timed phase;
- emits `BPS6_TRIAL` metadata with mode, density, rounds, sample count,
  observed event count, elapsed nanoseconds and deterministic checksum;
- emits `BPS6_COMPLETE=true` only after all protocol phases pass.

The trusted RemoteOps observer may separately read Linux kernel powercap
`energy_uj` and `max_energy_range_uj` for package/DRAM RAPL domains.
**RAPL counts cover the complete physical package/DRAM domain and other
active processes on the host. They cannot be attributed unambiguously to
the BPS-6 process.** A delta modulo one counter range is only correct if
there was at most one wrap between readings; long-running phases require
sufficiently frequent sampling and an explicit uncertainty report.

Do **not** equate the Rust process timing boundary with the external
observer's sampled boundary; stdout buffering and host scheduling cause
marker-observation delays. Require a negative control and an order-reversed
paired protocol. Reject impossible durations, missing markers or counters,
missing CI provenance and incomplete reports. Do not report an energy
speedup based on one noisy run or uncalibrated idle subtraction.

`BPS6_PROTOCOL physical_energy_measured=false` is the statement that
**this executable** does not have an energy sensor. A separate, verified,
trustworthy observer may publish host-wide physical RAPL observations
with their own provenance and ambiguity flags.

## Qualifications

Exact-head Rust checks:

    cargo fmt --all -- --check
    cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
    cargo test -p flat-algebraic-attention boolean_physical_protocol
    cargo run -p flat-algebraic-attention --example bps6_physical_trial -- --smoke

The full trial is intentionally **not** part of normal CI and must not
run inside a privileged production process. Its RemoteOps physical job
is bounded, source SHA pinned, locked to the trusted Dell runner,
networkless and unprivileged in the measured container; the host can
read RAPL counters but may not execute checked-out code as root.

The standard FLAT-ATTENTION whole-workspace Rust/WGPU, MAA, CodeQL,
supply-chain and semver gates must all pass on final PR HEAD before merge.

## Future TDI decision

After independent physical qualification, TDI must archive the exact
original timestamps, RAPL sensor domains, rollover handling, counter
capabilities, host load and run ordering, along with paired duration and
energy summaries **including negative outcomes and uncertainties**.

Only an independently calibrated cost model may associate additive
Boolean logical-work counters with joules. No physical claim of
BPS-6 energy superiority, and no FP32 LSTM quality or speed comparison,
is established by this work.

## External primary source

Linux kernel developers, *Power Capping Framework*, energy monitoring
attributes `energy_uj` and `max_energy_range_uj`:
https://github.com/torvalds/linux/blob/master/Documentation/power/powercap/powercap.rst
