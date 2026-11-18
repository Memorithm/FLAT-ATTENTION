# BPS-13 — Work-efficient parallel scan for recurrently connected Boolean neurons

**Research only.** This change extends BPS-11/BPS-12 from 64 independent
memory lanes to the **cross-lane recurrent connections** already defined
by BPS-2 and BPS-3. It does not replace the public FLAT attention API or
change existing training, source authentication or production execution.

Code: `crates/flat-algebraic-attention/src/boolean_parallel_rotating_scan.rs`.
Smoke: `crates/flat-algebraic-attention/examples/bps13_coupled_parallel_smoke.rs`.

## Behavioral advance

BPS-2 composes maps of the form

```text
R(keep,set,shift)(state) = (keep & rotate_left(state,shift)) | set
```

Two time-ordered rotating maps compose into a single rotating map.
This algebra is associative, **not commutative**; the order of the
forget/write maps and rotating predecessor connections is retained.
The existing BPS-2/BPS-3 implementation used an inclusive Hillis-Steele
scan requiring O(N log N) transition compositions.

BPS-13 introduces `scan_rotating_states_parallel`, an identity-padded
Blelloch prefix scan that uses O(N) transition work, O(N) temporary
memory, and O(log N) theoretical dependency depth. It schedules at
most eight Rust scoped workers on independent tree blocks. Small
stages remain sequential to limit thread overhead. Requests for zero
workers use one; requests above eight are clamped to eight.

New APIs:
- `BooleanRotatingProgram::forward_scan_parallel`: preserves BPS-2's
  input-dependent output mask and genuinely recurrent cross-lane shifts.
- `BooleanEdgeProgram::forward_scan_parallel`: scans all sampled ticks
  with identity steps for absent rising fronts, matching BPS-3.
- `BooleanEdgeProgram::forward_event_scan_parallel`: first detects
  rising edges, then scans only E eventful transitions (O(E) scan
  compositions) and reconstructs T requested outputs.

An event-free window retains the precise initial state. A held-high
input generates no repeated event; a later fall and re-arm produces
a new rising event. Nonzero input baselines and arbitrary shifts in
0..64 are supported. The actual event detector must still inspect all
T supplied input samples; output reconstruction costs O(T).
Thus event-sparse scanning is not proof of O(E) total acquisition cost.

## Exact-head checks

```bash
cargo fmt --all -- --check
cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
cargo test -p flat-algebraic-attention boolean_parallel_rotating_scan
cargo run -p flat-algebraic-attention --example bps13_coupled_parallel_smoke
```

The deterministic test oracle checks dense serial BPS-2/BPS-3 output,
the existing Hillis-Steele scan, varying input baselines, mixed
rotation shifts, random 64-bit gates, irregular lengths including
2,048+ timesteps, and workers 0/1/2/4/8/64. Neither a faster
wall-clock runtime nor a Joule improvement has been measured.

Production routing, BPS-10 authenticated frame semantics and
External Memorithm/RemoteOps power measurement remain untouched.
Qualification on the Dell T430 must independently compare serial,
old scan and BPS-13 at the same machine workload and resource budget.
