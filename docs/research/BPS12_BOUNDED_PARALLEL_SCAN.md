# BPS-12: bounded threaded execution of packed Boolean temporal scans

**Research-only. No production routing promotion.**

Source: `crates/flat-algebraic-attention/src/boolean_parallel_scan.rs`.
Executable: `crates/flat-algebraic-attention/examples/bps12_bounded_parallel_smoke.rs`.

The input-only Boolean memory of BPS-1 admits an associative transition
composition. BPS-11 introduced a Blelloch prefix scan with O(T) transition
work. BPS-12 adds an implementation using **real Rust scoped CPU threads**
at eligible scan levels. It preserves the BPS-1 recurrence exactly, with
64 packed independent one-bit lanes per `u64` input word.

At each upsweep or downsweep stage, a disjoint tree block accesses only
its two end nodes. The implementation slices the transition vector at
integral block boundaries and schedules at most 8 scoped workers.
No sharing, locks, unsafe code, background daemon, or external service
is introduced. When the tree has fewer than 2048 nodes or fewer than
four blocks per requested worker, the stage runs serially. The
`requested_workers` argument is bounded to [1,8]; zero means one.

Complexity: O(T) transition operations, O(T) temporary storage and
O(log T) theoretical dependency depth. Actual parallel throughput
depends on workload size, CPU layout, scheduling overhead, NUMA and
memory traffic. There is **no measured CPU speedup or energy reduction**.
The algorithm retains all sampled inputs; it does not claim sparse
event-driven avoidance or equivalent classical LSTM accuracy.

## Exact-head qualification

The test suite includes:
- Deterministic randomized gate tables and nonzero initial memory;
- empty, power-of-two, non-power-of-two, just-over-boundary and large streams;
- worker requests 0, 1, 2, 3, 4, 8, 32, with clamp behavior;
- ordered set/clear gate updates (composition is not commutative);
- stage-by-stage equality of explicitly threaded and serial tree stages;
- direct comparison with serial recurrence and BPS-11 reference scan.

```bash
cargo fmt --all -- --check
cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
cargo test -p flat-algebraic-attention boolean_parallel_scan
cargo run -p flat-algebraic-attention --example bps12_bounded_parallel_smoke
```

Ordinary full workspace, host-only oracle, WGPU, SemVer, supply-chain
and CodeQL checks on the exact HEAD remain merge gates.

Next step after correctness is physical Dell T430 CPU-only qualification
through **Memorithm/RemoteOps** (and only RemoteOps): compare BPS-1 serial,
BPS-11 Blelloch serial, and BPS-12 bounded workers on identical retained
inputs while measuring process isolation, p50/p95 latency, throughput,
actual physical energy (if a calibrated meter is accessible), and NUMA
placement. Record unfavorable outcomes and idle power. Do not
reconfigure production services or change CPU policy in this research
PR. Do not transplant BPS-12 to a classical LSTM without proving
associativity under its different recurrent gates.
