# BPS-11: work-efficient Boolean temporal scan

**Status: research-only.** Implemented in
`crates/flat-algebraic-attention/src/boolean_work_efficient_scan.rs`.
This improves the algorithmic work of the BPS-1 Boolean recurrent-memory
prefix-scan without changing neuron, state, gate, or output semantics.
It does not change `flat_attention::api::v1` or promote a production route.

## Exact same recurrence

For one input bit (64 independent lanes are packed into each `u64`):

```text
c[t] = (f(x[t]) & c[t-1]) | w(x[t])
h[t] = c[t] & o(x[t])
```

A transition `(F,W)` maps `c` to `(F & c) | W`. The ordered
composition of A followed by B is `(B.F & A.F, B.W | (B.F & A.W))`.
This operator is associative **but not commutative**.

The prior Hillis-Steele scan does O(T log T) Boolean-transition
compositions and has O(log T) dependency depth. BPS-11 adds an
identity-padded Blelloch up-sweep/down-sweep, with an exclusive prefix
converted to an inclusive prefix for each input. If
`P = next_power_of_two(T)`, it performs `2(P-1)+T` compositions and
stores `P` transitions. Since `P < 2T` for T>0, the algorithm uses
**O(T) work, O(T) memory and O(log T) dependency depth**. The identity
padding keeps irregular sequence lengths exact. Empty sequences return
empty outputs; representational padding overflow falls back to the
exact serial reference.

The current new implementation executes each tree level **serially**.
It is designed to allow independent subtrees to be parallelized later
under physical measurement. Reduced abstract work does not establish
less latency, power, or joules. It is not a classical LSTM and it does
not train the cross-lane recurrent connections.

## Deterministic qualification

The tests compare the exact outputs with both the serial reference and
the existing Hillis-Steele scan, including empty, irregular and
power-of-two sequence lengths, random truth-table masks, nonzero
initial memory, boundary bit lanes, and order-sensitive overwrite
transitions. One separate standalone smoke prints only correctness.

```bash
cargo fmt --all -- --check
cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
cargo test -p flat-algebraic-attention boolean_work_efficient_scan
cargo run -p flat-algebraic-attention --example bps11_work_efficient_scan_smoke
```

Normal exact-HEAD project gates remain required before merging. A future
performance claim must compare serial, Hillis-Steele and Blelloch scans
under identical machine, workload, affinity and energy measurement.

## Antminer AL1 hardware boundary

The **Bitmain Antminer AL1** advertises 15.6 TH/s on BLAKE3 for Alephium
at 3,510 W nominal, according to the Bitmain manufacturer specifications:
https://support.bitmain.com/hc/en-us/articles/35412695385881-ALPH-Miner-AL1-Specifications

Alephium's documented PoW uses **double unkeyed BLAKE3** over a
serialized block header:
https://docs.alephium.org/integration/mining/

BPS-10 separately uses **single keyed BLAKE3 MACs** with arbitrary
versioned-frame and checkpoint contents:
https://github.com/Memorithm/FLAT-ATTENTION/blob/main/docs/research/BPS10_AUTHENTICATED_STREAM.md

The AL1's mining throughput is not a measure of programmable Boolean
LSTM operations and cannot be transferred to keyed MAC throughput.
No verified AL1 firmware or host interface is known in this project
to expose arbitrary-message keyed BLAKE3 or user-defined Boolean gate
and recurrent-state computation. Neither direct neural acceleration
nor a BPS-10 cryptographic hardware adapter is implemented here.
Evaluate **only** after obtaining a documented, testable arbitrary-job
interface. Do not flash unofficial firmware or send secret keys to
untrusted ASIC firmware merely to attempt reuse.

Further work: parallelize the tree stages with bounded CPU threads,
benchmark against existing scan and serial baseline, and investigate
hardware acceleration independently of this Boolean runtime.
