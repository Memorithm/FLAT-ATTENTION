# BPS-7 — Direct rising-front-index execution with sparse query readout

**Research only.** This milestone is a genuinely event-indexed input pathway
for BPS-3's exact Boolean recurrence. It neither reproduces generic
Transformer attention nor claims novel spiking neural networks, faster
wall-clock inference or reduced joules.

Implementation:
- `crates/flat-algebraic-attention/src/boolean_indexed_fronts.rs`
- `crates/flat-algebraic-attention/examples/bps7_indexed_front_smoke.rs`

## Two distinct input contracts

BPS-3 accepts sampled 64-bit words and must detect rising fronts on **all**
sample ticks, including those containing no front. Sampling takes O(T)
packed-word work regardless of event density.

BPS-7 adds a **different** source contract:

    IndexedFront { tick: u64, rising: u64 }

Each entry is the complete, nonzero mask of the 0→1 transitions at a
strictly increasing 1-based tick. The upstream source must already have
identified **all** rising fronts. **Missing timestamps do not mean a
fall, a sample of zero, a re-arming event, or no unknown event.**

The new `BooleanEdgeProgram::forward_indexed_queries` accepts a
bounded `horizon`, a sorted, strictly increasing front sequence,
sorted requested observation ticks and an initial recurrent state.

- Every front at time t updates memory **before** a query at t.
- Quiet intervals are exact identity for BPS-3.
- The static readout mask applies only when an output is requested.
- An event after the last query still updates the returned final state
  at the end of the declared horizon.
- Zero events, zero queries, zero horizon, sparse million-tick gaps
  and nonzero initial memory are handled explicitly.
- Zero front masks, tick zero, nonmonotonic/duplicate timestamps,
  and events/queries beyond the horizon fail closed.

For E supplied front timestamps and Q requested observations, the
algorithm performs **O(E+Q) packed-word recurrence/query operations**
and allocates O(Q) output words, independent of the numeric horizon.
This bound **excludes** any cost to generate/authenticate the upstream
front stream and does not imply lower physical latency or energy.

## Source equivalence certification

`sampled_indexed_fronts` is a convenience reference converter from
original sampled BPS-3 input words and their explicit baseline. It costs
O(T) to find the fronts.

`certify_indexed_fronts_against_samples` independently checks that
the provided indexed feed contains **exactly** all rising bits of those
samples, including held-high baseline, falling/rearming and simultaneous
multi-bit fronts. It detects missing, fabricated and bit-altered events.
It also costs O(T+E), so calling it in the inference critical path would
erase the event-only input-work advantage.

Syntactic validation of an external feed (nonzero bit masks, ordered
timestamps and bounds) does **not** establish that it is trustworthy or
complete. In a real integration, this requires an upstream publisher with
a frozen event definition, provenance and completeness/sequence contract.
Without source authentication, BPS-7 results describe the **declared
front sequence**, not necessarily actual sampled or physical input.

## Oracle qualification

The module tests all-tick and sparse-query exact equivalence against
`BooleanEdgeProgram::forward_dense` and the event-only temporal scan,
including deterministic randomized Boolean gates, readout masks,
rotations, nonzero baselines, falling/rearming, zero-event trajectories,
events between queries and post-query events.

It explicitly checks that the returned full recurrent state is
independent of the readout mask and includes any tail events.

    cargo fmt --all -- --check
    cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
    cargo test -p flat-algebraic-attention boolean_indexed_fronts
    cargo run -p flat-algebraic-attention --example bps7_indexed_front_smoke

The smoke example deliberately constructs a sparse sampled sequence,
converts and certifies it **offline**, then queries only 128 of 4096
output ticks. Since its event producer inspected all 4096 sampled
words, it is a **correctness benchmark, not an end-to-end efficiency
claim**. It prints `physical_energy_measured=false`.

## BPS-8 stateful windowed execution

[BPS-8 watermarked streaming](BPS8_WATERMARKED_STREAM.md) adds a
transactional incremental counterpart to BPS-7's whole-history indexed
query API: closed non-overlapping event windows, sparse query readouts,
constant persistent state, and no retained previous front vectors.
A source watermark is **not** authenticated evidence that the publisher
delivered every real event; offline sampled-oracle certification remains
a separate O(T) qualification task.

## Integration and next gates

Possible consumers are source-authenticated asynchronous signals for
V888-BOOLEEN and SML-GENIUS, without importing their production runtime
or changing their semantics. A separate versioned adapter is required
before any cross-repository deployment.

BPS-6's trusted RemoteOps RAPL physical protocol is unaffected by this
change; its pinned SHA and exact original BPS-6 modules remain intact.
BPS-7 is not admitted as a physically measured energy optimization.
Any future comparison must match observable outputs, event publisher
costs, background load and target CPU counters.

Do not infer SNN novelty, memory quality, superiority to FP32 LSTMs,
or Transformer-level end-to-end gains from the reduced counted
recurrence alone.
