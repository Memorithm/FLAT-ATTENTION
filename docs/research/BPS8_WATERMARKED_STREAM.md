# BPS-8 — Transactional watermarked rising-front streaming

**Status:** research-only Rust implementation; no default attention-backend
promotion, device-energy claim, or direct model-training claim.

Files:
- `crates/flat-algebraic-attention/src/boolean_watermarked_stream.rs`
- `crates/flat-algebraic-attention/examples/bps8_watermarked_stream_smoke.rs`

BPS-7 takes an entire indexed event feed and full requested-query sequence.
BPS-8 instead owns a persistent 64-bit recurrent state and receives
**successive, non-overlapping, source-declared complete windows**. It does
not keep the earlier front vectors or earlier query responses.

## Contract: closed watermarks and input completeness

`WatermarkedEdgeRuntime::new(program, baseline_tick, initial_state)`
declares that the caller has already accounted for every event through
`baseline_tick`, including any pre-baseline state. The runtime does not
reconstruct an initial state from missing source history.

Each call:

    advance_window(watermark, fronts, queries)

submits every nonzero front in the interval
`(previous_watermark, watermark]`, plus optional query ticks within that
same interval. All ticks must increase strictly **within each input list**.
The output associated with a query at the same tick as an event includes
that event's Boolean recurrence update. Events after the final query
still update persistent state through the watermark.

Once a watermark is accepted, a later query or event at or before it is
**rejected**. This prevents implicit history rewrites and out-of-order
data from being silently accepted.

The runtime checks ordering, duplicate ticks, out-of-range data, zero
front masks, stale queries/events, and cumulative counter overflow.
It validates the **entire batch before mutating** persistent state. A
later malformed event or query cannot leave a partially advanced
state. Windows containing no queries and/or no events are valid; the
watermark must nevertheless advance strictly.

**Fundamental caveat:** a watermark is an assertion by the upstream event
publisher that the interval is complete. No code can infer an omitted
front from its absence in a sparse feed. Therefore BPS-8's local
validation does not prove physical-source authenticity, missed-event
freedom or lossless input capture. Its correctness against raw samples
requires a separately complete, observed sample stream and comparison
through BPS-7's `certify_indexed_fronts_against_samples`.

A deterministic negative-control test deliberately omits a legitimate
event: the watermarked runtime accepts the syntactically correct but
untruthful feed and diverges from the independent dense oracle, whereas
sampled-source certification rejects it. This distinguishes code
correctness from source-completeness claims.

## Execution and memory bounds

For successive windows containing total E indexed events and Q explicit
queries, runtime work is O(E+Q) packed-word steps, excluding source
acquisition, front encoding, validation of any original raw stream,
thread scheduling, and requested output transport. The persistent
runtime keeps one packed state word, the previous watermark and a
constant number of counters, i.e. O(1) persistent memory.

A window allocates O(Q_window) output words and consumes
caller-owned O(E_window+Q_window) input slices. Streaming removes the
need for BPS-7's retained **whole-history** event and query vectors
if a publisher already supplies bounded batches. It does not by itself
bound a malicious caller's submitted *single* window allocation; a
transport/adapter must enforce batch limits before submitting it.

If the publisher instead scans all T raw input samples to find fronts,
the **end-to-end input processing remains at least O(T)**. There is no
experimentally established latency advantage, energy saving or
FP32/Transformer quality improvement here. The BPS-6 physical
RemoteOps job is unaffected and is pinned to its earlier reviewed
source commit; it must not automatically execute BPS-8 as a benchmark.

## Deterministic qualification

Comparisons include:
- bit-identical outputs against BPS-3's dense sampled oracle and
  BPS-7's complete indexed-query path over arbitrary recurrent Boolean
  gate masks, readout masks, nonzero baseline states and shift choices;
- irregular windows, multiple query/front events, same-tick order,
  zero-event windows, huge tick gaps and `u64::MAX` watermark;
- events occurring after the last requested query;
- late, nonmonotone, duplicated, zero-mask and over-horizon input
  rejection with transactional **whole-state** equivalence on error;
- explicit missed-event negative control plus independent sampled
  certification that detects it.

Run the exact-head full workspace gates and focused tests:

    cargo fmt --all -- --check
    cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
    cargo test -p flat-algebraic-attention boolean_watermarked_stream
    cargo run -p flat-algebraic-attention --example bps8_watermarked_stream_smoke

The smoke partitions 4096 reference samples into 16 closed windows
and requests only sparse outputs. It retains no previous window events
inside the runtime but independently constructs and certifies the
sample-derived event feed for correctness. It reports no physical energy.

## BPS-9 sequence and epoch admission

[BPS-9 sequenced front frames](BPS9_SEQUENCED_FRONTS.md) add a
source-bound epoch, gap-free frame sequence, predecessor watermark and
deterministic content/chain checksum to BPS-8. Invalid frames put the
session in quarantine without mutating the last accepted Boolean state.
A contiguous sampled-word reference publisher proves equivalence to
the samples provided and correctly handles cross-frame re-arming.
The checksum is **not** a cryptographic signature, and upstream
source completeness/authentication remains a separate production gate.

## Next source-authenticated integration gate

Before use in a live V888/SML/FLAT control loop, a separate versioned
publisher adapter must bind an authenticated monotonically sequenced
stream to source-level completeness and epoch provenance, handle loss,
restart and duplication, and refuse uncertain watermarks.
Integration requires independent destination repository tests and
RemoteOps-only hardware orchestration. No such publisher is implemented
or implied by this milestone.
