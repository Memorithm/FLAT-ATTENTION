# PVP independent re-audit — 7 October 2026

Status: historical-source and offline-evidence audit; **no newly qualified GPU performance result**.

## Outcome and provenance

The published R11 raw reductions are reproducible from the archived logs. The
audit found no discrepancy in the 447 emitted p50/p95 rows: each uses its thirty
raw samples and the frozen nearest-rank indices (15th and 29th sorted samples).
The important limitations concern what was compared, what the timer includes,
and what the correctness checkpoints establish.

Historical execution source:
`ddb6f0144821dfa29d268601e7087e5001278e52`.

Audited archival source:
`bff0fa4f74f86c64073c376a11c60b6d5b227c49`, FLAT-ATTENTION PR #343.

Primary local evidence, authored by Memorithm on 6–7 October 2026:
[original protocol](PVP3D_THREE_CANDIDATE_PREREGISTRATION.md),
[R11 protocol](PVP3D_THOR_R11_TEN_DIAGNOSTIC_PROTOCOL.md),
[campaign report](PVP3D_THOR_REPETITIONS_AND_DELL_2026_10_07.md),
[R11 archive](evidence/pvp3d-thor-repeat-2026-10-06/r11/),
[benchmark source](../../examples/pvp3d_three_candidate_bench.rs), and
[percentile/order helpers](../../examples/support/pvp3d_protocol.rs).

No historical log, rejected panel, failed invocation, or frozen protocol is
rewritten by this audit. No samples are pooled with Dell, R8, or other sources.
The unchanged ten-process battery remains rejected for confirmation:
nine complete processes passed their pre/post whole-state checks; one failed
the pre-timing oracle with 16,320 incorrect words. One measuring-phase
occupancy snapshot was also indeterminate. Accepted confirmation rows remain
zero.

## Recomputed raw effects

For each process and admissible geometry:

```text
reduction_percent = 100 * (1 - candidate_p50_ns / vec4_p50_ns)
```

The reported effect is the median of these process-level reductions. For an
even number of processes, the median averages the two middle reductions.
It is not the ratio of separately aggregated latency medians. Positive values
mean lower latency; negative values mean slower execution. The baseline
column separately reports the median process p50 in microseconds.

| K | G | Emitted cohorts | Median vec4 p50, us | Median fused2 reduction | Median tile8 reduction |
|---:|---:|---:|---:|---:|---:|
| 256 | 128 | 10 | 307.906 | 7.22% | 14.95% |
| 256 | 512 | 10 | 265.231 | 9.50% | 19.60% |
| 256 | 2048 | 10 | 313.268 | 10.01% | 17.75% |
| 1024 | 128 | 10 | 411.281 | 7.72% | 17.02% |
| 1024 | 512 | 10 | 418.870 | 8.00% | 15.28% |
| 1024 | 2048 | 10 | 421.272 | 8.25% | 13.02% |
| 4096 | 128 | 10 | 486.284 | 6.96% | 13.14% |
| 4096 | 512 | 10 | 493.346 | 6.99% | 10.77% |
| 4096 | 2048 | 10 | 551.364 | 6.57% | 2.06% |
| 16384 | 128 | 10 | 564.841 | 5.13% | 9.49% |
| 16384 | 512 | 10 | 634.277 | 5.79% | 2.16% |
| 16384 | 2048 | 10 | 832.831 | 5.01% | -19.25% |
| 65536 | 128 | 10 | 721.637 | 3.86% | 2.86% |
| 65536 | 512 | 10 | 935.808 | 4.11% | -14.20% |
| 262144 | 128 | 9 | 934.627 | 0.81% | -10.23% |

The first fourteen panels include the measurements emitted by the process
that subsequently failed. The last panel has nine cohorts because that
process failed before its K262144/G128 timing. These are retained diagnostics,
not a winner selected by dropping the failure.

Selected process-level reduction ranges retain both tails:

| K | G | Cohorts | fused2 minimum to maximum | tile8 minimum to maximum |
|---:|---:|---:|---:|---:|
| 256 | 512 | 10 | 6.75% to 53.60% | 12.19% to 59.00% |
| 4096 | 128 | 10 | 5.28% to 7.19% | 12.68% to 13.52% |
| 16384 | 2048 | 10 | 4.12% to 5.86% | -21.33% to -2.30% |
| 262144 | 128 | 9 | -2.10% to 1.57% | -13.40% to -4.64% |

At K256/G512, vec4 process p50 ranges from 210.610 to 662.266 us (a factor
of 3.14). The process with the latter baseline has fused2 p50 307.276 us and
tile8 p50 271.517 us. Its 59.00% tile8 reduction is therefore particularly
sensitive to the slower baseline. The median 19.60% is the appropriate
descriptive estimate for this archived panel; it is still unqualified.
A 19.60% latency reduction corresponds to approximately 1.244x speed, not
a 19.60-fold acceleration.

Verification inputs are all ten
`r11/block-{1,2}/run-{1,2,3,4,5}/pvp3d.log` files.
The source helper selects nearest-rank p50/p95, while the comparison above
aggregates ratios across processes. Neither the min/max ranges nor the
previous >=5% engineering rule constitutes a confidence interval or
significance test.

## What the comparison measures

All three candidates already consume the same projected address-major
`[K, ceil(G/128)] vec4<u32>` bitplane matrix. The baseline is PVP2 vec4,
not an object/graph implementation, a dense floating-point model, or the
scalar WGPU candidate. The reductions above are incremental prefix-fusion
effects against an already projected representation. They do not measure
the total architectural benefit of projecting an ANF bank.

The fusion is deliberately limited:

| K | Butterfly stages | vec4 dispatches | fused2 dispatches | tile8 dispatches |
|---:|---:|---:|---:|---:|
| 256 | 8 | 8 | 7 | 6 |
| 262144 | 18 | 18 | 17 | 16 |

Fused2 combines the first two stages in registers and delegates the suffix
from stride four to vec4. Tile8 combines the first three stages in workgroup
memory and delegates the suffix from stride eight. Their elimination of one
or two dispatches is a count of explicit source dispatches, not a measured
kernel-time saving or an upper bound on all future algorithms.

Consequently, a small-panel reduction near 20% does not establish a ceiling
on the complete PVP programme. It also does not establish that broader
fusion will win: register pressure, occupancy, memory layout, barriers and
workgroup count must be measured for every new candidate.

Sources: [vec4 carrier](../../src/pvp_vec4.rs),
[fused2 carrier](../../src/pvp_fused2.rs),
[tile8 carrier](../../src/pvp_tile8.rs), and
[hardware bootstrap](PASCAL_VECTOR_PROJECTION_BOOTSTRAP.md).
The WGSL vector type is a representation/codegen candidate; the current
records do not establish a physical register-allocation claim.

## Repeated CPU preparation and timing scope

Within every timed dispatch call, the current carriers create parameter
bytes, a uniform buffer and bind group for each explicit stage, and encode
a compute pass for each stage. The timer also includes command-encoder
creation, submission and completed device wait. Pipeline creation,
completed source reset and correctness readbacks are outside the timer.

This is the prospectively declared resident wall-latency scope, so these
costs do not make the historical result incorrectly calculated. They do make
it unsuitable for attributing the measured reduction to GPU arithmetic
alone. PVP3d requests no optional device features and records no GPU
timestamps.

A separately preregistered prepared carrier should retain immutable stage
uniforms and bind groups per geometry/state, then compare its unchanged
transform semantics against the historical carrier. A new measurement
should retain wall latency while reporting preparation/encoding and
submit/completion costs separately. Open WGPU timestamp queries may add
device-command duration when supported; absence must be explicit rather
than replaced by wall time. Their instrumentation and readback costs must
remain outside the original timing comparison.

The existing [PVP3b example](../../examples/pvp3b_fused2_bench.rs) already
contains capability-filtered WGPU timestamp-query code. Its existence is a
reusable implementation resource, not physical PVP3d timing evidence.

## Correctness coverage is pre/post, not every measured execution

The historical banc checks all words before warmups and once after the
last measured invocation for each candidate. The final check reads the
actual last measured state, without an intervening reset or replacement
dispatch. This preserves the intended exact checkpoint semantics.

It does not check the intermediate states produced by all thirty
measurements. Given an observed intermittent defect, a passing process
cannot establish that every intermediate execution was exact. No
intermediate error is asserted to have occurred in the passing processes;
it was not observed because these states were not read.

A distinct untimed stress test should check every requested repetition and
retain phase, geometry, candidate and bounded mismatch output. Inserting
these readbacks into the frozen latency banc would change resource reuse,
synchronization and cache behavior, so it cannot be presented as an
unchanged replication of R11.

## Error morphology and localization limits

[R3](evidence/pvp3d-thor-repeat-2026-10-06/r3/mismatch-delta.csv),
[R4](evidence/pvp3d-thor-repeat-2026-10-06/r4/mismatch-delta.csv), and
[R5](evidence/pvp3d-thor-repeat-2026-10-06/r5/mismatch-delta.csv)
show that every recorded wrong word equals the preceding geometry's oracle
word at that index. This proves stale returned content for these failures.
It does not establish where that content became stale.

Additional offline observations from the retained mismatch deltas:

| Campaign | Wrong words | Consecutive wrong-word ranges | Observed granularity |
|---|---:|---:|---|
| R3 | 16864 | 23 | 22 ranges start at an index divisible by 32; 21 lengths are divisible by 32 |
| R4 | 160 | 5 | Five ranges of 32 u32 words, each starting on a 128-byte boundary |
| R5 | 272 | 9 | Eight 32-word ranges and one 16-word range; starts divisible by 32 words |
| R6 | 41080 | 29 | Every recorded incorrect actual word is zero |
| R7 | 2384 | 10 | Every incorrect actual word is zero; all range starts and lengths align to 64 bytes |

The byte alignment is an observation about recorded indices, not a
measurement of the GPU or CPU cache-line size. A cache/coherence or transfer
granularity explanation is a hypothesis.

For the subset-zeta butterfly, address zero is never a destination. Its
row must therefore equal the initial fixture. The R3/R11 first mismatches
include that invariant row; R6/R7 also corrupt the address-zero row.
These observations make a problem solely in arithmetic updates of
destination addresses insufficient to explain the returned arrays. Source
upload, reset/copy, output copy, mapping/coherence, resource reuse and
backend/driver behavior still require discrimination.

R11 prints only the first eight wrong tuples; those values agree with R3.
Its other 16,312 wrong actual values were counted but not printed. Do not
extend the previous-oracle equality claim to all R11 wrong words.

## Upstream mapping-barrier hypothesis

Primary upstream source: **andyleiserson, 25 March 2026**,
[wgpu issue #9306](https://github.com/gfx-rs/wgpu/issues/9306), opened with
correctness/bug labels and still open when checked on 7 October 2026.
The issue asks whether mapping needs a preceding barrier and explicitly
questions backend automatic synchronization guarantees.

The locked [Cargo dependencies](../../Cargo.lock) use wgpu-core 30.0.1.
That release's official
[`resource.rs` source](https://docs.rs/wgpu-core/30.0.1/src/wgpu_core/resource.rs.html#808-818)
contains the issue-linked TODO in `try_map_async`, which sets the tracker
to the host-map use while noting that the transition is ignored.

This is a concrete upstream reason to test mapping transitions. It is not
proof that #9306 causes Thor's failures, a driver attribution, or evidence
that an explicit transition resolves them.

## New untimed diagnostic and its limits

The separate [paired diagnostic](../../examples/pvp_stage_diagnostic.rs),
under its [prospective protocol](PVP_READBACK_ISOLATION_PROTOCOL.md),
tests source-after-upload, state-after-reset and state-after-complete-transform
against the host oracle, using two fresh readbacks for each observation.
Its fixed two-geometry transition preserves equal 4 MiB allocation sizes:
K65536/G512 followed by K262144/G128, for twelve rounds and all three
candidates. The control mode uses normal mapping; the second mode adds an
explicit MAP_READ transition to both readbacks before submission.

A disagreement between the two readbacks is evidence of an inconsistent
observation from one source/submission; agreement does not prove the device
state is correct, because both paths may share a defect. Full word equality,
not only checksums, remains the pass condition.

Important scope limits:

- Reading source before reset adds a submission and wait absent at that point
  in the historical sequence. Success cannot resolve the historical failure.
- The diagnostic checks three phases, not every individual butterfly stage.
- It does not cover the K16384/G2048 failures from R6/R7.
- Fixtures differ between the two geometries but repeat across rounds;
  an identically stale previous instance of the same geometry can be invisible.
- It is untimed; neither mode establishes a speedup or qualification.
- A phase failure localizes the earliest failed observation, not necessarily
  the earliest device operation that failed.

Physical executions need source/binary identity, retained RemoteOps
reservation/occupancy records and restoration evidence. Different modes
and subsequent hypotheses remain separate experiments. A missing adapter,
unsupported geometry or failed oracle must be explicit, never silently
shrunk or replaced with CPU performance evidence.

## Next evidence gates

1. Run the untimed phase diagnostic, retaining both mapping modes and all
   failures; preserve the unchanged historical reproduction separately.
2. If readbacks disagree or invariant rows fail, add a discriminating device
   probe/nonce or independent observation before attributing the cause.
3. Isolate any earliest failing transfer or transform stage; retain a
   per-repetition stress test with full oracle equality.
4. Only after the correctness question is closed, compare the prepared
   carrier and additional fusion candidates in a newly frozen protocol.
5. Evaluate architectural benefit against a declared reference and include
   projection/setup costs for the intended SML ANF workload.

FLAT owns carrier realization and qualification. SML owns ANF semantics,
model-aligned interpretation and eventual self-sufficiency. SciRust and NNIS
may reuse proven preparation, observation and harness contracts after
destination review; no mandatory model dependency, default route,
autotuning policy, throughput or quality claim follows from this audit.
