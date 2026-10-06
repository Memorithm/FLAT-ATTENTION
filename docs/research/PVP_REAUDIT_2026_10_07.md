# PVP independent re-audit — 7 October 2026

Status: source/evidence audit, native readback controls and packed-address ANF
correctness; **no newly qualified GPU performance result**.

The new paired-copy diagnostic reproduces automatic-readback disagreement in
all three automatic processes; three explicit-transition processes pass.
The separate full-grid follow-up passes pre/post oracle checks in ten of ten
processes. The new packed-address candidate passes 39 ANF cases and their
inverses on both software Vulkan and native Thor. These results are detailed
below and remain separate from historical timing evidence.

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

## Evidence gates defined before the controls below

This was the initial investigation sequence. The paired-readback diagnostic
and full-grid control are now completed and reported in the following sections;
broader stress, prepared-plan timing and model-aligned evidence remain open.

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

## Physical paired-readback result — 7 October 2026

This subsequent experiment ran on Thor's NVIDIA Tegra NVIDIA Thor adapter,
Vulkan driver 580.00. It completed six fresh processes in the prospectively
declared automatic/explicit alternating order. The two modes used the same
source and binary. These are correctness diagnostics with no latency samples
or performance admission.

Recorded identities:

- Diagnostic source: `f913d53748edd40ebcb696395aad6a8b9a61893e`.
- Diagnostic binary SHA256:
  `ae21676eed86e25c919dd7171aed7188feff65e5e5a024fc3a0a5418cbb7774c`.
- Protocol preregistration:
  `0b7563abcd93878ada12b716878fd26315089cf2`.
- Controller preregistration:
  `29df571ff9b2ebb7ec33ff28f108de650a6b116c`.
- Host evidence directory: `/var/tmp/remoteops-pvp-audit.2ayxqncf`.
- Execution interval: 2026-10-06 23:07:50.388997375 to
  23:08:39.779821894 UTC, i.e. 7 October in Europe/Paris.

The [published diagnostic archive](evidence/pvp-readback-isolation-2026-10-07/)
retains this experiment separately from the historical R11 record. The
independent offline audit read all six complete logs and verified their
504 comparisons per process, terminal failed-comparison counts and matching
controller exits. Each process has 168 observations; each observation
compares A to oracle, B to oracle, and A to B.

| Process | Mode | Exit | Comparisons | Failed comparisons | Observations with A/B disagreement |
|---:|---|---:|---:|---:|---:|
| 1 | Automatic mapping | 1 | 504 | 2 | 1 |
| 2 | Explicit MAP_READ | 0 | 504 | 0 | 0 |
| 3 | Automatic mapping | 1 | 504 | 210 | 105 |
| 4 | Explicit MAP_READ | 0 | 504 | 0 | 0 |
| 5 | Automatic mapping | 1 | 504 | 144 | 72 |
| 6 | Explicit MAP_READ | 0 | 504 | 0 | 0 |

All 178 disagreeing observations in automatic mode have the same asymmetry:
**A differs from the oracle; B matches it exactly.** The A/oracle and A/B
mismatch counts also agree in every observation. Consequently, 356 failed
comparison rows are 178 observations counted twice, not 356 independent
failures. Across the three automatic processes, all 504 B/oracle comparisons
pass. Both A/oracle and B/oracle pass in every observation of all three
explicit-transition processes.

Automatic-mode disagreements by phase:

| Phase | Process 1 | Process 3 | Process 5 | Total disagreeing observations |
|---|---:|---:|---:|---:|
| Source, no compute shader | 0 | 17 | 3 | 20 |
| State after reset | 0 | 35 | 29 | 64 |
| State after full transform | 1 | 53 | 40 | 94 |
| All phases | 1 | 105 | 72 | 178 |

The cleanest isolated failure is process 1, round 10 (zero-based),
K262144/G128, tile8 transformed phase: A has 32 wrong words, B has zero,
and A/B differs at those same 32 words. The first reported index is 11648.
This discrepancy occurs between copies/readbacks of one unchanged source in
one submission. A deterministic arithmetic error producing one wrong device
state cannot by itself explain these different observations.

Process 5's first failure is round 3, K65536/G512, fused2 reset. Its eight
reported wrong A values at indices 16–23 equal the independently recomputed
*transformed* fixture, while the reset oracle expects the *initial* fixture;
B is exact. This is additional phase-stale returned content. The comparison
was recomputed from the frozen xorshift fixture and scalar subset-zeta
network. Only those reported tuples support this particular equality claim.

The source-phase failures require no compute shader at all. Together with
A/B disagreement they narrow the investigation toward upload visibility,
copies, readback mapping/coherence and resource reuse. They do not identify
which operation or software layer is responsible. In particular, two-copy
disagreement is not a standalone proof of a CPU-cache fault, a driver fault,
or upstream issue #9306 causality.

The explicit MAP_READ transition is now supported as a **candidate workaround
by this small controlled diagnostic**: three automatic processes fail and
three explicit processes pass. The successful explicit processes each contain
twelve rounds and 504 exact comparisons. This does not establish permanent
elimination of the intermittent failure, unchanged historical reproduction,
coverage of the R6/R7 geometry, or any speedup. A new complete stress and
full historical-grid investigation remain separate evidence gates.

The controller retained incomplete occupancy observations for processes
3, 5 and 6. They remain unknown for admission, including the otherwise exact
explicit process 6; no accepted timing rows follow. The archived restoration
snapshot at 23:08:40.609181601 UTC records all four services active, with
their original Restart policies and RefuseManualStart=no. The controller
records `reservation_released_exit=0`. Unknown observations and failed
automatic processes are not dropped.

Verification sources are the six `block-1/run-N/diagnostic.log` files,
`block-1/control.log`, source/protocol/controller revision receipts,
binary identity before/after, occupancy records, and
`block-1/restored-services.txt` in the
[published diagnostic archive](evidence/pvp-readback-isolation-2026-10-07/).
The separate complete-grid follow-up uses
[its own archive](evidence/pvp-readback-full-grid-2026-10-07/); its results
are not included in the six-process counts above.

## Full-grid explicit-readback control — 7 October 2026

The separately preregistered follow-up completed **ten of ten fresh processes**
with exact whole-state pre/post oracle checks. Every process exited zero.
This is evidence for the explicit MAP_READ transition on the tested native
Vulkan workload, not proof of an upstream root cause or permanent elimination
of all intermittent readback errors.

Recorded identities:

- Execution source: `2fc5ad2ca4f5c27e2b1fdbb0ed328e6820e273af`.
- Protocol source: the same commit, recorded before execution.
- Frozen controller: `a3be4b8ddc2cb0462117f763965415b83824989d`.
- Benchmark binary SHA256:
  `4cc80be9dbafc4084c060099138f1eb07b7ae422ecc88dc1528cc5347ef6a951`.
- Physical host directory: `/var/tmp/remoteops-pvp-audit-full.jyalp5pb`.
- Mode: `--measure-map-transition`; historical `--measure` is unchanged.

The [full-grid archive](evidence/pvp-readback-full-grid-2026-10-07/)
contains all ten logs, including rejected geometries and incomplete observation
records. The grid yielded 15 measured geometries and 6 capability-rejected
geometries per process: **450 measured rows, 180 rejected rows and 13,500 raw
timing samples**. Independent offline reduction verified every emitted p50/p95
against its thirty samples and the frozen nearest-rank helper.

This campaign deliberately produces **zero accepted performance rows**.
The controller additionally retained incomplete occupancy observations in
runs 1, 2 and 5. These remain unknown; passing the numerical checks does not
convert them into accepted occupancy. The other observations do not change
the campaign's diagnostic-only status. Pre/post checks cover the observed
states, not every one of the thirty intermediate timed states.

For transparency, the following raw effects use the same median of
process-level latency ratios defined earlier. They are not pooled with R11,
the paired-readback experiment, Dell or any other campaign.

| K | G | Processes | Median vec4 p50, us | Median fused2 reduction | Median tile8 reduction |
|---:|---:|---:|---:|---:|---:|
| 256 | 128 | 10 | 236.346 | 9.39% | 19.01% |
| 256 | 512 | 10 | 259.106 | 9.28% | 19.20% |
| 256 | 2048 | 10 | 285.151 | 9.89% | 17.66% |
| 1024 | 128 | 10 | 390.619 | 7.84% | 17.07% |
| 1024 | 512 | 10 | 415.776 | 8.13% | 15.62% |
| 1024 | 2048 | 10 | 420.887 | 8.27% | 12.65% |
| 4096 | 128 | 10 | 487.207 | 6.80% | 13.07% |
| 4096 | 512 | 10 | 494.707 | 7.13% | 10.79% |
| 4096 | 2048 | 10 | 554.011 | 6.84% | 1.90% |
| 16384 | 128 | 10 | 569.535 | 5.57% | 9.45% |
| 16384 | 512 | 10 | 638.029 | 5.88% | 2.26% |
| 16384 | 2048 | 10 | 841.779 | 5.02% | -18.61% |
| 65536 | 128 | 10 | 725.738 | 3.53% | 2.18% |
| 65536 | 512 | 10 | 924.951 | 4.06% | -15.22% |
| 262144 | 128 | 10 | 929.391 | 0.63% | -10.53% |

These results reproduce the old pattern: approximately 19% raw tile8 latency
reduction on small panels, falling gains and eventual tile8 regressions as
the geometry grows. They compare the existing three address-major kernels;
**they do not measure the new packed-address kernel**. The readback transition
lies outside the timer interval, but the timer still includes command encoding,
submission and completion. The old baseline still constructs per-stage
uniforms/bind groups. No architectural, GPU-only or end-to-end SML gain follows.

## Packed-address ANF correctness on native Thor

The same independently reserved session subsequently ran the separate
`pvp_wgpu_packed` functional test, using an immutable binary copied before
execution. The actual adapter reports NVIDIA Tegra NVIDIA Thor, IntegratedGpu,
Vulkan; this is native device correctness evidence, separate from lavapipe.

- Source: `2fc5ad2ca4f5c27e2b1fdbb0ed328e6820e273af`.
- Test binary SHA256:
  `ef351a581fe653abb3dd6e3f33729c580fabcb52afce1b80e59bdc9e8c39b0df`.
- Result: **39 direct-ANF truth-table comparisons and 39 inverse comparisons
  pass**, with padding checks; test exit zero and
  `PVP_PACKED_COMPLETE cases=39` retained in the log.
- Earlier software Vulkan execution passes the same 39-case corpus. It is
  portability/correctness evidence only.

The candidate uses one `vec4<u32>` to carry 128 addresses of one Boolean
gate. For K >= 128, seven low-address Pascal/subset-zeta stages execute within
local values before any cross-vector suffix. K < 128 executes only its actual
stages and keeps padded addresses zero. This changes the carrier compared with
the old layout, which packs 128 gates at one address. It is a concrete
implementation of additional fusion, not just a planned dispatch reduction.

The prepared plan retains state, immutable per-stage uniforms and bind groups;
`encode` allocates no new buffers or bind groups. Its suffix pairs were also
independently checked for bounds, same-gate pairing, unique destinations and
dispatch coverage over 1,515,269 pairs. No native latency measurement of this
candidate has been made. Padding can increase physical storage for small K,
and conversion/preparation costs must be charged according to the declared
resident or end-to-end boundary. The next performance protocol must compare
prepared plans on both sides and retain a gate-major reference, rather than
crediting removal of baseline setup costs to the mathematics alone.

The GPU corpus reaches K=2048; it does not yet qualify this new carrier at
the historical K=262144 failure geometry. Seven local-value stages do not
establish seven stages in physical registers: register allocation and possible
spills remain unmeasured. Preparing a caller-supplied device buffer validates
geometry, size and usage, not its contents; canonical padding requires upload
from the validated host constructors or equivalent caller validation.

## Restoration and remaining evidence gates

The full-grid controller records reservation release with exit zero. Its
restoration snapshot and the final independent inspection at
2026-10-06 23:21:28.306112 UTC confirm all four services active, original
Restart policies restored, RefuseManualStart=no, temporary restoration timers
and trace instances removed, and the cooperative lock available. The shared
historical benchmark executable was restored to its original SHA256
`7aef67026ad088504478885cf9f2b7c864218f6fb3b0792bd8609fd907d3f414`;
the binaries actually executed in this follow-up remain immutable in its
separate build directory.

The [independent inspection receipt](evidence/pvp-readback-full-grid-2026-10-07/final-restoration-inspection.json)
retains these exact service, controller, timer, guard, trace, lock and historical
binary observations. The [inspection source](evidence/pvp-readback-full-grid-2026-10-07/post-inspection-source.txt)
performs read-only host checks; saving its receipt creates no GPU qualification.

The observation asymmetry and explicit-transition control justify further
investigation of native transfer/readback visibility. They do not prove the
driver or upstream issue #9306 is responsible. Broader stress, affected
historical geometries and other physical adapters remain to be checked.
Prepared-plan throughput, GPU timestamps where supported, paid conversion
costs and a declared SML ANF reference remain separate speed-evidence gates.
FLAT is not promoted to a required SML dependency by these results.
