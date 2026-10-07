# PVP canonical end-to-end study — prospective protocol

Status: prospective bounded native diagnostic with confirmation gates. Source and all harness/controller bytes must be frozen before native timing.

## Scope and fixed hypothesis

Compare existing prepared vec4 and packed7 implementations on Thor Vulkan. Candidate packed7 was selected from the earlier resident diagnostic; this new cohort must not be pooled with that selection evidence.

Input and output are identical canonical gate-major u64 banks. This is an explicitly declared API endpoint, not an assertion that every SML caller uses this representation. Use the existing scalar input adapters and the newly exact scalar vec4 reverse adapter. Adapter costs are real costs of these implementations; they do not establish an optimal SIMD-transpose baseline. Hardware-register allocation, model quality and default routing remain unclaimed.

The hot kernels and historical frozen binaries remain unchanged.

## Fixed panel

Ordered geometries: (4096,128), (16384,2048), (65536,512), (262144,128), (16384,129), (65536,129).
Banks: boundary, sparse4, dense32, same frozen direct-monomial reference; fixture round = trial % 3.
Arms: vec4, packed7.
Modes: prepared, fresh_resources.
Per arm/case/mode: five warmups and 20 measured trials. Paired order alternates, balancing both arm positions over 20 repeats.
Ten prospectively numbered, separately acquired/released cooperative leases, one fresh process per lease. No replacement of failed attempts or observation gaps.

Per complete process: 1440 measured timings + 360 warmup timings, 7200 complete comparisons. Full ten-process cohort: 14400 measured timings, 3600 warmup timings, 72000 comparisons. A rejected geometry is retained and prevents a complete-panel confirmation.

## Timing and correctness

Both modes measure one forward invocation from canonical input to canonical output:
input adapter + guarded physical-word creation + byte encoding;
mode-specific preparation;
fenced upload;
encode/submit/wait;
one fresh explicit MAP_READ readback;
canonical reverse conversion and padding validation.

Prepared pipelines/state/plans are retained outside the interval. Fresh_resources allocates a new pipeline/state/plan inside every interval on an already-created device; compiler/driver caches can persist. This is fresh-resource warm-context cost, not cold adapter/device startup. Initial pipeline costs are separately logged. No amortization extrapolation is made.

Total wall latency is the complete contiguous interval. Component timers are diagnostic subtotals; compare total wall values, not reconstructed sums. Pipeline/state destruction and model/device initialization are outside the interval and must not be claimed included. Timer instrumentation is common to both arms.

After stopping the timer, perform a second fresh explicit MAP_READ readback without changing the source. Compare A/oracle, B/oracle, A/B including every physical word, padding and four outside-binding sentinels, plus canonical output/oracle. No source readback is inserted into the timed interval. Independent host input round-trips and direct monomial cross-checks precede timing. Every reported forward state is verified; inverse latency is not an endpoint of this study.

A COMPLETE marker precedes a final assertion, so success requires process exit 0, complete unique records, no mismatches, exact flags and valid binary/source identities.

## Frozen descriptive and conditional decision rule

For each of the 36 geometry/bank/mode endpoints, pair vec4 and packed7 by measured repeat within each process. Report 20 ratios and their process median, then all ten process medians, p50/p95 raw latency and full regressions. Never pool repetitions as independent samples.

Minimum meaningful ratio is 1.05 (vec4 wall / packed7 wall). Conditional on independently qualified admission and independently justified session sampling, require all ten process medians strictly above 1.05 for each endpoint claimed. A one-sided sign test for a median ratio at most 1.05 gives p = 1/1024 for ten successes; Bonferroni across all 36 fixed endpoints gives 36/1024 = 0.03515625. This rule is intentionally conservative, and applies only under independent-session assumptions. Publish failures even when a subset passes.

Do not compute a confirmatory p-value or label a gain validated while the independent admission/sampling review is missing, a lease has unknown/foreign contamination, a process fails correctness, or record coverage is incomplete. A clean cooperative run alone does not close independent review. Diagnostic ratios remain descriptive. Any SML throughput claim additionally requires SML-owned banks and destination requalification.

## Host observation and restoration

Use the frozen Rust v2 device-user observer, existing lifecycle trace and existing rejection checker. Five complete preflight checks precede each invocation. During execution sample once per loop with a one-second pause, retaining every incomplete/foreign observation. Such a sample rejects confirmation for the entire lease; timings may remain raw diagnostic evidence.

Pause only the four already-authorized services and preserve their original policies. Each lease has its own namespace, cooperative lock, runtime manual-start guards, RuntimeMaxSec=25min and independent 30-minute restoration timer. Single process timeout is 900 seconds. Inspect restoration independently after each lease before the next lease is started. No blind replay after an ambiguous side effect, no other host-service pause and no vendor SDK or management library.

The observation checker and its lifecycle trace remain weaker than continuous hostile-process device fencing. This campaign cannot itself declare the admission implementation independently reviewed.
