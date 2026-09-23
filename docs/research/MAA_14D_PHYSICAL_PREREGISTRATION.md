# MAA-14d physical adaptive-repair qualification preregistration

Status: preregistered physical-qualification protocol. No WGPU sparse-carrier result, latency result, speedup claim, bandwidth claim, or production promotion is made by this document.

## Entry evidence

MAA-14d follows the preserved MAA-14c `CONFIRMATORY_PASS`:

- exploratory implementation merge: PR #279;
- confirmatory preregistration merge: PR #280;
- confirmatory execution merge: PR #281;
- preserved confirmatory observation merge: PR #282;
- confirmed deployable trigger: `repair_if_coverage_below_7_8`;
- trigger rule: `8 * base_count < 7 * medium_count`;
- base route: unchanged `tiered_recall`;
- repair envelope: unchanged Hamming-medium `d <= 72`;
- signature width: 128 bits;
- projector seed: `0x514b_5349_474e_3133`;
- tiered-recall inner threshold: Hamming `d <= 68`;
- tiered-recall outer rule: `68 < d <= 72 && ||K|| >= 1.5`.

No threshold, projector, norm rule, candidate ordering, or repair rule may be retuned in MAA-14d.

## Physical question

Can the confirmed 7/8 repair policy reduce **measured end-to-end attention latency** on a declared physical WGPU device after accounting for:

1. structural routing;
2. base/medium candidate construction;
3. the 7/8 repair decision;
4. candidate materialization/canonicalization;
5. host-to-device transfer required by the selected representation;
6. sparse numerical attention;
7. synchronization/readback included in the declared timing boundary;

while reproducing the exact host candidate set and numerical result?

Logical candidate-count reduction is an explanatory variable only. It is not physical-performance evidence.

## Gate A — exact WGPU sparse carrier before timing claims

The repository currently has the authoritative per-key `StructuralCandidateSet` semantics on the host. MAA-14d MUST NOT infer that an existing paged/page-selected WGPU path is equivalent to this per-key carrier.

Before any physical timing result is promotable, a WGPU-compatible numerical carrier MUST be qualified that consumes either:

- the exact canonical `StructuralCandidateSet`; or
- a different device representation proven to encode exactly the same per-query key identities and ascending execution order.

Required Gate-A evidence:

- exact candidate IDs for every query row match the host structural oracle;
- duplicate/out-of-range/empty effective rows retain fail-closed semantics;
- all-accept candidate sets reproduce dense FLAT O/LSE within the existing declared WGPU tolerance;
- arbitrary sparse candidate sets reproduce the authoritative host masked/sparse oracle;
- causal intersection, if exercised, preserves exact candidate identity;
- candidate representation conversion has exact deterministic accounting;
- source revision and WGPU runtime/device identity are recorded.

A page-granular or block-granular path may be used only when an explicit proof/test demonstrates exact equivalence for the measured candidate geometry. Similar density is insufficient.

Gate A makes no latency or speedup claim.

## Router placement

The first MAA-14d physical implementation is allowed to keep candidate routing on the host, because the current WGPU Boolean router does not yet encode the complete tiered-recall + norm + 7/8 count rule.

If routing remains on the host:

- its wall time is included in the end-to-end adaptive arm;
- candidate construction/materialization wall time is included;
- required upload/transfer wall time is included;
- the result may be called an end-to-end **host-routed WGPU** qualification only.

A later device-resident router is a distinct candidate. Before performance comparison it MUST prove exact candidate-set parity with the frozen host rule and bind parity/performance evidence to the same source revision, WGPU runtime, adapter, backend and driver identity.

## Frozen execution arms

For identical Q/K/V data and geometry on one adapter, measure:

1. `dense`
   - normal dense FLAT WGPU execution under the same synchronization/readback boundary;

2. `base_tiered_recall`
   - frozen tiered-recall candidate set;
   - sparse carrier only, no adaptive widening;

3. `adaptive_7_8_precomputed`
   - final confirmed 7/8 candidate set precomputed outside the timed region;
   - isolates sparse numerical-carrier potential;

4. `adaptive_7_8_end_to_end`
   - route + base/medium construction + 7/8 decision + materialization + required upload + sparse execution;

5. `matched_random_7_8_end_to_end`
   - exact per-case/row final cardinality matched to the adaptive arm;
   - preserves the same base candidates;
   - uses the frozen MAA-14c matched-random construction semantics with a physical-series seed frozen in the implementation preregistration before measurement;

6. `always_repair_medium`
   - complete Hamming-medium envelope;

7. `all_accept_structural`
   - all valid keys through the sparse carrier;
   - routing/carrier-overhead control against dense.

No arm may silently fall back to dense execution while being reported as sparse/adaptive.

## Frozen benchmark families

The first physical implementation MUST include at least these shapes, subject to exact carrier support:

- batch = 1;
- q_heads = 8;
- kv_heads = 2 where the carrier supports grouped attention exactly, otherwise q_heads = kv_heads = 1 for the first Gate-A qualification;
- query length / KV length: 128 and 512;
- head dimension: 64 and 128;
- causal = false for the first mandatory comparison;
- causal = true only after exact causal candidate-intersection parity passes;
- f32 reference dtype for the first qualification.

A smaller correctness-only shape may be added but cannot replace the declared physical families.

## Measurement protocol

For every physical performance record:

- warm-up iterations per arm: **5**;
- measured iterations per arm: **50**;
- execute arms in a deterministic rotating order across iterations so one arm is not always first/last;
- synchronize the device at the end of each measured arm boundary;
- collect wall-clock nanoseconds for the exact declared boundary;
- report p50, p95 and p99;
- preserve raw sample counts or a deterministic digest of the sample vector;
- do not remove outliers post hoc.

Software Vulkan, lavapipe and WARP may run Gate-A correctness and benchmark-plumbing checks, but their timings cannot support a physical-device performance claim.

## Required timing decomposition

Where technically measurable without perturbing the boundary beyond usefulness, report:

- signature/routing construction;
- candidate base/medium construction;
- 7/8 trigger decision;
- candidate canonicalization/materialization;
- host-to-device candidate/index upload;
- sparse numerical dispatch;
- device synchronization;
- result readback;
- total end-to-end latency.

At minimum, the implementation MUST provide:

- precomputed sparse numerical latency;
- end-to-end adaptive latency;
- dense latency;

under explicitly identical synchronization/readback boundaries.

If a substage cannot be isolated without changing execution semantics, report it as part of a named combined stage rather than estimating it.

## Required physical provenance

Every retained result MUST bind:

- exact 40-hex source commit;
- benchmark/protocol schema version;
- complete invocation/environment controls;
- adapter name;
- WGPU backend;
- WGPU runtime identity (currently `wgpu/30.0.1` where applicable);
- driver/runtime identity when exposed;
- operating system and architecture;
- shape, dtype, causal mode;
- warmups/repeats;
- exact candidate counts and executed pair counts;
- candidate-set digest;
- output/result digest.

Use the repository benchmark-manifest conventions where they apply. Checksums are corruption/reproducibility evidence, not cryptographic attestation.

## Numerical and structural gate

A timing sample family is inadmissible unless:

1. device candidate IDs exactly match the frozen host candidate set for the arm;
2. no effective empty row is silently repaired by dense/zero fallback;
3. all-accept structural execution reproduces dense semantics;
4. sparse O/LSE matches the authoritative host oracle within the declared WGPU tolerance;
5. retained-softmax-mass diagnostics are computed for the exact same candidate identities;
6. adaptive 7/8 keeps the confirmed host policy unchanged;
7. matched-random cardinality is exact.

## Physical decision rule

No numerical speedup threshold larger than zero is assumed.

For one exact qualified device/workload, `adaptive_7_8_end_to_end` is physically favorable only if:

- all Gate-A and numerical/structural gates pass;
- its p50 end-to-end latency is strictly below dense p50;
- its p95 end-to-end latency is strictly below dense p95;
- the measured candidate/result identities correspond to the confirmed 7/8 policy;
- no hidden dense fallback occurs.

p99 is reported but is not a promotion gate in the first 50-sample series because tail estimation is coarse at that sample count.

If only `adaptive_7_8_precomputed` beats dense while the end-to-end arm does not, preserve this as **kernel-potential evidence only**. It does not authorize a speedup claim for adaptive routing.

If the end-to-end arm is slower, preserve the negative result. Do not retune 7/8 on physical timings.

## Relationship to ElasticXxx

ElasticXxx may later bound adaptive rounds/resources using versioned physical observations. It cannot convert a physically slower or numerically unqualified MAA-14d candidate into a valid FLAT route.

## Promotion boundary

A favorable MAA-14d result is device/workload-specific physical evidence. It still does not authorize default production routing.

Production consideration requires:

- at least one real physical adapter result;
- reproducible exact-head rerun;
- retained numerical/quality gates;
- explicit handling of unsupported shapes/backends;
- no regression of dense/default FLAT semantics.

## Non-claims

MAA-14d does not by itself establish:

- cross-device speedup;
- model-quality improvement;
- real-language-model generalization;
- energy savings unless independently measured;
- production readiness;
- novelty.
