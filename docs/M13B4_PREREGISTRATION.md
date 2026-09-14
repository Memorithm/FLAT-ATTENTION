# M13B.4 preregistration — two-plane overlap and first-token readiness

Status: preregistered protocol. No performance result is claimed by this document.

## Objective

Qualify whether a Boolean control plane can prepare block/page admission decisions early enough to reduce end-to-end attention latency while the numerical plane processes admitted work.

The systems hypothesis remains:

```text
T_boolean_front_end + T_flat_survivors < T_flat_dense
```

This milestone does not assume that bitwise work is intrinsically faster than floating-point work.

## Frozen terminology

- **prefill completion**: the point at which resident numerical K/V state and the corresponding Boolean K/KV signatures for the prefill prefix are both committed and visible to decode.
- **first decode token**: the first generated-token attention invocation after prefill completion.
- **first-token ready**: the first decode token consumes Boolean metadata derived from the complete committed prefix without rebuilding that prefix metadata on the first decode critical path.
- **Boolean plane**: Q-signature generation plus Boolean admission/routing work only.
- **numerical plane**: exact FLAT attention work over admitted blocks/pages; numerical K/V remain authoritative for BIKV experiments.
- **overlap**: measured temporal overlap between Boolean-plane work for the next consumable unit and numerical-plane work for the current unit. Source-code interleaving alone is not evidence of overlap.

## H0 / H1

### H0 — no useful overlap

For the preregistered workloads, two-plane execution does not reduce first-token or steady-state decode latency relative to the matched non-overlapped Boolean-router path once router, dispatch, synchronization and surviving numerical work are all included.

### H1 — useful overlap

For at least one preregistered workload, two-plane execution reduces end-to-end latency relative to the matched non-overlapped Boolean-router path while preserving the exact declared admission semantics and numerical-quality tolerances.

A positive H1 outcome must not be generalized to untested devices, models, sequence lengths or routing densities.

## Required baselines

Every qualification run must include:

1. dense qualified FLAT decode;
2. M13B.3 Boolean router + exact FLAT survivors without intentional overlap;
3. M13B.4 two-plane execution with identical Boolean signatures, thresholds and survivor set;
4. when meaningful, a matched-density random mask control using the same numerical path.

The Boolean rule, threshold/top-k policy and any density target must be fixed before confirmatory measurements.

## Correctness gates before timing

Timing is invalid unless all gates below pass on the exact benchmark SHA:

- prefill K/KV signatures cover the entire committed prefix required by the first decode token;
- first decode token does not rebuild historical K/KV signatures;
- admitted block/page IDs equal the M13B.2/M13B.3 deterministic CPU oracle for the same signatures and rule;
- all-accept matches the qualified dense numerical path within its existing tolerance contract;
- partial admission matches M13B.1 semantics;
- no use-before-ready race is observable under repeated stress execution;
- dense fallback remains available and explicit.

## Scheduling variants

The first qualification must distinguish, rather than pool, these variants:

- **serial/matched**: Boolean router completes before numerical survivor execution begins;
- **multi-dispatch overlap candidate**: Boolean and numerical work use separate dispatches with explicit dependency/synchronization evidence;
- **same-dispatch/fused candidate**: only if a backend-valid implementation exists and preserves the same oracle semantics.

Unsupported variants are recorded as `NOT_APPLICABLE`; they are not simulated.

## Mandatory trace events

The runtime trace must expose monotonically ordered timestamps or backend timing intervals for at least:

- query representation ready;
- Q-signature start/end;
- Boolean routing start/end;
- survivor metadata ready;
- numerical K/V staging start;
- numerical attention start/end;
- synchronization wait start/end where applicable;
- output ready.

Prefill additionally records:

- numerical K/V commit;
- K/KV Boolean signature commit;
- the point at which both become decode-visible.

Host wall-clock traces may be used only when GPU timestamps are unavailable and must be labeled as host measurements.

## Required metrics

Report separately for first decode token and steady-state decode:

- end-to-end latency;
- Q-signature latency;
- Boolean-router latency;
- surviving numerical-attention latency;
- synchronization/wait time;
- dispatch count;
- retained block/page density;
- Boolean metadata bytes read;
- numerical K/V bytes avoided when directly measurable or, otherwise, an explicitly labeled analytical estimate;
- `numerical_KV_bytes_avoided / Boolean_metadata_bytes_read`;
- dense-target recall and false-negative rate;
- O/LSE error against the declared dense oracle where applicable;
- downstream quality only when an end-to-end model qualification is actually executed.

Do not infer energy use unless it is directly measured.

## Repetition and uncertainty

For each device/workload/variant:

- warm-up count is fixed in the experiment manifest before confirmatory timing;
- measured repetition count is fixed before confirmatory timing;
- report median and at least one dispersion statistic or percentile interval;
- preserve every failed run and every negative outcome with its failure classification.

No holdout workload may be used to choose routing thresholds, synchronization strategy or dispatch topology.

## Decision rule

M13B.4 may be described as demonstrating useful overlap only if all correctness gates pass and at least one preregistered workload satisfies both:

1. lower measured end-to-end latency than the matched non-overlapped Boolean-router baseline beyond the chosen uncertainty criterion; and
2. no regression outside the predeclared quality/admission tolerance.

Otherwise the result is retained as negative or inconclusive. A source-level concurrency structure, reduced operation count, or lower logical byte count is not sufficient evidence.

## Evidence manifest

Every retained record must identify:

- FLAT-ATTENTION commit SHA;
- device and backend;
- driver/runtime versions when exposed;
- model/revision or synthetic workload identifier;
- precision;
- batch, heads/KV heads, context length and head dimension;
- Boolean signature width and rule;
- threshold/top-k and retained density;
- warm-up and repetition counts;
- timing source (GPU timestamp vs host wall clock);
- selected scheduling variant;
- result disposition: positive, negative, inconclusive or not-applicable.

## Non-claims

This preregistration contains no speedup, throughput, bandwidth, quality or first-token result. It only freezes the qualification semantics and evidence requirements for M13B.4 before confirmatory measurements.