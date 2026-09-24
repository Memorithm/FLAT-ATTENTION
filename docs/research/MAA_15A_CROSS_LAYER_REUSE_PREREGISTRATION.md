# MAA-15a cross-layer reuse preregistration

Status: preregistered research protocol. No cross-layer reuse result, quality result, physical-memory result, latency result, speedup claim, or production promotion is made by this document.

## Motivation and source boundary

MAA-15a follows the merged MAA-14 correctness foundations: deterministic structural candidates, retained-softmax-mass diagnostics, confirmed tiered-recall/adaptive-repair evidence, exact u32 device layout, and the correctness-first WGPU sparse carrier.

DeepSeek-AI's *DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression* is used only as an external hypothesis source for separating cross-layer cache reuse and sparse-index reuse. Its CSA2, CED, FP4, SWA Bounded Replay, quality, memory, and performance results are not FLAT evidence.

## Research question

For a fixed set of attention layers with the same declared geometry and causal domain, can later layers reuse an earlier layer's structural K/V representation and/or candidate selection state while preserving the declared FLAT numerical/quality gate and reducing independently accounted work or storage?

The first phase is host-only and correctness-first. It does not time WGPU kernels.

## Frozen mode vocabulary

Every evaluated layer is assigned exactly one mode:

- `Full`: owns its source representation and computes its own candidate selection;
- `Reindex`: reuses a previously declared source representation but computes a fresh candidate selection for the current layer/query state;
- `Reuse`: reuses both the declared source representation and the latest compatible candidate selection.

The implementation must not infer a compatible source. The caller must provide an explicit source-layer identity.

## Reuse dimensions

MAA-15a treats these as independent switches/evidence dimensions:

1. main structural K/V source reuse;
2. indexer/selection-key source reuse where such a representation exists;
3. final candidate/Top-K identity reuse.

A combined `Reuse` mode is not evidence that each individual dimension is safe.

## Identity contract

A reusable source record must bind, at minimum:

- source layer id;
- destination layer id;
- attention geometry;
- causal/non-causal mode;
- representation/schema id;
- materialization epoch;
- candidate-policy id when candidate reuse is requested;
- exact candidate fingerprint when candidate reuse is requested.

Reuse fails closed on any mismatch. Layer number equality alone is insufficient.

## Frozen host arms

For the same deterministic panel, evaluate:

1. `full_full`: both layers independently materialize source representation and candidates;
2. `kv_reuse_reindex`: destination reuses source representation but computes its own candidates;
3. `kv_reuse_candidate_reuse`: destination reuses source representation and source candidates;
4. `candidate_reuse_only_control`: destination uses its own representation but is forced through the source candidate set;
5. `matched_density_random`: candidate count matched to the reused-candidate arm with deterministic random identities;
6. `all_accept`: complete candidate set control.

The first implementation may use a two-layer host fixture. It must not claim general multi-layer behavior from that fixture.

## Frozen synthetic panel

The first host panel is deterministic and development-only:

- batch = 1;
- heads = 1;
- sequence length = 128;
- head dimension = 16;
- causal = false;
- f32 scalar/reference arithmetic;
- at least 256 paired source/destination layer cases generated from a frozen seed;
- source/destination Q/K/V differ by a deterministic bounded perturbation so reuse is non-trivial;
- one negative-control subset deliberately breaks reuse compatibility.

The exact seed and perturbation construction must be committed before observation in the implementation PR and then remain unchanged.

## Required evidence

Per arm and aggregate:

- exact source/destination identity;
- admitted candidate count and candidate fingerprint;
- candidate recall against the destination layer's own Full candidate set;
- O and LSE error against destination Full/dense reference;
- retained destination-reference softmax mass where available;
- representation bytes logically reused versus independently materialized;
- selection/index evaluations avoided;
- fail-closed mismatch counts;
- deterministic replay digest.

Logical bytes/evaluations are explanatory evidence only; they are not physical memory, bandwidth, or latency measurements.

## Decision rule

MAA-15a host evidence may advance a specific reuse mode to MAA-15b only if:

- every identity/negative-control gate behaves as preregistered;
- no incompatible source is silently accepted;
- all-accept reproduces the dense reference within the existing scalar tolerance;
- the mode's candidate/attention error is no worse than the frozen acceptance limits declared in the implementation before observation;
- the matched-random control is retained even when it performs similarly or better.

No mode is promoted merely because it avoids more logical work.

## Hierarchical-pool boundary

MAA-15a does not yet implement the MAA-15b hierarchical candidate pool. It only establishes exact cross-layer reuse semantics. MAA-15b may use MAA-15a's compatible source/candidate records but requires a separate preregistration.

## Physical boundary

No WGPU timing, physical traffic, HBM/DRAM, energy, or speed claim is admissible in MAA-15a. MAA-15c/15d remain separate gates after host and hierarchical-candidate qualification.

## Non-goals

MAA-15a does not:

- implement DeepSeek CSA2;
- change the stable FLAT API;
- introduce CED or SWA;
- replace MAA-14;
- alter any previously frozen MAA threshold;
- claim real-model quality;
- claim cross-layer KV reuse is universally valid;
- infer speedup from candidate counts or logical byte accounting.
