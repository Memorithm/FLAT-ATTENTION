# MAA-14 adaptive repair routing bootstrap

Status: research-only. This programme starts after the MAA-13d confirmatory result and the MAA-13e structural sparse-carrier integration. It does not alter, reinterpret, or retune the frozen MAA-13e physical qualification protocol.

## Motivation

FLAT already owns:

- a dense streaming-softmax numerical oracle;
- Boolean/F2/Zhegalkin admission research;
- retained-softmax-mass diagnostics from MAA-13c;
- confirmed tiered-recall routing from MAA-13d;
- a deterministic bridge into the FA-V888 structural sparse path from MAA-13e.

An external architecture article on Cadence is treated only as a hypothesis source for one general systems idea: spend additional computation only where a previous state or partial computation is judged insufficient. FLAT does not import Cadence code, equilibrium semantics, or performance claims.

The Memorithm adaptation is narrower and falsifiable:

    initial structural route
        -> exact sparse FLAT pass
        -> separately qualified repair signal
        -> widen selected query rows only
        -> exact sparse FLAT pass
        -> ... -> optional dense terminal control

The repair mechanism is monotonic. It may add candidates, but it may not silently remove candidates admitted by an earlier round.

## MAA-14a — deterministic repair transport

MAA-14a implements only the transport/state machine:

- one frozen base StructuralCandidateSet;
- ordered, per-row expansion tiers;
- an explicit per-row repair depth;
- deterministic materialization back into StructuralCandidateSet;
- exact admitted-pair accounting before and after a repair;
- duplicate, out-of-range, malformed-state and exhausted-repair failures are explicit;
- an optional schedule may provide an all-key dense terminal state.

The caller supplies rows requiring repair. MAA-14a intentionally does not define a residual, confidence threshold, quality metric, latency policy, or learned controller.

This separation prevents the mechanics from becoming tuned to one favorable signal after observations.

## MAA-14b — preregister deployable repair signals

The first structural-only protocol is frozen in [`MAA_ADAPTIVE_REPAIR_SIGNAL_PREREGISTRATION.md`](MAA_ADAPTIVE_REPAIR_SIGNAL_PREREGISTRATION.md).

Before observing confirmatory evidence, freeze a bounded family of signals that can be computed without dense QK scoring.

Candidate signal families may use only already available sparse-pass or router state, for example:

- initial-route margin/tier metadata;
- sparse online-softmax statistics such as LSE evolution that do not require the omitted keys;
- router disagreement across independently defined Boolean/F2/Zhegalkin views;
- structural uncertainty from V888/topology controls;
- recurrent state-change indicators supplied by an explicitly versioned external caller.

The exact signal family and thresholds must be preregistered before confirmatory observation.

Dense exact-score retained softmax mass from MAA-13c may be used as a non-deployable oracle target for evaluation. It must not be smuggled into the runtime trigger.

## MAA-14c — matched host evidence

Required arms:

1. dense FLAT reference;
2. one-shot frozen sparse route;
3. adaptive repair from the same initial route;
4. matched-density random expansion;
5. deterministic non-adaptive widening;
6. all-accept/dense terminal control where the schedule provides it.

Required evidence:

- final admitted/executed pair counts;
- number of repaired rows and rounds;
- O and LSE error versus dense FLAT;
- retained dense-reference softmax mass as diagnostic evidence;
- empty/exhausted repair outcomes;
- per-round monotonicity;
- exact policy/source/dataset identities;
- held-out confirmation distinct from tuning/exploration.

Logical candidate reductions are not latency, bandwidth, energy, or memory-traffic results.

## MAA-14d — portable physical candidate

Only after MAA-14c host evidence supports a frozen hypothesis:

- construct a portable WGPU candidate;
- preserve the MAA-14 transport/policy separation;
- measure repair-control overhead separately from numerical attention;
- compare initial-only, adaptive, deterministic widening and dense paths on the same device;
- record first-pass and total-to-final-result latency separately;
- report quality together with latency and physical traffic evidence.

No production routing changes before exact-head device qualification.

## Cross-project boundaries

### SML-GENIUS

SML may independently study bounded rewritable Delta-KV/associative memory and recurrent repair. It must not turn that research into an implicit Transformer KV cache dependency for FLAT.

### ElasticXxx

ElasticXxx may own generic resource budgets and adaptation constraints. It does not own FLAT numerical correctness or repair semantics.

### Forge

Forge may propose schedules, signal parameters, or topology candidates. Every candidate must be independently requalified by FLAT before any promotion.

### CCOS-Enterprise

Observed, derived, and hypothetical memory/evidence states must remain distinct. Simulation or imagined repair branches are not factual provenance.

## Non-goals

MAA-14 does not:

- claim that attention is unnecessary;
- replace softmax by an equilibrium solver;
- alter the stable api::v1 contract;
- change the already frozen MAA-13e experiment;
- use dense exact-score mass as a deployable trigger;
- claim speedup from host candidate counts;
- promote a learned repair controller before matched controls and holdout evidence.
