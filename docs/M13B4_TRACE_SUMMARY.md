# M13B.4 per-unit trace summary

Status: research-only evidence contract. This document adds no speed, overlap,
throughput, bandwidth, quality, or hardware-generalization claim.

`api::research_boolean_overlap::metrics::summarize_decode_trace` derives exact
per-unit accounting from an already validated M13B.4 first-token or
steady-state decode trace plus a harness-observed dispatch count.

The summary reports:

- query-representation-ready to output-ready time;
- Q-signature interval;
- Boolean-routing interval;
- survivor-metadata-ready to numerical-K/V-staging-start gap;
- numerical K/V staging interval up to numerical-attention start;
- numerical-attention interval;
- output-finalization interval;
- explicit synchronization-wait interval when the trace contains the required
  wait pair;
- harness-reported dispatch count.

An absent synchronization pair is represented as `None`, not zero. This keeps
"no explicit wait event was recorded" distinct from "an explicit wait was
recorded with zero duration". The survivor-to-staging value is called a gap,
not a stall cause: the trace alone does not identify why the interval exists.

The contract fails closed for an invalid source trace, prefill scope, or zero
reported dispatches. It performs no timing-domain correlation and therefore
cannot prove cross-unit overlap. Cross-unit temporal intersection remains the
separate `correlation::measure_cross_unit_overlap` contract. Neither contract
establishes that measured overlap is useful or that total latency improved.

This closes only the backend-neutral accounting portion of the M13B.4
instrumentation deliverable. Real backend traces, first-token versus
steady-state measurements, dispatch/synchronization evidence, matched serial
controls, and preregistered hardware qualification remain required before any
performance interpretation.

## Canonical trace envelope

Validated `M13B4Trace` values now expose `canonical_json()` under the stable
`flat.m13b4-trace.v1` research schema. The envelope records the timing source,
scheduling variant, scope and every ordered event/timestamp with fixed enum
spellings. Invalid traces fail before serialization.

The canonical envelope exists so evidence stores such as KVLab can retain and
hash exact trace bytes rather than scraping logs or reconstructing enum names.
It is descriptive evidence only: serialization does not infer cross-unit overlap,
GPU concurrency, speedup, physical traffic, quality, or first-token improvement.
