# RAE-1 — sampled rising-edge front-end (research only)

Status: deterministic Rust host oracle; no default-route, public API, KV, SML,
training, or GPU changes. This document defines a proposed control primitive,
not a demonstrated replacement for dense attention.

## Definition

Given bit-packed state X[t] and explicitly supplied initial state X[t0],
a logical event at sampling instant t is:

```text
E[t] = X[t] AND NOT X[t_previous]
```

This expresses a **sampled rising front** (0 -> 1), not an active state,
not an analog derivative, not a physical Dirac impulse, and not automatically
a biologically faithful spiking neuron. A sustained high bit has one front;
a falling transition rearms the bit without generating a positive event.
The caller must provide a strictly increasing logical tick and an honest
initial baseline. Intermediate transitions between samples are unobservable.

The `flat_algebraic_attention::rising_edge::RisingEdgeDetector` oracle:
- validates nonzero width, exact number of u64 words and zero padding bits;
- emits packed rising bits into a caller-owned buffer without per-sample allocation;
- records logical tick, active bit count and rising bit count;
- rejects stale tick, malformed bitset or output width *transactionally*;
- does not infer forgotten transitions across gaps or perform temporal alignment.

The bitset must be read every sample, so event detection takes
O(ceil(width / 64)) logical word operations per sample even on a quiescent
signal. It is not an asynchronous hardware interrupt source.

## Research boundary

FLAT-ATTENTION retains its canonical numerical attention and dense oracle.
RAE-1 may only provide an **additional candidate-event signal** for future
MAA Boolean routing. It does not determine which K/V positions are relevant,
which numerical scores can be omitted or what attention mass is retained.
Any event-to-block mapping must preserve canonical key identity, mask,
causality, batch/head/position, and metadata epoch. No event must *not*
silently suppress a necessary decode step: a query change, persistent
active state or nonlocal retrieval can require work without a new front.

SML-GENIUS has no native Transformer Q/K/V semantics. A future model-owned
adapter may experiment with front events derived from SBG routing fields
and ANF/recurrent state changes to trigger atlas page activation, while
keeping the persistent state, native memory, learning contracts and
requalification in SML. Copying this primitive into SML is not authorized
as an already-qualified inter-repository integration.

Oscillator phase, timestamp coincidence, refractory windows and thresholds
are **not** implemented. They require separate semantics: sampled fronts
cannot recover pulses between samples, while globally synchronous phases do
not prove content-addressable long-term memory.

## Correctness tests

- stable high state emits no repeated fronts;
- falling state re-arms without positive event;
- simultaneous fronts across word boundaries;
- baseline high is not reported as a synthetic new event;
- high padding bits, mismatched lengths and repeated ticks fail closed;
- validation failures preserve both state and output;
- quiescent states emit no positive events.

Run:
```bash
cargo test -p flat-algebraic-attention rising_edge
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
```

## Proposed next experiment (not implemented in RAE-1)

Compare, on a frozen stream of actual model states:

1. numerical FLAT dense reference;
2. established MAA Boolean-only admission;
3. rising-front-triggered MAA candidate admission;
4. matched-density random and static-active-bit controls;
5. adversarial missed-important-key and unchanged-state/query-change controls.

Report false-negative key omissions, retained softmax mass, downstream quality,
exact event counts, byte-accurate state and index overhead, routing time,
K/V staging/traffic, effective latency and tokens/s by exact SHA/device.
Use CYCLOID-FLAT-1 cost/fidelity rules to evaluate qualified survivor sets,
not to invent evidence. A state transition alone has no relevance guarantee.
Do not change default runtime selection, promote a quality claim, or claim
physical energy savings before all required software and real-device gates.

## Sources and related project contracts

- MAA research: `docs/research/MULTI_ALGEBRA_ATTENTION.md`.
- Cost/fidelity gating: `docs/research/CYCLOID_FLAT_1_COSTED_SELECTION.md`.
- Authoritative FLAT ROADMAP, off-main ecosystem, ML maturity and DA-LUC overlays.
- Yao et al., "Spike-driven Transformer", NeurIPS 2023,
  https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html
  (a related spiking-attention paradigm; does not establish RAE-1 speed).
- Lee et al., "Spiking Transformer with Spatial-Temporal Attention", CVPR 2025,
  https://openaccess.thecvf.com/content/CVPR2025/html/Lee_Spiking_Transformer_with_Spatial-Temporal_Attention_CVPR_2025_paper.html
  (temporal dependencies matter; not RAE-1 model validation).
