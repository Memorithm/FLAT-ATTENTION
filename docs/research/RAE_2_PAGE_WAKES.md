# RAE-2 — rising-edge page wakes and persistent logical activation

Status: **research-only Rust functional prototype**. No change to the public
`flat_attention::api::v1`, numerical attention outputs, default GPU path,
SML-GENIUS model semantics, physical residency, or K/V cache policy.

## Motivation

RAE-1 (`rising_edge.rs`) implemented the exact logical transition

```text
front[t] = state[t] & !state[t-1]
```

RAE-2 implements **an actual downstream operation** rather than only counting
events: route set front bits through a fixed feature-to-page lookup, coalesce
duplicate page wake-ups and preserve a per-page logical latch between ticks.

It uses `RisingEdgePageRouter::sample_into`, which takes a caller-owned
packed feature state and two caller-owned page-bitset output buffers:

```text
(bit state[t], previous bit state[t-1])
             |
             v
       rising front bits
             |
             v
  feature-to-logical-page map
             |
             +--> per-tick unique page wake mask
             |
             +--> newly latched page mask
                         |
                         v
                persistent page latch state
```

An active page **does not need another wake** while its feature is held high.
After a falling edge and a later rising edge, the same page wakes again,
although the latch remains set. Features with identical target pages
coalesce to one unique page wake in a tick.

The persistent latch represents logical activation history, **not**
physical cache residency, a live tensor, correctness of numerical
attention, or a promise that a K/V page was retained.

## Contract

- Nonzero bit width, nonzero page count, fixed one-page-per-feature mapping.
- Reject out-of-range destination pages and mismatched map sizes.
- Explicit initial baseline and monotonically increasing logical ticks.
- Initial high features latch their logical pages, but do not issue
  synthetic time-zero wake events.
- If any input or caller-owned output dimensions are invalid, leave the
  complete router and both outputs unchanged.
- Zero allocation on the valid hot `sample_into` path after construction.
- Work remains O(ceil(feature_width/64) + number_of_rising_features
  + ceil(page_count/64)) for each logical tick (clearing output masks
  includes page count); it is **not** an interrupt-driven hardware kernel.
- Memory includes previous/front feature bitsets, logical latch bitset,
  an explicit `usize` feature-page map, two caller-owned output bitsets,
  and Rust container overhead. No compression claim is made.

## Critical safety and scientific controls

**A zero wake is not permission to skip numerical attention.**
A held-high feature may remain relevant when a new query arrives.
A model can also require work after a falling edge, after a K/V update,
or after an attention-mask/policy change. Therefore the wake mask is
strictly a *positive scheduling hint*, not an attention eligibility mask.

A future integration into FLAT must either:

1. use wake events solely to trigger independent metadata work while
   retaining fully qualified numerical attention; or
2. supply a separately verified current-query/key survivor and
   retained-mass/error contract with causal/mask/version identity,
   full dense fallback and adversarial missing-dominant-key controls.

The RAE-2 example and tests do **not** make that second guarantee.

SML-GENIUS has native Boolean/SBG/MOR/recurrent/page semantics, not
Transformer Q/K/V. An SML-owned implementation could test fronts on a
chosen SBG routing field to drive atlas activation; it would require a
separate model contract, real learned tasks, ablations and destination
qualification. FLAT-ATTENTION must not become a required SML runtime.

## Run and validate

```bash
cargo run -p flat-algebraic-attention --example rising_edge_pages
cargo test -p flat-algebraic-attention edge_pages
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
```

Functional checks cover coalescing, sustained high, rearm, baseline-high
initialization, multiword bitsets, output transactional failures, invalid
mappings and the mandatory negative control showing that unchanged
feature state does not mean a new attention query can be skipped.

## Next bounded experiment

Compare on **same pinned model/trace** and **same sequence of queries**:
- baseline every-tick active-feature scanning/page work;
- any-transition (rising OR falling) activation;
- rising-edge-only activation;
- matched-density, deterministic random wake control;
- real numerical attention with original dense and MAA Boolean-only
  admission for correctness.

Report exact logical event/page counts separately from measured K/V
traffic, memory, latency, energy (if instrumented), tokens/s and
downstream model quality. Use external query/KV invalidation signals;
do not infer them from feature fronts. A hybrid of rising and falling
events might outperform strict rising-only semantics — RAE-2 does not
prejudge the outcome.

Related contracts:
`docs/research/RAE_1_RISING_EDGE_FRONTEND.md`,
`docs/research/MULTI_ALGEBRA_ATTENTION.md`,
`docs/research/CYCLOID_FLAT_1_COSTED_SELECTION.md`,
`ROADMAP.md`, and authoritative off-main ecosystem / ML maturity /
DA-LUC overlays.

No novelty, latency, energy or general-model superiority is asserted
by these host-only tests.
