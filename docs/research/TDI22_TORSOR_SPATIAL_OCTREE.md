# TDI-22 spatial torsor octree — research CPU reference V1

Status: **implemented experimental scalar Rust hierarchy, not a production or GPU attention kernel**.

Implementation: `flat_algebraic_attention::torsor_octree` in
`crates/flat-algebraic-attention/src/torsor_octree.rs`.
Contract: `tdi22-spatial-octree-f64-reference-v1`.

This design extends the existing, opt-in
[TDI-22 six-channel dense reference](TDI22_TORSOR_FLAT6_ADAPTER.md).
It uses the same `TorsorKey` and `TorsorQuery` types, but does not
change FLAT's public `api::v1`, numerical default, GPU routing, or
the scientific authority of Memorithm/TDI.

## Mathematical contract

For each key `i`, `R_i` is the resultant, `M_i(P_i)` its own
reduced moment, and `P_i` its source position. At query position `Q`:

```text
M_i(Q) = M_i(P_i) + (P_i - Q) x R_i
s_i = scale * [v . R_i + omega . M_i(Q)]
```

The octree partitions keys into disjoint leaves by their three spatial
coordinates. Every parent contains the sum of its children's resultants
and moments reduced at the parent **mass-weighted position center**:

```text
n_parent = sum_children n_child
G_parent = (sum_children n_child * G_child) / n_parent
R_parent = sum_children R_child
M_parent(G_parent) = sum_children [M_child(G_child) +
                                  (G_child - G_parent) x R_child]
```

This is a *linear wrench reduction*. `R` and `M(Q)` for the total
bank are preserved in real arithmetic regardless of tree subdivision.
`R . M` is independent of the reduction point. Actual f64 outputs
have normal floating-point rounding error.

## Query algorithm

`PreparedTorsorOctree::build(keys, leaf_capacity, max_depth)`
constructs at most 4,096 keys with `1 <= max_depth <= 32`.
Degenerate/duplicate positions stop subdivision and stay exact in a leaf.

`forward(query, scale, theta)` evaluates **non-causal** attention.
`scale` must be positive and finite; `theta` must be finite and
nonnegative:

- `theta = 0`: open all internal nodes, evaluate every individual key
  with direct TDI-22 Varignon pairing, and compute online softmax.
- `theta > 0`: if a node is distant and does not contain Q in its
  bounding box, it may be summarized when
  `extent / distance(Q, G_node) < theta`.
  **Near nodes must open.** This is the correct direction of the
  Barnes–Hut opening test.
- Each compressed node becomes a single surrogate key with **mean**
  resultant/moment and a score bias `log(n_node)`, preserving the
  correct multiplicity if all member logits coincide.
- Individual exact keys and compressed nodes share one numerically stable
  online softmax, with `log_sum_exp` returned.
- The result includes `exact_keys`, `summarized_nodes` and
  `covered_keys`. A valid traversal accounts for all keys once.

## Critical limitation: reduction is not softmax equivalence

A single torsor can represent the **sum** of wrenches, but in general:

```text
log(sum_i exp(s_i)) != log(n) + mean_i(s_i)
sum_i softmax(s_i) * R_i != mean_i(R_i)
```

Consequently, nonzero `theta` is an *approximate* non-causal attention
method. Its output error is not bounded by `theta` alone; geometric
distance does not bound semantic/logit variation. The original
discussion's blanket `O(N log N)` claim is not proven for arbitrary
spatial distributions: tree construction and worst-case traversal
depend on geometry, occupancy, and the chosen threshold.

At `theta=0`, the exact path is compared to an independently
written dense f64 reference. This does **not** establish bitwise
equivalence to the existing f32 six-channel FLAT bridge or scientific
TDI f64 engine.

The module intentionally does not support causal prefixes, learned
keys/queries, backward gradients, GPU/WGPU, dtype quantization, a
compressed KV cache, or claiming model-quality/performance gains.
None of these is inferred from summary-node counts.

## Reproduce the code-level gate

```bash
cargo fmt --all -- --check
cargo test -p flat-algebraic-attention torsor_octree
cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
```

Tests cover bottom-up versus independent direct reduction, the
reduction-point invariant, `theta=0` versus independent dense
softmax, mass-aware summaries for nearly identical logits,
near/far opening, coverage, deliberate approximation divergence,
degenerate locations, and invalid input rejection.

## Next qualification gate

Before selecting this hierarchy for a model:

1. Freeze a corpus with geometric and semantic heterogeneity, and
   compare `theta=0` dense reference against a preregistered
   theta/leaf-capacity grid.
2. Measure head-specific O/LSE errors, output error and downstream
   quality; include adversarial distant keys with sharply different
   logits and uniform/random controls.
3. Run paired, bounded CPU tests only under Memorithm/RemoteOps control,
   with exact SHA, host, toolchain, build profile, memory and
   median/p95 latency recorded.
4. Do not merge this research-only module into a production route or
   integrate it into SciRust/SML-GENIUS/NNIS until those destinations
   independently accept an explicit versioned contract.
