# Compact preselection bootstrap

Status: research-only implementation and synthetic qualification suite. No stable API, default route, model-quality, GPU or performance promotion.

Input: user-provided Gemini discussion reviewed on 2026-09-30, proposing low-dimensional screening, sparse traversal and online softmax in Rust. It is a hypothesis source, not a verified CSA2 implementation or benchmark. This slice implements an independently specified coordinate-projection baseline; it does not reproduce CSA2, learn a projection, compress numerical V, or share KV between layers.

## Read before continuing

Read the repository AGENTS.md, published ROADMAP.md and applicable off-main ecosystem, ML maturity and DA-LUC overlays. This document supplements them; it never changes a frozen MAA-13/14/15 protocol or authorizes an existing promotion gate.

Companion execution plan: [COMPACT_PRESELECTION_ROADMAP.md](COMPACT_PRESELECTION_ROADMAP.md).

## Mathematical and implementation boundary

For original head dimension D and an explicit strictly increasing coordinate list C of length R:

```
projected_K[j,t] = K[j,C[t]]
screen_score(i,j) = scale(D) * sum_t f64(Q[i,C[t]]) * f64(projected_K[j,t])
```

`scale(D)` is the ordinary FLAT configured scale, defaulting to the original dimension's inverse square root, not R. The selector is a deterministic coordinate-subspace ablation, not a learned or generally quality-preserving projection. Selection uses f64 accumulation; downstream exact-on-selected-keys numerical execution retains the existing f32 FLAT contract.

Causal eligibility is applied before ranking. Keep at most the explicit per-query budget, break score ties by smallest original logical key position, then canonicalize to the existing `StructuralCandidateSet`. Batch/head identity and original positions are retained. A bounded heap avoids a full per-query score array.

The implementation lives in `flat_algebraic_attention::compact_preselection`. That research crate consumes the existing FLAT types through a host-only dependency. It does not create a second attention kernel, mask schema or dense numerical oracle.

The result feeds `forward_reference_structural_sparse` directly with original Q/K/V. That existing numerical oracle streams only admitted keys with online softmax. Its matched dense-masked oracle and an independent two-pass f64 test oracle check the result on the same candidates. The all-accept control is checked against ordinary dense FLAT.

## Required safety properties

- Validate full Q/K lengths, shape products and all input values, including unprojected/future coordinates.
- Reject empty, unordered, duplicate or out-of-range projection coordinates.
- Explicitly represent a zero-budget empty candidate set; numerical execution must reject it, not divide by zero or invent a fallback.
- Support odd dimensions and sequence tails without a fixed SIMD lane or block-size assumption.
- Do not assume portable SIMD, AVX-512, NEON, one-cycle FMA, alignment, lock freedom, or register residency from this scalar Rust implementation.
- Do not reuse a projected snapshot across calls, layer changes or cache mutations. This slice builds a call-local snapshot and supplies no reuse authority.
- No RoPE, positional-bias, GQA/MQA, cross-layer sharing, native Boolean KV, GPU execution or model runtime integration is qualified here.

## Memory and work interpretation

The temporary compact K payload has B*H*N*R f32 scalars. Original numerical K/V remains retained and authoritative. The reported `projected_key_payload_bytes` excludes full K/V, candidate IDs, offsets, coordinate metadata, vector capacities, allocator overhead, and Q/output buffers. It is not total storage or compression evidence.

Q/K validation and projection construction read original inputs. They are not free. `evaluated_pairs`, `evaluated_score_components` and `selected_pairs` are logical counts only. No physical DRAM/HBM/PCIe traffic, latency, energy or throughput follows from them.

With a bounded budget k, returned candidate storage is O(B*H*N*min(k,N)); there is no dense Boolean mask or N-by-N score/probability buffer. The all-accept k=N control has quadratic candidate metadata and must never be presented as sparse-memory evidence. Screening still scores every eligible key in the projected space, so this is not a subquadratic prefill-compute claim.

## Synthetic qualification and negative control

The integration suite covers dense all-accept equivalence; sparse versus dense-masked and independent f64 parity; batches/heads; N=19, D=17 tails; causal-before-budget selection; future-value noninterference; deterministic ties; explicit empty selection; malformed inputs; finite selector accumulation at f32 extrema; and exact logical counters.

A deliberately adverse omitted-coordinate fixture makes a low-dimensional screen discard the dominant full-dimensional key. Exact numerical attention on the survivors cannot restore an omitted key. This is a required negative control, not a failed test to remove or tune away.

The frozen descriptive panel has two synthetic fixtures and five arms: all accept, one-coordinate top-2, full-coordinate top-2, recency top-2 and a deterministic density-matched random top-2. It emits JSON lines tagged `flat.compact-preselection-smoke/v1` with original selected positions, retained full-score softmax mass, output error and executed pairs. Only the last query row's quality diagnostics are emitted; the executed-pair count covers all five rows. None of these records is a real-model or confirmatory-holdout result.

```bash
cargo test --locked -p flat-algebraic-attention --test compact_preselection
cargo test --locked -p flat-algebraic-attention --test compact_preselection emit_frozen_quality_panel -- --nocapture
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
```

Required repository CI must pass on the exact final PR head before merge. The workflow log is the execution evidence; this document alone is not a test result.
