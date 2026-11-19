# ISING-1 — Sparse Ising coupling of attention vectors (research only)

Status: scalar Rust **candidate** in `flat-algebraic-attention::ising_attention`; not a public attention API, not active in the default execution path, and not a performance or model-quality result.

## Hypothesis and mathematical contract

For **one already-normalized attention vector** `p`, assign a spin `s_i ∈ {-1,+1}` to each index. Define the symmetric, zero-diagonal sparse Ising model

```text
E(s) = -sum_i h_i*s_i - sum_(i<j) J_ij*s_i*s_j
h_i = (1/2)*log(p_i/(1-p_i))  for 0 < p_i < 1
```

This calibration makes the uncoupled `J=0` Bernoulli marginals exactly `p_i` at unit inverse temperature. A scalar damped parallel mean-field oracle uses

```text
m_i(new) = (1-damping)*m_i(old)
         + damping*tanh(h_i + sum_j J_ij*m_j(old))
p_i(new) = (1+m_i(new))/2
attention_i(new) = p_i(new)/sum_j p_j(new)
```

The normalized output is a new attention distribution. Its numerical change is the **hypothesis** to qualify; it is not an exact Ising posterior and is not guaranteed to improve quality. Iterations may stop at the budget without converging. No claim of monotonic energy descent applies to synchronous damped mean-field updates.

**Mask semantics:** exact zero entries are immutable and cannot affect other entries through `J`. Exactly one entries remain pinned. These boundary rules preserve explicit structural/padding/causal masks only when those masks have already been converted to exact zero attention weights. The caller must still apply causal masking *before* this experimental oracle.

**Structural constraints:** edges are undirected, unique, in ascending `(left,right)` order, with `left < right`; malformed topology and nonfinite values are rejected. The implementation stores only adjacency lists, `O(n + |J|)` memory; it does not create an `n x n` matrix. Dense couplings would themselves require quadratic storage and are not an accepted sparsity claim.

## Runnable API (isolated from production)

```rust
use flat_algebraic_attention::ising_attention::{
    ising_reweight_attention, IsingCoupling, IsingMeanFieldConfig,
};

let attention = [0.8, 0.15, 0.05];
let couplings = [IsingCoupling {
    left: 0,
    right: 1,
    weight: 0.1,
}];
let result = ising_reweight_attention(
    &attention,
    &couplings,
    IsingMeanFieldConfig::default(),
)?;
println!("{:?}", result.weights);
println!("converged: {}", result.converged);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Run focused tests:

```bash
cargo test -p flat-algebraic-attention --lib ising_attention
cargo fmt --all -- --check
cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
```

## Minimum ablation before promotion

1. Reference `J=0`: bit-for-bit preservation of the supplied attention vector.
2. Positive and negative sparse couplings at identical edge count; deterministic seeds and frozen candidate topology.
3. Matched random and local/structural topology controls at identical density; report edge metadata bytes.
4. Dense FLAT reference output/O/LSE, plus any downstream quality metric with the same input masks, and errors for non-converged cases.
5. CPU wall time, allocation and numerical error first; device-level resource/performance only once an actual portable WGPU path exists.

The research module does **not** execute Q/K/V attention, learn the `J` matrix, provide backward gradients, or become a default router. Its `J` graph can later be compared with V888-derived **structural priors**, but that would be a separately qualified experiment; no biological equivalence is asserted.

SML-GENIUS remains **non-attention** by its own architecture contract. Only independently qualified sparse energetic circuit/page activation could be considered there, without importing Q/K/V, token-score matrices, or Transformer cache semantics.

## Prior art and interpretation

- Gilhan Kim & Daniel K. Park, *Boltzmann Attention: Learnable Ising Couplings for Cooperative Attention*, 10 June 2026: https://arxiv.org/abs/2606.12478
- Rui Wang, *Variational-Ising-Attention (VIA): Tailored Attention Matters for Science*, 26 July 2026: https://arxiv.org/abs/2607.23634
- FLAT authoritative engineering plan: `ROADMAP.md`; off-main ecosystem and ML maturity overlays must be read before any promotion.
