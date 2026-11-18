# RAE-8 — executable scalar hybrid attention with dense fallback

**Status:** research-only host implementation, opt-in under `flat-algebraic-attention`. This is **not** a new default FLAT attention route, GPU device policy, complete model adapter, or physical bandwidth claim.

RAE-7 returns a norm-bounded repaired page mask. RAE-8 makes this computationally real on a single numerical query: evaluate selected `q·k * scale` scores, run the RAE-7 norm-bound repair, compute additional QK only for promoted original pages, and accumulate values through stable max-shifted causal softmax. The executor can instead compute full dense causal attention when an external invalidation occurs or sparse qualification fails.

### Research input contract

`rae_hybrid_execution::execute_rae_hybrid(query, policy)` accepts one finite `f64` query, flat row-major K and V arrays, original page IDs, causal/padding eligibility, nonnegative score scale, an impulse-proposed packed page mask and version-pinned per-page max K norms and *exact current causal eligible-key counts*.

The executor checks tensor geometry, original page IDs and per-page eligible counts, enforces causal eligibility before scoring, reuses already calculated initial QK scores, and returns `RaeNumericExecution` with scalar output, route, key-row score evaluations and accumulated V row counts. This is logical per-row accounting, **not** measured DRAM traffic. The implementation scans page IDs/eligibility, allocates the score cache, and currently sorts candidate page metadata; these costs must be measured in any runtime assessment.

### Fail-closed behavior

- A trusted external context invalidation always selects full dense causal attention.
- An empty initial selected set, malformed page mask, mismatched causal counts, stale or malformed norm certificate, or failed RAE-7 budgets leads to `DenseFallback` with no sparse mask.
- If a V component in **selected** pages exceeds the policy's independently asserted absolute bound, the executor switches to dense. Omitted V components cannot be checked without reading omitted V; the calling runtime must prove the value bound over those components for that exact pinned generation.
- Invalid tensor shapes or nonfinite executed Q/K/V/scores fail explicitly instead of fabricating outputs.
- Matching generation IDs are **not** a substitute for source provenance, and a sparse result here is **not** an authorization to replace normal FLAT numerical attention.

### Deterministic qualification

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked -p flat-algebraic-attention rae_hybrid_execution
cargo run --locked -p flat-algebraic-attention --example rae8_numeric_hybrid
```

The tests execute actual selected and full dense softmax outputs, show an example where two selected K/V rows replace three dense rows with a conservative componentwise error check, reject sparse on stale generations/incorrect counts/value bounds, check empty event selection, and prove causally excluded future keys with nonfinite payload are not read.

### Next evidence gate

Use one genuine model query/KV snapshot and production-authoritative page max-norm metadata, causal counts and V bounds. Compare RAE-8 with an independent FLAT dense oracle on a preregistered model holdout; record exact output/LSE error, fallback rate, total CPU/GPU utilization, metadata/sorting costs, physical K/V reads, peak memory, latency and task quality. Keep all device qualification through **Memorithm/RemoteOps**. No BANC v888 topology or SML native boolean-state semantics is inferred from the synthetic test.
