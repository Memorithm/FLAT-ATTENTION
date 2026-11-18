# RAE-7 — bounded repair of impulse-proposed attention pages

**Status:** opt-in, research-only Rust host planner in `flat-algebraic-attention`. Not wired to the stable `flat_attention::api::v1` or a GPU/production runtime.

## Functional improvement

RAE-5 supplies an event-proposed set of ORIGINAL attention pages, touching only changed packed feature words when the upstream producer provides an exact dirty-word stream. RAE-6 can bound the omitted softmax mass, but rejects a candidate when its bound is too weak.

RAE-7 adds `rae_hybrid_repair::plan_bounded_hybrid_attention`:

1. Validate the original page mask, its padding, selected eligible QK logits, per-page maximum K-norm metadata, query/key/value/mask generations and numerical budgets through RAE-6.
2. If an independently supplied query/context invalidation is set, return `DenseFallback` with no sparse selection exposed.
3. If the initial certificate passes, return its original mask unchanged.
4. Otherwise rank **eligible omitted pages** by `log(N_p) + ||q|| * scale * Kmax_p` (the upper bound on their softmax partition contribution), with original page ID as a deterministic tie-breaker.
5. Promote at most `max_promotions` pages, recomputing the conservative dropped-mass bound after each promotion without evaluating QK for any omitted or newly promoted page.
6. Return `BoundedSparse` only if **both** the mass and componentwise output-error research budgets pass. Otherwise return `DenseFallback`, not an uncertified sparse mask.

For a newly promoted page, `N_p * exp(-||q|| * scale * Kmax_p)` is a lower bound on its future *retained* partition contribution, while omitted pages use their **upper** contributions. The baseline selected pages contribute their exact caller-provided logits. Computing the future QK for promoted pages is the executor's responsibility.

This may preserve more useful work than immediately abandoning a weak event-only mask. It does **not** establish an optimal choice, speedup, energy reduction, lower physical K/V traffic, correct model output, or safe production gate. In particular, `max_promotions = 0` leaves only the initial proposal eligible.

## Contract and fault behavior

- The caller must guarantee that selected scores, original page IDs, causal eligibility and pinned metadata refer to the same query. Equal generation numbers alone do not prove provenance.
- All eligible keys of initially selected pages must be represented by their exact logits. Per-page eligible counts already exclude future/padded/masked positions. RAE-6's no-additive-bias and nonnegative-scale restrictions apply unchanged.
- An empty initially selected eligible-key set is **not** handled as a sparse plan: the underlying RAE-6 validation returns `NoSelectedEligibleKeys` and callers must choose the dense reference.
- Invalid masks, stale generations, invalid budgets, nonfinite data and other certificate errors return `Err` and must also cause dense execution at the calling boundary.
- The returned `DenseFallback` intentionally contains no page mask. It requires the authoritative full causal numerical attention path; the original event proposal cannot replace that path.
- RAE-7 does not change RAE-5 wake semantics or imply that missing impulses make a page irrelevant. The external context-invalidation signal must be independently trusted; it is not inferred from impulse absence.
- The bound uses f64 and one-ULP positive rounding, not interval-certified floating-point arithmetic. Model-level qualification must assess worst-case errors before any real execution admission.

## Example and qualification

```bash
cargo fmt --all -- --check
cargo clippy --locked -p flat-algebraic-attention --all-targets -- -D warnings
cargo test --locked -p flat-algebraic-attention rae_hybrid_repair
cargo run --locked -p flat-algebraic-attention --example rae7_hybrid_repair
```

The deterministic tests include a high-risk omitted page that is promoted without its QK logit, comparison with an independent dense softmax-mass oracle, promotion-budget exhaustion, external invalidation, deterministic tied risks, a page above index 63, and version/budget failures.

**Next integration gate:** connect the planner to an actual model query with RAE-5 proposals, exact selected logits, model-owned version-pinned page norms, causal counts and full dense fallback. Benchmark total metadata maintenance, sorting, selected QK/V reads, numerical error and tokens/s on real hardware through **Memorithm/RemoteOps**. A paper certificate or a passing host test alone does not authorize altering the production attention default.
