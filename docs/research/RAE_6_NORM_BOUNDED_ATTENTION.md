# RAE-6 — norm-bounded omitted-page softmax mass without omitted QK

Status: conditional mathematical bound implemented as a Rust
**research-only host oracle**. Not a production attention-skip permit,
GPU path, KV-cache manager, measured speedup or real-model validation.

## Goal

REA-1/2/3/5 detect and schedule events, and REA-4 measures the
*dense-reference* attention mass of pages selected by events. But
REA-4 requires all dense QK logits and therefore cannot itself
avoid reading or computing omitted K vectors.

RAE-6 provides a mathematical *upper bound* on omitted softmax
probability **without providing omitted QK logits**, conditioned on
correct versioned page metadata and the exact score model.

Let the causal logit for eligible key \\(i\\) be:

\\[
s_i = (q\\cdot k_i)\\,c,\\qquad c\\ge 0.
\\]

For page \\(p\\), an authoritative precomputed bound is
\\(K_p^{\\max}\\ge\\max_{i\\in p}\\|k_i\\|_2\\).
Let \\(Q_c = c\\|q\\|_2\\), and \\(N_p\\) be the exact
number of eligible keys in that page for the CURRENT causal mask.
By Cauchy–Schwarz:

\\[
s_i\\le Q_c K_p^{\\max}.
\\]

For the retained (selected) pages, compute the exact selected-key
logits \\(s_i\\) as usual, and let:

\\[
S=\\sum_{i\\in\\mathrm{selected}}\\exp(s_i),\\qquad
U=\\sum_{p\\in\\mathrm{omitted}}N_p\\exp(Q_c K_p^{\\max}).
\\]

Then the mathematically exact omitted softmax mass satisfies:

\\[
m_{\\mathrm{omitted}}\\le\\frac{U}{S+U}.
\\]

Assuming every value component is bounded in absolute value by
\\(B\\), the componentwise output difference between full
softmax attention and renormalized selected-key attention obeys
\\(\\Delta\\le 2B\\,m_{\\mathrm{omitted}}\\).

The Rust implementation uses log-sum-exp and a stable ratio, with
a one-ULP upward rounding step on computed positive bounds. This
is **not formally interval-certified floating-point arithmetic**:
its output must not be treated as a mathematically machine-checked
upper bound at very tight thresholds. Exact production admission
would require verified outward-rounding/interval error analysis.

## Functional implementation

`crates/flat-algebraic-attention/src/rae_norm_certificate.rs`
exports `certify_omitted_page_mass`, which receives:

- bit-packed mask of selected ORIGINAL page IDs;
- per-original-page currently eligible key counts and upper bounds
  on key \\(\\ell_2\\) norms, **not** omitted logits or keys;
- all exact selected eligible logits for this query;
- nonnegative scaled query norm and maximum absolute V-component bound;
- version IDs for the query's K generation, norm metadata K generation,
  query V generation, V-bound generation, query mask and eligible counts.

The API rejects stale K norms, stale V bounds and stale causal counts,
including nonzero bitset tail padding, missing selected logits,
nonfinite scores/bounds, invalid page geometry and overflowing terms.
It enforces up to 65,536 original pages and 1,048,576 eligible keys.

The returned struct reports selected and omitted eligible-key counts,
selected original pages and upper estimates of omitted mass and output
difference. The budget-predicate is explicitly **diagnostic**: it does
not enable runtime attention skipping.

## Preconditions that still need engineering evidence

- Page norms must be computed **beforehand** from all current K vectors
  and must upper-bound those norms. A K update invalidates its page
  bound; stale metadata must never be reused. Recomputing norms can
  impose full scans elsewhere.
- Causal/padding/length eligibility counts are query-specific.
  Current mask counts must reflect all future/past exclusion rules;
  an unrelated mask change requires re-evaluation.
- Scores must have the stated dot-product form and scaling.
  Positional biases, additive logit terms, nonstandard kernels or
  normalization must have separate valid upper bounds added to the
  formula. This implementation deliberately rejects silently
  interpreting arbitrary kernels as this form.
- Exact selected-key logits are still required, so this is not
  Boolean-only attention or a Q-free model.
- The supplied version IDs must be minted by authoritative model
  ownership. Equality of caller-supplied integers alone cannot
  attest provenance; it is only a fail-closed interface contract.
- The V-component maximum must cover current values and current
  generation. It can be expensive to maintain.
- Norm estimates and floating rounding must be qualified
  conservatively, not accepted as proof merely by passing unit tests.
- The model has to evaluate the remaining selected work and preserve
  exact attention eligibility semantics. Full attention remains the
  fallback if any required bound is absent or stale.

## Falsification examples

The Rust tests include:

1. Selected logit \\(10\\), omitted zero-norm key: the exact omitted
   logit is *not* provided to the certificate; the bound is small.
2. Omitted page with high key norm: the conservative bound is nearly
   one and a strict skip budget is rejected.
3. A causally ineligible future page with enormous norm contributes
   no upper mass (eligible count zero).
4. Key, V or mask generation mismatch rejects the certificate.
5. Selected-logit count, page-mask geometry, nonfinite values and
   bound-overflow errors fail closed.
6. Original page IDs crossing bit 63/64, with multiple eligible
   keys/page.
7. Cross-check the bound against RAE-4's independently computed
   dense causal oracle in a tiny deterministic fixture.

These are not representative perplexity/accuracy benchmarks, and
do not establish speedup.

## Experimental path

The next genuine model evaluation must pair RAE-5 candidate page masks
with **actual** per-query Q and selected K scores in FLAT, certified
per-page key norms and exact causal counts. Benchmark complete
metadata maintenance, selection, fallback, selected QK, V reads and
output renormalization **against the original dense baseline**.
Include adversarial query/context changes and K/V generation updates.

All device experiments must run through Memorithm/RemoteOps with
pinned code and logged workload context. No NVIDIA dependency is
required or authorized.

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked -p flat-algebraic-attention rae_norm_certificate
cargo test --locked --workspace
```

Related: RAE-4 `docs/research/RAE_4_CAUSAL_NUMERIC_MASS.md`,
RAE-5 `docs/research/RAE_5_DIRTY_WORD_ROUTING.md`,
the FLOP/bandwidth scenarios in the Cycloid work, and
`docs/research/MULTI_ALGEBRA_ATTENTION.md`.
