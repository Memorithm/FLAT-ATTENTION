# RAE-4 — causal numerical mass of event-selected FLAT pages

Status: research-only scalar Rust oracle, distinct from SML-GENIUS native
SBG/MOR and from the default FLAT attention kernel.

## Research question

RAE-1/2/3 show when rising fronts occur and which logical pages they
could wake. But selecting fewer pages does **not** prove retained attention
quality. RAE-4 measures the probability mass actually retained by a
particular page mask, using independent *already-computed* dense logits
and an authoritative causal/padding/length eligibility mask.

For eligible key positions \\(i\\), original logits \\(s_i\\), selected
page set \\(P\\) and original key-to-page mapping \\(p(i)\\):

\\[
m_{P}=\\frac{\\sum_{i\\in E,\\ p(i)\\in P}e^{s_i-s_{\\max}}}
                  {\\sum_{i\\in E}e^{s_i-s_{\\max}}},\\qquad
m_{\\mathrm{drop}}=1-m_{P}.
\\]

If every component of \\(V\\) is bounded by \\(B\\) in absolute value,
the existing `softmax_mass::output_error_bound` reports the conservative
componentwise upper bound \\(2(1-m_P)B\\) for renormalized retained
attention (subject to the underlying bound assumptions).

## Implementation

`crates/flat-algebraic-attention/src/rae_numeric_mass.rs` exposes
`audit_causal_page_mass` and `RaeNumericMassAudit`:

- Apply caller-supplied eligibility **before** selecting logits or
  computing the dense partition; no future/padded key leakage.
- Preserve original page identities, including multiple keys/page.
- Compare candidate selected keys to **all** eligible keys, using
  the previously implemented stable scalar softmax-mass reference.
- Return eligible/selected key counts, selected page count,
  retained/dropped mass, and conservative output-error bound.
- Reject malformed widths, selected mask padding, out-of-range page IDs,
  empty causal reference, invalid eligible logits and invalid value bound.
- Bound the oracle to 1,048,576 keys and 65,536 pages.
- Include a diagnostic `within_oracle_budgets` predicate with explicit
  mass and output-error limits. This predicate is **not** runtime
  authorization to skip attention.

Tests include: adversarial removal of a dominant key, equal-score
half-mass case, a causally excluded high logit, all-pages parity,
multiple keys/page and zero selected eligible keys.

## What these results cannot prove

Because **all logits are provided**, the oracle computes or reads the
dense baseline outside this function. Its results are NOT sparse
execution performance, saved memory, reduced K/V traffic or end-to-end
language-model quality. It never substitutes SML-GENIUS native MOR page
labels for FLAT Q/K score truth, and never changes default attention
routes or GPU kernels.

The next scientific panel must pair FLAT-RAE-3 selected page masks with
genuine dense causal per-query logits, original key->page indexing,
and actual model V bounds. Include dense, matched-density random and
structural masks, held-state query updates and dominant-key-drop cases.
Only a separately qualified runtime with explicit fallback may
consider using RAE-produced page masks for executable attention.

Run:
```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked -p flat-algebraic-attention rae_numeric_mass
cargo test --locked --workspace
```

Related `docs/research/RAE_3_MATCHED_POLICY_REPLAY.md`,
`docs/research/MULTI_ALGEBRA_ATTENTION.md` and
`ROADMAP.md`. Follow the off-main research and ML maturity overlays.
