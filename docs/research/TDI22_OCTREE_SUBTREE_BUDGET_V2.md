# TDI-22 octree — research-only subtree-local output budget V2

Status: **opt-in numerical reference**, not production routing, not a
precision-certified IEEE-754 interval bound and not a speedup claim.

Version: `tdi22-octree-local-output-budget-v2`.
Implementation: `crates/flat-algebraic-attention/src/torsor_octree.rs`.

## Why this is a new opt-in path

V1 accepts a node with scaled score span `S` only when

`range_root(component) * tanh(S) <= budget(component)`

for all six resultant/moment coordinates. This over-penalizes compact
subtrees when the **entire bank** contains dissimilar far-away groups.
V2 retains V1's spatial and score-spread gates but separates:

1. conditional output reweighting **inside each compressed node**, whose
   relevant six ranges are those of that node, transported to the query;
2. relative partition-mass error **between nodes**, whose mixture outputs
   can differ across the root's full global ranges.

The query's dense reference is unchanged; only the optional six-output
screen is different.

## Exact-arithmetic derivation

For any compressed group `g`, let the scaled per-key logits be `s_i`,
their arithmetic mean be `a_g`, their upper-bounded span be `S_g`,
and the exact and surrogate (mass-aware) partition masses be

`Z_g = sum_i exp(s_i)` and `Z'_g = mass_g * exp(a_g)`.

Hoeffding's lemma gives

`0 <= log(Z_g / Z'_g) <= H_g := S_g²/8`.

For exact leaves, `Z'_g = Z_g` and `H_g = 0`. With
`H := max_compressed_g H_g`, the global dense versus surrogate
log-sum-exp difference is nonnegative and at most `H` in exact
arithmetic. Their distribution over *groups* has total variation
bounded conservatively by `tanh(H/2)`.

Within each summarized group, the original normalized key weights
versus the uniform weights used by the averaged torsor have total
variation bounded conservatively by `tanh(S_g/2)`. For a bounded
scalar variable, differences of expectations are at most its range
multiplied by this total variation.

Therefore the V2 six-coordinate absolute-error bounds are:

```text
local_internal_error[c] = max_{compressed group g} (
    subtree_range_g[c] * tanh(S_g/2)
)
mass_renormalization_error[c] = root_range[c] * tanh(H/2)
total_bound[c] = local_internal_error[c] + mass_renormalization_error[c]
LSE_bound = H
```

Here `c` runs over three resultant and three transported-moment
coordinates; non-compressed groups contribute zero conditional
approximation error. The two components follow by a triangle
inequality between the exact conditional group expectations, the
uniform conditional group expectations at exact group masses, and the
uniform expectations at surrogate masses. Taking the maximum
conditional bound is justified since exact group masses sum to one.

The root and subtree ranges are computed from coordinatewise
origin-reduced `(R,C)` envelopes with the TDI transport convention.
The score span comes from the query-projected `(R,C)` envelope,
never a heuristic distance-only value.

V2 checks prospective cumulative maxima **before** admitting every
node, ensuring that all six budgets remain satisfied. Non-finite
spans, overflows, invalid budgets, non-finite query parameters and
non-finite output ranges fail closed or force exact expansion.

These are mathematical bounds under exact arithmetic. They do **not**
include IEEE-754 directed-rounding envelopes, float-cancellation
certificates, trained-model quality, causal attention, or GPU proofs.

## API and reproduction

```rust
let result = tree.forward_with_local_output_budgets(
    query,             // TorsorQuery
    0.01,              // positive scale
    0.5,               // geometric opening ratio
    1.0,               // maximum per-compressed-node score spread
    budget,            // six TorsorOutputErrorBudget values
)?;
```

This returns a `TorsorBudgetedOutput` with the same
`TorsorTreeOutput` counters and `TorsorAnalyticalErrorBounds` schema
as V1, but V2's `log_sum_exp_abs` is the Hoeffding upper bound `H`
rather than V1's maximum raw span.

Run the independent dense-oracle tests:

```bash
cargo test -p flat-algebraic-attention --lib torsor_octree -- --nocapture
cargo fmt --all -- --check
```

The deterministic adversarial **two separated clusters** test
requires V2 to summarize both clusters with zero exact keys at a
six-coordinate budget where V1 still expands. Additional deterministic
varied twists and limits compare every output to the independent dense
softmax and its declared bound. NaN, infinite and invalid budgets are
rejected. The existing frozen V1 33-line CSV schema remains untouched.

## Evidence and promotion restrictions

The earlier physical Dell CPU baseline is the unchanged
RemoteOps workflow execution
[44406248794](https://github.com/Memorithm/RemoteOps/actions/runs/44406248794);
it measured **V1**, not V2. These V2 unit tests are mathematical and
numerical correctness evidence, **not** a real-hardware V2 benchmark.
A reproducible paired Dell comparison must use the exclusive
`Memorithm/RemoteOps` control plane, pinned V2 SHA, immutable V1
baseline, matching fixtures, explicit load-admission guards, and
bounded measurements before claiming speedup or production maturity.


## Opt-in paired benchmark panel (research V2)

The existing `tdi22_octree_qualification` executable keeps its original
**four panels × two queries × four modes** by default, with exactly
**33 CSV lines including the header**. No existing V1 evidence,
RemoteOps pin, checksum, or CSV validator is retroactively changed.

The explicit `--compare-local` opt-in adds:

- mode `subtree_local_budget` (V2), adjacent to the original V1
  `six_output_budget` on the same constructed octree;
- panel `separated_clusters`: 50% of keys near x=0 and the rest near
  x=100, a large inter-cluster resultant difference, small intra-group
  resultant changes, and a distant aligned linear query that does not
  introduce unrelated moment errors.

Its output has **five panels × two queries × five modes = 50 data
rows (51 CSV lines)**, with the same 19 columns, identical keys,
oracle and build provenance within every paired panel/query. Both
V1 and V2 run against the *independent dense* torsor/softmax oracle,
and fail if any of the six reported bounds or the LSE bound are
violated. The optional mode does **not** claim equal execution cost
or training quality.

```bash
cargo test -p flat-algebraic-attention --example tdi22_octree_qualification
cargo run --release -p flat-algebraic-attention --example tdi22_octree_qualification -- --keys 1024 --samples 7 --compare-local > tdi22-v1-v2-paired.csv
```

Rust CI also requires both the frozen 33-line default and the new
51-line opt-in shape. The timestamps in the CSV are unpinned per-query
host observations: they do **not** qualify physical performance
without RemoteOps admission, a pinned SHA, resource provenance,
scheduler isolation and multiple paired independent-process runs.
