# MAA-9: matched evidence integrity

Scope: research-only host qualification. No stable API, production router,
attention kernel, or numerical policy is promoted by this work.

## Failure being repaired

Before MAA-9, the evidence constructors compared survivor cardinalities, but
cardinality does not prove set inclusion. For a Boolean mask admitting block IDs
`{0, 2}`, a general survivor batch could relabel those blocks as candidate IDs
`{1, 3}`. Both arms had two survivors and zero declared relevant labels, so the
old matched-evidence checks accepted the mismatch. The same problem permitted
using an oracle from another route or Boolean metadata generation.

## Binding contract

`SurvivorSet` now retains one immutable clone of its qualification route and the
recomposition policy. Route equality includes the selected domains and all M13B
metadata: mask/schema/geometry, query and key signatures, signature schema/width,
and optional Boolean-KV schema/generation. It also remembers the first candidate
whose ID differs from its supplied Boolean block index, including rejected
candidates. General batches continue to permit arbitrary IDs; matched block-space
evidence does not.

Both the low-level `MatchedAttentionEvidence::new` and the derived
`derive_matched_host_evidence` path require the original route and policy, identity
candidate-to-block mapping, exact Boolean control cardinality, complete bounded
candidate geometry, and actual membership of every survivor in the Boolean mask.
Input ordering may vary without changing candidate identity.

This is structural host provenance, not cryptographic attestation. The workload
identity string is still caller-supplied. Numerical Q/K/V, predicate definitions,
reference labels and timing collection are not authenticated by the route
snapshot. Equal latency sample counts alone do not prove identical hardware,
warmup, execution ordering, or measurement scope.

## Realizable count and latency summaries

For a universe of `N` candidates, `S` survivors, `R` relevant candidates and
`K` retained relevant candidates, an intersection can exist only when:

```text
max(0, R - (N - S)) <= K <= min(R, S)
```

For a nested multi-algebra arm `M` inside Boolean arm `B`, also require:

```text
K_B - min(K_B, S_B - S_M) <= K_M <= K_B
```

Removing one candidate cannot remove two distinct relevant candidates.
The constructors check these bounds without overflowing `usize`.

Latency minima/maxima denote observed extrema, not loose bounds. For `n > 0`
measurements with observed minimum `a`, maximum `b` and total `t`, require:

```text
(n - 1) * a + b <= t <= (n - 1) * b + a
```

For one sample this requires `a == b == t`; for two samples, `t == a + b`.
Arithmetic uses `u128`, including the full `u64` count/value boundary.
No synthetic test timing is a hardware performance measurement.

## Validation

The dedicated `maa-host` workflow checks formatting, strict Clippy, debug and
release tests, and strict documentation on the exact PR head. Full repository
CI, WGPU qualification, CodeQL, semver and supply-chain checks remain required.
No gate is disabled or replaced by this additional host gate.

```bash
cargo test -p flat-algebraic-attention --locked --all-targets
cargo test -p flat-algebraic-attention --locked --release --all-targets
```

The integration suite enumerates all 16 four-bit Boolean masks, 16 affine-input
acceptance masks and 16 relevance masks: 4,096 matched records. Independent
bounded set enumeration validates the intersection bounds, and explicit sample
sequence enumeration validates latency extrema. Negative controls cover
candidate relabeling, rejected-ID relabeling, another mask with equal counts,
metadata generation/signature changes, domain selection changes, and forged
Boolean control counts. The 63/64 bit boundary is included.

These are bounded correctness experiments, not an empirical attention-quality
or speedup result. Consult the exact-head CI run for execution status.

## Autonomous continuation protocol

At each recurring Research Autopilot pass, reread `AGENTS.md`, its required
off-main overlays, `ROADMAP.md`, the MAA contract, and current PRs. Resume the
existing MAA PR before opening a duplicate. Correct failures on its own branch,
inspect reviews, require all applicable green gates on the final head SHA, merge
with that SHA pinned, and reconstruct the next slice from current main. Preserve
unrelated concurrent work. If no hardware is available, advance host validation
without manufacturing device evidence. Commit reproducible outcomes and retain
negative controls before moving to the next experiment.

Next planned slices, not completed by MAA-9:

- MAA-10: executable numerical matched harness, frozen cases, per-domain ablations,
  density-matched random/structural controls, dense-derived relevance targets,
  explicit block-to-score geometry, O/LSE error and end-to-end host cost. Distinguish
  logical candidate counts from executed scalar Q.K operations and physical I/O.
- MAA-11: reproducible machine-readable evidence with actual input/predicate hashes,
  environment/measurement-scope metadata, repeated trials, independent holdouts,
  and separate readiness/defer versus permanent rejection accounting for max-plus.
- MAA-GPU: a specific preregistered portable candidate only after host qualification;
  hardware parity and timing evidence before any runtime promotion.

The route/generation binding and realizable-intersection rules are potential
cross-project evidence guards for BooleanLab, KVLab, and SciRust. No downstream
integration or compatibility is claimed here; inspect each owner's actual
contract before transferring a generic primitive. Keep attention policy owned by
FLAT and avoid a circular dependency.
