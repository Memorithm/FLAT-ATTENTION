# MAA-11a: holdout policy-freeze preregistration

Protocol version: `maa-holdout-policy-freeze/v1`.

This contract is committed before any confirmatory MAA holdout observation is
implemented. Its purpose is to make policy drift and dataset leakage explicit
and fail-closed rather than relying on prose discipline alone.

## Question

Can a Multi-Algebra Attention policy be frozen with enough identity information
that a later confirmatory observation is accepted only when it uses exactly the
predeclared policy, feature schema, algebraic route, recomposition rule, source
revision and holdout dataset?

This is a provenance/integrity gate. It is not a model-quality or performance
experiment.

## Frozen identity

A frozen policy manifest binds all of the following:

- schema version;
- human-readable policy identifier;
- source revision identifier;
- selected algebraic route/domain set;
- recomposition policy;
- SHA-256 digest of the predicate/policy payload;
- SHA-256 digest of the feature-schema payload;
- SHA-256 digest of the tuning/exploratory dataset identity;
- SHA-256 digest of the confirmatory/holdout dataset identity.

Digests are identity tokens supplied by the experiment harness; this slice does
not add a hashing dependency or claim to hash external data itself. The Rust
contract validates canonical 64-hex SHA-256 text and normalizes it to lowercase.

The tuning and confirmatory dataset digests must differ. An identical digest is
rejected as a holdout-boundary violation.

## Confirmatory binding rule

A later confirmatory context is valid only when all frozen fields match exactly:

```text
observed.policy_id             == frozen.policy_id
observed.source_revision       == frozen.source_revision
observed.route_domains         == frozen.route_domains
observed.recomposition_policy  == frozen.recomposition_policy
observed.predicate_digest      == frozen.predicate_digest
observed.feature_schema_digest == frozen.feature_schema_digest
observed.dataset_digest        == frozen.confirmatory_dataset_digest
```

The tuning dataset digest is never accepted in place of the confirmatory digest.
A mismatch must name the field that drifted and fail closed.

## Acceptance criteria frozen before implementation

1. Empty/blank policy IDs and source revisions fail closed.
2. SHA-256 digests require exactly 64 hexadecimal characters and are normalized
   to lowercase.
3. A manifest with identical tuning and confirmatory dataset digests fails
   closed.
4. The manifest stores the exact ordered route domains selected by the existing
   `AlgebraicRoute` and the exact `RecompositionPolicy`.
5. A matching confirmatory context validates successfully.
6. Mismatches in policy ID, source revision, route, recomposition policy,
   predicate digest, feature-schema digest, or holdout dataset digest each fail
   closed with distinct error evidence.
7. Supplying the frozen tuning dataset digest as the observed confirmatory
   dataset fails closed.
8. The manifest is immutable after construction; confirmatory validation does
   not mutate it.
9. Tests cover all four algebra domains in one frozen route and a smaller
   Boolean+F2 route, proving that route identity is not inferred from a policy
   label alone.
10. Existing MAA debug/release tests, deterministic smokes, documentation,
    repository CI/WGPU, semver, CodeQL and supply-chain gates remain green.

## Anti-leakage boundary

This slice cannot prove that a human never inspected holdout data before freezing
policy. It enforces the machine-checkable part of the protocol: once a frozen
manifest exists, an observation labeled confirmatory cannot silently substitute a
new policy, new features, new route, new source revision, or different dataset.
Git history provides the temporal evidence that this preregistration document and
manifest implementation precede later confirmatory result commits.

## Explicit non-claims

This work does not establish generalization, novelty, speedup, quality, sparsity,
latency, bandwidth, energy efficiency, or production readiness. It does not
change `flat_attention::api::v1`, default routing, or any GPU kernel. A later
MAA-11b experiment must freeze concrete policy/dataset digests in a separate
pre-result commit before producing confirmatory numerical observations.
