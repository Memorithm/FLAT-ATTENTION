# FDAL5 — DA-LUC deterministic dynamic precision tiers

FDAL5 defines a research-only host oracle for assigning logical KV token
segments to explicit DA-LUC representation tiers. It does not mutate, transcode,
move, evict or materialize a KV payload and it does not promote a runtime path.

## Boundary

FLAT-ATTENTION owns the attention-facing tier selection semantics and the
reproducible deterministic baselines. A tier is only a caller-declared pair of
FDAL0 K and V representation descriptors. FDAL5 does **not** copy SLHAv2 tile or
cache-policy internals and does not define an NNIS physical layout.

No tier name or ordering is inferred from nominal bit width. The caller supplies
tiers in explicit selection-priority order and supplies an exact segment quota
for every tier.

## Versioned semantics

`DA_LUC_TIER_ROUTING_VERSION = 1`.

A routing plan records:

- DA-LUC KV schema version;
- routing policy;
- logical `kv_len`;
- fixed logical `segment_size`;
- one canonical assignment for every segment;
- the selected caller-defined tier id for each segment.

The final partial segment is permitted and ends exactly at `kv_len`.

Every tier representation is validated by constructing the corresponding FDAL0
contract over the same logical shape/layout. Invalid K/V representation choices
fail closed before a plan is emitted.

## Exact quotas

Every tier must have exactly one explicit quota. Quotas must sum to the exact
number of logical segments. There is no implicit fallback tier and no unassigned
segment.

This is a routing budget, not a storage/compression claim. FDAL1 exact storage
accounting remains the authority for effective bits/value of any materialized
representation.

## Deterministic baselines

### Recency

Segments are ranked newest first. Higher logical segment index means newer.
Equal-age ambiguity does not exist because every segment has a unique index.

### Attention mass

The caller supplies one finite, non-negative scalar per segment. Larger mass
ranks first. Equal mass uses lower segment index as the deterministic tie-break.

The supplied mass is logical routing evidence only. It is not physical bandwidth,
latency, quality or performance evidence.

## Explicit transitions

`DalucTierRoutingPlan::transitions_from` returns the exact segments whose tier id
changes between two compatible plans. Returning this list does not apply any
change. A runtime/cache owner must perform every transition explicitly and must
use its own qualified representation conversion/storage protocol.

There are no implicit precision transitions in FDAL5.

## Reproducible commands

Run the dedicated FDAL5 oracle tests:

```bash
cargo test --test fdal5_da_luc_tiering -- --nocapture
```

Run the complete host workspace:

```bash
cargo test --workspace --all-features
```

After merge, dispatch the exact qualification workflow:

```bash
gh workflow run fdal5-da-luc-tiering.yml \
  --repo Memorithm/FLAT-ATTENTION \
  --ref main
```

These are routing-semantics tests only. They do not constitute a DA-LUC physical
Thor performance, memory, quality, latency or tokens/second trial.

## Promotion boundary

FDAL5 v1 deliberately does not:

- materialize different tier payloads;
- move or evict pages;
- claim an optimal routing policy;
- claim a memory/compression ratio from nominal tier widths;
- use semantic/PHR-Lite routing;
- copy SLHAv2 cache internals;
- define an NNIS CUDA/NVRTC adapter;
- promote a GPU/runtime default.

PHR-Lite remains a later controlled experiment and must beat deterministic
recency and attention-mass baselines under the same storage and quality budget
before any promotion can be considered.
