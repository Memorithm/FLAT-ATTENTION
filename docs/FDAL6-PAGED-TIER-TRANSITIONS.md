# FDAL6: cache-scoped page tier transitions

Research-only planning API. This extends the cache-scoped binding introduced by PR #200; it does not change the stable `api::v1` contract or numerical attention routing.

## Contract

`DalucCacheScopedPagedTierBinding` retains the validated `DalucKvViewContract`, an independent tier catalog and the existing opaque cache checkpoint. Read-only access is provided by `binding()`, `contract()` and `tiers()`.

`next.transitions_from(&previous, &cache)` returns an immutable `DalucPagedTierTransitionPlan`. Its version is `DA_LUC_PAGED_TIER_TRANSITION_VERSION = 1`.

Let `A[p]` and `B[p]` be the previous and next declared tier IDs for logical page `p`. After validation, the result contains exactly the pages where `A[p] != B[p]`, in increasing logical-page order. Each record carries:

- logical and physical page IDs;
- the exact half-open token interval, including a partial last page;
- generation and branch epoch;
- previous and next tier IDs.

Unchanged tier IDs are omitted. This is a difference between **declared assignments**, not evidence that either representation exists in device memory. Different IDs with identical descriptors still produce an assignment-change record.

## Preconditions and rejection behavior

Both bindings must independently validate against the same currently borrowed live cache. A foreign previous binding fails before the next binding is considered. A foreign next binding is also rejected even when all exposed metadata is identical.

The complete captured view contracts must be equal. This intentionally rejects layout-only differences, different base representation descriptors and different attention geometry. The first version does not attempt a generalized layout conversion.

The catalogs must contain exactly the same tier IDs with identical K/V representation descriptors. Catalog order may change, because FDAL5 uses caller order as selection priority. Reusing an ID with a different key bit order, value dtype or other descriptor is rejected. Adding or removing even an unused tier is conservatively rejected.

The output allocation uses `try_reserve_exact` for the number of changed assignments. Failure is an explicit error. The plan borrows the two immutable bindings instead of cloning their catalogs and page assignment vectors.

## Revalidation is required

`plan.validate_cache(&cache)` reruns the binding and compatibility checks. Append makes a fixed-length plan stale. Truncate/reappend invalidates its branch, and reset/reappend invalidates its generation even when the length returns to its old value. Externally recorded writes remain fail-closed.

The plan does not borrow or lock the cache for its whole lifetime. A successful validation is only a point-in-time metadata check: it is not an execution lease, synchronization fence or protection against later mutation. A future executor must keep its own appropriate exclusive scope and GPU ordering across validation and actuation.

## What this does not establish

An empty plan does not prove payload equality. The catalog contains representation descriptors, not hashes of codebook contents or K/V bytes. Matching descriptors do not establish identical codebooks. The checkpoint establishes process-local cache-instance/lifecycle scope, not content identity or GPU completion.

No API in this tranche transcodes, copies, demotes, promotes or evicts a page. Representation tier changes must not be translated into `KvResidencyEventKind` automatically. There is no latency, memory-saving, model-quality or end-to-end speedup claim.

## Qualification

The dedicated FDAL6 workflow checks the exact PR head with Rust 1.89, strict format/lint/documentation and a mandatory WGPU adapter. Its three test suites are:

```sh
cargo test --locked --features wgpu --test fdal6_da_luc_paged_binding -- --nocapture
cargo test --locked --features wgpu --test fdal6_da_luc_cache_scoped_binding -- --nocapture
cargo test --locked --features wgpu --test fdal6_da_luc_paged_transitions -- --nocapture
```

The transition regressions cover no-op plans and read-only capture, deterministic ordering and partial pages, reordered compatible catalogs, redefined K/V descriptors, changed catalog membership, changed row layout, foreign instances, stale previous/next bindings, stale replay after append/truncate/reset and external recording. These are structural/device-API checks, not numerical transcode parity or representative-model benchmarks.

## Next execution and ecosystem gates

FLAT remains the owner of physical page mapping and attention semantics. KVLab can use the declared changes as experimental planning metadata, but any serialized evidence must carry its own provenance and cannot recreate the opaque in-process cache scope.

Before an executor in FLAT, SLHAv2 or NNIS acts on these records, add explicit materialization/codebook identity, supported codec conversion, resource bounds, synchronization, rollback and independent numerical validation. ElasticXxx can orchestrate such a qualified executor; a planning record alone must never be reported as completed actuation or measured memory release.

Cross-model KV transfer is a separate experiment: source and receiver model identities, tokenization, layer/head mapping, RoPE convention and held-out output quality cannot be inferred from this page-tier contract. No cross-model mapper is implemented here.
