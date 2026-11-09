# RAE-3 — matched policy replay with independently required page masks

Status: executable **research-only** host oracle in `flat-algebraic-attention`.
No changes to the default FLAT numerical attention path, stable public API,
SML model architecture, memory allocator or GPU kernel.

## Purpose

Compare actual candidate page selections by five policies over the **same**
chronological bit-packed input trace and an **independently supplied** set
of required original page IDs at each step. It is no longer sufficient to
report "few events": every saved page selection must be checked for missing
required work. The input contract does not certify independent truth
provenance; it must be established by the experiment harness.

Core module: `crates/flat-algebraic-attention/src/rae_policy_audit.rs`.

| Policy | Selected page set |
|---|---|
| AllPages | All original pages; structural no-miss control |
| RisingOnly | Pages addressed by 0 -> 1 front bits |
| AnyTransition | Pages addressed by either 0 -> 1 or 1 -> 0 bits |
| ActiveState | Pages addressed by currently high bits |
| RisingWithContextInvalidation | Any-transition pages, plus active pages if a trusted external context invalidation occurred |

The "hybrid" arm is **not guaranteed safe**. A required page may have no
active bit, even during invalidation. A query can change while a source bit
remains high; a single rising-front trace cannot establish numerical
attention equivalence.

The output retains each candidate's actual per-step packed page mask,
counts of selected, independently required, missing and excess pages,
and raw rising/falling feature counts. Totals are over a paired trace,
using original page IDs. Since the full masks are retained, page-index
comparisons are reproducible rather than inferred from aggregate counts.

AllPages is a structural upper bound, NOT a measured dense attention
throughput baseline; "required" is provided externally, NOT automatically
calculated from a dense QK softmax. These are **logical** page-selection
counts, not measured physical DRAM reads or total computational costs.

The Rust oracle rejects malformed input dimensions, out-of-range mapping
destinations, high padding bits, repeated/non-increasing logical tick
values, empty traces and excessive dimensions. The research replay is
bounded to 10,000 steps and 65,536 features/pages. It stores each row's
five selected masks so storage grows with sampled steps and page count.
These are explicit implementation limits, not workload measurements.

## Example and validation

```bash
cargo run --locked -p flat-algebraic-attention --example rae_paired_policy
cargo test --locked -p flat-algebraic-attention rae_policy_audit
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
```

The example deliberately includes a sustained high feature across a new
query that requires another page; rising-only and the naive hybrid fail
the independently supplied reference. Reporting this failure is
scientifically necessary and not a reason to weaken the reference.

## Next genuine model integration

- SML-GENIUS owns native SBG (field `R`), MOR decisions, learned
  expert/page selection, recurrent state, task labels and context/memory
  invalidation; it must export a **source-proven** paired trace with
  stable original page IDs and no mandatory dependency on FLAT.
- FLAT can consume appropriately mapped MAA survivor sets or qualified
  attention key/block masks and exact numerical attention oracles.
- Compare against existing MAA Boolean-only, relevant dense numeric
  reference, all-pages and matched-density controls. SML's independent
  MOR page oracle is not interchangeable with dense attention quality.
- Quantify false negatives and exact feature/page overhead before
  promoting an event-driven candidate. For zero omissions on one
  dataset, evaluate held-out adversarial changes in query, K/V, mask,
  semantic state, runtime hints and memory generations.
- Only hardware benchmarks through Memorithm/RemoteOps can support
  latency, bandwidth or energy claims, with exact source/device IDs.
- A physical speedup does not prove new scientific novelty.

Do not merge a runtime policy that drops numerical work based solely on
the absence of a 0 -> 1 transition.

Related: `docs/research/RAE_1_RISING_EDGE_FRONTEND.md`,
`docs/research/RAE_2_PAGE_WAKES.md`,
`docs/research/MULTI_ALGEBRA_ATTENTION.md` and
`docs/research/CYCLOID_FLAT_1_COSTED_SELECTION.md`.
