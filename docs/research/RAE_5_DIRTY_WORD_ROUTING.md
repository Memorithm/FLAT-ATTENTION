# RAE-5 — dirty-word deltas rather than full feature-state scanning

**Research-only Rust implementation**, based on the merged FLAT RAE-1/2/3/4
experiments. It is not a default attention path, GPU kernel or model gate.

## Problem removed — with a critical precondition

RAE-1 and RAE-2 scan all packed Boolean feature words on every logical
cycle. Even when no 0 -> 1 edge exists, this incurs
`O(ceil(feature_width / 64))` host word reads, before page dispatch.

RAE-5 introduces `rae_dirty_words::DirtyWordPageRouter`. Instead of
receiving a full snapshot per cycle, it accepts a strictly ascending,
deduplicated list of packed **word indices and their new full values**:

```rust
DirtyFeatureWord { index: 3, next: 0b0010 }
```

Only words identified as dirty are examined. A successful empty-delta
cycle reads **zero packed feature-state words** inside this router.
For a supplied dirty word, old/new Boolean arithmetic detects its
rising and falling features. Positive fronts are mapped to page IDs.
Repeated page targets coalesce through a tick-stamped page generation
table without scanning or clearing a full page bitset per tick.

The router keeps a persistent previous-state bitset, activation count,
latched-page bitset and per-page last-wake tick. Logical latches represent
history, not actual page residency or a K/V tensor. Both output page
vectors are caller-owned. Their capacity may grow on a busy tick;
zero allocation or zero memory traffic is **not** claimed.

On a valid tick, the router performs work proportional to supplied
dirty words and rising bits (+ emitted distinct page IDs), instead of
the full number of packed feature words. **However the upstream producer
must already know every changed word.** If computing that dirty list
requires comparing the entire state, total-system complexity may be
unchanged or worse.

## Fault contract and trust boundary

- Every changed packed u64 word must be supplied **exactly once** in
  increasing index order, including falling-only changes.
- Deltas use full replacement word values, not XOR masks.
- Initial high bits are latched, not treated as synthetic rising fronts.
- A valid empty delta is legal; it moves the logical clock forward.
- Reject invalid widths/pages/maps, non-monotonic ticks, unsorted or
  repeated indices, out-of-range words and tail-padding violations.
- Every validation failure leaves the router and caller-owned output
  vectors unchanged.
- `first_snapshot_mismatch` compares the maintained state to an
  independent authoritative full snapshot and exposes the first wrong
  packed-word index. This reconciliation **does scan all words** and
  needs a producer-controlled cadence/version/rollback policy.
- If the upstream source silently omits a dirty word, output may be
  wrong until reconciliation. The algorithm cannot infer missing events.
- Dirty signals never authorize excluding a page from numerical
  attention; query updates, causal masking, K/V writes, memory
  generations and model routing may change without a front.
- No artificial speedup claim: visited-word counts are algorithmic,
  not measured DRAM reads, CPU time or energy.

## Falsifiable test suite

- Bit-exact replay against **RAE-2 full-snapshot page wake routing**,
  including original feature IDs 63/64, aliasing, held-high state,
  falling and later rearming, exact active/popcount evidence.
- Empty-delta advances the clock without inspecting feature words.
- Malformed padding, repeated/invalid word indices and duplicate ticks
  are transactionally rejected.
- A deliberately omitted dirty notification creates a mismatch that
  the full-reference reconciliation detects; correcting the missing
  word restores parity.
- Initial high baseline emits no wake until falling and rearming.

## Implementing an actual lower-read upstream source

The **next** gated step is to source dirty u64 indices directly from a
native Boolean computation that already knows its changed destinations.
For SML-GENIUS this should be a model-owned R-field mutation/write
surface, not a Transformer QKV hook or a mandatory FLAT dependency.
Benchmark independently:

1. full packed scan (RAE-2);
2. dirty-word routing, with upstream notification generation cost included;
3. periodic authoritative reconciliation cost;
4. equal task/model quality and page-select correctness.

Only actual measured latency, CPU cache/DRAM counters or power and end-to-end
model task quality can establish system benefit. For physical experiments
use Memorithm/RemoteOps with exact pinned SHA and no NVIDIA SDK.

## Validation commands

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked -p flat-algebraic-attention rae_dirty_words
cargo test --locked --workspace
```

Related contracts:
`docs/research/RAE_1_RISING_EDGE_FRONTEND.md`,
`docs/research/RAE_2_PAGE_WAKES.md`,
`docs/research/RAE_3_MATCHED_POLICY_REPLAY.md`,
`docs/research/RAE_4_CAUSAL_NUMERIC_MASS.md`.
