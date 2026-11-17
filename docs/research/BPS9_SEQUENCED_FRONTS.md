# BPS-9 — Source-bound sequenced front frames and sampled reference publisher

**Research-only, CPU Rust.** BPS-9 is a *functional admission protocol* for
BPS-8's watermarked recurrence. It is not a cryptographic authentication
implementation, a physical benchmark, or a production FLAT attention backend.

Source: `crates/flat-algebraic-attention/src/boolean_sequenced_publisher.rs`
Smoke: `crates/flat-algebraic-attention/examples/bps9_sequenced_front_smoke.rs`

## Problem solved

BPS-8 rejects stale/overlapping windows but does not establish whether
a publisher has dropped an entire intermediate frame or restarted a
sequence midstream. BPS-9 adds versioned frames with:

- fixed version `FRONT_WIRE_VERSION = 1`;
- source identifier and source epoch bound to an external `StreamAnchor`;
- gap-free `u64` frame sequence, with checked increment;
- predecessor watermark and newly closed watermark;
- predecessor and current deterministic frame commitments;
- the complete ordered front-bit/timestamp list for that closed interval.

The consumer validates all fields, exact chain continuity, size limits,
frame checksum and BPS-8 front/query constraints before accepting a
frame. If *anything* fails, the stream is **quarantined**; the last good
recurrent state, commitment, sequence, and watermark remain unchanged.
Quarantine cannot be cleared using untrusted data in the same runtime.
Resumption requires an *independently authenticated new StreamAnchor
and corresponding memory state*, provisioned by the host integration.

The consumer does not store earlier frames. Beyond the caller's owned
current frame and returned query outputs, the persistent runtime is O(1)
for one 64-bit recurrent state, a constant amount of source/session
state, and counters. For an already indexed event feed, recurrent
work is O(E+Q) for events E and requested queries Q, exclusive of source
acquisition, serialization, transport and cryptographic verification.

## Exact semantics and source limitations

The `SampledFrontPublisher` consumes *contiguous consecutive* sampled
64-bit words; it records the previous word across frames and computes
`rise = current & !previous` at each tick. That handles falling edges,
re-arming and held-high bits between windows exactly. Empty sample
frames, size excess, timestamp overflow and sequence exhaustion fail
without modifying the publisher's state.

This publisher proves correspondence only to the samples actually
received. It cannot detect missing physical samples or certify that the
upstream hardware/network delivered the complete original source.

`FrontFrame::claimed` also allows a pre-indexed external producer
to submit a self-described event frame with a deterministic checksum.
**This frame is not independently trusted**: its source ID/epoch and
watermark are only assertions. The checksum detects accidental
corruption and broken continuity under the established local session.
It is intentionally **not SHA-256, an HMAC, a digital signature, source
authentication or tamper resistance against an adversary able to
rewrite the frame and recompute its checksum**.

Even a perfectly sequenced, internally consistent stream can omit a
genuine rising event while claiming to be complete. A required negative
control demonstrates that such a fabricated but formally well-formed
feed differs from the BPS-3 dense oracle; independent sampled-source
certification detects the missing front. BPS-9 therefore does **not**
solve the external transport's authentic identity or physical
completeness problem.

A production adapter must define and enforce externally:

1. Mutual authentication/integrity for the actual transport and
   hardware-source identity, with verified binding to `SourceEpoch`.
2. A trustworthy epoch, monotonically increasing sequence and
   restart/checkpoint authority. A fresh local instance without a
   trusted persistent checkpoint cannot detect a replayed entire session.
3. Explicit loss detection for raw input samples (if any), watermark
   completeness semantics, bounded frame sizes, and rejected
   inconsistent clock epochs.
4. Separate identity/authorization and data-plane tests in the actual
   destination repository before any production routing.

No unreviewed auto-recovery, implicit re-initialization or default
backend promotion is implemented in this milestone.

## Qualification

Deterministic Rust tests include:

- exact comparisons of sequenced sampled publisher output against
  BPS-3 dense recurrence and BPS-7 indexed query across several
  baselines, shifts, batch boundaries and 197 sampled ticks;
- falling/re-arming exactly at consecutive frame boundaries;
- missing sequence and replay, changed source/epoch, malformed
  version, changed predecessor commitment and mutated event masks;
- size limits and consumer query validation before core mutation;
- atomic publisher rejection for empty samples, excessive windows,
  `u64` tick overflow and sequence exhaustion;
- honest negative control demonstrating that a forged, self-consistent
  frame can omit a genuine event and still pass local integrity checks.

CI gate for the exact PR HEAD:

    cargo fmt --all -- --check
    cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
    cargo test -p flat-algebraic-attention boolean_sequenced_publisher
    cargo run -p flat-algebraic-attention --example bps9_sequenced_front_smoke

All normal Rust/WGPU, MAA host, Semver, supply-chain and CodeQL checks
must pass before merge. A smoke result is not a hardware or Joule
measurement. The older BPS-6 RemoteOps RAPL protocol remains pinned
to its own source SHA and must not be changed or automatically
scheduled as part of this work.

## Integration candidates

An eventual V888/SML event publisher or FLAT async runtime may use the
same *conceptual* source/epoch/sequence checkpoint contract, but it
requires a separate consumer-specific adapter and full destination CI.
Do not insert a new mandatory FLAT runtime dependency into SML or
assume compatible event semantics without independent tests.
