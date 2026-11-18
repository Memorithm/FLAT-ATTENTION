# BPS-10 — Keyed frame authentication and durable-head checkpoint recovery

**Scope:** Rust research component, not a production-authenticated input service,
not physical measurements, and not an external Memorithm/RemoteOps deployment.

Source: `crates/flat-algebraic-attention/src/boolean_authenticated_frames.rs`
Example: `crates/flat-algebraic-attention/examples/bps10_authenticated_checkpoint_smoke.rs`

## Specific advancement over BPS-9

BPS-9 provides ordered frames and a 64-bit checksum which is expressly
*non-cryptographic*. An attacker who can rewrite frames can recompute that
checksum. BPS-10 adds an independently provisioned **256-bit keyed BLAKE3 MAC**
for each complete frame. Its canonical little-endian input binds:

- a domain-separation string and BPS-10 authentication format version;
- `key_id` (public identifier, not a key or secret);
- previous full **256-bit MAC**, forming a cryptographic chain;
- BPS-9 wire version, source ID and epoch, frame sequence and predecessor
  64-bit checksum;
- previous and next watermarks;
- the declared length, complete ordered front timestamps and rising-bit masks;
- the original BPS-9 checksum (defense in depth, not required for MAC security).

All MAC comparisons use `blake3::Hash` constant-time equality, **never raw
`[u8;32]` equality**. The implementation pins official BLAKE3 1.8.7 with
pure-Rust execution so no project-authored C/ASM or vendor SDK is required.
The dependency lock must be generated and reviewed using Cargo on the
exact toolchain; no invented registry checksums.

An `AuthenticatedFrontRuntime` wraps BPS-9's `SequencedFrontRuntime`.
On a wrong key ID, missing/altered predecessor MAC, incorrect MAC or
failed BPS-9 check, the stream is **quarantined without changing the
previously accepted Boolean state, sequence, watermark, commitment or
MAC**. No untrusted same-session unquarantine exists.

`AuthenticatedSampledPublisher` emits authenticated frames from
**contiguous sampled words**, preserving BPS-9 rising-front semantics and
the previous input word across frame boundaries. As before, producing
such a frame scans each supplied sampled word and costs O(T) upstream.

## Restart checkpoint contract

The session produces a **MAC-authenticated checkpoint** containing:

- version, key ID, BPS-9 source/epoch, next sequence, last BPS-9 checksum,
  last closed watermark and full recurrent memory;
- the previous full authenticated frame MAC;
- a model binding over **all Boolean gate masks, readout mask, rotation
  and limits**, preventing restoration under a different program/limits;
- a separate keyed BLAKE3 checkpoint MAC with a **different domain separator**
  from frame MACs.

To restore, the caller must supply the checkpoint, current model, limits,
exact expected key ID and secret, **and a matching `TrustedCheckpointHead`
from independently trusted durable storage**. The checkpoint MAC,
trusted-head MAC, source ID/epoch, model and limits must match before
instantiating a fresh stream.

**Crucial anti-rollback condition:** the separately trusted head must
advance and be persisted *atomically* with the accepted checkpoint and
external source cursor. Passing a checkpoint and a head copied from the
same untrusted blob is NOT rollback protection. An older validly signed
checkpoint remains cryptographically authentic, but is rejected when
the trusted durable head has advanced. Availability after crash requires
a trusted source that can replay or resume from the persisted source
cursor; this module has **no persistence, key store, process supervisor
or durable source-cursor implementation**. Recovery may be impossible
if the producer has lost historical data.

No raw key bytes are stored inside publisher/runtime structs. Every
operation receives a borrowed `&[u8;32]` supplied by the caller. The
project does not generate keys or provide process memory pinning,
automatic zeroization of caller memory, rotation, revocation or secure
external transport. The demo/test key is deterministic and **must never
be used for an actual deployment**.

## What this does NOT establish

A MAC authenticates data possession by an authorized principal **sharing
the same secret**, not unique hardware identity or truth of its
underlying physical observations. A compromised/key-holding producer
can sign a missing-event lie. A deterministic negative-control test
demonstrates this and compares the result with BPS-3's dense oracle.

`source_id` and `epoch_id` are bound to the MAC but are **not
independently authenticated** here. Correct key-to-source authorization,
strong entropy, fresh key provisioning, trusted persistent highwater
state, mutual identity verification, input completeness, fault handling,
transport replay filtering, and epoch rekey are integration contracts
for an external service. A compromised key cannot be rescued by an
in-band checksum.

No FP32 LSTM superiority, measurable latency, Joule savings, encrypted
transport, production readiness or ML 5/5 qualification is claimed.
The BPS-6 RemoteOps physical workflow remains source-pinned to its
independently reviewed BPS-6 SHA and is not modified.

## Deterministic acceptance

The new tests cover:

- MAC-authenticated frames from contiguous samples against BPS-3 and
  BPS-7 dense/indexed oracle results for multiple recurrence shifts;
- mutations of version, key ID, previous MAC, source/epoch, sequence,
  watermark, event payload, weak checksum, full MAC and wrong key;
- fail-closed quarantine with **exact prior state/sequence/watermark/MAC**;
- valid contiguous checkpoint restore versus uninterrupted operation;
- rejected rollback of an older *valid* checkpoint against the newest
  trusted head;
- wrong model, limits, source, key and checkpoint-state tampering;
- a maliciously incomplete but correctly MAC-authenticated frame
  negative control.

Commands (Rust 1.89):

    cargo fmt --all -- --check
    cargo clippy -p flat-algebraic-attention --all-targets -- -D warnings
    cargo test -p flat-algebraic-attention boolean_authenticated_frames
    cargo run -p flat-algebraic-attention --example bps10_authenticated_checkpoint_smoke

Full exact-HEAD `ci` (Rust/WGPU), `maa-host`, `semver`, `supply-chain`
and `codeql` must succeed before merge. Cryptographic claims are
limited to keyed authentication under BLAKE3's assumptions; formal
security proofs and external deployment remain out of scope.
