//! BPS-10: keyed BLAKE3 authentication and checkpoint-bound recovery.
//!
//! Security scope: a 256-bit shared secret authenticates frame *bytes*
//! between principals already authorized to possess that key. No secret is
//! compiled into this crate, persisted here, or included in Debug output.
//! Key rotation and cryptographic identity binding are transport duties.
//!
//! This does NOT prove that a compromised/authorized sender delivered every
//! real front. Checkpoint rollback is detected only relative to an independent,
//! durable, trusted checkpoint head supplied on restore. Signing a checkpoint
//! alone cannot make it fresh. This module performs no file I/O.
//!
//! Research-only; BPS-3 dense recurrence remains the numerical oracle.

use blake3::{Hash, Hasher};

use super::boolean_edge_memory::BooleanEdgeProgram;
use super::boolean_sequenced_publisher::{
    AcceptedFrontWindow, FrameLimits, FrontFrame, FrontFrameError, SampledFrontPublisher,
    SequencedFrontRuntime, SourceEpoch, StreamAnchor,
};

pub const BPS10_AUTH_VERSION: u16 = 1;
const FRAME_DOMAIN: &[u8] = b"Memorithm.FLAT.BPS10.frame.keyed.v1";
const CHECKPOINT_DOMAIN: &[u8] = b"Memorithm.FLAT.BPS10.checkpoint.keyed.v1";
const MODEL_DOMAIN: &[u8] = b"Memorithm.FLAT.BPS10.program.v1";

fn put_u16(h: &mut Hasher, x: u16) {
    h.update(&x.to_le_bytes());
}
fn put_u64(h: &mut Hasher, x: u64) {
    h.update(&x.to_le_bytes());
}
fn put_usize(h: &mut Hasher, x: usize) {
    put_u64(h, u64::try_from(x).expect("64-bit count required by BPS-10 protocol"));
}
fn put_source(h: &mut Hasher, source: SourceEpoch) {
    put_u64(h, source.source_id);
    put_u64(h, source.epoch_id);
}
fn same_tag(a: &[u8; 32], b: &[u8; 32]) -> bool {
    // NEVER compare raw MAC byte slices: blake3::Hash Eq is constant-time.
    Hash::from_bytes(*a) == Hash::from_bytes(*b)
}

fn model_binding(program: BooleanEdgeProgram, limits: FrameLimits) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(MODEL_DOMAIN);
    for gate in [program.forget, program.write] {
        put_u64(&mut h, gate.when_false);
        put_u64(&mut h, gate.when_true);
    }
    put_u64(&mut h, program.readout_mask);
    h.update(&[program.shift()]);
    put_usize(&mut h, limits.max_fronts);
    put_usize(&mut h, limits.max_queries);
    put_u64(&mut h, limits.max_advance_ticks);
    *h.finalize().as_bytes()
}

/// A MAC-authenticated BPS-9 frame. Public fields are wire fields and MUST be
/// verified by AuthenticatedFrontRuntime before touching Boolean memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticatedFrontFrame {
    pub auth_version: u16,
    pub key_id: u64,
    pub previous_mac: [u8; 32],
    pub frame: FrontFrame,
    pub mac: [u8; 32],
}

fn compute_frame_mac(key: &[u8; 32], tagged: &AuthenticatedFrontFrame) -> [u8; 32] {
    let mut h = Hasher::new_keyed(key);
    h.update(FRAME_DOMAIN);
    put_u16(&mut h, tagged.auth_version);
    put_u64(&mut h, tagged.key_id);
    h.update(&tagged.previous_mac);
    let frame = &tagged.frame;
    put_u16(&mut h, frame.version);
    put_source(&mut h, frame.source);
    put_u64(&mut h, frame.sequence);
    put_u64(&mut h, frame.prior_commitment);
    put_u64(&mut h, frame.closed_from);
    put_u64(&mut h, frame.watermark);
    put_usize(&mut h, frame.fronts.len());
    for front in &frame.fronts {
        put_u64(&mut h, front.tick);
        put_u64(&mut h, front.rising);
    }
    put_u64(&mut h, frame.commitment);
    *h.finalize().as_bytes()
}

/// Only an already authorized sender should have access to this secret.
/// This function cannot authenticate the genuine source of a supplied frame
/// against an adversary *also* holding the same key.
#[must_use]
pub fn authenticate_frame(
    frame: FrontFrame,
    key_id: u64,
    previous_mac: [u8; 32],
    secret: &[u8; 32],
) -> AuthenticatedFrontFrame {
    let mut tagged = AuthenticatedFrontFrame {
        auth_version: BPS10_AUTH_VERSION,
        key_id,
        previous_mac,
        frame,
        mac: [0; 32],
    };
    tagged.mac = compute_frame_mac(secret, &tagged);
    tagged
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthError {
    Quarantined,
    UnsupportedVersion,
    KeyIdMismatch,
    PreviousMacMismatch,
    MacMismatch,
    Frame(FrontFrameError),
    CheckpointMacMismatch,
    TrustedHeadMismatch,
    ModelMismatch,
    SourceMismatch,
}

/// Checkpoint contains the BPS-9 source/sequence/commitment/watermark
/// together with the full Boolean recurrent memory and previous MAC.
/// This is data, not a trustworthy recovery authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticatedCheckpoint {
    pub auth_version: u16,
    pub key_id: u64,
    pub source: SourceEpoch,
    pub next_sequence: u64,
    pub prior_commitment: u64,
    pub closed_through: u64,
    pub memory_state: u64,
    pub previous_frame_mac: [u8; 32],
    pub model_binding: [u8; 32],
    pub mac: [u8; 32],
}

fn compute_checkpoint_mac(key: &[u8; 32], checkpoint: &AuthenticatedCheckpoint) -> [u8; 32] {
    let mut h = Hasher::new_keyed(key);
    h.update(CHECKPOINT_DOMAIN);
    put_u16(&mut h, checkpoint.auth_version);
    put_u64(&mut h, checkpoint.key_id);
    put_source(&mut h, checkpoint.source);
    put_u64(&mut h, checkpoint.next_sequence);
    put_u64(&mut h, checkpoint.prior_commitment);
    put_u64(&mut h, checkpoint.closed_through);
    put_u64(&mut h, checkpoint.memory_state);
    h.update(&checkpoint.previous_frame_mac);
    h.update(&checkpoint.model_binding);
    *h.finalize().as_bytes()
}

/// This EXACT head must come from a separately trusted durable monotonic
/// authority. Returning it from the same untrusted checkpoint is NOT valid.
/// A key-valid older checkpoint must be rejected after a newer head is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrustedCheckpointHead {
    pub source: SourceEpoch,
    pub key_id: u64,
    pub checkpoint_mac: [u8; 32],
}

/// Constructed from a trusted anchor and program. No key is retained:
/// caller passes the high-entropy 32-byte secret for each operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticatedFrontRuntime {
    core: SequencedFrontRuntime,
    source: SourceEpoch,
    key_id: u64,
    last_mac: [u8; 32],
    program: BooleanEdgeProgram,
    limits: FrameLimits,
    quarantined: bool,
}

impl AuthenticatedFrontRuntime {
    #[must_use]
    pub const fn new(
        program: BooleanEdgeProgram,
        state: u64,
        anchor: StreamAnchor,
        limits: FrameLimits,
        key_id: u64,
        initial_mac: [u8; 32],
    ) -> Self {
        Self {
            core: SequencedFrontRuntime::new(program, state, anchor, limits),
            source: anchor.source,
            key_id,
            last_mac: initial_mac,
            program,
            limits,
            quarantined: false,
        }
    }

    #[must_use]
    pub const fn state(&self) -> u64 {
        self.core.state()
    }

    #[must_use]
    pub const fn closed_through(&self) -> u64 {
        self.core.closed_through()
    }

    #[must_use]
    pub const fn next_sequence(&self) -> u64 {
        self.core.next_sequence()
    }

    #[must_use]
    pub const fn previous_mac(&self) -> [u8; 32] {
        self.last_mac
    }

    #[must_use]
    pub const fn quarantined(&self) -> bool {
        self.quarantined || self.core.quarantined()
    }

    /// Any failure blocks this session permanently. The failure itself may
    /// change only the quarantine flag; never the last accepted memory state,
    /// watermark, BPS-9 commitment, sequence or accepted MAC.
    pub fn ingest(
        &mut self,
        incoming: &AuthenticatedFrontFrame,
        queries: &[u64],
        key: &[u8; 32],
    ) -> Result<AcceptedFrontWindow, AuthError> {
        if self.quarantined() {
            return Err(AuthError::Quarantined);
        }
        let checked = self.validate_and_advance(incoming, queries, key);
        if checked.is_err() {
            self.quarantined = true;
        }
        checked
    }

    fn validate_and_advance(
        &mut self,
        incoming: &AuthenticatedFrontFrame,
        queries: &[u64],
        key: &[u8; 32],
    ) -> Result<AcceptedFrontWindow, AuthError> {
        if incoming.auth_version != BPS10_AUTH_VERSION {
            return Err(AuthError::UnsupportedVersion);
        }
        if incoming.key_id != self.key_id {
            return Err(AuthError::KeyIdMismatch);
        }
        if !same_tag(&incoming.previous_mac, &self.last_mac) {
            return Err(AuthError::PreviousMacMismatch);
        }
        if !same_tag(&compute_frame_mac(key, incoming), &incoming.mac) {
            return Err(AuthError::MacMismatch);
        }
        // The BPS-9 consumer independently checks version/source/epoch,
        // gap-free sequence and input shape. This runs only after the MAC.
        let receipt = self
            .core
            .ingest(&incoming.frame, queries)
            .map_err(AuthError::Frame)?;
        self.last_mac = incoming.mac;
        Ok(receipt)
    }

    /// Pure snapshot. The caller must durably persist this checkpoint
    /// atomically with a matching TrustedCheckpointHead from an independent
    /// storage authority before treating it as restartable.
    pub fn checkpoint(&self, key: &[u8; 32]) -> Result<AuthenticatedCheckpoint, AuthError> {
        if self.quarantined() {
            return Err(AuthError::Quarantined);
        }
        let mut checkpoint = AuthenticatedCheckpoint {
            auth_version: BPS10_AUTH_VERSION,
            key_id: self.key_id,
            source: self.source,
            next_sequence: self.core.next_sequence(),
            prior_commitment: self.core.prior_commitment(),
            closed_through: self.core.closed_through(),
            memory_state: self.core.state(),
            previous_frame_mac: self.last_mac,
            model_binding: model_binding(self.program, self.limits),
            mac: [0; 32],
        };
        checkpoint.mac = compute_checkpoint_mac(key, &checkpoint);
        Ok(checkpoint)
    }

    /// Restore ONLY if an independently persisted high-water head exactly
    /// matches the authenticated checkpoint. A stale head supplied alongside
    /// the stale checkpoint is not trustworthy and cannot prevent rollback.
    pub fn restore(
        checkpoint: &AuthenticatedCheckpoint,
        durable_head: &TrustedCheckpointHead,
        program: BooleanEdgeProgram,
        limits: FrameLimits,
        expected_key_id: u64,
        key: &[u8; 32],
    ) -> Result<Self, AuthError> {
        if checkpoint.auth_version != BPS10_AUTH_VERSION {
            return Err(AuthError::UnsupportedVersion);
        }
        if checkpoint.key_id != expected_key_id || durable_head.key_id != expected_key_id {
            return Err(AuthError::KeyIdMismatch);
        }
        if checkpoint.source != durable_head.source {
            return Err(AuthError::SourceMismatch);
        }
        if !same_tag(&compute_checkpoint_mac(key, checkpoint), &checkpoint.mac) {
            return Err(AuthError::CheckpointMacMismatch);
        }
        if !same_tag(&checkpoint.mac, &durable_head.checkpoint_mac) {
            return Err(AuthError::TrustedHeadMismatch);
        }
        if !same_tag(&checkpoint.model_binding, &model_binding(program, limits)) {
            return Err(AuthError::ModelMismatch);
        }
        let anchor = StreamAnchor {
            source: checkpoint.source,
            next_sequence: checkpoint.next_sequence,
            prior_commitment: checkpoint.prior_commitment,
            closed_through: checkpoint.closed_through,
        };
        Ok(Self::new(
            program,
            checkpoint.memory_state,
            anchor,
            limits,
            expected_key_id,
            checkpoint.previous_frame_mac,
        ))
    }
}

/// Contiguous sampled publisher with a 32-byte MAC chain. It authenticates
/// data *relative to the caller-provisioned key*, not physical completeness.
/// The secret is never stored by the publisher.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthenticatedSampledPublisher {
    source: SampledFrontPublisher,
    key_id: u64,
    previous_mac: [u8; 32],
}

impl AuthenticatedSampledPublisher {
    #[must_use]
    pub const fn new(
        anchor: StreamAnchor,
        baseline_sample: u64,
        key_id: u64,
        initial_mac: [u8; 32],
    ) -> Self {
        Self {
            source: SampledFrontPublisher::new(anchor, baseline_sample),
            key_id,
            previous_mac: initial_mac,
        }
    }

    pub fn publish(
        &mut self,
        samples: &[u64],
        max_samples: usize,
        key: &[u8; 32],
    ) -> Result<AuthenticatedFrontFrame, FrontFrameError> {
        let frame = self.source.publish_contiguous_samples(samples, max_samples)?;
        let tagged = authenticate_frame(frame, self.key_id, self.previous_mac, key);
        self.previous_mac = tagged.mac;
        Ok(tagged)
    }

    #[must_use]
    pub const fn anchor(&self) -> StreamAnchor {
        self.source.anchor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boolean_indexed_fronts::sampled_indexed_fronts;
    use crate::boolean_scan_lstm::UnaryGate;

    // Deterministic test fixture ONLY; never use a repeated-byte key in real
    // deployment. Production secrets must be generated externally.
    const KEY: [u8; 32] = [0x39; 32];
    const WRONG_KEY: [u8; 32] = [0x92; 32];
    const KEY_ID: u64 = 9;
    const INITIAL_MAC: [u8; 32] = [0x45; 32];
    const ANCHOR: StreamAnchor = StreamAnchor {
        source: SourceEpoch { source_id: 31, epoch_id: 7 },
        next_sequence: 1,
        prior_commitment: 0x9a12_f443_1202_42d2,
        closed_through: 0,
    };
    const LIMITS: FrameLimits = FrameLimits {
        max_fronts: 128,
        max_queries: 128,
        max_advance_ticks: 128,
    };

    fn program(shift: u8) -> BooleanEdgeProgram {
        BooleanEdgeProgram::new(
            UnaryGate { when_false: u64::MAX, when_true: u64::MAX },
            UnaryGate::default(), u64::MAX, shift,
        ).unwrap()
    }

    fn make_frame() -> AuthenticatedFrontFrame {
        let payload = FrontFrame::claimed(
            ANCHOR, 4,
            vec![super::super::boolean_indexed_fronts::IndexedFront {
                tick: 2, rising: 1,
            }],
        );
        authenticate_frame(payload, KEY_ID, INITIAL_MAC, &KEY)
    }

    #[test]
    fn complete_authenticated_stream_matches_dense_and_bps7_oracles() {
        for shift in [0, 1, 7, 63] {
            let p = program(shift);
            let inputs = [1, 1, 0, 1, 0, 5, 5, 0, 7, 0, 7];
            let reference = p.forward_dense(&inputs, 0, 1);
            let mut publisher = AuthenticatedSampledPublisher::new(ANCHOR, 0, KEY_ID, INITIAL_MAC);
            let mut runtime = AuthenticatedFrontRuntime::new(p, 1, ANCHOR, LIMITS, KEY_ID, INITIAL_MAC);
            let mut output = Vec::new();
            let mut start = 0usize;
            for end in [1, 4, 7, 11] {
                let tagged = publisher.publish(&inputs[start..end], 64, &KEY).unwrap();
                let q: Vec<u64> = (start as u64 + 1..=end as u64).collect();
                let accepted = runtime.ingest(&tagged, &q, &KEY).unwrap();
                output.extend(accepted.data.outputs);
                start = end;
            }
            assert_eq!(output, reference);
            let indexed = p.forward_indexed_queries(
                &sampled_indexed_fronts(&inputs, 0),
                &(1..=inputs.len() as u64).collect::<Vec<_>>(),
                inputs.len() as u64, 1,
            ).unwrap();
            assert_eq!(runtime.state(), indexed.final_state);
            assert!(!runtime.quarantined());
            assert_eq!(runtime.next_sequence(), 5);
        }
    }

    #[test]
    fn tampering_each_protected_field_and_wrong_key_quarantines_atomically() {
        for case in 0..11 {
            let mut runtime = AuthenticatedFrontRuntime::new(
                program(1), 1, ANCHOR, LIMITS, KEY_ID, INITIAL_MAC,
            );
            let original = make_frame();
            let mut changed = original.clone();
            match case {
                0 => changed.auth_version += 1,
                1 => changed.key_id += 1,
                2 => changed.previous_mac[0] ^= 1,
                3 => changed.frame.source.source_id += 1,
                4 => changed.frame.source.epoch_id += 1,
                5 => changed.frame.sequence += 1,
                6 => changed.frame.closed_from += 1,
                7 => changed.frame.watermark += 1,
                8 => changed.frame.fronts[0].rising ^= 2,
                9 => changed.frame.commitment ^= 1,
                10 => changed.mac[3] ^= 1,
                _ => unreachable!(),
            }
            let before = runtime.clone();
            assert!(runtime.ingest(&changed, &[3], &KEY).is_err(), "case={case}");
            assert!(runtime.quarantined());
            assert_eq!(runtime.state(), before.state());
            assert_eq!(runtime.next_sequence(), before.next_sequence());
            assert_eq!(runtime.closed_through(), before.closed_through());
            assert_eq!(runtime.previous_mac(), before.previous_mac());
            assert_eq!(runtime.ingest(&original, &[3], &KEY), Err(AuthError::Quarantined));
        }
        let mut runtime = AuthenticatedFrontRuntime::new(
            program(1), 1, ANCHOR, LIMITS, KEY_ID, INITIAL_MAC,
        );
        assert_eq!(runtime.ingest(&make_frame(), &[3], &WRONG_KEY), Err(AuthError::MacMismatch));
        assert!(runtime.quarantined());
    }

    #[test]
    fn stale_replay_fails_even_with_valid_old_mac() {
        let p = program(1);
        let mut producer = AuthenticatedSampledPublisher::new(ANCHOR, 0, KEY_ID, INITIAL_MAC);
        let first = producer.publish(&[1, 0], 3, &KEY).unwrap();
        let second = producer.publish(&[1, 0], 3, &KEY).unwrap();
        let mut consumer = AuthenticatedFrontRuntime::new(p, 1, ANCHOR, LIMITS, KEY_ID, INITIAL_MAC);
        consumer.ingest(&first, &[1, 2], &KEY).unwrap();
        consumer.ingest(&second, &[3, 4], &KEY).unwrap();
        let previous = consumer.clone();
        assert!(consumer.ingest(&first, &[1], &KEY).is_err());
        assert!(consumer.quarantined());
        assert_eq!(consumer.state(), previous.state());
        assert_eq!(consumer.next_sequence(), previous.next_sequence());
        assert_eq!(consumer.previous_mac(), previous.previous_mac());
    }

    #[test]
    fn verified_checkpoint_resume_matches_uninterrupted_recurrence() {
        let p = program(7);
        let mut publisher = AuthenticatedSampledPublisher::new(ANCHOR, 0, KEY_ID, INITIAL_MAC);
        let first = publisher.publish(&[1, 1, 0, 1], 4, &KEY).unwrap();
        let second = publisher.publish(&[0, 3, 3, 0], 4, &KEY).unwrap();
        let mut continuous = AuthenticatedFrontRuntime::new(p, 1, ANCHOR, LIMITS, KEY_ID, INITIAL_MAC);
        continuous.ingest(&first, &[1, 4], &KEY).unwrap();
        let checkpoint = continuous.checkpoint(&KEY).unwrap();
        let durable_head = TrustedCheckpointHead {
            source: checkpoint.source,
            key_id: checkpoint.key_id,
            checkpoint_mac: checkpoint.mac,
        };
        let mut restored = AuthenticatedFrontRuntime::restore(
            &checkpoint, &durable_head, p, LIMITS, KEY_ID, &KEY,
        ).unwrap();
        let direct = continuous.ingest(&second, &[6, 8], &KEY).unwrap();
        let replay = restored.ingest(&second, &[6, 8], &KEY).unwrap();
        assert_eq!(direct.data, replay.data);
        assert_eq!(continuous.state(), restored.state());
        assert_eq!(continuous.previous_mac(), restored.previous_mac());
        assert_eq!(continuous.next_sequence(), restored.next_sequence());
    }

    #[test]
    fn checkpoint_tamper_key_model_and_durable_rollback_rejected() {
        let p = program(1);
        let mut publisher = AuthenticatedSampledPublisher::new(ANCHOR, 0, KEY_ID, INITIAL_MAC);
        let first = publisher.publish(&[1, 0], 2, &KEY).unwrap();
        let second = publisher.publish(&[1, 0], 2, &KEY).unwrap();
        let mut consumer = AuthenticatedFrontRuntime::new(p, 1, ANCHOR, LIMITS, KEY_ID, INITIAL_MAC);
        consumer.ingest(&first, &[2], &KEY).unwrap();
        let older = consumer.checkpoint(&KEY).unwrap();
        consumer.ingest(&second, &[4], &KEY).unwrap();
        let latest = consumer.checkpoint(&KEY).unwrap();
        let head = TrustedCheckpointHead {
            source: latest.source, key_id: latest.key_id, checkpoint_mac: latest.mac,
        };
        assert_eq!(
            AuthenticatedFrontRuntime::restore(&older, &head, p, LIMITS, KEY_ID, &KEY),
            Err(AuthError::TrustedHeadMismatch)
        );
        assert_eq!(
            AuthenticatedFrontRuntime::restore(&latest, &head, p, LIMITS, KEY_ID, &WRONG_KEY),
            Err(AuthError::CheckpointMacMismatch)
        );
        assert_eq!(
            AuthenticatedFrontRuntime::restore(&latest, &head, program(7), LIMITS, KEY_ID, &KEY),
            Err(AuthError::ModelMismatch)
        );
        let changed_limits = FrameLimits { max_fronts: 64, ..LIMITS };
        assert_eq!(
            AuthenticatedFrontRuntime::restore(&latest, &head, p, changed_limits, KEY_ID, &KEY),
            Err(AuthError::ModelMismatch)
        );
        let mut tampered = latest.clone();
        tampered.memory_state ^= 1;
        assert_eq!(
            AuthenticatedFrontRuntime::restore(&tampered, &head, p, LIMITS, KEY_ID, &KEY),
            Err(AuthError::CheckpointMacMismatch)
        );
        let other_head = TrustedCheckpointHead {
            source: SourceEpoch { epoch_id: 99, ..latest.source }, ..head
        };
        assert_eq!(
            AuthenticatedFrontRuntime::restore(&latest, &other_head, p, LIMITS, KEY_ID, &KEY),
            Err(AuthError::SourceMismatch)
        );
    }

    #[test]
    fn authenticated_but_incomplete_source_remains_unprovable() {
        let p = program(1);
        let inputs = [1, 0, 1];
        // A sender who possesses the key can sign an omitted-event lie.
        let lying = authenticate_frame(
            FrontFrame::claimed(ANCHOR, 3, vec![]),
            KEY_ID, INITIAL_MAC, &KEY,
        );
        let mut consumer = AuthenticatedFrontRuntime::new(p, 1, ANCHOR, LIMITS, KEY_ID, INITIAL_MAC);
        let got = consumer.ingest(&lying, &[3], &KEY).unwrap().data.outputs[0];
        assert_ne!(got, p.forward_dense(&inputs, 0, 1)[2]);
        assert!(!consumer.quarantined());
    }

    #[test]
    fn publisher_errors_do_not_change_mac_chain() {
        let mut publisher = AuthenticatedSampledPublisher::new(ANCHOR, 0, KEY_ID, INITIAL_MAC);
        let snapshot = publisher.clone();
        assert_eq!(
            publisher.publish(&[], 32, &KEY),
            Err(FrontFrameError::EmptySampleWindow)
        );
        assert_eq!(publisher, snapshot);
    }
}
