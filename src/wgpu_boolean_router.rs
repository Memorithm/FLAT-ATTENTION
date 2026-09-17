//! M13B.3 WGPU Boolean front-end routing before numerical K/V staging.
//!
//! This module consumes only Boolean Q/K signatures and writes one admission
//! flag per candidate block. It deliberately has no K/V buffer binding: the
//! resulting admission plane can therefore be produced before a caller stages
//! numerical K/V. Compaction and numerical attention remain separate steps.

use core::fmt;
use std::fmt::Write as _;

use crate::api::boolean_attention_mask::{BooleanAttentionMask, BooleanAttentionMaskError};
use crate::api::boolean_attention_signature::{
    BooleanAttentionSignature, BooleanAttentionSignatureError, HammingAdmissionRule,
};
use crate::fingerprint::fnv1a64;
use crate::wgpu_internal;

const ROUTER_WORKGROUP_SIZE: u32 = 64;

/// Canonical research-only parity record for FLAT BKV-4 / KVLab BKV-K7.
pub const BOOLEAN_KV_WGPU_PARITY_SCHEMA: &str = "flat.boolean-kv-wgpu-parity.v1";

const BOOLEAN_ROUTER_WGSL: &str = r#"
struct Params {
    key_count: u32,
    words_per_signature: u32,
    max_distance: u32,
    _padding: u32,
};

@group(0) @binding(0) var<storage, read> query_words: array<u32>;
@group(0) @binding(1) var<storage, read> key_words: array<u32>;
@group(0) @binding(2) var<storage, read_write> admissions: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn boolean_route(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let key_index = global_id.x;
    if (key_index >= params.key_count) {
        return;
    }

    var distance = 0u;
    let key_base = key_index * params.words_per_signature;
    for (var word = 0u; word < params.words_per_signature; word = word + 1u) {
        distance = distance + countOneBits(query_words[word] ^ key_words[key_base + word]);
    }

    admissions[key_index] = select(0u, 1u, distance <= params.max_distance);
}
"#;

/// Fail-closed validation or WGPU pipeline error for M13B.3 routing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum WgpuBooleanRouterError {
    ZeroKeys,
    Signature(BooleanAttentionSignatureError),
    Mask(BooleanAttentionMaskError),
    ArithmeticOverflow,
    IndexSpaceExceeded {
        value: usize,
    },
    BufferTooSmall {
        binding: &'static str,
        required_bytes: u64,
        actual_bytes: u64,
    },
    DispatchLimit {
        required: u32,
        maximum: u32,
    },
    Pipeline(String),
}

impl fmt::Display for WgpuBooleanRouterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroKeys => {
                write!(f, "Boolean WGPU router requires at least one key signature")
            }
            Self::Signature(error) => write!(f, "invalid Boolean attention signature: {error}"),
            Self::Mask(error) => write!(f, "invalid Boolean attention mask: {error}"),
            Self::ArithmeticOverflow => {
                write!(f, "Boolean WGPU router size computation overflowed")
            }
            Self::IndexSpaceExceeded { value } => write!(
                f,
                "Boolean WGPU router value {value} exceeds the WGSL u32 index space"
            ),
            Self::BufferTooSmall {
                binding,
                required_bytes,
                actual_bytes,
            } => write!(
                f,
                "Boolean WGPU router {binding} buffer has {actual_bytes} bytes, requires {required_bytes}"
            ),
            Self::DispatchLimit { required, maximum } => write!(
                f,
                "Boolean WGPU router requires {required} workgroups, device maximum is {maximum}"
            ),
            Self::Pipeline(message) => write!(f, "Boolean WGPU router pipeline failed: {message}"),
        }
    }
}

impl std::error::Error for WgpuBooleanRouterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Signature(error) => Some(error),
            Self::Mask(error) => Some(error),
            _ => None,
        }
    }
}

impl From<BooleanAttentionSignatureError> for WgpuBooleanRouterError {
    fn from(value: BooleanAttentionSignatureError) -> Self {
        Self::Signature(value)
    }
}

impl From<BooleanAttentionMaskError> for WgpuBooleanRouterError {
    fn from(value: BooleanAttentionMaskError) -> Self {
        Self::Mask(value)
    }
}

/// Validated Boolean-only routing plan. No numerical K/V storage is referenced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanWgpuRouterPlan {
    query_words: Vec<u32>,
    key_words: Vec<u32>,
    oracle_mask: BooleanAttentionMask,
    signature_bits: usize,
    key_count: u32,
    words_per_signature: u32,
    max_distance: u32,
}

impl BooleanWgpuRouterPlan {
    /// Build a WGPU-compatible u32 representation and exact CPU oracle.
    ///
    /// WGPU's portable WGSL storage integer is u32, so each canonical u64 word
    /// is split into low/high u32 halves without changing its physical byte
    /// count. Canonical zero tail bits guarantee that padding cannot affect the
    /// Hamming result.
    ///
    /// # Errors
    ///
    /// Rejects an empty key set, mismatched signature widths, invalid rules,
    /// address-space overflow, or any signature rejected by the canonical
    /// M13B.2 contract.
    pub fn new(
        query: &BooleanAttentionSignature,
        keys: &[BooleanAttentionSignature],
        rule: HammingAdmissionRule,
    ) -> Result<Self, WgpuBooleanRouterError> {
        if keys.is_empty() {
            return Err(WgpuBooleanRouterError::ZeroKeys);
        }

        let query_words = split_u64_words(query.words());
        let words_per_signature = u32::try_from(query_words.len()).map_err(|_| {
            WgpuBooleanRouterError::IndexSpaceExceeded {
                value: query_words.len(),
            }
        })?;
        let key_count = u32::try_from(keys.len())
            .map_err(|_| WgpuBooleanRouterError::IndexSpaceExceeded { value: keys.len() })?;
        let max_distance = u32::try_from(rule.max_distance).map_err(|_| {
            WgpuBooleanRouterError::IndexSpaceExceeded {
                value: rule.max_distance,
            }
        })?;

        let key_word_capacity = query_words
            .len()
            .checked_mul(keys.len())
            .ok_or(WgpuBooleanRouterError::ArithmeticOverflow)?;
        let mut key_words = Vec::with_capacity(key_word_capacity);
        let mut admissions = Vec::with_capacity(keys.len());
        for key in keys {
            admissions.push(rule.admits(query, key)?);
            key_words.extend(split_u64_words(key.words()));
        }

        Ok(Self {
            query_words,
            key_words,
            oracle_mask: BooleanAttentionMask::from_admissions(&admissions)?,
            signature_bits: query.bits(),
            key_count,
            words_per_signature,
            max_distance,
        })
    }

    #[must_use]
    pub fn query_words(&self) -> &[u32] {
        &self.query_words
    }

    #[must_use]
    pub fn key_words(&self) -> &[u32] {
        &self.key_words
    }

    #[must_use]
    pub const fn signature_bits(&self) -> usize {
        self.signature_bits
    }

    #[must_use]
    pub const fn key_count(&self) -> u32 {
        self.key_count
    }

    #[must_use]
    pub const fn words_per_signature(&self) -> u32 {
        self.words_per_signature
    }

    #[must_use]
    pub const fn max_distance(&self) -> u32 {
        self.max_distance
    }

    #[must_use]
    pub fn oracle_mask(&self) -> &BooleanAttentionMask {
        &self.oracle_mask
    }

    fn params(&self) -> [u32; 4] {
        [
            self.key_count,
            self.words_per_signature,
            self.max_distance,
            0,
        ]
    }

    fn query_bytes(&self) -> Result<u64, WgpuBooleanRouterError> {
        bytes_for_u32(self.query_words.len())
    }

    fn key_bytes(&self) -> Result<u64, WgpuBooleanRouterError> {
        bytes_for_u32(self.key_words.len())
    }

    fn admission_bytes(&self) -> Result<u64, WgpuBooleanRouterError> {
        bytes_for_u32(self.key_count as usize)
    }
}

/// Fail-closed errors for the Boolean-KV WGPU-vs-CPU parity record.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BooleanKvWgpuParityError {
    AdmissionCountMismatch {
        expected: usize,
        actual: usize,
    },
    NonBinaryAdmission {
        index: usize,
        value: u32,
    },
    CandidateSetMismatch {
        cpu_admitted_blocks: Vec<usize>,
        wgpu_admitted_blocks: Vec<usize>,
    },
}

impl fmt::Display for BooleanKvWgpuParityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AdmissionCountMismatch { expected, actual } => write!(
                f,
                "Boolean-KV WGPU readback contains {actual} admission flags, expected {expected}"
            ),
            Self::NonBinaryAdmission { index, value } => write!(
                f,
                "Boolean-KV WGPU admission flag {index} has non-binary value {value}"
            ),
            Self::CandidateSetMismatch {
                cpu_admitted_blocks,
                wgpu_admitted_blocks,
            } => write!(
                f,
                "Boolean-KV WGPU candidate set {:?} differs from CPU oracle {:?}",
                wgpu_admitted_blocks, cpu_admitted_blocks
            ),
        }
    }
}

impl std::error::Error for BooleanKvWgpuParityError {}

/// Exact candidate-set parity evidence for one already executed WGPU router plan.
///
/// Construction accepts the actual u32 admission-buffer readback from the same
/// [`BooleanWgpuRouterPlan`] used to encode the device dispatch. It fails closed
/// unless every flag is binary. The complete WGPU candidate set is retained
/// alongside the deterministic CPU Hamming oracle even when they differ, so a
/// negative parity result cannot disappear. `require_exact_match()` is the
/// fail-closed gate before any performance comparison. The record binds the
/// signature geometry, threshold and exact packed input words and contains no
/// timing or performance field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BooleanKvWgpuParityEvidence {
    signature_bits: usize,
    words_per_signature: u32,
    max_distance: u32,
    key_count: u32,
    query_words: Vec<u32>,
    key_words: Vec<u32>,
    cpu_admitted_blocks: Vec<usize>,
    wgpu_admitted_blocks: Vec<usize>,
    exact_candidate_set_match: bool,
}

impl BooleanKvWgpuParityEvidence {
    /// Validate WGPU readback against the exact CPU oracle carried by `plan`.
    pub fn from_readback(
        plan: &BooleanWgpuRouterPlan,
        observed_admissions: &[u32],
    ) -> Result<Self, BooleanKvWgpuParityError> {
        let expected = plan.key_count as usize;
        if observed_admissions.len() != expected {
            return Err(BooleanKvWgpuParityError::AdmissionCountMismatch {
                expected,
                actual: observed_admissions.len(),
            });
        }
        let mut wgpu_admitted_blocks = Vec::new();
        for (index, &value) in observed_admissions.iter().enumerate() {
            match value {
                0 => {}
                1 => wgpu_admitted_blocks.push(index),
                _ => return Err(BooleanKvWgpuParityError::NonBinaryAdmission { index, value }),
            }
        }
        let cpu_admitted_blocks = plan.oracle_mask.admitted_blocks();
        let exact_candidate_set_match = wgpu_admitted_blocks == cpu_admitted_blocks;
        Ok(Self {
            signature_bits: plan.signature_bits,
            words_per_signature: plan.words_per_signature,
            max_distance: plan.max_distance,
            key_count: plan.key_count,
            query_words: plan.query_words.clone(),
            key_words: plan.key_words.clone(),
            cpu_admitted_blocks,
            wgpu_admitted_blocks,
            exact_candidate_set_match,
        })
    }

    #[must_use]
    pub fn cpu_admitted_blocks(&self) -> &[usize] {
        &self.cpu_admitted_blocks
    }

    #[must_use]
    pub fn wgpu_admitted_blocks(&self) -> &[usize] {
        &self.wgpu_admitted_blocks
    }

    #[must_use]
    pub const fn exact_candidate_set_match(&self) -> bool {
        self.exact_candidate_set_match
    }

    /// Fail closed before any CPU-vs-WGPU performance comparison.
    pub fn require_exact_match(&self) -> Result<(), BooleanKvWgpuParityError> {
        if self.exact_candidate_set_match {
            Ok(())
        } else {
            Err(BooleanKvWgpuParityError::CandidateSetMismatch {
                cpu_admitted_blocks: self.cpu_admitted_blocks.clone(),
                wgpu_admitted_blocks: self.wgpu_admitted_blocks.clone(),
            })
        }
    }

    /// Canonical compact JSON for cross-project retention.
    ///
    /// The trailing FNV-1a checksum detects accidental corruption only. It is
    /// not cryptographic attestation and must not be used as proof of device
    /// identity or trustworthy execution.
    #[must_use]
    pub fn canonical_json(&self) -> String {
        let mut payload = String::with_capacity(
            256 + (self.query_words.len() + self.key_words.len()) * 12
                + (self.cpu_admitted_blocks.len() + self.wgpu_admitted_blocks.len()) * 12,
        );
        write!(
            payload,
            "{{\"schema\":\"{}\",\"signature_bits\":{},\"key_count\":{},\"words_per_signature\":{},\"max_distance\":{},\"query_words_u32\":",
            BOOLEAN_KV_WGPU_PARITY_SCHEMA,
            self.signature_bits,
            self.key_count,
            self.words_per_signature,
            self.max_distance,
        )
        .expect("writing to String cannot fail");
        write_u32_array(&mut payload, &self.query_words);
        payload.push_str(",\"key_words_u32\":");
        write_u32_array(&mut payload, &self.key_words);
        payload.push_str(",\"cpu_admitted_blocks\":");
        write_usize_array(&mut payload, &self.cpu_admitted_blocks);
        payload.push_str(",\"wgpu_admitted_blocks\":");
        write_usize_array(&mut payload, &self.wgpu_admitted_blocks);
        write!(
            payload,
            ",\"exact_candidate_set_match\":{}}}",
            self.exact_candidate_set_match
        )
        .expect("writing to String cannot fail");
        let closing = payload.pop();
        debug_assert_eq!(closing, Some('}'));
        let checksum = fnv1a64(payload.as_bytes());
        write!(
            payload,
            ",\"parity_checksum\":{{\"algorithm\":\"fnv1a64\",\"value\":\"{checksum:016x}\"}}}}"
        )
        .expect("writing to String cannot fail");
        payload
    }
}

fn write_u32_array(output: &mut String, values: &[u32]) {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{value}").expect("writing to String cannot fail");
    }
    output.push(']');
}

fn write_usize_array(output: &mut String, values: &[usize]) {
    output.push('[');
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        write!(output, "{value}").expect("writing to String cannot fail");
    }
    output.push(']');
}

/// Compiled M13B.3 Boolean-only WGPU routing pipeline.
pub struct WgpuBooleanRouterPipeline {
    pipeline: wgpu::ComputePipeline,
}

impl WgpuBooleanRouterPipeline {
    /// Compile the portable Boolean router on an existing FLAT WGPU device.
    ///
    /// # Errors
    ///
    /// Returns the exact WGPU validation message if shader or pipeline
    /// creation fails.
    pub fn new(device: &wgpu::Device) -> Result<Self, WgpuBooleanRouterError> {
        let pipeline = wgpu_internal::create_pipeline(
            device,
            BOOLEAN_ROUTER_WGSL,
            "flat-m13b3-boolean-router",
            "boolean_route",
        )
        .map_err(WgpuBooleanRouterError::Pipeline)?;
        Ok(Self { pipeline })
    }

    /// Encode routing before numerical K/V staging.
    ///
    /// The caller owns and uploads the Boolean query/key signature buffers.
    /// `admissions` receives one u32 flag per candidate block. This method has
    /// intentionally no K/V binding and performs no submission or readback.
    ///
    /// # Errors
    ///
    /// Fails before encoding when a buffer is too small or the device dispatch
    /// limit cannot represent the plan.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        query: &wgpu::Buffer,
        keys: &wgpu::Buffer,
        admissions: &wgpu::Buffer,
        plan: &BooleanWgpuRouterPlan,
    ) -> Result<(), WgpuBooleanRouterError> {
        validate_buffer("query", query, plan.query_bytes()?)?;
        validate_buffer("keys", keys, plan.key_bytes()?)?;
        validate_buffer("admissions", admissions, plan.admission_bytes()?)?;

        let workgroups = plan.key_count.div_ceil(ROUTER_WORKGROUP_SIZE);
        let maximum = device.limits().max_compute_workgroups_per_dimension;
        if workgroups > maximum {
            return Err(WgpuBooleanRouterError::DispatchLimit {
                required: workgroups,
                maximum,
            });
        }

        let params = plan.params();
        let params_bytes = wgpu_internal::encode_u32(&params);
        let params_buffer = wgpu_internal::create_uniform_buffer_init(
            device,
            "flat-m13b3-boolean-router-params",
            &params_bytes,
        );
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-m13b3-boolean-router-bind-group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: admissions.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat-m13b3-boolean-router"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        drop(pass);
        Ok(())
    }
}

fn split_u64_words(words: &[u64]) -> Vec<u32> {
    let mut packed = Vec::with_capacity(words.len().saturating_mul(2));
    for &word in words {
        packed.push(word as u32);
        packed.push((word >> 32) as u32);
    }
    packed
}

fn bytes_for_u32(elements: usize) -> Result<u64, WgpuBooleanRouterError> {
    let bytes = elements
        .checked_mul(core::mem::size_of::<u32>())
        .ok_or(WgpuBooleanRouterError::ArithmeticOverflow)?;
    u64::try_from(bytes).map_err(|_| WgpuBooleanRouterError::ArithmeticOverflow)
}

fn validate_buffer(
    binding: &'static str,
    buffer: &wgpu::Buffer,
    required_bytes: u64,
) -> Result<(), WgpuBooleanRouterError> {
    let actual_bytes = buffer.size();
    if actual_bytes < required_bytes {
        return Err(WgpuBooleanRouterError::BufferTooSmall {
            binding,
            required_bytes,
            actual_bytes,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_preserves_cpu_hamming_oracle_and_u64_bytes() {
        let query = BooleanAttentionSignature::new(65, vec![0b1011, 1]).unwrap();
        let keys = vec![
            BooleanAttentionSignature::new(65, vec![0b1011, 1]).unwrap(),
            BooleanAttentionSignature::new(65, vec![0b0011, 1]).unwrap(),
            BooleanAttentionSignature::new(65, vec![0, 0]).unwrap(),
        ];
        let plan =
            BooleanWgpuRouterPlan::new(&query, &keys, HammingAdmissionRule::new(1, 65).unwrap())
                .unwrap();

        assert_eq!(plan.key_count(), 3);
        assert_eq!(plan.words_per_signature(), 4);
        assert_eq!(plan.max_distance(), 1);
        assert_eq!(plan.query_words().len() * 4, query.words().len() * 8);
        assert_eq!(plan.oracle_mask().admitted_blocks(), vec![0, 1]);
        assert_eq!(plan.key_words().len(), 12);
    }

    #[test]
    fn bkv7_parity_evidence_binds_exact_inputs_and_candidate_set() {
        let query = BooleanAttentionSignature::new(65, vec![0b1011, 1]).unwrap();
        let keys = vec![
            BooleanAttentionSignature::new(65, vec![0b1011, 1]).unwrap(),
            BooleanAttentionSignature::new(65, vec![0b0011, 1]).unwrap(),
            BooleanAttentionSignature::new(65, vec![0, 0]).unwrap(),
        ];
        let plan =
            BooleanWgpuRouterPlan::new(&query, &keys, HammingAdmissionRule::new(1, 65).unwrap())
                .unwrap();
        let evidence = BooleanKvWgpuParityEvidence::from_readback(&plan, &[1, 1, 0]).unwrap();
        assert_eq!(evidence.cpu_admitted_blocks(), &[0, 1]);
        assert_eq!(evidence.wgpu_admitted_blocks(), &[0, 1]);
        assert!(evidence.exact_candidate_set_match());
        evidence.require_exact_match().unwrap();
        let json = evidence.canonical_json();
        assert!(json.contains(BOOLEAN_KV_WGPU_PARITY_SCHEMA));
        assert!(json.contains("\"signature_bits\":65"));
        assert!(json.contains("\"key_count\":3"));
        assert!(json.contains("\"cpu_admitted_blocks\":[0,1]"));
        assert!(json.contains("\"wgpu_admitted_blocks\":[0,1]"));
        assert!(json.contains("\"exact_candidate_set_match\":true"));
        assert!(json.contains("\"parity_checksum\":{\"algorithm\":\"fnv1a64\""));
    }

    #[test]
    fn bkv7_parity_rejects_nonbinary_length_and_candidate_drift() {
        let query = BooleanAttentionSignature::new(8, vec![0]).unwrap();
        let keys = vec![
            BooleanAttentionSignature::new(8, vec![0]).unwrap(),
            BooleanAttentionSignature::new(8, vec![1]).unwrap(),
        ];
        let plan =
            BooleanWgpuRouterPlan::new(&query, &keys, HammingAdmissionRule::new(0, 8).unwrap())
                .unwrap();
        assert!(matches!(
            BooleanKvWgpuParityEvidence::from_readback(&plan, &[1]),
            Err(BooleanKvWgpuParityError::AdmissionCountMismatch { .. })
        ));
        assert!(matches!(
            BooleanKvWgpuParityEvidence::from_readback(&plan, &[1, 2]),
            Err(BooleanKvWgpuParityError::NonBinaryAdmission { index: 1, value: 2 })
        ));
        let mismatch = BooleanKvWgpuParityEvidence::from_readback(&plan, &[0, 1]).unwrap();
        assert!(!mismatch.exact_candidate_set_match());
        assert!(matches!(
            mismatch.require_exact_match(),
            Err(BooleanKvWgpuParityError::CandidateSetMismatch { .. })
        ));
        assert!(mismatch
            .canonical_json()
            .contains("\"exact_candidate_set_match\":false"));
    }

    #[test]
    fn empty_keys_and_width_mismatch_fail_closed() {
        let query = BooleanAttentionSignature::new(8, vec![0]).unwrap();
        let rule = HammingAdmissionRule::new(1, 8).unwrap();
        assert_eq!(
            BooleanWgpuRouterPlan::new(&query, &[], rule),
            Err(WgpuBooleanRouterError::ZeroKeys)
        );

        let key = BooleanAttentionSignature::new(7, vec![0]).unwrap();
        assert!(matches!(
            BooleanWgpuRouterPlan::new(&query, &[key], rule),
            Err(WgpuBooleanRouterError::Signature(
                BooleanAttentionSignatureError::WidthMismatch { .. }
            ))
        ));
    }
}
