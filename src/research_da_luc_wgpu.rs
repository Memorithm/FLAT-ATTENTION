//! FDAL3 portable WGPU candidate for a narrow DA-LUC q_len=1 subset.
//!
//! This module is intentionally research-only and fail-closed. It consumes the
//! FDAL1 compressed planes directly enough to avoid materializing dense K or V:
//! K scores come from a query-local codebook LUT plus packed K indices, while V
//! values are reconstructed from packed U8 values, F32 scales and U8 zero points
//! in shader registers during accumulation.
//!
//! The first candidate supports no sparse residuals, paging, F16/BF16 planes,
//! MSB0 streams, dense V, or sub-byte K/V streams. Unsupported contracts are
//! rejected before dispatch. Scalar/register conversion remains present, so this
//! module makes no "zero dequantization" or performance claim.

use super::*;
use crate::api::research_da_luc::{
    DalucBackendCapabilities, DalucBitOrder, DalucCodebookScope, DalucFloatDType,
    DalucKvViewContract, DalucKvViewError, DalucPaddingRule, DalucResidualSemantics, DalucRowOrder,
    DalucStorageTopology, DalucValueRepresentation, DalucZeroPointStorage,
};
use crate::{wgpu_internal, FlatAttentionError};
use core::fmt;
use std::sync::mpsc;

use super::decode::DalucQlen1DecodeConfig;

/// Version of the first portable direct-compressed WGPU candidate.
pub const DA_LUC_WGPU_CANDIDATE_VERSION: u16 = 1;

const MAX_HEAD_DIM: usize = 128;
const MAX_LUT_ENTRIES: usize = 2048;
const SHADER: &str = include_str!("../shaders/flat_da_luc_decode.wgsl");

/// Exact host-validated geometry accepted by the first WGPU candidate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DalucWgpuPlan {
    contract: DalucKvViewContract,
    config: DalucQlen1DecodeConfig,
    kv_capacity: usize,
    subspaces: usize,
    groups: usize,
    query_elements: usize,
    output_elements: usize,
    lse_elements: usize,
    scale: f32,
}

impl DalucWgpuPlan {
    /// Capabilities declared by this exact candidate revision.
    ///
    /// This is deliberately narrower than the FDAL0 schema. Missing capability
    /// means rejection and never authorizes dense conversion or a hidden
    /// fallback.
    #[must_use]
    pub const fn backend_capabilities() -> DalucBackendCapabilities {
        DalucBackendCapabilities {
            supports_f16: false,
            supports_bf16: false,
            supports_f32: true,
            supports_lsb0: true,
            supports_msb0: false,
            supports_batch_token_head: false,
            supports_batch_head_token: true,
            supports_shared_codebook: true,
            supports_per_kv_head_codebook: true,
            supports_contiguous: true,
            supports_paged: false,
            supports_groupwise_affine_values: true,
            supports_signed_symmetric_values: false,
            supports_u8_zero_points: true,
            supports_u16_zero_points: false,
            supports_sparse_coordinates: false,
            supports_sparse_bitmap: false,
            supports_no_padding: false,
            supports_zero_filled_padding: true,
            minimum_plane_alignment_bytes: 4,
            max_packed_index_bits: 8,
            max_groupwise_value_bits: 8,
        }
    }

    /// Validate that a DA-LUC contract is exactly representable by FDAL3 v1.
    pub fn new(
        contract: DalucKvViewContract,
        config: DalucQlen1DecodeConfig,
    ) -> Result<Self, DalucWgpuCandidateError> {
        contract.validate_for_backend(Self::backend_capabilities())?;

        if contract.keys.codebook_dtype != DalucFloatDType::F32 {
            return Err(DalucWgpuCandidateError::UnsupportedCandidate(
                "F32 K codebook required",
            ));
        }
        if contract.keys.index_bits != 8 || contract.keys.index_bit_order != DalucBitOrder::Lsb0 {
            return Err(DalucWgpuCandidateError::UnsupportedCandidate(
                "8-bit LSB0 K indices required",
            ));
        }
        if contract.keys.residual != DalucResidualSemantics::None {
            return Err(DalucWgpuCandidateError::UnsupportedCandidate(
                "K residuals are not implemented by FDAL3 v1",
            ));
        }
        if contract.layout.row_order != DalucRowOrder::BatchHeadToken {
            return Err(DalucWgpuCandidateError::UnsupportedCandidate(
                "BatchHeadToken row order required",
            ));
        }
        if contract.layout.padding != DalucPaddingRule::ZeroFilledToAlignment {
            return Err(DalucWgpuCandidateError::UnsupportedCandidate(
                "zero-filled aligned planes required",
            ));
        }

        let kv_capacity = match contract.layout.topology {
            DalucStorageTopology::Contiguous { capacity_tokens } => capacity_tokens,
            DalucStorageTopology::Paged { .. } => {
                return Err(DalucWgpuCandidateError::UnsupportedCandidate(
                    "paged topology is not implemented by FDAL3 v1",
                ));
            }
        };

        let groups = match contract.values {
            DalucValueRepresentation::GroupwiseAffine {
                storage_bits,
                group_size,
                scale_dtype,
                zero_point,
                bit_order,
                residual,
            } => {
                if storage_bits != 8
                    || scale_dtype != DalucFloatDType::F32
                    || zero_point != DalucZeroPointStorage::U8
                    || bit_order != DalucBitOrder::Lsb0
                    || residual != DalucResidualSemantics::None
                {
                    return Err(DalucWgpuCandidateError::UnsupportedCandidate(
                        "FDAL3 v1 V requires U8 LSB0 groupwise affine, F32 scales, U8 zero points, no residual",
                    ));
                }
                contract.shape.value_head_dim / group_size
            }
            DalucValueRepresentation::Dense { .. } => {
                return Err(DalucWgpuCandidateError::UnsupportedCandidate(
                    "dense V is not the direct-compressed FDAL3 v1 candidate",
                ));
            }
        };

        if contract.shape.key_head_dim > MAX_HEAD_DIM
            || contract.shape.value_head_dim > MAX_HEAD_DIM
        {
            return Err(DalucWgpuCandidateError::UnsupportedCandidate(
                "key/value head dimension exceeds FDAL3 v1 portable limit",
            ));
        }

        let subspaces = contract.shape.key_head_dim / contract.keys.subspace_dim;
        let lut_entries = checked_product(
            &[subspaces, contract.keys.codebook_entries],
            "FDAL3 LUT entries",
        )?;
        if lut_entries > MAX_LUT_ENTRIES {
            return Err(DalucWgpuCandidateError::UnsupportedCandidate(
                "query-local codebook LUT exceeds FDAL3 v1 workgroup budget",
            ));
        }

        let query_elements = checked_product(
            &[
                contract.shape.batch,
                contract.shape.q_heads,
                contract.shape.key_head_dim,
            ],
            "FDAL3 query elements",
        )?;
        let output_elements = checked_product(
            &[
                contract.shape.batch,
                contract.shape.q_heads,
                contract.shape.value_head_dim,
            ],
            "FDAL3 output elements",
        )?;
        let lse_elements = checked_product(
            &[contract.shape.batch, contract.shape.q_heads],
            "FDAL3 LSE elements",
        )?;
        let scale = config
            .attention
            .resolved_scale(contract.shape.key_head_dim)?;

        for (label, value) in [
            ("kv_len", contract.shape.kv_len),
            ("kv_capacity", kv_capacity),
            ("key_head_dim", contract.shape.key_head_dim),
            ("value_head_dim", contract.shape.value_head_dim),
            ("q_heads", contract.shape.q_heads),
            ("kv_heads", contract.shape.kv_heads),
            ("batch", contract.shape.batch),
            ("subspace_dim", contract.keys.subspace_dim),
            ("codebook_entries", contract.keys.codebook_entries),
            ("value groups", groups),
            ("query_position", config.query_position),
        ] {
            if u32::try_from(value).is_err() {
                return Err(DalucWgpuCandidateError::IndexSpaceExceeded(label));
            }
        }

        Ok(Self {
            contract,
            config,
            kv_capacity,
            subspaces,
            groups,
            query_elements,
            output_elements,
            lse_elements,
            scale,
        })
    }

    #[must_use]
    pub const fn contract(self) -> DalucKvViewContract {
        self.contract
    }

    #[must_use]
    pub const fn config(self) -> DalucQlen1DecodeConfig {
        self.config
    }

    #[must_use]
    pub const fn query_elements(self) -> usize {
        self.query_elements
    }

    #[must_use]
    pub const fn output_elements(self) -> usize {
        self.output_elements
    }

    #[must_use]
    pub const fn lse_elements(self) -> usize {
        self.lse_elements
    }

    /// The defining FDAL3 property: this candidate never asks for a dense K/V
    /// reconstruction before dispatch.
    #[must_use]
    pub const fn materializes_dense_kv(self) -> bool {
        false
    }
}

/// Readback from one FDAL3 candidate dispatch.
#[derive(Debug, Clone, PartialEq)]
pub struct DalucWgpuOutput {
    /// Canonical `[batch, q_heads, value_head_dim]` output.
    pub output: Vec<f32>,
    /// `[batch, q_heads]` log-sum-exp values.
    pub lse: Vec<f32>,
}

/// Typed rejection or WGPU failure for the FDAL3 research candidate.
#[derive(Debug)]
#[non_exhaustive]
pub enum DalucWgpuCandidateError {
    Contract(DalucKvViewError),
    Oracle(DalucOracleError),
    Attention(FlatAttentionError),
    UnsupportedCandidate(&'static str),
    QueryLength {
        expected: usize,
        actual: usize,
    },
    NonFiniteQuery {
        index: usize,
    },
    IndexSpaceExceeded(&'static str),
    DispatchLimit {
        actual: usize,
        maximum: u32,
    },
    DeviceBufferLimit {
        label: &'static str,
        required_bytes: u64,
        maximum_bytes: u64,
    },
    Unavailable,
    Device(String),
    PipelineValidation(String),
    Readback(String),
}

impl fmt::Display for DalucWgpuCandidateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(formatter, "DA-LUC WGPU contract rejected: {error}"),
            Self::Oracle(error) => write!(formatter, "DA-LUC WGPU payload rejected: {error}"),
            Self::Attention(error) => write!(formatter, "DA-LUC WGPU attention config rejected: {error}"),
            Self::UnsupportedCandidate(reason) => {
                write!(formatter, "DA-LUC WGPU candidate does not support {reason}")
            }
            Self::QueryLength { expected, actual } => write!(
                formatter,
                "DA-LUC WGPU query length {actual} does not match expected {expected}"
            ),
            Self::NonFiniteQuery { index } => {
                write!(formatter, "DA-LUC WGPU query contains non-finite value at {index}")
            }
            Self::IndexSpaceExceeded(label) => {
                write!(formatter, "DA-LUC WGPU {label} exceeds u32 index space")
            }
            Self::DispatchLimit { actual, maximum } => write!(
                formatter,
                "DA-LUC WGPU requires {actual} workgroups, device maximum is {maximum}"
            ),
            Self::DeviceBufferLimit {
                label,
                required_bytes,
                maximum_bytes,
            } => write!(
                formatter,
                "DA-LUC WGPU buffer {label} requires {required_bytes} bytes, device maximum is {maximum_bytes}"
            ),
            Self::Unavailable => formatter.write_str("WGPU adapter unavailable for DA-LUC candidate"),
            Self::Device(error) => write!(formatter, "DA-LUC WGPU device failure: {error}"),
            Self::PipelineValidation(error) => {
                write!(formatter, "DA-LUC WGPU pipeline validation failed: {error}")
            }
            Self::Readback(error) => write!(formatter, "DA-LUC WGPU readback failed: {error}"),
        }
    }
}

impl std::error::Error for DalucWgpuCandidateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            Self::Oracle(error) => Some(error),
            Self::Attention(error) => Some(error),
            _ => None,
        }
    }
}

impl From<DalucKvViewError> for DalucWgpuCandidateError {
    fn from(value: DalucKvViewError) -> Self {
        Self::Contract(value)
    }
}

impl From<DalucOracleError> for DalucWgpuCandidateError {
    fn from(value: DalucOracleError) -> Self {
        Self::Oracle(value)
    }
}

impl From<FlatAttentionError> for DalucWgpuCandidateError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Attention(value)
    }
}

/// Standalone correctness-first portable WGPU execution context.
pub struct WgpuDalucQlen1Candidate {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
    max_workgroups_per_dimension: u32,
    max_storage_buffer_binding_size: u64,
}

impl fmt::Debug for WgpuDalucQlen1Candidate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WgpuDalucQlen1Candidate")
            .field("adapter_name", &self.adapter_name)
            .finish_non_exhaustive()
    }
}

impl WgpuDalucQlen1Candidate {
    /// Create the portable pipeline. No fallback is attempted when WGPU is
    /// unavailable or shader validation fails.
    pub fn new() -> Result<Self, DalucWgpuCandidateError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|_| DalucWgpuCandidateError::Unavailable)?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flat-fdal3-da-luc"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .map_err(|error| DalucWgpuCandidateError::Device(format!("request_device: {error}")))?;
        let pipeline = wgpu_internal::create_pipeline(
            &device,
            SHADER,
            "flat-fdal3-da-luc",
            "flat_da_luc_decode",
        )
        .map_err(DalucWgpuCandidateError::PipelineValidation)?;
        let limits = device.limits();
        Ok(Self {
            device,
            queue,
            pipeline,
            adapter_name,
            max_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        })
    }

    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Execute one direct-compressed q_len=1 request and read back O/LSE.
    ///
    /// `payload.decode_keys()` and `payload.decode_values()` are never called.
    /// Packed byte streams are staged as word-aligned compressed buffers only;
    /// no dense K or V tensor is constructed on host or device.
    pub fn forward(
        &self,
        payload: &DalucOraclePayload,
        query: &[f32],
        config: DalucQlen1DecodeConfig,
    ) -> Result<DalucWgpuOutput, DalucWgpuCandidateError> {
        payload.validate()?;
        let plan = DalucWgpuPlan::new(payload.contract, config)?;
        validate_query(query, plan.query_elements)?;

        let workgroups = checked_product(
            &[payload.contract.shape.batch, payload.contract.shape.q_heads],
            "FDAL3 workgroups",
        )?;
        if workgroups > self.max_workgroups_per_dimension as usize {
            return Err(DalucWgpuCandidateError::DispatchLimit {
                actual: workgroups,
                maximum: self.max_workgroups_per_dimension,
            });
        }

        let query_bytes = wgpu_internal::encode_f32(query)
            .ok_or(DalucWgpuCandidateError::IndexSpaceExceeded("query bytes"))?;
        let codebook_bytes = f32_plane_bytes(
            &payload.key_codebook,
            codebook_elements(payload.contract)?,
            DalucFloatDType::F32,
        )?;
        let key_index_bytes = u8_plane_words(&payload.key_indices)?;
        let value_bytes = u8_plane_words(&payload.values)?;
        let scale_count = checked_product(
            &[
                payload.contract.shape.batch,
                payload.contract.shape.kv_heads,
                plan.kv_capacity,
                plan.groups,
            ],
            "FDAL3 V scale count",
        )?;
        let scale_bytes =
            f32_plane_bytes(&payload.value_scales, scale_count, DalucFloatDType::F32)?;
        let zero_point_bytes = u8_plane_words(&payload.value_zero_points)?;

        let q_buffer = self.upload("Q", &query_bytes)?;
        let codebook_buffer = self.upload("K codebook", &codebook_bytes)?;
        let key_index_buffer = self.upload("K indices", &key_index_bytes)?;
        let value_buffer = self.upload("V packed", &value_bytes)?;
        let scale_buffer = self.upload("V scales", &scale_bytes)?;
        let zero_point_buffer = self.upload("V zero points", &zero_point_bytes)?;

        let combined_elements = plan.output_elements.checked_add(plan.lse_elements).ok_or(
            DalucWgpuCandidateError::IndexSpaceExceeded("output elements"),
        )?;
        let output_bytes = wgpu_internal::f32_bytes(combined_elements)
            .ok_or(DalucWgpuCandidateError::IndexSpaceExceeded("output bytes"))?;
        self.ensure_buffer_limit("O|LSE", output_bytes)?;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-fdal3-da-luc-o-lse"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let group_size = match payload.contract.values {
            DalucValueRepresentation::GroupwiseAffine { group_size, .. } => group_size,
            DalucValueRepresentation::Dense { .. } => unreachable!("plan rejects dense V"),
        };
        let params = [
            to_u32("kv_len", payload.contract.shape.kv_len)?,
            to_u32("kv_capacity", plan.kv_capacity)?,
            to_u32("key_head_dim", payload.contract.shape.key_head_dim)?,
            to_u32("value_head_dim", payload.contract.shape.value_head_dim)?,
            to_u32("q_heads", payload.contract.shape.q_heads)?,
            to_u32("kv_heads", payload.contract.shape.kv_heads)?,
            to_u32("batch", payload.contract.shape.batch)?,
            to_u32("subspace_dim", payload.contract.keys.subspace_dim)?,
            to_u32("codebook_entries", payload.contract.keys.codebook_entries)?,
            u32::from(matches!(
                payload.contract.keys.codebook_scope,
                DalucCodebookScope::PerKvHead
            )),
            to_u32("value_group_size", group_size)?,
            plan.scale.to_bits(),
            u32::from(config.attention.causal),
            to_u32("query_position", config.query_position)?,
            0,
            0,
        ];
        let params_bytes = wgpu_internal::encode_u32(&params);
        let params_buffer = wgpu_internal::create_uniform_buffer_init(
            &self.device,
            "flat-fdal3-da-luc-params",
            &params_bytes,
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flat-fdal3-da-luc-bind-group"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: q_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: codebook_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: key_index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: value_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scale_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: zero_point_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-fdal3-da-luc-decode"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-fdal3-da-luc-decode"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(to_u32("workgroups", workgroups)?, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        let mut combined = self.download(&output, combined_elements)?;
        let lse = combined.split_off(plan.output_elements);
        Ok(DalucWgpuOutput {
            output: combined,
            lse,
        })
    }

    fn upload(
        &self,
        label: &'static str,
        bytes: &[u8],
    ) -> Result<wgpu::Buffer, DalucWgpuCandidateError> {
        let size = u64::try_from(bytes.len().max(4))
            .map_err(|_| DalucWgpuCandidateError::IndexSpaceExceeded(label))?;
        self.ensure_buffer_limit(label, size)?;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.queue.write_buffer(&buffer, 0, bytes);
        }
        Ok(buffer)
    }

    fn ensure_buffer_limit(
        &self,
        label: &'static str,
        required_bytes: u64,
    ) -> Result<(), DalucWgpuCandidateError> {
        if required_bytes > self.max_storage_buffer_binding_size {
            return Err(DalucWgpuCandidateError::DeviceBufferLimit {
                label,
                required_bytes,
                maximum_bytes: self.max_storage_buffer_binding_size,
            });
        }
        Ok(())
    }

    fn download(
        &self,
        source: &wgpu::Buffer,
        elements: usize,
    ) -> Result<Vec<f32>, DalucWgpuCandidateError> {
        let bytes = wgpu_internal::f32_bytes(elements).ok_or(
            DalucWgpuCandidateError::IndexSpaceExceeded("readback bytes"),
        )?;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-fdal3-da-luc-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-fdal3-da-luc-readback"),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .map_err(|error| DalucWgpuCandidateError::Readback(format!("map callback: {error}")))?
            .map_err(|error| DalucWgpuCandidateError::Readback(format!("map read: {error:?}")))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| DalucWgpuCandidateError::Readback(format!("map range: {error}")))?;
        let decoded =
            wgpu_internal::decode_f32(&mapped, elements).map_err(|error| match error {
                wgpu_internal::DecodeF32Failure::Overflow => {
                    DalucWgpuCandidateError::Readback("decoded f32 length overflow".into())
                }
                wgpu_internal::DecodeF32Failure::LengthMismatch {
                    actual_bytes,
                    expected_bytes,
                } => DalucWgpuCandidateError::Readback(format!(
                    "decoded f32 bytes {actual_bytes} do not match expected {expected_bytes}"
                )),
            })?;
        drop(mapped);
        staging.unmap();
        Ok(decoded)
    }
}

fn validate_query(query: &[f32], expected: usize) -> Result<(), DalucWgpuCandidateError> {
    if query.len() != expected {
        return Err(DalucWgpuCandidateError::QueryLength {
            expected,
            actual: query.len(),
        });
    }
    if let Some((index, _)) = query
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(DalucWgpuCandidateError::NonFiniteQuery { index });
    }
    Ok(())
}

fn f32_plane_bytes(
    plane: &DalucOraclePlane,
    elements: usize,
    dtype: DalucFloatDType,
) -> Result<Vec<u8>, DalucWgpuCandidateError> {
    let mut values = Vec::with_capacity(elements);
    for index in 0..elements {
        values.push(read_float(plane, index, dtype)?);
    }
    wgpu_internal::encode_f32(&values).ok_or(DalucWgpuCandidateError::IndexSpaceExceeded(
        "F32 plane bytes",
    ))
}

/// Convert a compressed byte plane into native u32 words while preserving each
/// payload byte exactly. This is word-alignment staging, not dense K/V decode.
fn u8_plane_words(plane: &DalucOraclePlane) -> Result<Vec<u8>, DalucWgpuCandidateError> {
    let logical_bytes = plane.logical_bytes();
    let source = &plane.bytes()[..logical_bytes];
    let word_count = logical_bytes.div_ceil(4).max(1);
    let mut words = Vec::with_capacity(word_count);
    for word_index in 0..word_count {
        let start = word_index * 4;
        let mut bytes = [0u8; 4];
        for (offset, byte) in bytes.iter_mut().enumerate() {
            if let Some(value) = source.get(start + offset) {
                *byte = *value;
            }
        }
        words.push(u32::from_le_bytes(bytes));
    }
    Ok(wgpu_internal::encode_u32(&words))
}

fn to_u32(label: &'static str, value: usize) -> Result<u32, DalucWgpuCandidateError> {
    u32::try_from(value).map_err(|_| DalucWgpuCandidateError::IndexSpaceExceeded(label))
}
