//! Real portable GPU execution for the fused FLAT-ATTENTION forward kernels.
//!
//! M5 adds an optional subgroup-assisted dot-product reduction. M6 adds a
//! portable `vec4<f32>` Q/K/V storage path for head dimensions 64 and 128.
//! Subgroup reduction keeps priority when selected; otherwise M6 chooses vec4
//! only for the two qualified dimensions and preserves the M4 scalar Q4 GPU
//! kernel as the fallback. No path silently falls back to CPU.

use std::fmt;
use std::sync::{mpsc, Arc};

use super::{
    validate_input, AttentionShape, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
    FLAT_FWD_SUBGROUP_WGSL, FLAT_FWD_WGSL, WGSL_MAX_HEAD_DIM, WGSL_QUERY_ROWS,
};

const FLAT_FWD_VEC4_WGSL: &str = include_str!("../shaders/flat_fwd_vec4.wgsl");

/// Runtime policy controlling whether the M5 subgroup kernel may be selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WgpuSubgroupPolicy {
    /// Select subgroup when the adapter reports it; otherwise use portable Q4.
    #[default]
    Auto,
    /// Always use a qualified portable Q4 GPU kernel.
    Disable,
    /// Require native subgroup support and a valid subgroup pipeline.
    Require,
}

/// Concrete fused kernel generation selected for a dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuKernelVariant {
    /// Qualified M4 scalar-storage Q4 kernel.
    Q4Portable,
    /// M6 portable Q4 kernel using `vec4<f32>` Q/K/V storage transactions.
    Q4Vec4Portable,
    /// M5 native-subgroup Q4 reduction kernel.
    Q4Subgroup,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WgpuFlatAttentionError {
    Core(FlatAttentionError),
    Unavailable,
    RequiredSubgroupUnavailable,
    UnsupportedHeadDim {
        actual: usize,
        maximum: usize,
    },
    DispatchLimit {
        axis: &'static str,
        actual: usize,
        maximum: u32,
    },
    IndexSpaceExceeded {
        elements: usize,
    },
    ForeignBuffer,
    ResidentLength {
        tensor: &'static str,
        actual: usize,
        expected: usize,
    },
    Execution(String),
}

impl fmt::Display for WgpuFlatAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => write!(f, "{err}"),
            Self::Unavailable => write!(f, "no compatible WGPU adapter/device is available"),
            Self::RequiredSubgroupUnavailable => write!(
                f,
                "the selected WGPU adapter does not provide required subgroup support"
            ),
            Self::UnsupportedHeadDim { actual, maximum } => write!(
                f,
                "head_dim {actual} exceeds portable WGSL maximum {maximum}"
            ),
            Self::DispatchLimit {
                axis,
                actual,
                maximum,
            } => write!(
                f,
                "WGPU dispatch axis {axis} requires {actual} workgroups, device maximum is {maximum}"
            ),
            Self::IndexSpaceExceeded { elements } => write!(
                f,
                "packed WGPU index space requires {elements} f32 elements, exceeding u32 addressing"
            ),
            Self::ForeignBuffer => write!(f, "resident buffer belongs to a different WGPU context"),
            Self::ResidentLength {
                tensor,
                actual,
                expected,
            } => write!(
                f,
                "resident tensor {tensor} contains {actual} elements, expected {expected}"
            ),
            Self::Execution(message) => write!(f, "WGPU execution failed: {message}"),
        }
    }
}

impl std::error::Error for WgpuFlatAttentionError {}

impl From<FlatAttentionError> for WgpuFlatAttentionError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

/// An f32 storage buffer owned by one FLAT WGPU context.
pub struct WgpuResidentBuffer {
    buffer: Arc<wgpu::Buffer>,
    len: usize,
    owner: usize,
}

impl WgpuResidentBuffer {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn raw_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// Packed resident `[O | LSE]` result.
pub struct WgpuResidentAttentionOutput {
    combined: WgpuResidentBuffer,
    output_len: usize,
    lse_len: usize,
}

impl WgpuResidentAttentionOutput {
    pub fn output_len(&self) -> usize {
        self.output_len
    }

    pub fn lse_len(&self) -> usize {
        self.lse_len
    }

    pub fn combined(&self) -> &WgpuResidentBuffer {
        &self.combined
    }
}

struct WgpuFlatAttentionInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    vec4_pipeline: wgpu::ComputePipeline,
    adapter_name: String,
    max_workgroups_per_dimension: u32,
    subgroup_supported: bool,
    subgroup_size_range: Option<(u32, u32)>,
    kernel_variant: WgpuKernelVariant,
    vectorization_enabled: bool,
}

#[derive(Clone)]
pub struct WgpuFlatAttention {
    inner: Arc<WgpuFlatAttentionInner>,
}

impl WgpuFlatAttention {
    /// Create a context using automatic subgroup selection and M6 vectorization.
    pub fn new() -> Result<Self, WgpuFlatAttentionError> {
        Self::with_subgroup_policy_and_vectorization(WgpuSubgroupPolicy::Auto, true)
    }

    /// Create a context with explicit subgroup policy and M6 vectorization on.
    pub fn with_subgroup_policy(
        policy: WgpuSubgroupPolicy,
    ) -> Result<Self, WgpuFlatAttentionError> {
        Self::with_subgroup_policy_and_vectorization(policy, true)
    }

    /// Create a context with independently controlled subgroup and vec4 paths.
    ///
    /// `vectorization_enabled = false` is intentionally public so the M4 scalar
    /// baseline can be reproduced and benchmarked against M6 on the same device.
    pub fn with_subgroup_policy_and_vectorization(
        policy: WgpuSubgroupPolicy,
        vectorization_enabled: bool,
    ) -> Result<Self, WgpuFlatAttentionError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or(WgpuFlatAttentionError::Unavailable)?;

        let adapter_features = adapter.features();
        let subgroup_supported = adapter_features.contains(wgpu::Features::SUBGROUP);
        if policy == WgpuSubgroupPolicy::Require && !subgroup_supported {
            return Err(WgpuFlatAttentionError::RequiredSubgroupUnavailable);
        }

        let adapter_limits = adapter.limits();
        let subgroup_size_range = subgroup_supported.then_some((
            adapter_limits.min_subgroup_size,
            adapter_limits.max_subgroup_size,
        ));
        let request_subgroup = subgroup_supported && policy != WgpuSubgroupPolicy::Disable;
        let required_features = if request_subgroup {
            wgpu::Features::SUBGROUP
        } else {
            wgpu::Features::empty()
        };

        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("flat-attention-q4"),
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|err| WgpuFlatAttentionError::Execution(format!("request_device: {err}")))?;

        let (pipeline, kernel_variant) = if request_subgroup {
            match create_pipeline(
                &device,
                FLAT_FWD_SUBGROUP_WGSL,
                "flat-attention-forward-q4-subgroup",
            ) {
                Ok(pipeline) => (pipeline, WgpuKernelVariant::Q4Subgroup),
                Err(_error) if policy == WgpuSubgroupPolicy::Auto => (
                    create_pipeline(&device, FLAT_FWD_WGSL, "flat-attention-forward-q4")
                        .map_err(WgpuFlatAttentionError::Execution)?,
                    WgpuKernelVariant::Q4Portable,
                ),
                Err(error) => {
                    return Err(WgpuFlatAttentionError::Execution(format!(
                        "required subgroup pipeline: {error}"
                    )))
                }
            }
        } else {
            (
                create_pipeline(&device, FLAT_FWD_WGSL, "flat-attention-forward-q4")
                    .map_err(WgpuFlatAttentionError::Execution)?,
                WgpuKernelVariant::Q4Portable,
            )
        };

        let vec4_pipeline = create_pipeline(
            &device,
            FLAT_FWD_VEC4_WGSL,
            "flat-attention-forward-q4-vec4",
        )
        .map_err(|error| {
            WgpuFlatAttentionError::Execution(format!("M6 vec4 pipeline: {error}"))
        })?;
        let max_workgroups_per_dimension = device.limits().max_compute_workgroups_per_dimension;

        Ok(Self {
            inner: Arc::new(WgpuFlatAttentionInner {
                device,
                queue,
                pipeline,
                vec4_pipeline,
                adapter_name,
                max_workgroups_per_dimension,
                subgroup_supported,
                subgroup_size_range,
                kernel_variant,
                vectorization_enabled,
            }),
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.inner.adapter_name
    }

    pub fn max_workgroups_per_dimension(&self) -> u32 {
        self.inner.max_workgroups_per_dimension
    }

    pub fn subgroup_supported(&self) -> bool {
        self.inner.subgroup_supported
    }

    pub fn subgroup_size_range(&self) -> Option<(u32, u32)> {
        self.inner.subgroup_size_range
    }

    /// M5 context-level reduction selection. Kept stable for M5 callers/tests.
    pub fn kernel_variant(&self) -> WgpuKernelVariant {
        self.inner.kernel_variant
    }

    pub fn vectorization_enabled(&self) -> bool {
        self.inner.vectorization_enabled
    }

    /// Effective kernel generation that will be used for `head_dim`.
    pub fn kernel_variant_for_head_dim(&self, head_dim: usize) -> WgpuKernelVariant {
        if self.inner.kernel_variant == WgpuKernelVariant::Q4Subgroup {
            WgpuKernelVariant::Q4Subgroup
        } else if self.inner.vectorization_enabled && matches!(head_dim, 64 | 128) {
            WgpuKernelVariant::Q4Vec4Portable
        } else {
            WgpuKernelVariant::Q4Portable
        }
    }

    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<FlatAttentionOutput, WgpuFlatAttentionError> {
        shape.validate()?;
        let tensor_len = shape.tensor_len()?;
        validate_input("Q", q, tensor_len)?;
        validate_input("K", k, tensor_len)?;
        validate_input("V", v, tensor_len)?;

        let q_gpu = self.upload(q)?;
        let k_gpu = self.upload(k)?;
        let v_gpu = self.upload(v)?;
        let resident = self.forward_resident(&q_gpu, &k_gpu, &v_gpu, shape, config)?;
        self.download_attention(&resident)
    }

    pub fn upload(&self, data: &[f32]) -> Result<WgpuResidentBuffer, WgpuFlatAttentionError> {
        let bytes = encode_f32(data)?;
        let size = bytes.len().max(4) as u64;
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-resident-input"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.inner.queue.write_buffer(&buffer, 0, &bytes);
        }
        Ok(WgpuResidentBuffer {
            buffer: Arc::new(buffer),
            len: data.len(),
            owner: self.owner_id(),
        })
    }

    pub fn forward_resident(
        &self,
        q: &WgpuResidentBuffer,
        k: &WgpuResidentBuffer,
        v: &WgpuResidentBuffer,
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<WgpuResidentAttentionOutput, WgpuFlatAttentionError> {
        let dispatch = self.validate_dispatch(shape)?;
        self.validate_resident("Q", q, dispatch.tensor_len)?;
        self.validate_resident("K", k, dispatch.tensor_len)?;
        self.validate_resident("V", v, dispatch.tensor_len)?;
        let scale = config.resolved_scale(shape.head_dim)?;

        let combined_len = dispatch
            .tensor_len
            .checked_add(dispatch.lse_len)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let output_bytes = bytes_for_f32_len(combined_len)?;
        let output = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-o-lse"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [
            dispatch.seq_len,
            checked_u32(shape.head_dim)?,
            dispatch.batch_heads,
            u32::from(config.causal),
            scale.to_bits(),
            0,
            0,
            0,
        ];
        let params_bytes = encode_u32(&params);
        let params_buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-params"),
            size: params_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.inner
            .queue
            .write_buffer(&params_buffer, 0, &params_bytes);

        let (pipeline, label) = self.pipeline_for_head_dim(shape.head_dim);
        let bind_group = self
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.query_workgroups, dispatch.batch_heads, 1);
        }
        self.inner.queue.submit(Some(encoder.finish()));

        Ok(WgpuResidentAttentionOutput {
            combined: WgpuResidentBuffer {
                buffer: Arc::new(output),
                len: combined_len,
                owner: self.owner_id(),
            },
            output_len: dispatch.tensor_len,
            lse_len: dispatch.lse_len,
        })
    }

    pub fn download_attention(
        &self,
        resident: &WgpuResidentAttentionOutput,
    ) -> Result<FlatAttentionOutput, WgpuFlatAttentionError> {
        self.ensure_owner(&resident.combined)?;
        let expected = resident
            .output_len
            .checked_add(resident.lse_len)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        if resident.combined.len != expected {
            return Err(WgpuFlatAttentionError::Execution(
                "resident output metadata is inconsistent".into(),
            ));
        }
        let mut values = self.download_buffer(&resident.combined.buffer, expected)?;
        let lse = values.split_off(resident.output_len);
        Ok(FlatAttentionOutput {
            output: values,
            lse,
        })
    }

    fn pipeline_for_head_dim(&self, head_dim: usize) -> (&wgpu::ComputePipeline, &'static str) {
        match self.kernel_variant_for_head_dim(head_dim) {
            WgpuKernelVariant::Q4Subgroup => (
                &self.inner.pipeline,
                "flat-attention-forward-q4-subgroup",
            ),
            WgpuKernelVariant::Q4Vec4Portable => (
                &self.inner.vec4_pipeline,
                "flat-attention-forward-q4-vec4",
            ),
            WgpuKernelVariant::Q4Portable => {
                (&self.inner.pipeline, "flat-attention-forward-q4")
            }
        }
    }

    fn validate_dispatch(
        &self,
        shape: AttentionShape,
    ) -> Result<DispatchGeometry, WgpuFlatAttentionError> {
        shape.validate()?;
        if shape.head_dim > WGSL_MAX_HEAD_DIM {
            return Err(WgpuFlatAttentionError::UnsupportedHeadDim {
                actual: shape.head_dim,
                maximum: WGSL_MAX_HEAD_DIM,
            });
        }
        let tensor_len = shape.tensor_len()?;
        let lse_len = shape.lse_len()?;
        let combined_len = tensor_len
            .checked_add(lse_len)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        if combined_len > u32::MAX as usize {
            return Err(WgpuFlatAttentionError::IndexSpaceExceeded {
                elements: combined_len,
            });
        }

        let batch_heads = shape
            .batch
            .checked_mul(shape.heads)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let query_workgroups = shape.seq_len.div_ceil(WGSL_QUERY_ROWS);
        let maximum = self.inner.max_workgroups_per_dimension;
        if query_workgroups > maximum as usize {
            return Err(WgpuFlatAttentionError::DispatchLimit {
                axis: "x/query_tiles",
                actual: query_workgroups,
                maximum,
            });
        }
        if batch_heads > maximum as usize {
            return Err(WgpuFlatAttentionError::DispatchLimit {
                axis: "y/batch_heads",
                actual: batch_heads,
                maximum,
            });
        }

        Ok(DispatchGeometry {
            tensor_len,
            lse_len,
            batch_heads: checked_u32(batch_heads)?,
            seq_len: checked_u32(shape.seq_len)?,
            query_workgroups: checked_u32(query_workgroups)?,
        })
    }

    fn validate_resident(
        &self,
        tensor: &'static str,
        buffer: &WgpuResidentBuffer,
        expected: usize,
    ) -> Result<(), WgpuFlatAttentionError> {
        self.ensure_owner(buffer)?;
        if buffer.len != expected {
            return Err(WgpuFlatAttentionError::ResidentLength {
                tensor,
                actual: buffer.len,
                expected,
            });
        }
        Ok(())
    }

    fn ensure_owner(&self, buffer: &WgpuResidentBuffer) -> Result<(), WgpuFlatAttentionError> {
        if buffer.owner != self.owner_id() {
            return Err(WgpuFlatAttentionError::ForeignBuffer);
        }
        Ok(())
    }

    fn owner_id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    fn download_buffer(
        &self,
        source: &wgpu::Buffer,
        len: usize,
    ) -> Result<Vec<f32>, WgpuFlatAttentionError> {
        let bytes = bytes_for_f32_len(len)?;
        let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flat-attention-readback"),
            });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
        self.inner.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.inner.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|err| WgpuFlatAttentionError::Execution(format!("map callback: {err}")))?
            .map_err(|err| WgpuFlatAttentionError::Execution(format!("map read: {err:?}")))?;

        let mapped = slice.get_mapped_range();
        let decoded = decode_f32(&mapped, len)?;
        drop(mapped);
        staging.unmap();
        Ok(decoded)
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    source: &'static str,
    label: &'static str,
) -> Result<wgpu::ComputePipeline, String> {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: "flat_attention_forward",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    let validation_error = pollster::block_on(device.pop_error_scope());
    match validation_error {
        Some(error) => Err(error.to_string()),
        None => Ok(pipeline),
    }
}

struct DispatchGeometry {
    tensor_len: usize,
    lse_len: usize,
    batch_heads: u32,
    seq_len: u32,
    query_workgroups: u32,
}

fn checked_u32(value: usize) -> Result<u32, WgpuFlatAttentionError> {
    u32::try_from(value).map_err(|_| WgpuFlatAttentionError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32_len(len: usize) -> Result<u64, WgpuFlatAttentionError> {
    let bytes = len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| WgpuFlatAttentionError::IndexSpaceExceeded { elements: len })
}

fn encode_f32(values: &[f32]) -> Result<Vec<u8>, WgpuFlatAttentionError> {
    let capacity = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let mut bytes = Vec::with_capacity(capacity);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(bytes)
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn decode_f32(bytes: &[u8], expected: usize) -> Result<Vec<f32>, WgpuFlatAttentionError> {
    let expected_bytes = expected
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    if bytes.len() != expected_bytes {
        return Err(WgpuFlatAttentionError::Execution(format!(
            "readback returned {} bytes, expected {expected_bytes}",
            bytes.len()
        )));
    }
    let mut values = Vec::with_capacity(expected);
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(values)
}
