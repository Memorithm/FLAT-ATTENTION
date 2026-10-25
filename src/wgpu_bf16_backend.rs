//! FML1 portable packed-bfloat16 I/O for FLAT-ATTENTION.
//!
//! Two bfloat16 scalars are stored per `u32`. bfloat16 is the upper half of
//! an IEEE binary32, so the shader decodes with an exact shift and needs no
//! native `f16`/`bf16` capability. All attention arithmetic and LSE remain
//! `f32`; O is re-encoded with round-to-nearest-even. The device result is
//! qualified against `forward_reference_bf16`. There is no CPU fallback.

use super::wgpu_internal;

use std::fmt;
use std::sync::{mpsc, Arc};

use super::{
    AttentionShape, Bf16, FlatAttentionBf16Output, FlatAttentionConfig, FlatAttentionError,
    FLAT_FWD_BF16_WGSL, WGSL_MAX_HEAD_DIM, WGSL_QUERY_ROWS,
};

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum WgpuBf16AttentionError {
    Core(FlatAttentionError),
    Unavailable,
    ShaderUnsupported(String),
    UnsupportedHeadDim {
        actual: usize,
    },
    OddPackedLength {
        actual: usize,
    },
    DispatchLimit {
        axis: &'static str,
        actual: usize,
        maximum: u32,
    },
    IndexSpaceExceeded {
        elements: usize,
    },
    DeviceBufferLimit {
        required_bytes: u64,
        maximum_bytes: u64,
    },
    /// bfloat16 shares binary32's exponent range, so finite inputs can
    /// overflow the FP32 score on the device, where values are not inspected.
    /// Host-slice entry points reject inputs whose conservative score bound
    /// `scale * head_dim * max|Q| * max|K|` is not below `f32::MAX`.
    ScoreBoundExceeded {
        bound: f64,
    },
    ForeignBuffer,
    ResidentLength {
        tensor: &'static str,
        actual: usize,
        expected: usize,
    },
    Execution(String),
}

impl fmt::Display for WgpuBf16AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::Unavailable => write!(f, "no compatible WGPU adapter/device is available"),
            Self::ShaderUnsupported(message) => {
                write!(f, "packed-bfloat16 WGSL pipeline is unsupported: {message}")
            }
            Self::UnsupportedHeadDim { actual } => write!(
                f,
                "packed-bfloat16 I/O supports head_dim 64 or 128, got {actual}"
            ),
            Self::OddPackedLength { actual } => write!(
                f,
                "packed-bfloat16 storage requires an even scalar count, got {actual}"
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
                "packed-bfloat16 index space requires {elements} elements, exceeding u32 addressing"
            ),
            Self::DeviceBufferLimit {
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "packed-bfloat16 attention requires {required_bytes} bytes per buffer, device maximum is {maximum_bytes}"
            ),
            Self::ScoreBoundExceeded { bound } => write!(
                f,
                "packed-bfloat16 score bound {bound:e} could overflow FP32 on the device"
            ),
            Self::ForeignBuffer => write!(
                f,
                "resident packed-bfloat16 buffer belongs to another WGPU context"
            ),
            Self::ResidentLength {
                tensor,
                actual,
                expected,
            } => write!(
                f,
                "resident packed-bfloat16 tensor {tensor} contains {actual} elements, expected {expected}"
            ),
            Self::Execution(message) => write!(f, "WGPU packed-bf16 execution failed: {message}"),
        }
    }
}

impl std::error::Error for WgpuBf16AttentionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Core(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FlatAttentionError> for WgpuBf16AttentionError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

pub struct WgpuResidentBf16Buffer {
    buffer: Arc<wgpu::Buffer>,
    len: usize,
    owner: usize,
}

impl WgpuResidentBf16Buffer {
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn raw_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

pub struct WgpuResidentBf16AttentionOutput {
    buffer: Arc<wgpu::Buffer>,
    output_len: usize,
    lse_len: usize,
    packed_words: usize,
    owner: usize,
}

impl WgpuResidentBf16AttentionOutput {
    #[must_use]
    pub fn output_len(&self) -> usize {
        self.output_len
    }

    #[must_use]
    pub fn lse_len(&self) -> usize {
        self.lse_len
    }

    pub fn packed_words(&self) -> usize {
        self.packed_words
    }

    #[must_use]
    pub fn raw_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

struct WgpuBf16AttentionInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
    max_workgroups_per_dimension: u32,
    max_storage_buffer_binding_size: u64,
    owner_id: usize,
}

#[derive(Clone)]
pub struct WgpuBf16Attention {
    inner: Arc<WgpuBf16AttentionInner>,
}

impl WgpuBf16Attention {
    /// Create the portable packed-bfloat16 executor.
    ///
    /// No shader-f16 feature is requested: the shader only contains baseline
    /// WGSL `u32` and `f32` types, bit shifts and bitcasts.
    pub fn new() -> Result<Self, WgpuBf16AttentionError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|_| WgpuBf16AttentionError::Unavailable)?;

        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flat-attention-fml1-packed-bf16"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .map_err(|error| WgpuBf16AttentionError::Execution(format!("request_device: {error}")))?;

        let pipeline = create_pipeline(&device)?;
        let device_limits = device.limits();
        let max_workgroups_per_dimension = device_limits.max_compute_workgroups_per_dimension;
        Ok(Self {
            inner: Arc::new(WgpuBf16AttentionInner {
                device,
                queue,
                pipeline,
                adapter_name,
                max_workgroups_per_dimension,
                max_storage_buffer_binding_size: device_limits.max_storage_buffer_binding_size,
                owner_id: super::next_resident_owner_id(),
            }),
        })
    }

    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.inner.adapter_name
    }

    pub fn upload_bf16(
        &self,
        data: &[Bf16],
    ) -> Result<WgpuResidentBf16Buffer, WgpuBf16AttentionError> {
        let bytes = encode_packed_bf16(data)?;
        let size = bytes.len().max(4) as u64;
        if size > self.inner.max_storage_buffer_binding_size {
            return Err(WgpuBf16AttentionError::DeviceBufferLimit {
                required_bytes: size,
                maximum_bytes: self.inner.max_storage_buffer_binding_size,
            });
        }
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-fml1-packed-bf16-input"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.inner.queue.write_buffer(&buffer, 0, &bytes);
        }
        Ok(WgpuResidentBf16Buffer {
            buffer: Arc::new(buffer),
            len: data.len(),
            owner: self.owner_id(),
        })
    }

    pub fn forward(
        &self,
        q: &[Bf16],
        k: &[Bf16],
        v: &[Bf16],
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<FlatAttentionBf16Output, WgpuBf16AttentionError> {
        let dispatch = self.validate_dispatch(shape)?;
        validate_bf16_input("Q", q, dispatch.tensor_len)?;
        validate_bf16_input("K", k, dispatch.tensor_len)?;
        validate_bf16_input("V", v, dispatch.tensor_len)?;
        validate_score_bound(q, k, shape.head_dim, config.resolved_scale(shape.head_dim)?)?;
        let q_gpu = self.upload_bf16(q)?;
        let k_gpu = self.upload_bf16(k)?;
        let v_gpu = self.upload_bf16(v)?;
        let resident = self.forward_resident(&q_gpu, &k_gpu, &v_gpu, shape, config)?;
        self.download_attention(&resident)
    }

    pub fn forward_resident(
        &self,
        q: &WgpuResidentBf16Buffer,
        k: &WgpuResidentBf16Buffer,
        v: &WgpuResidentBf16Buffer,
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<WgpuResidentBf16AttentionOutput, WgpuBf16AttentionError> {
        let dispatch = self.validate_dispatch(shape)?;
        self.validate_resident("Q", q, dispatch.tensor_len)?;
        self.validate_resident("K", k, dispatch.tensor_len)?;
        self.validate_resident("V", v, dispatch.tensor_len)?;
        let scale = config.resolved_scale(shape.head_dim)?;

        let output_words = dispatch.tensor_len / 2;
        let packed_words = output_words
            .checked_add(dispatch.lse_len)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let output_bytes = bytes_for_words(packed_words)?;
        let output = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-fml1-packed-bf16-o-lse"),
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
            label: Some("flat-attention-fml1-packed-bf16-params"),
            size: params_bytes.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.inner
            .queue
            .write_buffer(&params_buffer, 0, &params_bytes);

        let bind_group = self
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flat-attention-fml1-packed-bf16"),
                layout: &self.inner.pipeline.get_bind_group_layout(0),
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

        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flat-attention-fml1-packed-bf16"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-attention-fml1-packed-bf16"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inner.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.query_workgroups, dispatch.batch_heads, 1);
        }
        self.inner.queue.submit(Some(encoder.finish()));

        Ok(WgpuResidentBf16AttentionOutput {
            buffer: Arc::new(output),
            output_len: dispatch.tensor_len,
            lse_len: dispatch.lse_len,
            packed_words,
            owner: self.owner_id(),
        })
    }

    pub fn download_attention(
        &self,
        resident: &WgpuResidentBf16AttentionOutput,
    ) -> Result<FlatAttentionBf16Output, WgpuBf16AttentionError> {
        if resident.owner != self.owner_id() {
            return Err(WgpuBf16AttentionError::ForeignBuffer);
        }
        let bytes = bytes_for_words(resident.packed_words)?;
        let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-fml1-packed-bf16-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flat-attention-fml1-packed-bf16-readback"),
                });
        encoder.copy_buffer_to_buffer(&resident.buffer, 0, &staging, 0, bytes);
        self.inner.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.inner.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .map_err(|error| WgpuBf16AttentionError::Execution(format!("map callback: {error}")))?
            .map_err(|error| WgpuBf16AttentionError::Execution(format!("map read: {error:?}")))?;

        let mapped = slice
            .get_mapped_range()
            .map_err(|error| WgpuBf16AttentionError::Execution(format!("map range: {error}")))?;
        let decoded = decode_output(&mapped, resident.output_len, resident.lse_len)?;
        drop(mapped);
        staging.unmap();
        Ok(decoded)
    }

    fn validate_dispatch(
        &self,
        shape: AttentionShape,
    ) -> Result<DispatchGeometry, WgpuBf16AttentionError> {
        shape.validate()?;
        if shape.head_dim > WGSL_MAX_HEAD_DIM || !matches!(shape.head_dim, 64 | 128) {
            return Err(WgpuBf16AttentionError::UnsupportedHeadDim {
                actual: shape.head_dim,
            });
        }
        let tensor_len = shape.tensor_len()?;
        let lse_len = shape.lse_len()?;
        if tensor_len > u32::MAX as usize {
            return Err(WgpuBf16AttentionError::IndexSpaceExceeded {
                elements: tensor_len,
            });
        }
        let batch_heads = shape
            .batch
            .checked_mul(shape.heads)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let query_workgroups = shape.seq_len.div_ceil(WGSL_QUERY_ROWS);
        let maximum = self.inner.max_workgroups_per_dimension;
        if query_workgroups > maximum as usize {
            return Err(WgpuBf16AttentionError::DispatchLimit {
                axis: "x/query_tiles",
                actual: query_workgroups,
                maximum,
            });
        }
        if batch_heads > maximum as usize {
            return Err(WgpuBf16AttentionError::DispatchLimit {
                axis: "y/batch_heads",
                actual: batch_heads,
                maximum,
            });
        }
        // Packed bf16 Q/K/V and the O|LSE output are storage-bound; enforce the
        // device binding limit instead of relying on wgpu validation aborts.
        let maximum_binding_bytes = self.inner.max_storage_buffer_binding_size;
        let packed_words = tensor_len.div_ceil(2);
        let input_words = packed_words
            .checked_mul(3)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let required_words = input_words
            .checked_add(packed_words)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let required_bytes = bytes_for_words(required_words)?;
        if required_bytes > maximum_binding_bytes {
            return Err(WgpuBf16AttentionError::DeviceBufferLimit {
                required_bytes,
                maximum_bytes: maximum_binding_bytes,
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
        buffer: &WgpuResidentBf16Buffer,
        expected: usize,
    ) -> Result<(), WgpuBf16AttentionError> {
        if buffer.owner != self.owner_id() {
            return Err(WgpuBf16AttentionError::ForeignBuffer);
        }
        if buffer.len != expected {
            return Err(WgpuBf16AttentionError::ResidentLength {
                tensor,
                actual: buffer.len,
                expected,
            });
        }
        Ok(())
    }

    fn owner_id(&self) -> usize {
        self.inner.owner_id
    }
}

fn create_pipeline(device: &wgpu::Device) -> Result<wgpu::ComputePipeline, WgpuBf16AttentionError> {
    wgpu_internal::create_pipeline(
        device,
        FLAT_FWD_BF16_WGSL,
        "flat-attention-fml1-packed-bf16",
        "flat_attention_forward",
    )
    .map_err(WgpuBf16AttentionError::ShaderUnsupported)
}

struct DispatchGeometry {
    tensor_len: usize,
    lse_len: usize,
    batch_heads: u32,
    seq_len: u32,
    query_workgroups: u32,
}

fn validate_bf16_input(
    name: &'static str,
    data: &[Bf16],
    expected: usize,
) -> Result<(), FlatAttentionError> {
    if data.len() != expected {
        return Err(FlatAttentionError::LengthMismatch {
            tensor: name,
            actual: data.len(),
            expected,
        });
    }
    if let Some(index) = data.iter().position(|value| !value.is_finite()) {
        return Err(FlatAttentionError::NonFiniteInput {
            tensor: name,
            index,
        });
    }
    Ok(())
}

fn validate_score_bound(
    q: &[Bf16],
    k: &[Bf16],
    head_dim: usize,
    scale: f32,
) -> Result<(), WgpuBf16AttentionError> {
    let max_abs = |values: &[Bf16]| {
        values
            .iter()
            .map(|value| f64::from(value.to_f32().abs()))
            .fold(0.0f64, f64::max)
    };
    let bound = f64::from(scale) * head_dim as f64 * max_abs(q) * max_abs(k);
    if bound < f64::from(f32::MAX) {
        Ok(())
    } else {
        Err(WgpuBf16AttentionError::ScoreBoundExceeded { bound })
    }
}

fn checked_u32(value: usize) -> Result<u32, WgpuBf16AttentionError> {
    wgpu_internal::checked_u32(value)
        .ok_or(WgpuBf16AttentionError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_words(words: usize) -> Result<u64, WgpuBf16AttentionError> {
    let bytes = words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| WgpuBf16AttentionError::IndexSpaceExceeded { elements: words })
}

fn encode_packed_bf16(values: &[Bf16]) -> Result<Vec<u8>, WgpuBf16AttentionError> {
    if !values.len().is_multiple_of(2) {
        return Err(WgpuBf16AttentionError::OddPackedLength {
            actual: values.len(),
        });
    }
    let word_count = values.len() / 2;
    let capacity = word_count
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let mut bytes = Vec::with_capacity(capacity);
    for pair in values.chunks_exact(2) {
        let word = u32::from(pair[0].to_bits()) | (u32::from(pair[1].to_bits()) << 16);
        bytes.extend_from_slice(&word.to_ne_bytes());
    }
    Ok(bytes)
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    wgpu_internal::encode_u32(values)
}

fn decode_output(
    bytes: &[u8],
    output_len: usize,
    lse_len: usize,
) -> Result<FlatAttentionBf16Output, WgpuBf16AttentionError> {
    let output_words = output_len / 2;
    let expected_words = output_words
        .checked_add(lse_len)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let expected_bytes = expected_words
        .checked_mul(4)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    if bytes.len() != expected_bytes {
        return Err(WgpuBf16AttentionError::Execution(format!(
            "readback returned {} bytes, expected {expected_bytes}",
            bytes.len()
        )));
    }

    let mut words = Vec::with_capacity(expected_words);
    for chunk in bytes.chunks_exact(4) {
        words.push(u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    let mut output = Vec::with_capacity(output_len);
    for &word in &words[..output_words] {
        output.push(Bf16::from_bits((word & 0xffff) as u16));
        output.push(Bf16::from_bits((word >> 16) as u16));
    }
    let lse = words[output_words..]
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    Ok(FlatAttentionBf16Output { output, lse })
}
