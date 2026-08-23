//! M8 portable packed-binary16 I/O for FLAT-ATTENTION.
//!
//! WGPU/Naga 0.20 does not expose native WGSL `f16`, so this backend stores two
//! IEEE-754 binary16 scalars in each `u32`. The shader converts those pairs to
//! `f32` with WGSL pack/unpack builtins. All attention arithmetic and LSE remain
//! `f32`. The already-qualified f32 executor stays independent and is the only
//! fallback; no path silently executes on CPU.

use std::fmt;
use std::sync::{mpsc, Arc};

use super::{
    AttentionShape, FlatAttentionConfig, FlatAttentionError, FlatAttentionF16Output,
    FlatAttentionOutput, WgpuFlatAttention, WgpuFlatAttentionError, F16, FLAT_FWD_F16_WGSL,
    WGSL_MAX_HEAD_DIM, WGSL_QUERY_ROWS,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WgpuIoPrecision {
    F32,
    PackedF16,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WgpuF16AttentionError {
    Core(FlatAttentionError),
    F32(WgpuFlatAttentionError),
    Unavailable,
    PackedShaderUnsupported(String),
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
    ForeignBuffer,
    ResidentLength {
        tensor: &'static str,
        actual: usize,
        expected: usize,
    },
    Execution(String),
}

impl fmt::Display for WgpuF16AttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(error) => write!(f, "{error}"),
            Self::F32(error) => write!(f, "f32 fallback failed: {error}"),
            Self::Unavailable => write!(f, "no compatible WGPU adapter/device is available"),
            Self::PackedShaderUnsupported(message) => {
                write!(f, "packed-binary16 WGSL pipeline is unsupported: {message}")
            }
            Self::UnsupportedHeadDim { actual } => write!(
                f,
                "M8 packed-binary16 I/O supports head_dim 64 or 128, got {actual}"
            ),
            Self::OddPackedLength { actual } => write!(
                f,
                "packed-binary16 storage requires an even scalar count, got {actual}"
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
                "M8 packed index space requires {elements} elements, exceeding u32 addressing"
            ),
            Self::DeviceBufferLimit {
                required_bytes,
                maximum_bytes,
            } => write!(
                f,
                "packed-binary16 attention requires {required_bytes} bytes per buffer, device maximum is {maximum_bytes}"
            ),
            Self::ForeignBuffer => write!(
                f,
                "resident packed-binary16 buffer belongs to another WGPU context"
            ),
            Self::ResidentLength {
                tensor,
                actual,
                expected,
            } => write!(
                f,
                "resident packed-binary16 tensor {tensor} contains {actual} elements, expected {expected}"
            ),
            Self::Execution(message) => write!(f, "WGPU packed-f16 execution failed: {message}"),
        }
    }
}

impl std::error::Error for WgpuF16AttentionError {}

impl From<FlatAttentionError> for WgpuF16AttentionError {
    fn from(value: FlatAttentionError) -> Self {
        Self::Core(value)
    }
}

impl From<WgpuFlatAttentionError> for WgpuF16AttentionError {
    fn from(value: WgpuFlatAttentionError) -> Self {
        Self::F32(value)
    }
}

pub struct WgpuResidentF16Buffer {
    buffer: Arc<wgpu::Buffer>,
    len: usize,
    owner: usize,
}

impl WgpuResidentF16Buffer {
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

pub struct WgpuResidentF16AttentionOutput {
    buffer: Arc<wgpu::Buffer>,
    output_len: usize,
    lse_len: usize,
    packed_words: usize,
    owner: usize,
}

impl WgpuResidentF16AttentionOutput {
    pub fn output_len(&self) -> usize {
        self.output_len
    }

    pub fn lse_len(&self) -> usize {
        self.lse_len
    }

    pub fn packed_words(&self) -> usize {
        self.packed_words
    }

    pub fn raw_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

struct WgpuF16AttentionInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
    max_workgroups_per_dimension: u32,
    max_storage_buffer_binding_size: u64,
    owner_id: usize,
}

#[derive(Clone)]
pub struct WgpuF16Attention {
    inner: Arc<WgpuF16AttentionInner>,
}

impl WgpuF16Attention {
    /// Create the portable packed-binary16 executor.
    ///
    /// No native shader-f16 feature is requested: the shader only contains
    /// baseline WGSL `u32` and `f32` types plus pack/unpack conversion builtins.
    pub fn new() -> Result<Self, WgpuF16AttentionError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or(WgpuF16AttentionError::Unavailable)?;

        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("flat-attention-m8-packed-f16"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|error| WgpuF16AttentionError::Execution(format!("request_device: {error}")))?;

        let pipeline = create_pipeline(&device)?;
        let device_limits = device.limits();
        let max_workgroups_per_dimension = device_limits.max_compute_workgroups_per_dimension;
        Ok(Self {
            inner: Arc::new(WgpuF16AttentionInner {
                device,
                queue,
                pipeline,
                adapter_name,
                max_workgroups_per_dimension,
                max_storage_buffer_binding_size: u64::from(
                    device_limits.max_storage_buffer_binding_size,
                ),
                owner_id: super::next_resident_owner_id(),
            }),
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.inner.adapter_name
    }

    pub fn io_precision(&self) -> WgpuIoPrecision {
        WgpuIoPrecision::PackedF16
    }

    pub fn upload_f16(&self, data: &[F16]) -> Result<WgpuResidentF16Buffer, WgpuF16AttentionError> {
        let bytes = encode_packed_f16(data)?;
        let size = bytes.len().max(4) as u64;
        if size > self.inner.max_storage_buffer_binding_size {
            return Err(WgpuF16AttentionError::DeviceBufferLimit {
                required_bytes: size,
                maximum_bytes: self.inner.max_storage_buffer_binding_size,
            });
        }
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-m8-packed-f16-input"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.inner.queue.write_buffer(&buffer, 0, &bytes);
        }
        Ok(WgpuResidentF16Buffer {
            buffer: Arc::new(buffer),
            len: data.len(),
            owner: self.owner_id(),
        })
    }

    pub fn forward(
        &self,
        q: &[F16],
        k: &[F16],
        v: &[F16],
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<FlatAttentionF16Output, WgpuF16AttentionError> {
        let dispatch = self.validate_dispatch(shape)?;
        validate_f16_input("Q", q, dispatch.tensor_len)?;
        validate_f16_input("K", k, dispatch.tensor_len)?;
        validate_f16_input("V", v, dispatch.tensor_len)?;
        let q_gpu = self.upload_f16(q)?;
        let k_gpu = self.upload_f16(k)?;
        let v_gpu = self.upload_f16(v)?;
        let resident = self.forward_resident(&q_gpu, &k_gpu, &v_gpu, shape, config)?;
        self.download_attention(&resident)
    }

    pub fn forward_resident(
        &self,
        q: &WgpuResidentF16Buffer,
        k: &WgpuResidentF16Buffer,
        v: &WgpuResidentF16Buffer,
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<WgpuResidentF16AttentionOutput, WgpuF16AttentionError> {
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
            label: Some("flat-attention-m8-packed-f16-o-lse"),
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
            label: Some("flat-attention-m8-packed-f16-params"),
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
                label: Some("flat-attention-m8-packed-f16"),
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
                    label: Some("flat-attention-m8-packed-f16"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-attention-m8-packed-f16"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inner.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.query_workgroups, dispatch.batch_heads, 1);
        }
        self.inner.queue.submit(Some(encoder.finish()));

        Ok(WgpuResidentF16AttentionOutput {
            buffer: Arc::new(output),
            output_len: dispatch.tensor_len,
            lse_len: dispatch.lse_len,
            packed_words,
            owner: self.owner_id(),
        })
    }

    pub fn download_attention(
        &self,
        resident: &WgpuResidentF16AttentionOutput,
    ) -> Result<FlatAttentionF16Output, WgpuF16AttentionError> {
        if resident.owner != self.owner_id() {
            return Err(WgpuF16AttentionError::ForeignBuffer);
        }
        let bytes = bytes_for_words(resident.packed_words)?;
        let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-m8-packed-f16-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flat-attention-m8-packed-f16-readback"),
                });
        encoder.copy_buffer_to_buffer(&resident.buffer, 0, &staging, 0, bytes);
        self.inner.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.inner.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|error| WgpuF16AttentionError::Execution(format!("map callback: {error}")))?
            .map_err(|error| WgpuF16AttentionError::Execution(format!("map read: {error:?}")))?;

        let mapped = slice.get_mapped_range();
        let decoded = decode_output(&mapped, resident.output_len, resident.lse_len)?;
        drop(mapped);
        staging.unmap();
        Ok(decoded)
    }

    fn validate_dispatch(
        &self,
        shape: AttentionShape,
    ) -> Result<DispatchGeometry, WgpuF16AttentionError> {
        shape.validate()?;
        if shape.head_dim > WGSL_MAX_HEAD_DIM || !matches!(shape.head_dim, 64 | 128) {
            return Err(WgpuF16AttentionError::UnsupportedHeadDim {
                actual: shape.head_dim,
            });
        }
        let tensor_len = shape.tensor_len()?;
        let lse_len = shape.lse_len()?;
        if tensor_len > u32::MAX as usize {
            return Err(WgpuF16AttentionError::IndexSpaceExceeded {
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
            return Err(WgpuF16AttentionError::DispatchLimit {
                axis: "x/query_tiles",
                actual: query_workgroups,
                maximum,
            });
        }
        if batch_heads > maximum as usize {
            return Err(WgpuF16AttentionError::DispatchLimit {
                axis: "y/batch_heads",
                actual: batch_heads,
                maximum,
            });
        }
        // Packed Q/K/V and the O|LSE output are storage-bound; enforce the
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
            return Err(WgpuF16AttentionError::DeviceBufferLimit {
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
        buffer: &WgpuResidentF16Buffer,
        expected: usize,
    ) -> Result<(), WgpuF16AttentionError> {
        if buffer.owner != self.owner_id() {
            return Err(WgpuF16AttentionError::ForeignBuffer);
        }
        if buffer.len != expected {
            return Err(WgpuF16AttentionError::ResidentLength {
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

/// Shape-based convenience router. It never falls back to CPU.
pub struct WgpuPreferredAttention {
    f32: WgpuFlatAttention,
    packed_f16: Option<WgpuF16Attention>,
}

impl WgpuPreferredAttention {
    pub fn new() -> Result<Self, WgpuF16AttentionError> {
        let f32 = WgpuFlatAttention::new()?;
        let packed_f16 = match WgpuF16Attention::new() {
            Ok(context) => Some(context),
            Err(WgpuF16AttentionError::Unavailable)
            | Err(WgpuF16AttentionError::PackedShaderUnsupported(_)) => None,
            Err(error) => return Err(error),
        };
        Ok(Self { f32, packed_f16 })
    }

    pub fn io_precision_for_head_dim(&self, head_dim: usize) -> WgpuIoPrecision {
        if self.packed_f16.is_some() && matches!(head_dim, 64 | 128) {
            WgpuIoPrecision::PackedF16
        } else {
            WgpuIoPrecision::F32
        }
    }

    pub fn adapter_name(&self) -> &str {
        self.packed_f16
            .as_ref()
            .map_or_else(|| self.f32.adapter_name(), WgpuF16Attention::adapter_name)
    }

    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<FlatAttentionOutput, WgpuF16AttentionError> {
        if self.io_precision_for_head_dim(shape.head_dim) == WgpuIoPrecision::PackedF16 {
            let tensor_len = shape.tensor_len()?;
            validate_f32_for_quantization("Q", q, tensor_len)?;
            validate_f32_for_quantization("K", k, tensor_len)?;
            validate_f32_for_quantization("V", v, tensor_len)?;
            let q16: Vec<F16> = q.iter().copied().map(F16::from_f32).collect();
            let k16: Vec<F16> = k.iter().copied().map(F16::from_f32).collect();
            let v16: Vec<F16> = v.iter().copied().map(F16::from_f32).collect();
            if q16
                .iter()
                .chain(&k16)
                .chain(&v16)
                .any(|value| !value.is_finite())
            {
                return self.f32.forward(q, k, v, shape, config).map_err(Into::into);
            }
            let result = self
                .packed_f16
                .as_ref()
                .expect("precision selection requires packed-f16 context")
                .forward(&q16, &k16, &v16, shape, config)?;
            Ok(FlatAttentionOutput {
                output: result.output.into_iter().map(F16::to_f32).collect(),
                lse: result.lse,
            })
        } else {
            self.f32.forward(q, k, v, shape, config).map_err(Into::into)
        }
    }
}

fn create_pipeline(device: &wgpu::Device) -> Result<wgpu::ComputePipeline, WgpuF16AttentionError> {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flat-attention-m8-packed-f16"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(FLAT_FWD_F16_WGSL)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("flat-attention-m8-packed-f16"),
        layout: None,
        module: &shader,
        entry_point: "flat_attention_forward",
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });
    let validation_error = pollster::block_on(device.pop_error_scope());
    match validation_error {
        Some(error) => Err(WgpuF16AttentionError::PackedShaderUnsupported(
            error.to_string(),
        )),
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

fn validate_f16_input(
    name: &'static str,
    data: &[F16],
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

fn validate_f32_for_quantization(
    name: &'static str,
    data: &[f32],
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

fn checked_u32(value: usize) -> Result<u32, WgpuF16AttentionError> {
    u32::try_from(value).map_err(|_| WgpuF16AttentionError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_words(words: usize) -> Result<u64, WgpuF16AttentionError> {
    let bytes = words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| WgpuF16AttentionError::IndexSpaceExceeded { elements: words })
}

fn encode_packed_f16(values: &[F16]) -> Result<Vec<u8>, WgpuF16AttentionError> {
    if values.len() % 2 != 0 {
        return Err(WgpuF16AttentionError::OddPackedLength {
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
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn decode_output(
    bytes: &[u8],
    output_len: usize,
    lse_len: usize,
) -> Result<FlatAttentionF16Output, WgpuF16AttentionError> {
    let output_words = output_len / 2;
    let expected_words = output_words
        .checked_add(lse_len)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    let expected_bytes = expected_words
        .checked_mul(4)
        .ok_or(FlatAttentionError::ShapeOverflow)?;
    if bytes.len() != expected_bytes {
        return Err(WgpuF16AttentionError::Execution(format!(
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
        output.push(F16::from_bits((word & 0xffff) as u16));
        output.push(F16::from_bits((word >> 16) as u16));
    }
    let lse = words[output_words..]
        .iter()
        .copied()
        .map(f32::from_bits)
        .collect();
    Ok(FlatAttentionF16Output { output, lse })
}
