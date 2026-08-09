//! Real portable GPU execution for the fused FLAT-ATTENTION forward kernel.
//!
//! This backend never substitutes the CPU oracle. Construction fails explicitly
//! when no WGPU adapter/device is available, and dispatch errors are surfaced to
//! the caller. Q/K/V and the packed O+LSE result stay resident unless the caller
//! explicitly requests a readback.

use std::fmt;
use std::sync::{mpsc, Arc};

use super::{
    validate_input, AttentionShape, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
    FLAT_FWD_WGSL, WGSL_MAX_HEAD_DIM,
};

/// Errors produced by the real WGPU execution path.
#[derive(Debug, Clone, PartialEq)]
pub enum WgpuFlatAttentionError {
    /// Shape/input/config validation from the backend-neutral core.
    Core(FlatAttentionError),
    /// No adapter/device satisfying the portable contract could be acquired.
    Unavailable,
    /// The first portable kernel is intentionally bounded to 128 dimensions.
    UnsupportedHeadDim { actual: usize, maximum: usize },
    /// A dispatch dimension exceeds the selected device's workgroup-count limit.
    DispatchLimit {
        axis: &'static str,
        actual: usize,
        maximum: u32,
    },
    /// The packed shader index space would exceed `u32` addressing.
    IndexSpaceExceeded { elements: usize },
    /// A resident buffer came from another WGPU context.
    ForeignBuffer,
    /// A resident input has the wrong logical element count.
    ResidentLength {
        tensor: &'static str,
        actual: usize,
        expected: usize,
    },
    /// Device execution, mapping, or synchronization failed.
    Execution(String),
}

impl fmt::Display for WgpuFlatAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core(err) => write!(f, "{err}"),
            Self::Unavailable => write!(f, "no compatible WGPU adapter/device is available"),
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

/// A storage buffer resident on the context that created it.
pub struct WgpuResidentBuffer {
    buffer: Arc<wgpu::Buffer>,
    len: usize,
    owner: usize,
}

impl WgpuResidentBuffer {
    /// Logical number of `f32` elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the logical buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw WGPU buffer for future zero-copy SciRust integration.
    pub fn raw_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}

/// Resident result of one fused forward dispatch.
///
/// The underlying storage is packed as `[O | LSE]`. Keeping one storage buffer
/// lets the portable shader stay within the four-storage-buffer downlevel
/// contract while preserving both outputs required by backward recomputation.
pub struct WgpuResidentAttentionOutput {
    combined: WgpuResidentBuffer,
    output_len: usize,
    lse_len: usize,
}

impl WgpuResidentAttentionOutput {
    /// Number of `f32` elements occupied by O at the start of the buffer.
    pub fn output_len(&self) -> usize {
        self.output_len
    }

    /// Number of trailing `f32` elements occupied by LSE.
    pub fn lse_len(&self) -> usize {
        self.lse_len
    }

    /// Total packed `[O | LSE]` storage.
    pub fn combined(&self) -> &WgpuResidentBuffer {
        &self.combined
    }
}

struct WgpuFlatAttentionInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
    max_workgroups_per_dimension: u32,
}

/// WGPU device, queue, and compiled fused-attention pipeline.
#[derive(Clone)]
pub struct WgpuFlatAttention {
    inner: Arc<WgpuFlatAttentionInner>,
}

impl WgpuFlatAttention {
    /// Acquire a real WGPU adapter/device and compile the fused forward shader.
    ///
    /// The requested limits are `downlevel_defaults`: this intentionally keeps
    /// the baseline kernel inside a conservative portable profile. There is no
    /// CPU fallback when adapter acquisition fails.
    pub fn new() -> Result<Self, WgpuFlatAttentionError> {
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

        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("flat-attention"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|err| WgpuFlatAttentionError::Execution(format!("request_device: {err}")))?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-attention-forward"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(FLAT_FWD_WGSL)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-attention-forward"),
            layout: None,
            module: &shader,
            entry_point: "flat_attention_forward",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        let max_workgroups_per_dimension = device.limits().max_compute_workgroups_per_dimension;

        Ok(Self {
            inner: Arc::new(WgpuFlatAttentionInner {
                device,
                queue,
                pipeline,
                adapter_name,
                max_workgroups_per_dimension,
            }),
        })
    }

    /// Human-readable adapter name reported by WGPU.
    pub fn adapter_name(&self) -> &str {
        &self.inner.adapter_name
    }

    /// Device limit used to validate X/Y dispatch dimensions.
    pub fn max_workgroups_per_dimension(&self) -> u32 {
        self.inner.max_workgroups_per_dimension
    }

    /// Convenience path: validate and upload Q/K/V, execute one fused dispatch,
    /// and explicitly download O/LSE.
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

    /// Upload `f32` data to a storage buffer owned by this context.
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

    /// Execute the fused forward kernel on already-resident Q/K/V.
    ///
    /// Exactly one compute dispatch is encoded. No score/probability matrix and
    /// no host readback are created by this method.
    pub fn forward_resident(
        &self,
        q: &WgpuResidentBuffer,
        k: &WgpuResidentBuffer,
        v: &WgpuResidentBuffer,
        shape: AttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<WgpuResidentAttentionOutput, WgpuFlatAttentionError> {
        let (tensor_len, lse_len, batch_heads, seq_len) = self.validate_dispatch(shape)?;
        self.validate_resident("Q", q, tensor_len)?;
        self.validate_resident("K", k, tensor_len)?;
        self.validate_resident("V", v, tensor_len)?;
        let scale = config.resolved_scale(shape.head_dim)?;

        let combined_len = tensor_len
            .checked_add(lse_len)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        if combined_len > u32::MAX as usize {
            return Err(WgpuFlatAttentionError::IndexSpaceExceeded {
                elements: combined_len,
            });
        }
        let output_bytes = bytes_for_f32_len(combined_len)?;
        let output = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-o-lse"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [
            seq_len,
            u32::try_from(shape.head_dim).map_err(|_| {
                WgpuFlatAttentionError::IndexSpaceExceeded {
                    elements: shape.head_dim,
                }
            })?,
            batch_heads,
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

        let bind_group = self
            .inner
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("flat-attention-forward"),
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
                    label: Some("flat-attention-forward"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-attention-forward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inner.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(seq_len, batch_heads, 1);
        }
        self.inner.queue.submit(Some(encoder.finish()));

        Ok(WgpuResidentAttentionOutput {
            combined: WgpuResidentBuffer {
                buffer: Arc::new(output),
                len: combined_len,
                owner: self.owner_id(),
            },
            output_len: tensor_len,
            lse_len,
        })
    }

    /// Explicitly download and split the packed `[O | LSE]` resident result.
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

    fn validate_dispatch(
        &self,
        shape: AttentionShape,
    ) -> Result<(usize, usize, u32, u32), WgpuFlatAttentionError> {
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
        let maximum = self.inner.max_workgroups_per_dimension;
        if shape.seq_len > maximum as usize {
            return Err(WgpuFlatAttentionError::DispatchLimit {
                axis: "x/sequence",
                actual: shape.seq_len,
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
        Ok((
            tensor_len,
            lse_len,
            u32::try_from(batch_heads).map_err(|_| WgpuFlatAttentionError::IndexSpaceExceeded {
                elements: batch_heads,
            })?,
            u32::try_from(shape.seq_len).map_err(|_| {
                WgpuFlatAttentionError::IndexSpaceExceeded {
                    elements: shape.seq_len,
                }
            })?,
        ))
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
        let mut encoder =
            self.inner
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
