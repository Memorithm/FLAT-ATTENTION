//! Native WGPU execution for M10 GQA/MQA.
//!
//! This backend preserves physical KV-head cardinality. K/V buffers contain
//! exactly `batch * kv_heads * seq_len * head_dim` scalars and are never
//! expanded to `q_heads` before dispatch.

use std::fmt;
use std::sync::{mpsc, Arc};

use super::{
    validate_input, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
    GroupedAttentionShape, WgpuFlatAttentionError, FLAT_FWD_GROUPED_WGSL, WGSL_MAX_HEAD_DIM,
    WGSL_QUERY_ROWS,
};

pub struct WgpuGroupedResidentBuffer {
    buffer: Arc<wgpu::Buffer>,
    len: usize,
    owner: usize,
}

impl WgpuGroupedResidentBuffer {
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

pub struct WgpuGroupedResidentAttentionOutput {
    combined: WgpuGroupedResidentBuffer,
    output_len: usize,
    lse_len: usize,
}

impl WgpuGroupedResidentAttentionOutput {
    pub fn output_len(&self) -> usize {
        self.output_len
    }

    pub fn lse_len(&self) -> usize {
        self.lse_len
    }

    pub fn combined(&self) -> &WgpuGroupedResidentBuffer {
        &self.combined
    }
}

struct WgpuGroupedAttentionInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
    max_workgroups_per_dimension: u32,
}

#[derive(Clone)]
pub struct WgpuGroupedAttention {
    inner: Arc<WgpuGroupedAttentionInner>,
}

impl fmt::Debug for WgpuGroupedAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuGroupedAttention")
            .field("adapter_name", &self.inner.adapter_name)
            .finish_non_exhaustive()
    }
}

impl WgpuGroupedAttention {
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
                label: Some("flat-attention-grouped-q4"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .map_err(|err| WgpuFlatAttentionError::Execution(format!("request_device: {err}")))?;

        let pipeline = create_pipeline(
            &device,
            FLAT_FWD_GROUPED_WGSL,
            "flat-attention-forward-grouped-q4",
        )
        .map_err(|error| {
            WgpuFlatAttentionError::Execution(format!("M10 grouped pipeline: {error}"))
        })?;
        let max_workgroups_per_dimension = device.limits().max_compute_workgroups_per_dimension;

        Ok(Self {
            inner: Arc::new(WgpuGroupedAttentionInner {
                device,
                queue,
                pipeline,
                adapter_name,
                max_workgroups_per_dimension,
            }),
        })
    }

    pub fn adapter_name(&self) -> &str {
        &self.inner.adapter_name
    }

    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: GroupedAttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<FlatAttentionOutput, WgpuFlatAttentionError> {
        shape.validate()?;
        let q_len = shape.q_tensor_len()?;
        let kv_len = shape.kv_tensor_len()?;
        validate_input("Q", q, q_len)?;
        validate_input("K", k, kv_len)?;
        validate_input("V", v, kv_len)?;

        let q_gpu = self.upload(q)?;
        let k_gpu = self.upload(k)?;
        let v_gpu = self.upload(v)?;
        let resident = self.forward_resident(&q_gpu, &k_gpu, &v_gpu, shape, config)?;
        self.download_attention(&resident)
    }

    pub fn upload(
        &self,
        data: &[f32],
    ) -> Result<WgpuGroupedResidentBuffer, WgpuFlatAttentionError> {
        let bytes = encode_f32(data)?;
        let size = bytes.len().max(4) as u64;
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-grouped-resident-input"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.inner.queue.write_buffer(&buffer, 0, &bytes);
        }
        Ok(WgpuGroupedResidentBuffer {
            buffer: Arc::new(buffer),
            len: data.len(),
            owner: self.owner_id(),
        })
    }

    pub fn forward_resident(
        &self,
        q: &WgpuGroupedResidentBuffer,
        k: &WgpuGroupedResidentBuffer,
        v: &WgpuGroupedResidentBuffer,
        shape: GroupedAttentionShape,
        config: FlatAttentionConfig,
    ) -> Result<WgpuGroupedResidentAttentionOutput, WgpuFlatAttentionError> {
        let dispatch = self.validate_dispatch(shape)?;
        self.validate_resident("Q", q, dispatch.q_tensor_len)?;
        self.validate_resident("K", k, dispatch.kv_tensor_len)?;
        self.validate_resident("V", v, dispatch.kv_tensor_len)?;
        let scale = config.resolved_scale(shape.head_dim)?;

        let combined_len = dispatch
            .q_tensor_len
            .checked_add(dispatch.lse_len)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        let output_bytes = bytes_for_f32_len(combined_len)?;
        let output = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-grouped-o-lse"),
            size: output_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [
            dispatch.seq_len,
            checked_u32(shape.head_dim)?,
            checked_u32(shape.q_heads)?,
            checked_u32(shape.kv_heads)?,
            checked_u32(shape.batch)?,
            u32::from(config.causal),
            scale.to_bits(),
            0,
        ];
        let params_bytes = encode_u32(&params);
        let params_buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-attention-grouped-params"),
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
                label: Some("flat-attention-grouped-bind-group"),
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
                    label: Some("flat-attention-grouped-forward"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-attention-grouped-forward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inner.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.query_workgroups, dispatch.q_batch_heads, 1);
        }
        self.inner.queue.submit(Some(encoder.finish()));

        Ok(WgpuGroupedResidentAttentionOutput {
            combined: WgpuGroupedResidentBuffer {
                buffer: Arc::new(output),
                len: combined_len,
                owner: self.owner_id(),
            },
            output_len: dispatch.q_tensor_len,
            lse_len: dispatch.lse_len,
        })
    }

    pub fn download_attention(
        &self,
        resident: &WgpuGroupedResidentAttentionOutput,
    ) -> Result<FlatAttentionOutput, WgpuFlatAttentionError> {
        self.ensure_owner(&resident.combined)?;
        let expected = resident
            .output_len
            .checked_add(resident.lse_len)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        if resident.combined.len != expected {
            return Err(WgpuFlatAttentionError::Execution(
                "grouped resident output metadata is inconsistent".into(),
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
        shape: GroupedAttentionShape,
    ) -> Result<GroupedDispatchGeometry, WgpuFlatAttentionError> {
        shape.validate()?;
        if shape.head_dim > WGSL_MAX_HEAD_DIM {
            return Err(WgpuFlatAttentionError::UnsupportedHeadDim {
                actual: shape.head_dim,
                maximum: WGSL_MAX_HEAD_DIM,
            });
        }
        let q_tensor_len = shape.q_tensor_len()?;
        let kv_tensor_len = shape.kv_tensor_len()?;
        let lse_len = shape.lse_len()?;
        let combined_len = q_tensor_len
            .checked_add(lse_len)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        if combined_len > u32::MAX as usize || kv_tensor_len > u32::MAX as usize {
            return Err(WgpuFlatAttentionError::IndexSpaceExceeded {
                elements: combined_len.max(kv_tensor_len),
            });
        }

        let q_batch_heads = shape
            .batch
            .checked_mul(shape.q_heads)
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
        if q_batch_heads > maximum as usize {
            return Err(WgpuFlatAttentionError::DispatchLimit {
                axis: "y/batch_q_heads",
                actual: q_batch_heads,
                maximum,
            });
        }

        Ok(GroupedDispatchGeometry {
            q_tensor_len,
            kv_tensor_len,
            lse_len,
            q_batch_heads: checked_u32(q_batch_heads)?,
            seq_len: checked_u32(shape.seq_len)?,
            query_workgroups: checked_u32(query_workgroups)?,
        })
    }

    fn validate_resident(
        &self,
        tensor: &'static str,
        buffer: &WgpuGroupedResidentBuffer,
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

    fn ensure_owner(
        &self,
        buffer: &WgpuGroupedResidentBuffer,
    ) -> Result<(), WgpuFlatAttentionError> {
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
            label: Some("flat-attention-grouped-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flat-attention-grouped-readback"),
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

struct GroupedDispatchGeometry {
    q_tensor_len: usize,
    kv_tensor_len: usize,
    lse_len: usize,
    q_batch_heads: u32,
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
