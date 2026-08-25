//! WGPU executor for FLAT-R1 fused RoPE + GQA/MQA.
//!
//! Q and K are uploaded/stored unrotated. The compute shader performs head-local
//! RoPE in workgroup memory and immediately consumes the rotated values in the
//! fused online-softmax attention loop. No global rotated-Q/K buffer exists.

use super::wgpu_internal;

use std::fmt;
use std::sync::{mpsc, Arc};

use super::{
    validate_input, FlatAttentionConfig, FlatAttentionError, FlatAttentionOutput,
    GroupedAttentionShape, RotaryEmbeddingConfig, WgpuFlatAttentionError,
    FLAT_FWD_GROUPED_ROPE_WGSL, WGSL_MAX_HEAD_DIM, WGSL_QUERY_ROWS,
};

pub struct WgpuRotaryGroupedResidentBuffer {
    buffer: Arc<wgpu::Buffer>,
    len: usize,
    owner: usize,
}

impl WgpuRotaryGroupedResidentBuffer {
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

pub struct WgpuRotaryGroupedResidentOutput {
    combined: WgpuRotaryGroupedResidentBuffer,
    output_len: usize,
    lse_len: usize,
}

impl WgpuRotaryGroupedResidentOutput {
    #[must_use]
    pub fn output_len(&self) -> usize {
        self.output_len
    }

    #[must_use]
    pub fn lse_len(&self) -> usize {
        self.lse_len
    }

    #[must_use]
    pub fn combined(&self) -> &WgpuRotaryGroupedResidentBuffer {
        &self.combined
    }
}

struct WgpuRotaryGroupedAttentionInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    adapter_name: String,
    max_workgroups_per_dimension: u32,
    max_storage_buffer_binding_size: u64,
    owner_id: usize,
}

#[derive(Clone)]
pub struct WgpuRotaryGroupedAttention {
    inner: Arc<WgpuRotaryGroupedAttentionInner>,
}

impl fmt::Debug for WgpuRotaryGroupedAttention {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuRotaryGroupedAttention")
            .field("adapter_name", &self.inner.adapter_name)
            .finish_non_exhaustive()
    }
}

impl WgpuRotaryGroupedAttention {
    pub fn new() -> Result<Self, WgpuFlatAttentionError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|_| WgpuFlatAttentionError::Unavailable)?;
        let adapter_name = adapter.get_info().name;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("flat-r1-fused-rope-gqa"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        }))
        .map_err(|error| WgpuFlatAttentionError::Execution(format!("request_device: {error}")))?;

        let pipeline = create_pipeline(&device)?;
        let device_limits = device.limits();
        let max_workgroups_per_dimension = device_limits.max_compute_workgroups_per_dimension;
        Ok(Self {
            inner: Arc::new(WgpuRotaryGroupedAttentionInner {
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

    pub fn forward(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        shape: GroupedAttentionShape,
        config: FlatAttentionConfig,
        rotary: RotaryEmbeddingConfig,
    ) -> Result<FlatAttentionOutput, WgpuFlatAttentionError> {
        shape.validate()?;
        rotary.validate(shape.head_dim, shape.seq_len)?;
        let q_len = shape.q_tensor_len()?;
        let kv_len = shape.kv_tensor_len()?;
        validate_input("Q", q, q_len)?;
        validate_input("K", k, kv_len)?;
        validate_input("V", v, kv_len)?;

        let q_gpu = self.upload(q)?;
        let k_gpu = self.upload(k)?;
        let v_gpu = self.upload(v)?;
        let resident = self.forward_resident(&q_gpu, &k_gpu, &v_gpu, shape, config, rotary)?;
        self.download_attention(&resident)
    }

    pub fn upload(
        &self,
        data: &[f32],
    ) -> Result<WgpuRotaryGroupedResidentBuffer, WgpuFlatAttentionError> {
        let bytes = encode_f32(data)?;
        let size = bytes.len().max(4) as u64;
        if size > self.inner.max_storage_buffer_binding_size {
            return Err(WgpuFlatAttentionError::DeviceBufferLimit {
                required_bytes: size,
                maximum_bytes: self.inner.max_storage_buffer_binding_size,
            });
        }
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-r1-resident-input"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            self.inner.queue.write_buffer(&buffer, 0, &bytes);
        }
        Ok(WgpuRotaryGroupedResidentBuffer {
            buffer: Arc::new(buffer),
            len: data.len(),
            owner: self.owner_id(),
        })
    }

    pub fn forward_resident(
        &self,
        q: &WgpuRotaryGroupedResidentBuffer,
        k: &WgpuRotaryGroupedResidentBuffer,
        v: &WgpuRotaryGroupedResidentBuffer,
        shape: GroupedAttentionShape,
        config: FlatAttentionConfig,
        rotary: RotaryEmbeddingConfig,
    ) -> Result<WgpuRotaryGroupedResidentOutput, WgpuFlatAttentionError> {
        let dispatch = self.validate_dispatch(shape, rotary)?;
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
            label: Some("flat-r1-o-lse"),
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
            rotary.theta.to_bits(),
            dispatch.position_offset,
            0,
            0,
            0,
        ];
        let params_bytes = encode_u32(&params);
        let params_buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-r1-params"),
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
                label: Some("flat-r1-bind-group"),
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
                    label: Some("flat-r1-forward"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-r1-forward"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.inner.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dispatch.query_workgroups, dispatch.q_batch_heads, 1);
        }
        self.inner.queue.submit(Some(encoder.finish()));

        Ok(WgpuRotaryGroupedResidentOutput {
            combined: WgpuRotaryGroupedResidentBuffer {
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
        resident: &WgpuRotaryGroupedResidentOutput,
    ) -> Result<FlatAttentionOutput, WgpuFlatAttentionError> {
        self.ensure_owner(&resident.combined)?;
        let expected = resident
            .output_len
            .checked_add(resident.lse_len)
            .ok_or(FlatAttentionError::ShapeOverflow)?;
        if resident.combined.len != expected {
            return Err(WgpuFlatAttentionError::Execution(
                "R1 resident output metadata is inconsistent".into(),
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
        rotary: RotaryEmbeddingConfig,
    ) -> Result<RotaryGroupedDispatchGeometry, WgpuFlatAttentionError> {
        shape.validate()?;
        rotary.validate(shape.head_dim, shape.seq_len)?;
        if shape.head_dim > WGSL_MAX_HEAD_DIM {
            return Err(WgpuFlatAttentionError::UnsupportedHeadDim {
                actual: shape.head_dim,
                maximum: WGSL_MAX_HEAD_DIM,
            });
        }
        let final_position = rotary
            .position_offset
            .checked_add(shape.seq_len.saturating_sub(1))
            .ok_or(FlatAttentionError::PositionOverflow)?;
        if final_position > u32::MAX as usize {
            return Err(FlatAttentionError::PositionOverflow.into());
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

        // Q, K and V plus the packed O|LSE output are storage-bound; enforce
        // the device binding limit with typed errors instead of wgpu
        // validation aborts.
        let maximum_binding_bytes = self.inner.max_storage_buffer_binding_size;
        let q_bytes = bytes_for_f32_len(q_tensor_len)?;
        let kv_bytes = bytes_for_f32_len(kv_tensor_len)?;
        let output_bytes = bytes_for_f32_len(combined_len)?;
        let required_bytes = q_bytes.max(kv_bytes).max(output_bytes);
        if required_bytes > maximum_binding_bytes {
            return Err(WgpuFlatAttentionError::DeviceBufferLimit {
                required_bytes,
                maximum_bytes: maximum_binding_bytes,
            });
        }

        Ok(RotaryGroupedDispatchGeometry {
            q_tensor_len,
            kv_tensor_len,
            lse_len,
            q_batch_heads: checked_u32(q_batch_heads)?,
            seq_len: checked_u32(shape.seq_len)?,
            query_workgroups: checked_u32(query_workgroups)?,
            position_offset: checked_u32(rotary.position_offset)?,
        })
    }

    fn validate_resident(
        &self,
        tensor: &'static str,
        buffer: &WgpuRotaryGroupedResidentBuffer,
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
        buffer: &WgpuRotaryGroupedResidentBuffer,
    ) -> Result<(), WgpuFlatAttentionError> {
        if buffer.owner != self.owner_id() {
            return Err(WgpuFlatAttentionError::ForeignBuffer);
        }
        Ok(())
    }

    fn owner_id(&self) -> usize {
        self.inner.owner_id
    }

    fn download_buffer(
        &self,
        source: &wgpu::Buffer,
        len: usize,
    ) -> Result<Vec<f32>, WgpuFlatAttentionError> {
        let bytes = bytes_for_f32_len(len)?;
        let staging = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flat-r1-readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder =
            self.inner
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("flat-r1-readback"),
                });
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
        self.inner.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..bytes);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        let _ = self.inner.device.poll(wgpu::PollType::wait_indefinitely());
        receiver
            .recv()
            .map_err(|error| WgpuFlatAttentionError::Execution(format!("map callback: {error}")))?
            .map_err(|error| WgpuFlatAttentionError::Execution(format!("map read: {error:?}")))?;

        let mapped = slice
            .get_mapped_range()
            .map_err(|error| WgpuFlatAttentionError::Execution(format!("map range: {error}")))?;
        let decoded = decode_f32(&mapped, len)?;
        drop(mapped);
        staging.unmap();
        Ok(decoded)
    }
}

fn create_pipeline(device: &wgpu::Device) -> Result<wgpu::ComputePipeline, WgpuFlatAttentionError> {
    wgpu_internal::create_pipeline(
        device,
        FLAT_FWD_GROUPED_ROPE_WGSL,
        "flat-r1-fused-rope-gqa",
        "flat_attention_forward",
    )
    .map_err(|error| WgpuFlatAttentionError::Execution(format!("R1 pipeline validation: {error}")))
}

struct RotaryGroupedDispatchGeometry {
    q_tensor_len: usize,
    kv_tensor_len: usize,
    lse_len: usize,
    q_batch_heads: u32,
    seq_len: u32,
    query_workgroups: u32,
    position_offset: u32,
}

fn checked_u32(value: usize) -> Result<u32, WgpuFlatAttentionError> {
    wgpu_internal::checked_u32(value)
        .ok_or(WgpuFlatAttentionError::IndexSpaceExceeded { elements: value })
}

fn bytes_for_f32_len(len: usize) -> Result<u64, WgpuFlatAttentionError> {
    wgpu_internal::f32_bytes(len)
        .ok_or_else(|| WgpuFlatAttentionError::from(FlatAttentionError::ShapeOverflow))
}

fn encode_f32(values: &[f32]) -> Result<Vec<u8>, WgpuFlatAttentionError> {
    wgpu_internal::encode_f32(values)
        .ok_or_else(|| WgpuFlatAttentionError::from(FlatAttentionError::ShapeOverflow))
}

fn encode_u32(values: &[u32]) -> Vec<u8> {
    wgpu_internal::encode_u32(values)
}

fn decode_f32(bytes: &[u8], expected: usize) -> Result<Vec<f32>, WgpuFlatAttentionError> {
    match wgpu_internal::decode_f32(bytes, expected) {
        Ok(values) => Ok(values),
        Err(wgpu_internal::DecodeF32Failure::Overflow) => Err(WgpuFlatAttentionError::from(
            FlatAttentionError::ShapeOverflow,
        )),
        Err(wgpu_internal::DecodeF32Failure::LengthMismatch {
            actual_bytes,
            expected_bytes,
        }) => Err(WgpuFlatAttentionError::Execution(format!(
            "readback returned {actual_bytes} bytes, expected {expected_bytes}"
        ))),
    }
}
