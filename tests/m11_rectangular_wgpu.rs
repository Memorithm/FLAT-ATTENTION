use flat_attention::{
    forward_reference_grouped_asymmetric, AsymmetricGroupedAttentionShape, FlatAttentionConfig,
};

const SHADER: &str = include_str!("../shaders/flat_fwd_projection_rope_rect.wgsl");

#[test]
fn rectangular_shader_parses_and_validates() {
    let module = naga::front::wgsl::parse_str(SHADER).expect("M11 rectangular WGSL must parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("M11 rectangular WGSL must validate");
}

#[cfg(feature = "wgpu")]
mod device {
    use super::*;
    use std::sync::mpsc;
    use wgpu::util::DeviceExt;

    fn bytes_of_f32(values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn bytes_of_u32(values: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for &value in values {
            bytes.extend_from_slice(&value.to_ne_bytes());
        }
        bytes
    }

    fn deterministic_values(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 * 0.137) + phase).sin() * 0.6)
            .collect()
    }

    fn projection_to_head_major(
        data: &[f32],
        batch: usize,
        len: usize,
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let width = heads * head_dim;
        let mut out = vec![0.0; data.len()];
        for b in 0..batch {
            for h in 0..heads {
                for p in 0..len {
                    for d in 0..head_dim {
                        let src = (b * len + p) * width + h * head_dim + d;
                        let dst = ((b * heads + h) * len + p) * head_dim + d;
                        out[dst] = data[src];
                    }
                }
            }
        }
        out
    }

    fn head_major_to_projection(
        data: &[f32],
        batch: usize,
        len: usize,
        heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let width = heads * head_dim;
        let mut out = vec![0.0; data.len()];
        for b in 0..batch {
            for h in 0..heads {
                for p in 0..len {
                    for d in 0..head_dim {
                        let src = ((b * heads + h) * len + p) * head_dim + d;
                        let dst = (b * len + p) * width + h * head_dim + d;
                        out[dst] = data[src];
                    }
                }
            }
        }
        out
    }

    fn rotate_head_major(
        data: &mut [f32],
        batch: usize,
        heads: usize,
        len: usize,
        head_dim: usize,
        theta: f32,
        position_offset: usize,
    ) {
        for b in 0..batch {
            for h in 0..heads {
                for p in 0..len {
                    let base = ((b * heads + h) * len + p) * head_dim;
                    for pair in 0..head_dim / 2 {
                        let i = base + pair * 2;
                        let freq = theta.powf(-2.0 * pair as f32 / head_dim as f32);
                        let angle = (position_offset + p) as f32 * freq;
                        let (s, c) = angle.sin_cos();
                        let even = data[i];
                        let odd = data[i + 1];
                        data[i] = even * c - odd * s;
                        data[i + 1] = even * s + odd * c;
                    }
                }
            }
        }
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max)
    }

    fn run_case(
        query_len: usize,
        kv_len: usize,
        query_position_offset: usize,
        causal: bool,
    ) {
        let (batch, q_heads, kv_heads, head_dim) = (1usize, 4usize, 2usize, 8usize);
        let theta = 10_000.0f32;
        let shape = AsymmetricGroupedAttentionShape {
            batch,
            q_heads,
            kv_heads,
            query_len,
            kv_len,
            head_dim,
            query_position_offset,
        };
        let q_projection = deterministic_values(batch * query_len * q_heads * head_dim, 0.2);
        let k_projection = deterministic_values(batch * kv_len * kv_heads * head_dim, 0.8);
        let v_projection = deterministic_values(batch * kv_len * kv_heads * head_dim, 1.4);

        let mut q_head =
            projection_to_head_major(&q_projection, batch, query_len, q_heads, head_dim);
        let mut k_head = projection_to_head_major(&k_projection, batch, kv_len, kv_heads, head_dim);
        let v_head = projection_to_head_major(&v_projection, batch, kv_len, kv_heads, head_dim);
        rotate_head_major(
            &mut q_head,
            batch,
            q_heads,
            query_len,
            head_dim,
            theta,
            query_position_offset,
        );
        rotate_head_major(
            &mut k_head,
            batch,
            kv_heads,
            kv_len,
            head_dim,
            theta,
            0,
        );
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let expected =
            forward_reference_grouped_asymmetric(&q_head, &k_head, &v_head, shape, config)
                .unwrap();
        let expected_projection =
            head_major_to_projection(&expected.output, batch, query_len, q_heads, head_dim);

        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })) else {
            eprintln!("wgpu: no adapter, skipping M11 rectangular device parity");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("flat-m11-rectangular-test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))
        .expect("request M11 WGPU device");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("flat-m11-rectangular"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flat-m11-rectangular"),
            layout: None,
            module: &shader,
            entry_point: "flat_attention_forward",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let q_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("q"),
            contents: &bytes_of_f32(&q_projection),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let k_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("k"),
            contents: &bytes_of_f32(&k_projection),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let v_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("v"),
            contents: &bytes_of_f32(&v_projection),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_elements = q_projection.len();
        let lse_elements = batch * q_heads * query_len;
        let output_bytes = (output_elements + lse_elements) * 4;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("out-lse"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: output_bytes as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let scale = 1.0 / (head_dim as f32).sqrt();
        let params = [
            query_len as u32,
            kv_len as u32,
            head_dim as u32,
            q_heads as u32,
            kv_heads as u32,
            batch as u32,
            u32::from(causal),
            scale.to_bits(),
            theta.to_bits(),
            query_position_offset as u32,
            0,
            0,
        ];
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: &bytes_of_u32(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: k_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: v_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: out_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params_buf.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat-m11-rectangular"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("flat-m11-rectangular"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(query_len as u32, (batch * q_heads) as u32, 1);
        }
        encoder.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, output_bytes as u64);
        queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map M11 readback");
        let mapped = slice.get_mapped_range();
        let actual: Vec<f32> = mapped[..output_elements * 4]
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        drop(mapped);
        staging.unmap();

        let error = max_abs_diff(&actual, &expected_projection);
        assert!(error < 2e-4, "M11 rectangular max_abs_diff={error}");
    }

    #[test]
    fn rectangular_wgpu_matches_oracle_for_cross_attention() {
        run_case(3, 5, 0, false);
    }

    #[test]
    fn single_query_decode_matches_oracle_over_longer_kv() {
        run_case(1, 7, 6, true);
    }
}
