use flat_attention::{
    forward_reference_grouped, FlatAttentionConfig, GroupedAttentionShape,
};

const ATOL: f32 = 2.0e-4;
const RTOL: f32 = 1.0e-3;

#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 32) as u32
    }

    fn pick(&mut self, upper: usize) -> usize {
        self.next_u32() as usize % upper
    }

    fn finite(&mut self) -> f32 {
        let centered = self.next_u32() as f64 / u32::MAX as f64 * 2.0 - 1.0;
        (centered * 3.0) as f32
    }
}

fn random_vec(rng: &mut Lcg, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.finite()).collect()
}

fn assert_close(name: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{name}: length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = ATOL + RTOL * expected.abs();
        let error = (actual - expected).abs();
        assert!(
            error <= tolerance,
            "{name}[{index}]: actual={actual}, expected={expected}, abs_error={error}, tolerance={tolerance}"
        );
    }
}

#[test]
fn randomized_grouped_oracle_preserves_attention_invariants() {
    let mut rng = Lcg(0x4d38_5eed_cafe_babe);
    let q_head_choices = [1usize, 2, 4, 8];
    let head_dim_choices = [1usize, 2, 4, 8, 16, 32, 64, 128];

    for case in 0..64 {
        let batch = 1 + rng.pick(2);
        let q_heads = q_head_choices[rng.pick(q_head_choices.len())];
        let divisors: Vec<_> = (1..=q_heads)
            .filter(|candidate| q_heads % candidate == 0)
            .collect();
        let kv_heads = divisors[rng.pick(divisors.len())];
        let seq_len = 1 + rng.pick(33);
        let head_dim = head_dim_choices[rng.pick(head_dim_choices.len())];
        let shape = GroupedAttentionShape {
            batch,
            q_heads,
            kv_heads,
            seq_len,
            head_dim,
        };
        let q = random_vec(&mut rng, shape.q_tensor_len().unwrap());
        let k = random_vec(&mut rng, shape.kv_tensor_len().unwrap());
        let v = random_vec(&mut rng, shape.kv_tensor_len().unwrap());
        let causal = case % 2 == 0;
        let config = FlatAttentionConfig {
            causal,
            softmax_scale: None,
        };
        let result = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();

        assert!(result.output.iter().all(|value| value.is_finite()));
        assert!(result.lse.iter().all(|value| value.is_finite()));

        let ones = vec![1.0f32; shape.kv_tensor_len().unwrap()];
        let normalized = forward_reference_grouped(&q, &k, &ones, shape, config).unwrap();
        for (index, value) in normalized.output.iter().enumerate() {
            assert!(
                (*value - 1.0).abs() <= 3.0e-5,
                "case {case}: normalized output[{index}]={value}"
            );
        }

        if causal {
            let group_size = q_heads / kv_heads;
            let q_head_stride = seq_len * head_dim;
            let kv_head_stride = seq_len * head_dim;
            for batch_index in 0..batch {
                for q_head in 0..q_heads {
                    let kv_head = q_head / group_size;
                    let out_base =
                        (batch_index * q_heads + q_head) * q_head_stride;
                    let v_base =
                        (batch_index * kv_heads + kv_head) * kv_head_stride;
                    assert_close(
                        "causal first-query singleton visibility",
                        &result.output[out_base..out_base + head_dim],
                        &v[v_base..v_base + head_dim],
                    );
                }
            }
        }
    }
}

#[test]
fn extreme_but_finite_scores_remain_finite_and_normalized() {
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 8,
        kv_heads: 1,
        seq_len: 33,
        head_dim: 128,
    };
    let q = (0..shape.q_tensor_len().unwrap())
        .map(|index| if index % 2 == 0 { 1_000.0 } else { -1_000.0 })
        .collect::<Vec<_>>();
    let k = (0..shape.kv_tensor_len().unwrap())
        .map(|index| if index % 3 == 0 { -1_000.0 } else { 1_000.0 })
        .collect::<Vec<_>>();
    let v = vec![1.0f32; shape.kv_tensor_len().unwrap()];

    for causal in [false, true] {
        let result = forward_reference_grouped(
            &q,
            &k,
            &v,
            shape,
            FlatAttentionConfig {
                causal,
                softmax_scale: None,
            },
        )
        .unwrap();
        assert!(result.output.iter().all(|value| value.is_finite()));
        assert!(result.lse.iter().all(|value| value.is_finite()));
        for value in result.output {
            assert!((value - 1.0).abs() <= 3.0e-5);
        }
    }
}

#[cfg(feature = "wgpu")]
fn require_or_skip_grouped() -> Option<flat_attention::WgpuGroupedAttention> {
    match flat_attention::WgpuGroupedAttention::new() {
        Ok(executor) => Some(executor),
        Err(error) => {
            if std::env::var_os("FLAT_REQUIRE_WGPU").is_some() {
                panic!("M38 requires WGPU but adapter creation failed: {error}");
            }
            eprintln!("M38: no WGPU adapter, skipping device stress: {error}");
            None
        }
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn repeated_resident_dispatch_reuses_inputs_without_numerical_drift() {
    let Some(executor) = require_or_skip_grouped() else {
        return;
    };
    let shape = GroupedAttentionShape {
        batch: 1,
        q_heads: 8,
        kv_heads: 2,
        seq_len: 17,
        head_dim: 32,
    };
    let mut rng = Lcg(0x4d38_d15c_a7c4_0001);
    let q = random_vec(&mut rng, shape.q_tensor_len().unwrap());
    let k = random_vec(&mut rng, shape.kv_tensor_len().unwrap());
    let v = random_vec(&mut rng, shape.kv_tensor_len().unwrap());
    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let oracle = forward_reference_grouped(&q, &k, &v, shape, config).unwrap();
    let q_gpu = executor.upload(&q).unwrap();
    let k_gpu = executor.upload(&k).unwrap();
    let v_gpu = executor.upload(&v).unwrap();

    for iteration in 0..32 {
        let resident = executor
            .forward_resident(&q_gpu, &k_gpu, &v_gpu, shape, config)
            .unwrap();
        let actual = executor.download_attention(&resident).unwrap();
        assert_close(
            &format!("M38 repeated O iteration {iteration}"),
            &actual.output,
            &oracle.output,
        );
        assert_close(
            &format!("M38 repeated LSE iteration {iteration}"),
            &actual.lse,
            &oracle.lse,
        );
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn resident_kv_cache_reset_and_reuse_preserve_logical_length_contract() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    })) else {
        if std::env::var_os("FLAT_REQUIRE_WGPU").is_some() {
            panic!("M38 requires WGPU but no adapter was available");
        }
        return;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("flat-m38-kv-reset"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
    ))
    .expect("M38 request device");

    let mut cache = flat_attention::WgpuResidentKvCache::new(&device, 1, 2, 8, 16).unwrap();
    let make_source = |label: &'static str, rows: usize| {
        let bytes = (rows * 2 * 16 * std::mem::size_of::<f32>()) as u64;
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let k3 = make_source("flat-m38-k3", 3);
    let v3 = make_source("flat-m38-v3", 3);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m38-append-3"),
    });
    assert_eq!(cache.record_append(&mut encoder, &k3, &v3, 3).unwrap(), 3);
    queue.submit(Some(encoder.finish()));
    assert_eq!(cache.len(), 3);
    assert_eq!(cache.remaining_capacity(), 5);

    cache.reset();
    assert!(cache.is_empty());
    assert_eq!(cache.remaining_capacity(), 8);

    let k2 = make_source("flat-m38-k2", 2);
    let v2 = make_source("flat-m38-v2", 2);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flat-m38-append-2"),
    });
    assert_eq!(cache.record_append(&mut encoder, &k2, &v2, 2).unwrap(), 2);
    queue.submit(Some(encoder.finish()));
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.remaining_capacity(), 6);
}
