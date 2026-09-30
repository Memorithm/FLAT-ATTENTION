#![cfg(feature = "wgpu")]

use flat_attention::paged_kv::{PagedKvConfig, PagedKvTable};
use flat_attention::{
    FlatAttentionConfig, PackedU16PagedDecodePass, WgpuPackedPagedKvTable16,
    WgpuPackedU16PagedDecodeShadowPipeline, FLAT_DECODE_PAGED_U16_SHADOW_WGSL,
};

fn device() -> Option<wgpu::Device> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    })) {
        Ok(adapter) => adapter,
        Err(error) if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() => {
            eprintln!("WGPU adapter unavailable; packed-u16 shadow device test skipped: {error}");
            return None;
        }
        Err(error) => panic!("required WGPU adapter unavailable: {error}"),
    };

    let required_limits = wgpu::Limits {
        max_bind_groups: 5,
        ..wgpu::Limits::downlevel_defaults()
    };
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("flat-paged-u16-shadow-qualification"),
        required_features: wgpu::Features::empty(),
        required_limits,
        ..Default::default()
    })) {
        Ok((device, _queue)) => Some(device),
        Err(error) if std::env::var_os("FLAT_REQUIRE_WGPU").is_none() => {
            eprintln!("WGPU device unavailable; packed-u16 shadow device test skipped: {error}");
            None
        }
        Err(error) => panic!("required WGPU device unavailable: {error}"),
    }
}

#[test]
fn packed_u16_shadow_pipeline_compiles_on_selected_backend() {
    let Some(device) = device() else {
        return;
    };

    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flat-m16-paged-u16-shadow"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
            FLAT_DECODE_PAGED_U16_SHADOW_WGSL,
        )),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("flat-m16-paged-u16-shadow"),
        layout: None,
        module: &shader,
        entry_point: Some("flat_attention_decode_paged_u16_shadow"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    if let Some(error) = pollster::block_on(error_scope.pop()) {
        panic!("packed-u16 shadow pipeline validation failed: {error}");
    }

    drop(pipeline);
}

#[test]
fn packed_u16_shadow_pipeline_encodes_minimal_valid_dispatch() {
    let Some(device) = device() else {
        return;
    };

    let mut table = PagedKvTable::new(PagedKvConfig {
        page_size: 1,
        physical_pages: 1,
    })
    .unwrap();
    table.append(1).unwrap();
    let page_table = WgpuPackedPagedKvTable16::from_table(&table).unwrap();

    let pipeline = WgpuPackedU16PagedDecodeShadowPipeline::new(&device).unwrap();
    let q = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("u16-shadow-q"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let k = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("u16-shadow-k"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let v = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("u16-shadow-v"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let out_and_lse = pipeline.create_output_buffer(&device, 1, 2).unwrap();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("u16-shadow-encoder"),
    });

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let layout = pipeline
        .encode(
            &device,
            &mut encoder,
            PackedU16PagedDecodePass {
                q: &q,
                k: &k,
                v: &v,
                page_table: &page_table,
                out_and_lse: &out_and_lse,
                q_heads: 1,
                kv_heads: 1,
                head_dim: 2,
                config: FlatAttentionConfig {
                    causal: true,
                    softmax_scale: None,
                },
                theta: 10_000.0,
                q_rope_position: 0,
                q_causal_position: 0,
            },
        )
        .unwrap();

    assert_eq!(layout.q_elements, 2);
    assert_eq!(layout.output_elements, 2);
    assert_eq!(layout.lse_elements, 1);
    assert_eq!(layout.combined_elements, 3);
    let _command_buffer = encoder.finish();

    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("packed-u16 shadow encode validation failed: {error}");
    }
}
