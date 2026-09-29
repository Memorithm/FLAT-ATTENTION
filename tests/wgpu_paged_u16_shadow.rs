#![cfg(feature = "wgpu")]

use flat_attention::FLAT_DECODE_PAGED_U16_SHADOW_WGSL;

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
