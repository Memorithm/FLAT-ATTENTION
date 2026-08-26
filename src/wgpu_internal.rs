//! Shared WGPU host-side primitives.
//!
//! These helpers are error-type agnostic: they return plain values or
//! [`Option`], and each backend maps failures onto its own typed error at the
//! call site. This keeps one implementation per primitive while preserving
//! every module's public error surface exactly.

use wgpu::util::DeviceExt;

/// Narrow a host-side count to the WGSL u32 index space, or `None`.
pub(crate) fn checked_u32(value: usize) -> Option<u32> {
    u32::try_from(value).ok()
}

/// Byte size of `elements` f32 scalars, or `None` on overflow in either the
/// multiplication or the u64 conversion.
pub(crate) fn f32_bytes(elements: usize) -> Option<u64> {
    let bytes = elements.checked_mul(core::mem::size_of::<f32>())?;
    u64::try_from(bytes).ok()
}

/// Encode f32 scalars as native-endian device bytes, or `None` when the byte
/// capacity computation overflows.
pub(crate) fn encode_f32(values: &[f32]) -> Option<Vec<u8>> {
    let capacity = values.len().checked_mul(core::mem::size_of::<f32>())?;
    let mut bytes = Vec::with_capacity(capacity);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    Some(bytes)
}

/// Why a device readback could not be decoded as f32 scalars.
pub(crate) enum DecodeF32Failure {
    /// The expected element count cannot be sized in bytes.
    Overflow,
    /// The readback length does not match the expected element count.
    LengthMismatch {
        actual_bytes: usize,
        expected_bytes: usize,
    },
}

/// Decode native-endian device bytes into exactly `expected` f32 scalars.
pub(crate) fn decode_f32(bytes: &[u8], expected: usize) -> Result<Vec<f32>, DecodeF32Failure> {
    let Some(expected_bytes) = expected.checked_mul(core::mem::size_of::<f32>()) else {
        return Err(DecodeF32Failure::Overflow);
    };
    if bytes.len() != expected_bytes {
        return Err(DecodeF32Failure::LengthMismatch {
            actual_bytes: bytes.len(),
            expected_bytes,
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Encode u32 words as native-endian device bytes (uniform blocks).
pub(crate) fn encode_u32(values: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(core::mem::size_of_val(values));
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

/// Create and initialize an immutable uniform buffer from host bytes.
///
/// `DeviceExt::create_buffer_init` is the wgpu 30 convenience path for
/// mapped-at-creation initialization and avoids exposing fallible mapping to
/// every caller-owned pipeline implementation.
pub(crate) fn create_uniform_buffer_init(
    device: &wgpu::Device,
    label: &'static str,
    contents: &[u8],
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

/// Compile one WGSL compute pipeline under a validation error scope.
///
/// Returns the raw validation message on failure; callers wrap it in their own
/// error variant. Every FLAT kernel keeps its shader entry point explicit.
pub(crate) fn create_pipeline(
    device: &wgpu::Device,
    source: impl AsRef<str>,
    label: &'static str,
    entry_point: &'static str,
) -> Result<wgpu::ComputePipeline, String> {
    let error_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(source.as_ref().to_owned())),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: None,
        module: &shader,
        entry_point: Some(entry_point),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    match pollster::block_on(error_scope.pop()) {
        Some(error) => Err(error.to_string()),
        None => Ok(pipeline),
    }
}
