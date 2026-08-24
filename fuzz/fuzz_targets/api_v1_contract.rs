//! Fuzz the versioned api::v1 contract validation surface.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 32 {
        return;
    }
    let u32_at = |index: usize| -> usize {
        u32::from_le_bytes([
            data[index * 4],
            data[index * 4 + 1],
            data[index * 4 + 2],
            data[index * 4 + 3],
        ]) as usize
    };
    let shape = flat_attention::api::v1::AttentionShape {
        batch: u32_at(0),
        q_heads: u32_at(1),
        kv_heads: u32_at(2),
        query_len: u32_at(3),
        kv_len: u32_at(4),
        head_dim: u32_at(5) % 512,
        query_position_offset: u32_at(6),
    };
    if shape.validate().is_err() {
        return;
    }
    let config = flat_attention::api::v1::AttentionConfig {
        causal: data[0] & 1 == 1,
        softmax_scale: if data[1] & 1 == 1 {
            None
        } else {
            Some(f32::from_bits(u32::from_le_bytes([
                data[2], data[3], data[4], data[5],
            ])))
        },
    };
    // Validation must be total. An explicit scale is allowed to be rejected
    // as non-finite/non-positive, but nothing may panic.
    match config.validate(shape.head_dim) {
        Ok(()) => {}
        Err(flat_attention::api::v1::ApiError::InvalidScale(_)) => {}
        Err(error) => panic!("unexpected rejection of a validated shape: {error:?}"),
    }
});
