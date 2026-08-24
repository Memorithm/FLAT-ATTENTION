//! Fuzz the canonical shape contract and the scalar forward oracle.
//!
//! Invariants: validation, length arithmetic and the oracle must never panic;
//! successful oracle runs return exactly the documented element counts.

#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
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
    let batch = u32_at(0);
    let heads = u32_at(1);

    let shape = flat_attention::AttentionShape {
        batch,
        heads,
        seq_len: u32_at(2),
        head_dim: u32_at(3),
    };

    // Length arithmetic must be total; zero dimensions are rejected by the
    // oracle's own public validation path below.
    let Ok(tensor_len) = shape.tensor_len() else {
        return;
    };
    if batch == 0 || heads == 0 || shape.seq_len == 0 || shape.head_dim == 0 {
        return;
    }
    let Ok(lse_len) = shape.lse_len() else {
        return;
    };
    assert_eq!(lse_len, batch * heads * shape.seq_len);

    // The oracle only runs on deliberately small shapes so the fuzzer spends
    // its budget on logic, not on gigabyte allocations.
    if tensor_len == 0 || tensor_len > 64usize * 1024 {
        return;
    }
    let values = (0..tensor_len).map(|index| index as f32).collect::<Vec<_>>();
    let config = flat_attention::FlatAttentionConfig {
        causal: data[0] & 1 == 1,
        softmax_scale: None,
    };
    let output = flat_attention::forward_reference(&values, &values, &values, shape, config)
        .expect("validated finite inputs must execute");
    assert_eq!(output.output.len(), tensor_len);
    assert_eq!(output.lse.len(), lse_len);
});
