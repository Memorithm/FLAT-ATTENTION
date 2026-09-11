//! Host-only smoke example: run the scalar online-softmax oracle.
//!
//! This path needs no GPU and no `wgpu` feature. It is a contract demo, not a
//! performance claim.

use flat_attention::{forward_reference, AttentionShape, FlatAttentionConfig};

fn main() {
    let shape = AttentionShape {
        batch: 1,
        heads: 2,
        seq_len: 4,
        head_dim: 8,
    };
    let tensor_len = shape.tensor_len().expect("valid shape");
    let q: Vec<f32> = (0..tensor_len).map(|i| 0.01 * i as f32).collect();
    let k: Vec<f32> = (0..tensor_len).map(|i| 0.02 * i as f32).collect();
    let v: Vec<f32> = (0..tensor_len).map(|i| 0.03 * i as f32).collect();

    let config = FlatAttentionConfig {
        causal: true,
        softmax_scale: None,
    };
    let result = forward_reference(&q, &k, &v, shape, config).expect("oracle forward");

    println!(
        "hello_attention: B={} H={} N={} D={} causal={} O_len={} LSE_len={}",
        shape.batch,
        shape.heads,
        shape.seq_len,
        shape.head_dim,
        config.causal,
        result.output.len(),
        result.lse.len(),
    );
    println!(
        "O[0..8] = {:?}\nLSE     = {:?}",
        &result.output[..shape.head_dim.min(result.output.len())],
        result.lse
    );
    println!("finite_O={} finite_LSE={}", result.output.iter().all(|x| x.is_finite()), result.lse.iter().all(|x| x.is_finite()));
}
