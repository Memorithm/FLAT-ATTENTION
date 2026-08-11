// M18 correctness-first backward recomputation kernel.
//
// Portable storage contract:
//   packed_forward = Q | K | V | dO | O | LSE
//   packed_grads   = dQ | dK | dV
//
// Each invocation owns exactly one scalar gradient, avoiding floating-point
// atomics and cross-workgroup write races. Scores/probabilities are recomputed
// from Q/K/V and saved O/LSE; no N x N matrix is materialized.

const WORKGROUP_SIZE: u32 = 64u;

struct Params {
    batch: u32,
    heads: u32,
    seq_len: u32,
    head_dim: u32,
    causal: u32,
    scale_bits: u32,
    tensor_elems: u32,
    lse_elems: u32,
};

@group(0) @binding(0) var<storage, read> packed_forward: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_grads: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn q_offset() -> u32 { return 0u; }
fn k_offset() -> u32 { return params.tensor_elems; }
fn v_offset() -> u32 { return 2u * params.tensor_elems; }
fn do_offset() -> u32 { return 3u * params.tensor_elems; }
fn o_offset() -> u32 { return 4u * params.tensor_elems; }
fn lse_offset() -> u32 { return 5u * params.tensor_elems; }

fn head_base(batch_head: u32) -> u32 {
    return batch_head * params.seq_len * params.head_dim;
}

fn row_base(batch_head: u32, position: u32) -> u32 {
    return head_base(batch_head) + position * params.head_dim;
}

fn score(batch_head: u32, query_pos: u32, key_pos: u32, scale: f32) -> f32 {
    let qb = row_base(batch_head, query_pos);
    let kb = row_base(batch_head, key_pos);
    var dot = 0.0;
    var d = 0u;
    loop {
        if (d >= params.head_dim) { break; }
        dot += packed_forward[q_offset() + qb + d] * packed_forward[k_offset() + kb + d];
        d += 1u;
    }
    return dot * scale;
}

fn probability(batch_head: u32, query_pos: u32, key_pos: u32, scale: f32) -> f32 {
    let lse_index = batch_head * params.seq_len + query_pos;
    return exp(score(batch_head, query_pos, key_pos, scale) - packed_forward[lse_offset() + lse_index]);
}

fn delta(batch_head: u32, query_pos: u32) -> f32 {
    let qb = row_base(batch_head, query_pos);
    var value = 0.0;
    var d = 0u;
    loop {
        if (d >= params.head_dim) { break; }
        value += packed_forward[do_offset() + qb + d] * packed_forward[o_offset() + qb + d];
        d += 1u;
    }
    return value;
}

fn d_probability(batch_head: u32, query_pos: u32, key_pos: u32) -> f32 {
    let qb = row_base(batch_head, query_pos);
    let vb = row_base(batch_head, key_pos);
    var value = 0.0;
    var d = 0u;
    loop {
        if (d >= params.head_dim) { break; }
        value += packed_forward[do_offset() + qb + d] * packed_forward[v_offset() + vb + d];
        d += 1u;
    }
    return value;
}

@compute @workgroup_size(64, 1, 1)
fn flat_attention_backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gradient_elems = 3u * params.tensor_elems;
    let global_index = global_id.x;
    if (global_index >= gradient_elems || params.tensor_elems == 0u || params.lse_elems == 0u) {
        return;
    }

    let kind = global_index / params.tensor_elems;
    let local_index = global_index - kind * params.tensor_elems;
    let head_stride = params.seq_len * params.head_dim;
    let batch_head = local_index / head_stride;
    let within_head = local_index - batch_head * head_stride;
    let position = within_head / params.head_dim;
    let dim = within_head - position * params.head_dim;
    let scale = bitcast<f32>(params.scale_bits);

    var gradient = 0.0;

    if (kind == 0u) {
        // dQ[q,d] = sum_k dS[q,k] * scale * K[k,d]
        let query_pos = position;
        let key_limit = select(params.seq_len, query_pos + 1u, params.causal != 0u);
        let delta_q = delta(batch_head, query_pos);
        var key_pos = 0u;
        loop {
            if (key_pos >= key_limit) { break; }
            let p = probability(batch_head, query_pos, key_pos, scale);
            let ds = p * (d_probability(batch_head, query_pos, key_pos) - delta_q);
            let kb = row_base(batch_head, key_pos);
            gradient += ds * scale * packed_forward[k_offset() + kb + dim];
            key_pos += 1u;
        }
    } else if (kind == 1u) {
        // dK[k,d] = sum_q dS[q,k] * scale * Q[q,d]
        let key_pos = position;
        var query_pos = 0u;
        loop {
            if (query_pos >= params.seq_len) { break; }
            if (params.causal == 0u || key_pos <= query_pos) {
                let p = probability(batch_head, query_pos, key_pos, scale);
                let ds = p * (d_probability(batch_head, query_pos, key_pos) - delta(batch_head, query_pos));
                let qb = row_base(batch_head, query_pos);
                gradient += ds * scale * packed_forward[q_offset() + qb + dim];
            }
            query_pos += 1u;
        }
    } else {
        // dV[k,d] = sum_q P[q,k] * dO[q,d]
        let key_pos = position;
        var query_pos = 0u;
        loop {
            if (query_pos >= params.seq_len) { break; }
            if (params.causal == 0u || key_pos <= query_pos) {
                let p = probability(batch_head, query_pos, key_pos, scale);
                let qb = row_base(batch_head, query_pos);
                gradient += p * packed_forward[do_offset() + qb + dim];
            }
            query_pos += 1u;
        }
    }

    packed_grads[global_index] = gradient;
}
