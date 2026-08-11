// M19 correctness-first native GQA/MQA backward recomputation kernel.
//
// Portable storage contract:
//   packed_forward = Q | K | V | dO | O | LSE
//   packed_grads   = dQ | dK | dV
//
// Q/dQ use query-head cardinality. K/V/dK/dV retain physical KV-head
// cardinality. Each invocation owns one scalar gradient, so no floating-point
// atomics or cross-workgroup write races are required. Scores/probabilities are
// recomputed from Q/K/V and saved O/LSE; no N x N matrix is materialized.

const WORKGROUP_SIZE: u32 = 64u;

struct Params {
    batch: u32,
    q_heads: u32,
    kv_heads: u32,
    seq_len: u32,
    head_dim: u32,
    causal: u32,
    scale_bits: u32,
    q_elems: u32,
    kv_elems: u32,
    lse_elems: u32,
};

@group(0) @binding(0) var<storage, read> packed_forward: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed_grads: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

fn q_offset() -> u32 { return 0u; }
fn k_offset() -> u32 { return params.q_elems; }
fn v_offset() -> u32 { return params.q_elems + params.kv_elems; }
fn do_offset() -> u32 { return params.q_elems + 2u * params.kv_elems; }
fn o_offset() -> u32 { return 2u * params.q_elems + 2u * params.kv_elems; }
fn lse_offset() -> u32 { return 3u * params.q_elems + 2u * params.kv_elems; }

fn dq_offset() -> u32 { return 0u; }
fn dk_offset() -> u32 { return params.q_elems; }
fn dv_offset() -> u32 { return params.q_elems + params.kv_elems; }

fn q_head_base(batch: u32, q_head: u32) -> u32 {
    return (batch * params.q_heads + q_head) * params.seq_len * params.head_dim;
}

fn kv_head_base(batch: u32, kv_head: u32) -> u32 {
    return (batch * params.kv_heads + kv_head) * params.seq_len * params.head_dim;
}

fn q_row_base(batch: u32, q_head: u32, position: u32) -> u32 {
    return q_head_base(batch, q_head) + position * params.head_dim;
}

fn kv_row_base(batch: u32, kv_head: u32, position: u32) -> u32 {
    return kv_head_base(batch, kv_head) + position * params.head_dim;
}

fn mapped_kv_head(q_head: u32) -> u32 {
    return q_head / (params.q_heads / params.kv_heads);
}

fn score(batch: u32, q_head: u32, query_pos: u32, key_pos: u32, scale: f32) -> f32 {
    let kv_head = mapped_kv_head(q_head);
    let qb = q_row_base(batch, q_head, query_pos);
    let kb = kv_row_base(batch, kv_head, key_pos);
    var dot = 0.0;
    var d = 0u;
    loop {
        if (d >= params.head_dim) { break; }
        dot += packed_forward[q_offset() + qb + d] * packed_forward[k_offset() + kb + d];
        d += 1u;
    }
    return dot * scale;
}

fn probability(batch: u32, q_head: u32, query_pos: u32, key_pos: u32, scale: f32) -> f32 {
    let lse_index = (batch * params.q_heads + q_head) * params.seq_len + query_pos;
    return exp(score(batch, q_head, query_pos, key_pos, scale) - packed_forward[lse_offset() + lse_index]);
}

fn delta(batch: u32, q_head: u32, query_pos: u32) -> f32 {
    let qb = q_row_base(batch, q_head, query_pos);
    var value = 0.0;
    var d = 0u;
    loop {
        if (d >= params.head_dim) { break; }
        value += packed_forward[do_offset() + qb + d] * packed_forward[o_offset() + qb + d];
        d += 1u;
    }
    return value;
}

fn d_probability(batch: u32, q_head: u32, query_pos: u32, key_pos: u32) -> f32 {
    let kv_head = mapped_kv_head(q_head);
    let qb = q_row_base(batch, q_head, query_pos);
    let vb = kv_row_base(batch, kv_head, key_pos);
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
fn flat_attention_backward_grouped(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gradient_elems = params.q_elems + 2u * params.kv_elems;
    let global_index = global_id.x;
    if (global_index >= gradient_elems || params.q_elems == 0u || params.kv_elems == 0u || params.lse_elems == 0u || params.kv_heads == 0u || params.q_heads == 0u || (params.q_heads % params.kv_heads) != 0u) {
        return;
    }

    let scale = bitcast<f32>(params.scale_bits);
    let q_head_stride = params.seq_len * params.head_dim;
    let kv_head_stride = params.seq_len * params.head_dim;

    if (global_index < params.q_elems) {
        // dQ[q,d] = sum_k dS[q,k] * scale * K[k,d]
        let local_index = global_index;
        let batch_q_head = local_index / q_head_stride;
        let batch = batch_q_head / params.q_heads;
        let q_head = batch_q_head - batch * params.q_heads;
        let within_head = local_index - batch_q_head * q_head_stride;
        let query_pos = within_head / params.head_dim;
        let dim = within_head - query_pos * params.head_dim;
        let kv_head = mapped_kv_head(q_head);
        let key_limit = select(params.seq_len, query_pos + 1u, params.causal != 0u);
        let delta_q = delta(batch, q_head, query_pos);
        var gradient = 0.0;
        var key_pos = 0u;
        loop {
            if (key_pos >= key_limit) { break; }
            let p = probability(batch, q_head, query_pos, key_pos, scale);
            let ds = p * (d_probability(batch, q_head, query_pos, key_pos) - delta_q);
            let kb = kv_row_base(batch, kv_head, key_pos);
            gradient += ds * scale * packed_forward[k_offset() + kb + dim];
            key_pos += 1u;
        }
        packed_grads[dq_offset() + local_index] = gradient;
        return;
    }

    let kv_global = global_index - params.q_elems;
    let kind = kv_global / params.kv_elems;
    let local_index = kv_global - kind * params.kv_elems;
    let batch_kv_head = local_index / kv_head_stride;
    let batch = batch_kv_head / params.kv_heads;
    let kv_head = batch_kv_head - batch * params.kv_heads;
    let within_head = local_index - batch_kv_head * kv_head_stride;
    let key_pos = within_head / params.head_dim;
    let dim = within_head - key_pos * params.head_dim;
    let group_size = params.q_heads / params.kv_heads;
    let first_q_head = kv_head * group_size;
    let last_q_head = first_q_head + group_size;
    var gradient = 0.0;
    var q_head = first_q_head;
    loop {
        if (q_head >= last_q_head) { break; }
        var query_pos = 0u;
        loop {
            if (query_pos >= params.seq_len) { break; }
            if (params.causal == 0u || key_pos <= query_pos) {
                let p = probability(batch, q_head, query_pos, key_pos, scale);
                let qb = q_row_base(batch, q_head, query_pos);
                if (kind == 0u) {
                    let ds = p * (d_probability(batch, q_head, query_pos, key_pos) - delta(batch, q_head, query_pos));
                    gradient += ds * scale * packed_forward[q_offset() + qb + dim];
                } else {
                    gradient += p * packed_forward[do_offset() + qb + dim];
                }
            }
            query_pos += 1u;
        }
        q_head += 1u;
    }

    if (kind == 0u) {
        packed_grads[dk_offset() + local_index] = gradient;
    } else {
        packed_grads[dv_offset() + local_index] = gradient;
    }
}
