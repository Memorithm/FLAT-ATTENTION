// FLAT-ATTENTION portable fused forward kernel.
//
// Properties:
// - Q/K/V/O layout: [batch_heads, sequence, head_dim], contiguous row-major.
// - One workgroup computes one query row.
// - K and V are staged in workgroup memory in tiles of 8 rows.
// - Scores are consumed immediately by an online softmax; no N x N score or
//   probability matrix is ever materialized in storage memory.
// - O and LSE share one output storage buffer so the kernel requires only four
//   storage bindings (Q, K, V, OUT), compatible with portable downlevel limits.
// - head_dim is runtime-configurable from 1..128.
// - f32 is used end-to-end in this portable baseline.
//
// Output storage layout:
//   out_and_lse[0 .. tensor_elems] = O
//   out_and_lse[tensor_elems .. tensor_elems + batch_heads * seq_len] = LSE
//
// Dispatch contract:
//   x = seq_len
//   y = batch_heads
//   z = 1

const WORKGROUP_SIZE: u32 = 64u;
const KV_TILE: u32 = 8u;
const MAX_HEAD_DIM: u32 = 128u;
const NEG_MAX_F32: f32 = -3.402823466e38;

struct Params {
    seq_len: u32,
    head_dim: u32,
    batch_heads: u32,
    causal: u32,
    scale_bits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_and_lse: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> q_shared: array<f32, 128>;
var<workgroup> k_shared: array<f32, 1024>; // KV_TILE * MAX_HEAD_DIM
var<workgroup> v_shared: array<f32, 1024>;
var<workgroup> reduce_shared: array<f32, 64>;
var<workgroup> running_max_shared: f32;
var<workgroup> running_sum_shared: f32;
var<workgroup> alpha_shared: f32;
var<workgroup> p_shared: f32;

@compute @workgroup_size(64, 1, 1)
fn flat_attention_forward(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let query_pos = workgroup_id.x;
    let bh = workgroup_id.y;
    let lane = local_id.x;

    if (query_pos >= params.seq_len || bh >= params.batch_heads || params.head_dim == 0u || params.head_dim > MAX_HEAD_DIM) {
        return;
    }

    let scale = bitcast<f32>(params.scale_bits);
    let head_stride = params.seq_len * params.head_dim;
    let head_base = bh * head_stride;
    let query_base = head_base + query_pos * params.head_dim;
    let output_elems = params.batch_heads * head_stride;

    // Each lane owns at most two output dimensions because MAX_HEAD_DIM = 128
    // and WORKGROUP_SIZE = 64.
    let d0 = lane;
    let d1 = lane + WORKGROUP_SIZE;
    var acc0 = 0.0;
    var acc1 = 0.0;

    if (d0 < params.head_dim) {
        q_shared[d0] = q[query_base + d0];
    }
    if (d1 < params.head_dim) {
        q_shared[d1] = q[query_base + d1];
    }

    if (lane == 0u) {
        running_max_shared = NEG_MAX_F32;
        running_sum_shared = 0.0;
    }
    workgroupBarrier();

    var tile_start = 0u;
    loop {
        if (tile_start >= params.seq_len) {
            break;
        }
        let tile_rows = min(KV_TILE, params.seq_len - tile_start);
        let tile_elements = tile_rows * params.head_dim;

        // Cooperative K/V staging. Workgroup layout is padded to MAX_HEAD_DIM
        // so indexing stays simple and deterministic for every head_dim.
        var linear = lane;
        loop {
            if (linear >= tile_elements) {
                break;
            }
            let tile_row = linear / params.head_dim;
            let dim = linear - tile_row * params.head_dim;
            let global_index = head_base + (tile_start + tile_row) * params.head_dim + dim;
            let shared_index = tile_row * MAX_HEAD_DIM + dim;
            k_shared[shared_index] = k[global_index];
            v_shared[shared_index] = v[global_index];
            linear += WORKGROUP_SIZE;
        }
        workgroupBarrier();

        var tile_row = 0u;
        loop {
            if (tile_row >= tile_rows) {
                break;
            }
            let key_pos = tile_start + tile_row;

            // Causal rows can stop once the tile reaches future keys. The
            // condition is workgroup-uniform because query/key positions are.
            if (params.causal != 0u && key_pos > query_pos) {
                break;
            }

            let shared_row = tile_row * MAX_HEAD_DIM;
            var partial = 0.0;
            if (d0 < params.head_dim) {
                partial += q_shared[d0] * k_shared[shared_row + d0];
            }
            if (d1 < params.head_dim) {
                partial += q_shared[d1] * k_shared[shared_row + d1];
            }
            reduce_shared[lane] = partial;
            workgroupBarrier();

            // Fixed tree reduction: deterministic within this kernel.
            var offset = 32u;
            loop {
                if (offset == 0u) {
                    break;
                }
                if (lane < offset) {
                    reduce_shared[lane] += reduce_shared[lane + offset];
                }
                workgroupBarrier();
                offset = offset / 2u;
            }

            if (lane == 0u) {
                let score = reduce_shared[0] * scale;
                let previous_max = running_max_shared;
                let new_max = max(previous_max, score);
                let alpha = select(exp(previous_max - new_max), 0.0, running_sum_shared == 0.0);
                let p = exp(score - new_max);
                running_max_shared = new_max;
                running_sum_shared = running_sum_shared * alpha + p;
                alpha_shared = alpha;
                p_shared = p;
            }
            workgroupBarrier();

            if (d0 < params.head_dim) {
                acc0 = acc0 * alpha_shared + p_shared * v_shared[shared_row + d0];
            }
            if (d1 < params.head_dim) {
                acc1 = acc1 * alpha_shared + p_shared * v_shared[shared_row + d1];
            }
            workgroupBarrier();
            tile_row += 1u;
        }

        workgroupBarrier();
        tile_start += KV_TILE;
    }

    let inv_sum = 1.0 / running_sum_shared;
    if (d0 < params.head_dim) {
        out_and_lse[query_base + d0] = acc0 * inv_sum;
    }
    if (d1 < params.head_dim) {
        out_and_lse[query_base + d1] = acc1 * inv_sum;
    }
    if (lane == 0u) {
        let lse_index = output_elems + bh * params.seq_len + query_pos;
        out_and_lse[lse_index] = running_max_shared + log(running_sum_shared);
    }
}
