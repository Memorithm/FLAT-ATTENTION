// FLAT-ATTENTION M58 Q1 vec4 memory specialization for MHA.
//
// Contract: head_dim is 64 or 128, therefore every logical row is aligned to
// four f32 values. One workgroup computes exactly one query row. Q is held in
// registers (each lane owns one vec4 slice of the query row) instead of being
// staged through workgroup memory; K/V tiles are staged through workgroup
// memory with vec4 storage loads. This removes the Q4 multi-row tiling and
// reduction multiplexing that dominates small-head-count MHA prefill.
//
// Output and LSE keep the existing packed scalar layout [O | LSE].

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

@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out_and_lse: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> k_shared: array<f32, 1024>;
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

    if (query_pos >= params.seq_len || bh >= params.batch_heads ||
        (params.head_dim != 64u && params.head_dim != 128u)) {
        return;
    }

    let scale = bitcast<f32>(params.scale_bits);
    let head_dim4 = params.head_dim / 4u;
    let head_stride = params.seq_len * params.head_dim;
    let head_base = bh * head_stride;
    let head_stride4 = params.seq_len * head_dim4;
    let head_base4 = bh * head_stride4;
    let output_elems = params.batch_heads * head_stride;
    let query_base4 = head_base4 + query_pos * head_dim4;

    // Register-resident Q slice: lanes beyond head_dim4 hold an empty vec4.
    var q_reg = vec4<f32>(0.0);
    if (lane < head_dim4) {
        q_reg = q[query_base4 + lane];
    }

    // Register-resident output accumulator for this lane's vec4 slice.
    var acc = vec4<f32>(0.0);

    if (lane == 0u) {
        running_max_shared = NEG_MAX_F32;
        running_sum_shared = 0.0;
    }
    workgroupBarrier();

    let kv_limit = select(params.seq_len, query_pos + 1u, params.causal != 0u);

    var tile_start = 0u;
    loop {
        if (tile_start >= kv_limit) {
            break;
        }
        let tile_rows = min(KV_TILE, kv_limit - tile_start);
        let tile_vec4 = tile_rows * head_dim4;

        // Vectorized K/V staging: each storage expression loads four scalars.
        var linear4 = lane;
        loop {
            if (linear4 >= tile_vec4) {
                break;
            }
            let tile_row = linear4 / head_dim4;
            let vec_col = linear4 - tile_row * head_dim4;
            let global4 = head_base4 + (tile_start + tile_row) * head_dim4 + vec_col;
            let shared_base = tile_row * MAX_HEAD_DIM + vec_col * 4u;
            let packed_k = k[global4];
            let packed_v = v[global4];
            k_shared[shared_base] = packed_k.x;
            k_shared[shared_base + 1u] = packed_k.y;
            k_shared[shared_base + 2u] = packed_k.z;
            k_shared[shared_base + 3u] = packed_k.w;
            v_shared[shared_base] = packed_v.x;
            v_shared[shared_base + 1u] = packed_v.y;
            v_shared[shared_base + 2u] = packed_v.z;
            v_shared[shared_base + 3u] = packed_v.w;
            linear4 += WORKGROUP_SIZE;
        }
        workgroupBarrier();

        var tile_row = 0u;
        loop {
            if (tile_row >= tile_rows) {
                break;
            }
            let key_pos = tile_start + tile_row;
            if (params.causal != 0u && key_pos > query_pos) {
                break;
            }
            let shared_row = tile_row * MAX_HEAD_DIM;
            let shared_vec = shared_row / 4u + lane;

            var partial = 0.0;
            if (lane < head_dim4) {
                let k_slice = vec4<f32>(
                    k_shared[shared_vec * 4u],
                    k_shared[shared_vec * 4u + 1u],
                    k_shared[shared_vec * 4u + 2u],
                    k_shared[shared_vec * 4u + 3u],
                );
                partial = dot(q_reg, k_slice);
            }
            reduce_shared[lane] = partial;
            workgroupBarrier();

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
                let alpha = select(
                    exp(previous_max - new_max),
                    0.0,
                    running_sum_shared == 0.0,
                );
                let p = exp(score - new_max);
                running_max_shared = new_max;
                running_sum_shared = running_sum_shared * alpha + p;
                alpha_shared = alpha;
                p_shared = p;
            }
            workgroupBarrier();

            if (lane < head_dim4) {
                let v_slice = vec4<f32>(
                    v_shared[shared_vec * 4u],
                    v_shared[shared_vec * 4u + 1u],
                    v_shared[shared_vec * 4u + 2u],
                    v_shared[shared_vec * 4u + 3u],
                );
                acc = acc * alpha_shared + p_shared * v_slice;
            }
            workgroupBarrier();
            tile_row += 1u;
        }

        workgroupBarrier();
        tile_start += KV_TILE;
    }

    let query_base = head_base + query_pos * params.head_dim;
    if (lane < head_dim4) {
        out_and_lse[query_base + lane * 4u] = acc.x / running_sum_shared;
        out_and_lse[query_base + lane * 4u + 1u] = acc.y / running_sum_shared;
        out_and_lse[query_base + lane * 4u + 2u] = acc.z / running_sum_shared;
        out_and_lse[query_base + lane * 4u + 3u] = acc.w / running_sum_shared;
    }
    if (lane == 0u) {
        let lse_index = output_elems + bh * params.seq_len + query_pos;
        out_and_lse[lse_index] = running_max_shared + log(running_sum_shared);
    }
}
