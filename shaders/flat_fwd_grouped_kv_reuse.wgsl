// Phase O native GQA/MQA candidate: reuse each staged K/V tile across two
// query heads from the same physical KV group. The dispatch Y axis addresses
// (batch, kv_head, pair-within-group), so K/V are loaded once per Q-head pair.

const WORKGROUP_SIZE: u32 = 64u;
const QUERY_ROWS: u32 = 4u;
const Q_HEADS_PER_TILE: u32 = 2u;
const STATE_ROWS: u32 = QUERY_ROWS * Q_HEADS_PER_TILE;
const KV_TILE: u32 = 8u;
const MAX_HEAD_DIM: u32 = 128u;
const NEG_MAX_F32: f32 = -3.402823466e38;

struct Params {
    seq_len: u32,
    head_dim: u32,
    q_heads: u32,
    kv_heads: u32,
    batch: u32,
    causal: u32,
    scale_bits: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> k: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> v: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> out_and_lse: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> q_shared: array<f32, 1024>;
var<workgroup> k_shared: array<f32, 1024>;
var<workgroup> v_shared: array<f32, 1024>;
var<workgroup> reduce_shared: array<f32, 512>;
var<workgroup> running_max_shared: array<f32, 8>;
var<workgroup> running_sum_shared: array<f32, 8>;
var<workgroup> alpha_shared: array<f32, 8>;
var<workgroup> p_shared: array<f32, 8>;

@compute @workgroup_size(64, 1, 1)
fn flat_attention_forward(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let query_start = workgroup_id.x * QUERY_ROWS;
    let lane = local_id.x;

    if (query_start >= params.seq_len ||
        (params.head_dim != 64u && params.head_dim != 128u) ||
        params.q_heads == 0u || params.kv_heads == 0u) {
        return;
    }

    let group_size = params.q_heads / params.kv_heads;
    if (group_size < 2u) {
        return;
    }
    let group_tiles = (group_size + Q_HEADS_PER_TILE - 1u) / Q_HEADS_PER_TILE;
    let batch_kv = workgroup_id.y / group_tiles;
    let group_tile = workgroup_id.y - batch_kv * group_tiles;
    let batch_index = batch_kv / params.kv_heads;
    let kv_head = batch_kv - batch_index * params.kv_heads;
    if (batch_index >= params.batch || kv_head >= params.kv_heads) {
        return;
    }

    let q_group_start = kv_head * group_size;
    let q_head_start = q_group_start + group_tile * Q_HEADS_PER_TILE;
    let q_head_count = min(Q_HEADS_PER_TILE, q_group_start + group_size - q_head_start);
    let scale = bitcast<f32>(params.scale_bits);
    let head_stride = params.seq_len * params.head_dim;
    let q_batch_heads = params.batch * params.q_heads;
    let output_elems = q_batch_heads * head_stride;
    let head_dim4 = params.head_dim / 4u;
    let head_stride4 = params.seq_len * head_dim4;
    let kv_head_base4 = (batch_index * params.kv_heads + kv_head) * head_stride4;
    let d0 = lane;
    let d1 = lane + WORKGROUP_SIZE;

    var accumulators: array<f32, 16>;
    let q_tile_vec4 = q_head_count * QUERY_ROWS * head_dim4;
    var q_iter = 0u;
    loop {
        if (q_iter >= 4u) { break; }
        let q_linear4 = lane + q_iter * WORKGROUP_SIZE;
        if (q_linear4 < q_tile_vec4) {
            let q_head_local_stride4 = QUERY_ROWS * head_dim4;
            let q_head_local = q_linear4 / q_head_local_stride4;
            let within_head4 = q_linear4 - q_head_local * q_head_local_stride4;
            let qr = within_head4 / head_dim4;
            let vec_col = within_head4 - qr * head_dim4;
            let query_pos = query_start + qr;
            if (query_pos < params.seq_len) {
                let q_head = q_head_start + q_head_local;
                let q_head_base4 = (batch_index * params.q_heads + q_head) * head_stride4;
                let packed = q[q_head_base4 + query_pos * head_dim4 + vec_col];
                let shared_base =
                    (q_head_local * QUERY_ROWS + qr) * MAX_HEAD_DIM + vec_col * 4u;
                q_shared[shared_base] = packed.x;
                q_shared[shared_base + 1u] = packed.y;
                q_shared[shared_base + 2u] = packed.z;
                q_shared[shared_base + 3u] = packed.w;
            }
        }
        q_iter += 1u;
    }

    if (lane == 0u) {
        var state = 0u;
        loop {
            if (state >= STATE_ROWS) { break; }
            running_max_shared[state] = NEG_MAX_F32;
            running_sum_shared[state] = 0.0;
            alpha_shared[state] = 1.0;
            p_shared[state] = 0.0;
            state += 1u;
        }
    }
    workgroupBarrier();

    let query_limit = min(params.seq_len, query_start + QUERY_ROWS);
    let kv_limit = select(params.seq_len, query_limit, params.causal != 0u);
    var tile_start = 0u;
    loop {
        if (tile_start >= kv_limit) { break; }
        let tile_rows = min(KV_TILE, kv_limit - tile_start);
        let tile_vec4 = tile_rows * head_dim4;

        var kv_iter = 0u;
        loop {
            if (kv_iter >= 4u) { break; }
            let linear4 = lane + kv_iter * WORKGROUP_SIZE;
            if (linear4 < tile_vec4) {
                let tile_row = linear4 / head_dim4;
                let vec_col = linear4 - tile_row * head_dim4;
                let global4 = kv_head_base4 + (tile_start + tile_row) * head_dim4 + vec_col;
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
            }
            kv_iter += 1u;
        }
        workgroupBarrier();

        var tile_row = 0u;
        loop {
            if (tile_row >= tile_rows) { break; }
            let key_pos = tile_start + tile_row;
            let shared_row = tile_row * MAX_HEAD_DIM;
            var q_head_local = 0u;
            loop {
                if (q_head_local >= q_head_count) { break; }
                var qr = 0u;
                loop {
                    if (qr >= QUERY_ROWS) { break; }
                    let state = q_head_local * QUERY_ROWS + qr;
                    let query_pos = query_start + qr;
                    let participates = query_pos < params.seq_len &&
                        (params.causal == 0u || key_pos <= query_pos);
                    let q_shared_base = state * MAX_HEAD_DIM;
                    let reduce_base = state * WORKGROUP_SIZE;
                    var partial = 0.0;
                    if (participates && d0 < params.head_dim) {
                        partial += q_shared[q_shared_base + d0] * k_shared[shared_row + d0];
                    }
                    if (participates && d1 < params.head_dim) {
                        partial += q_shared[q_shared_base + d1] * k_shared[shared_row + d1];
                    }
                    reduce_shared[reduce_base + lane] = partial;
                    workgroupBarrier();

                    var offset = 32u;
                    loop {
                        if (offset == 0u) { break; }
                        if (lane < offset) {
                            reduce_shared[reduce_base + lane] +=
                                reduce_shared[reduce_base + lane + offset];
                        }
                        workgroupBarrier();
                        offset = offset / 2u;
                    }

                    if (lane == 0u) {
                        if (participates) {
                            let score = reduce_shared[reduce_base] * scale;
                            let previous_max = running_max_shared[state];
                            let new_max = max(previous_max, score);
                            let alpha = select(
                                exp(previous_max - new_max),
                                0.0,
                                running_sum_shared[state] == 0.0,
                            );
                            let probability = exp(score - new_max);
                            running_max_shared[state] = new_max;
                            running_sum_shared[state] =
                                running_sum_shared[state] * alpha + probability;
                            alpha_shared[state] = alpha;
                            p_shared[state] = probability;
                        } else {
                            alpha_shared[state] = 1.0;
                            p_shared[state] = 0.0;
                        }
                    }
                    workgroupBarrier();

                    let alpha = alpha_shared[state];
                    let probability = p_shared[state];
                    let acc_base = state * 2u;
                    if (d0 < params.head_dim) {
                        accumulators[acc_base] = accumulators[acc_base] * alpha +
                            probability * v_shared[shared_row + d0];
                    }
                    if (d1 < params.head_dim) {
                        accumulators[acc_base + 1u] = accumulators[acc_base + 1u] * alpha +
                            probability * v_shared[shared_row + d1];
                    }
                    workgroupBarrier();
                    qr += 1u;
                }
                q_head_local += 1u;
            }
            tile_row += 1u;
        }
        workgroupBarrier();
        tile_start += KV_TILE;
    }

    var q_head_local = 0u;
    loop {
        if (q_head_local >= q_head_count) { break; }
        let q_head = q_head_start + q_head_local;
        let q_head_base = (batch_index * params.q_heads + q_head) * head_stride;
        var qr = 0u;
        loop {
            if (qr >= QUERY_ROWS) { break; }
            let query_pos = query_start + qr;
            if (query_pos < params.seq_len) {
                let state = q_head_local * QUERY_ROWS + qr;
                let acc_base = state * 2u;
                let query_base = q_head_base + query_pos * params.head_dim;
                let inv_sum = 1.0 / running_sum_shared[state];
                if (d0 < params.head_dim) {
                    out_and_lse[query_base + d0] = accumulators[acc_base] * inv_sum;
                }
                if (d1 < params.head_dim) {
                    out_and_lse[query_base + d1] = accumulators[acc_base + 1u] * inv_sum;
                }
                if (lane == 0u) {
                    let q_batch_head = batch_index * params.q_heads + q_head;
                    let lse_index = output_elems + q_batch_head * params.seq_len + query_pos;
                    out_and_lse[lse_index] =
                        running_max_shared[state] + log(running_sum_shared[state]);
                }
            }
            qr += 1u;
        }
        q_head_local += 1u;
    }
}
