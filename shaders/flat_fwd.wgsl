// FLAT-ATTENTION portable fused forward kernel — Q4 tiled generation.
//
// One workgroup computes up to four consecutive query rows. Each K/V tile is
// staged once and reused by all four queries, reducing logical K/V storage loads
// relative to the qualified single-row M2/M3 baseline. Scores are consumed by
// online softmax and no N x N score/probability matrix is materialized.
//
// Dispatch contract:
//   x = ceil(seq_len / QUERY_ROWS)
//   y = batch_heads
//   z = 1

const WORKGROUP_SIZE: u32 = 64u;
const QUERY_ROWS: u32 = 4u;
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

var<workgroup> q_shared: array<f32, 512>;       // QUERY_ROWS * MAX_HEAD_DIM
var<workgroup> k_shared: array<f32, 1024>;      // KV_TILE * MAX_HEAD_DIM
var<workgroup> v_shared: array<f32, 1024>;
var<workgroup> reduce_shared: array<f32, 256>;  // QUERY_ROWS * WORKGROUP_SIZE
var<workgroup> running_max_shared: array<f32, 4>;
var<workgroup> running_sum_shared: array<f32, 4>;
var<workgroup> alpha_shared: array<f32, 4>;
var<workgroup> p_shared: array<f32, 4>;

@compute @workgroup_size(64, 1, 1)
fn flat_attention_forward(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let query_start = workgroup_id.x * QUERY_ROWS;
    let bh = workgroup_id.y;
    let lane = local_id.x;

    if (query_start >= params.seq_len || bh >= params.batch_heads || params.head_dim == 0u || params.head_dim > MAX_HEAD_DIM) {
        return;
    }

    let scale = bitcast<f32>(params.scale_bits);
    let head_stride = params.seq_len * params.head_dim;
    let head_base = bh * head_stride;
    let output_elems = params.batch_heads * head_stride;
    let d0 = lane;
    let d1 = lane + WORKGROUP_SIZE;

    var acc00 = 0.0;
    var acc01 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;
    var acc20 = 0.0;
    var acc21 = 0.0;
    var acc30 = 0.0;
    var acc31 = 0.0;

    // Stage up to four Q rows once. Invalid tail rows remain unused.
    var qr = 0u;
    loop {
        if (qr >= QUERY_ROWS) {
            break;
        }
        let query_pos = query_start + qr;
        let q_shared_base = qr * MAX_HEAD_DIM;
        if (query_pos < params.seq_len) {
            let query_base = head_base + query_pos * params.head_dim;
            if (d0 < params.head_dim) {
                q_shared[q_shared_base + d0] = q[query_base + d0];
            }
            if (d1 < params.head_dim) {
                q_shared[q_shared_base + d1] = q[query_base + d1];
            }
        }
        if (lane == 0u) {
            running_max_shared[qr] = NEG_MAX_F32;
            running_sum_shared[qr] = 0.0;
            alpha_shared[qr] = 1.0;
            p_shared[qr] = 0.0;
        }
        qr += 1u;
    }
    workgroupBarrier();

    // In causal mode, no query in this four-row tile can observe a key beyond
    // the final valid query row. This avoids staging fully-future K/V tiles.
    let query_limit = min(params.seq_len, query_start + QUERY_ROWS);
    let kv_limit = select(params.seq_len, query_limit, params.causal != 0u);

    var tile_start = 0u;
    loop {
        if (tile_start >= kv_limit) {
            break;
        }
        let tile_rows = min(KV_TILE, kv_limit - tile_start);
        let tile_elements = tile_rows * params.head_dim;

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
            let shared_row = tile_row * MAX_HEAD_DIM;

            qr = 0u;
            loop {
                if (qr >= QUERY_ROWS) {
                    break;
                }
                let query_pos = query_start + qr;
                let valid_query = query_pos < params.seq_len;
                let active = valid_query && (params.causal == 0u || key_pos <= query_pos);
                let q_shared_base = qr * MAX_HEAD_DIM;
                let reduce_base = qr * WORKGROUP_SIZE;

                var partial = 0.0;
                if (active && d0 < params.head_dim) {
                    partial += q_shared[q_shared_base + d0] * k_shared[shared_row + d0];
                }
                if (active && d1 < params.head_dim) {
                    partial += q_shared[q_shared_base + d1] * k_shared[shared_row + d1];
                }
                reduce_shared[reduce_base + lane] = partial;
                workgroupBarrier();

                var offset = 32u;
                loop {
                    if (offset == 0u) {
                        break;
                    }
                    if (lane < offset) {
                        reduce_shared[reduce_base + lane] += reduce_shared[reduce_base + lane + offset];
                    }
                    workgroupBarrier();
                    offset = offset / 2u;
                }

                if (lane == 0u) {
                    if (active) {
                        let score = reduce_shared[reduce_base] * scale;
                        let previous_max = running_max_shared[qr];
                        let new_max = max(previous_max, score);
                        let alpha = select(
                            exp(previous_max - new_max),
                            0.0,
                            running_sum_shared[qr] == 0.0,
                        );
                        let p = exp(score - new_max);
                        running_max_shared[qr] = new_max;
                        running_sum_shared[qr] = running_sum_shared[qr] * alpha + p;
                        alpha_shared[qr] = alpha;
                        p_shared[qr] = p;
                    } else {
                        alpha_shared[qr] = 1.0;
                        p_shared[qr] = 0.0;
                    }
                }
                workgroupBarrier();

                let alpha = alpha_shared[qr];
                let p = p_shared[qr];
                if (qr == 0u) {
                    if (d0 < params.head_dim) {
                        acc00 = acc00 * alpha + p * v_shared[shared_row + d0];
                    }
                    if (d1 < params.head_dim) {
                        acc01 = acc01 * alpha + p * v_shared[shared_row + d1];
                    }
                } else if (qr == 1u) {
                    if (d0 < params.head_dim) {
                        acc10 = acc10 * alpha + p * v_shared[shared_row + d0];
                    }
                    if (d1 < params.head_dim) {
                        acc11 = acc11 * alpha + p * v_shared[shared_row + d1];
                    }
                } else if (qr == 2u) {
                    if (d0 < params.head_dim) {
                        acc20 = acc20 * alpha + p * v_shared[shared_row + d0];
                    }
                    if (d1 < params.head_dim) {
                        acc21 = acc21 * alpha + p * v_shared[shared_row + d1];
                    }
                } else {
                    if (d0 < params.head_dim) {
                        acc30 = acc30 * alpha + p * v_shared[shared_row + d0];
                    }
                    if (d1 < params.head_dim) {
                        acc31 = acc31 * alpha + p * v_shared[shared_row + d1];
                    }
                }
                workgroupBarrier();
                qr += 1u;
            }
            tile_row += 1u;
        }

        workgroupBarrier();
        tile_start += KV_TILE;
    }

    // Normalize and store each valid query row.
    qr = 0u;
    loop {
        if (qr >= QUERY_ROWS) {
            break;
        }
        let query_pos = query_start + qr;
        if (query_pos < params.seq_len) {
            let query_base = head_base + query_pos * params.head_dim;
            let inv_sum = 1.0 / running_sum_shared[qr];
            if (qr == 0u) {
                if (d0 < params.head_dim) { out_and_lse[query_base + d0] = acc00 * inv_sum; }
                if (d1 < params.head_dim) { out_and_lse[query_base + d1] = acc01 * inv_sum; }
            } else if (qr == 1u) {
                if (d0 < params.head_dim) { out_and_lse[query_base + d0] = acc10 * inv_sum; }
                if (d1 < params.head_dim) { out_and_lse[query_base + d1] = acc11 * inv_sum; }
            } else if (qr == 2u) {
                if (d0 < params.head_dim) { out_and_lse[query_base + d0] = acc20 * inv_sum; }
                if (d1 < params.head_dim) { out_and_lse[query_base + d1] = acc21 * inv_sum; }
            } else {
                if (d0 < params.head_dim) { out_and_lse[query_base + d0] = acc30 * inv_sum; }
                if (d1 < params.head_dim) { out_and_lse[query_base + d1] = acc31 * inv_sum; }
            }
            if (lane == 0u) {
                let lse_index = output_elems + bh * params.seq_len + query_pos;
                out_and_lse[lse_index] = running_max_shared[qr] + log(running_sum_shared[qr]);
            }
        }
        qr += 1u;
    }
}
